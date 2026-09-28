"""Step 1: minimal end-to-end HRank pipeline on a tiny 3-conv net + real CIFAR-10.

Validates the three HRank primitives before we touch resnet-56/vgg16-bn:
  1. rank scoring   -- forward hook computes per-channel matrix rank of a
                        conv's post-ReLU output, averaged over a few batches
  2. masking        -- zero the lowest-rank output channels in a conv + its
                        paired BN (soft/multiplicative mask, shapes unchanged,
                        matching HRank's own approach -- not physical removal)
  3. the outer loop -- prune one conv layer at a time, fine-tune, move on,
                        reapplying every mask built so far after each step

Not chasing accuracy numbers here -- this is a toy net, the point is proving
the mechanism works end to end before scaling up.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMPRESS_RATE = 0.5  # prune 50% of channels per conv layer
RANK_BATCHES = 10  # batches used to estimate rank (HRank's `limit`)
BASELINE_EPOCHS = 50
FINETUNE_EPOCHS = 2  # per-layer finetune epochs (HRank uses ~15 on the real nets)


class SmallConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return self.fc(x)


def get_loaders(batch_size=128):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_tf
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_tf
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=2
    )
    return train_loader, test_loader


def evaluate(model, loader):
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100.0 * correct / total


def apply_mask(conv, bn, mask):
    mask = mask.to(conv.weight.device)

    with torch.no_grad():
        # zero out everything
        conv.weight.data *= mask.view(-1, 1, 1, 1)
        bn.weight.data *= mask
        bn.bias.data *= mask


# @param masks: for fine tuning; keep pruned channels at 0 during
def train_epochs(model, loader, epochs, masks=()):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            for conv, bn, mask in masks:
                apply_mask(
                    conv, bn, mask
                )  # keep pruned channels at zero (HRank's grad_mask)

        print(f"epoch {epoch + 1}/{epochs} done")


class RankAccumulator:
    """
    Running average of per-channel matrix rank across batches
    """

    def __init__(self):
        self.sum_rank: torch.Tensor | None = None
        self.total = 0

    def hook(self, module, inp, output):
        ranks = torch.linalg.matrix_rank(output.detach().float())  # (batch, channels)
        batch_sum = ranks.sum(0).cpu()  # each chan's rank summed across all batches

        self.sum_rank = (
            batch_sum if self.sum_rank is None else self.sum_rank + batch_sum
        )  # add to prev batch result's
        self.total += output.shape[0]

    def result(self):
        assert self.sum_rank is not None, "no batches were run through the hook"
        return (self.sum_rank / self.total).numpy()  # average rank per channel


def compute_rank(model, relu_layer, loader, num_batches):
    acc = RankAccumulator()
    handle = relu_layer.register_forward_hook(acc.hook)

    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= num_batches:
                break
            model(x.to(DEVICE))
    handle.remove()

    return acc.result()


def build_mask(rank, compress_rate):
    out_c = len(rank)
    pruned_num = int(compress_rate * out_c)
    order = (
        rank.argsort()
    )  # idx's ascending: lowest rank channels idx first = prune first

    mask = torch.ones(out_c)
    mask[order[:pruned_num]] = 0.0
    return mask


def main():
    train_loader, test_loader = get_loaders()
    model = SmallConvNet().to(DEVICE)

    print("== training baseline ==")
    train_epochs(model, train_loader, epochs=BASELINE_EPOCHS)

    baseline_acc = evaluate(model, test_loader)
    print(f"baseline accuracy: {baseline_acc:.2f}%")

    layers = [
        (model.conv1, model.bn1, model.relu1),
        (model.conv2, model.bn2, model.relu2),
        (model.conv3, model.bn3, model.relu3),
    ]

    masks = []
    for i, (conv, bn, relu) in enumerate(layers, start=1):
        print(f"\n== layer {i}: computing rank ==")

        rank = compute_rank(model, relu, train_loader, RANK_BATCHES)
        mask = build_mask(rank, COMPRESS_RATE)

        pruned = int(mask.numel() - mask.sum().item())
        print(
            f"pruning {pruned}/{mask.numel()}"  #  channels (lowest rank: {sorted(rank)[:pruned]})"
        )

        apply_mask(conv, bn, mask)  # zero out weight's according to mask
        masks.append((conv, bn, mask))

        # =========== finetuning ========
        print(f"== layer {i}: finetuning ==")
        train_epochs(model, train_loader, epochs=FINETUNE_EPOCHS, masks=masks)
        acc = evaluate(model, test_loader)
        print(f"accuracy after pruning+finetuning layer {i}: {acc:.2f}%")

    final_acc = evaluate(model, test_loader)
    print(
        f"\nbaseline: {baseline_acc:.2f}%  ->  final ({COMPRESS_RATE * 100:.0f}% pruned, 3 layers): {final_acc:.2f}%"
    )


if __name__ == "__main__":
    main()
