"""
Shared HRank primitives
=======================

- data loading (CIFAR-10)
- rank scoring
- masking
- per-layer prune -> finetune loop

* architecture-specific code (which convs to target, how to reach their activations)
  lives in vgg_prune.py / resnet_prune.py

--------------
tldr pseudocode

    STEP 1: GET MATRIX RANK
    -----
    for each conv layer, once, on the pretrained model:
        run a few batches through it
        => hook the layer's post-relu output
        => rank[layer] = average matrix_rank of each output channel across batches

    STEP 2: PRUNE (MASK) BASED ON LOWEST RANKED
    ------
    masks = []
    for each conv layer, in order (`compute_all_ranks` feeds `prune_and_finetune`):
        mask = zero out this layer's lowest-rank channels (+ its BN)
        masks.append(mask)
        finetune the whole network for a few epochs,
            reapplying every mask in `masks` after each optimizer step
            (so already-pruned channels can't drift back off zero)

    NOTE: this is all soft pruning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, test_loader


def evaluate(model, loader):
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100.0 * correct / total


def apply_mask(conv, bn, mask):
    mask = mask.to(conv.weight.device)
    with torch.no_grad():
        conv.weight.data *= mask.view(-1, 1, 1, 1)
        bn.weight.data *= mask
        bn.bias.data *= mask


def train_epochs(model, loader, epochs, masks=(), lr=0.01, lr_decay_step=(5, 10)):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(lr_decay_step), gamma=0.1
    )
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            for conv, bn, mask in masks:
                apply_mask(
                    conv, bn, mask
                )  # keep pruned channels at zero (HRank's grad_mask)
        scheduler.step()
        print(f"    epoch {epoch + 1}/{epochs} done")


class RankAccumulator:
    """
    Running average of per-channel matrix rank across batches, matching
    """

    def __init__(self):
        self.sum_rank: torch.Tensor | None = None
        self.total = 0

    def _accumulate(self, activation):
        ranks = torch.linalg.matrix_rank(
            activation.detach().float()
        )  # (batch, channels)
        batch_sum = ranks.sum(0).cpu()  # rank sum across batches
        self.sum_rank = (
            batch_sum if self.sum_rank is None else self.sum_rank + batch_sum
        )
        self.total += activation.shape[0]

    def hook(self, _module, _inp, output):  # for forward hook
        self._accumulate(output)

    def pre_hook(self, _module, inp):  # for pre-forward hooks
        self._accumulate(inp[0])

    def result(self):
        assert self.sum_rank is not None, "no batches were run through the hook yet"
        return (self.sum_rank / self.total).numpy()


def compute_all_ranks(model, loader, num_batches, registrars):
    """
    `registrars` is one `f(acc) -> handle` callable per target layer

    each registering acc.hook or acc.pre_hook on whatever module/site reaches that layer's activation
    (a plain forward hook for most layers, but e.g. resnet's shared per-block ReLU needs a pre-hook
    on the next conv instead).
    """
    accs = [RankAccumulator() for _ in registrars]
    handles = [register(acc) for register, acc in zip(registrars, accs)]

    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= num_batches:
                break
            model(x.to(DEVICE))
    for h in handles:
        h.remove()

    return [acc.result() for acc in accs]


def build_mask(rank, compress_rate):
    out_c = len(rank)
    pruned_num = int(compress_rate * out_c)
    order = rank.argsort()  # ascending: lowest rank first = prune first
    mask = torch.ones(out_c)
    mask[order[:pruned_num]] = 0.0
    return mask


def prune_and_finetune(
    model, convs, bns, ranks, compress_rate, loader, finetune_epochs
):
    """
    the outer loop for hrank:

    one conv layer at a time, prune it, fine-tune the whole network
    with every mask built so far reapplied after each step
    """

    masks = []
    for i, (conv, bn, rank, rate) in enumerate(
        zip(convs, bns, ranks, compress_rate), start=1
    ):
        n = len(convs)
        print(f"\n== LAYER {i}/{n}: PRUNING {rate * 100:.0f}% ==")
        mask = build_mask(rank, rate)
        pruned = int(mask.numel() - mask.sum().item())
        print(f"    pruning {pruned}/{mask.numel()} channels")

        apply_mask(conv, bn, mask)
        masks.append((conv, bn, mask))

        print(f"== LAYER {i}/{n}: FINETUNING ({finetune_epochs} epochs) ==")
        train_epochs(model, loader, epochs=finetune_epochs, masks=masks)
