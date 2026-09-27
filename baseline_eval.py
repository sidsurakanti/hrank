import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

ARCH = "cifar10_vgg16_bn"  # chenyaofo reports 94.16% top-1 for this one


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=2
    )

    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        ARCH,
        pretrained=True,
        trust_repo=True,  # pyright: ignore[reportArgumentType]
    )
    model = model.to(device).eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    print(f"Test accuracy: {acc:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    main()
