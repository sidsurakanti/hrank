"""
HRank reproduction on chenyaofo's pretrained resnet56 + CIFAR-10

NOTE: chenyaofo's `BasicBlock` reuses one `relu` module for both activations, so we
can't hook it directly per-site the way we hook VGG's distinct ReLU modules.

instead we reach each activation from its neighboring layer:
  - conv1's rank comes from a forward-pre-hook on conv2 (conv2's input is
    exactly conv1's post-relu activation)
  - conv2's rank comes from a forward-hook on the block itself (the block's
    output is exactly conv2's post-residual-relu activation)
"""

import os

import torch
import hrank_core as core

ARCH = "cifar10_resnet56"

# HRank's published compress-rate schedule for resnet_56 (github.com/lmbxmu/HRank/README.md)
COMPRESS_RATE = [0.1] + [0.60] * 35 + [0.0] * 2 + [0.6] * 6 + [0.4] * 3 + [0.1, 0.4] * 4
RANK_BATCHES = 10
FINETUNE_EPOCHS = int(os.environ.get("FINETUNE_EPOCHS", 15))  # HRank's own default epochs per layer


def main():
    train_loader, test_loader = core.get_loaders()

    print(f"== loading pretrained {ARCH} ==")
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        ARCH,
        pretrained=True,
        trust_repo=True,  # pyright: ignore[reportArgumentType]
    )
    model = model.to(core.DEVICE)

    baseline_acc = core.evaluate(model, test_loader)
    print(f"pretrained baseline accuracy: {baseline_acc:.2f}%")

    blocks = [
        block for layer in (model.layer1, model.layer2, model.layer3) for block in layer
    ]
    assert len(COMPRESS_RATE) == 1 + 2 * len(blocks)

    # flatten layers
    convs = [model.conv1] + [c for block in blocks for c in (block.conv1, block.conv2)]
    bns = [model.bn1] + [b for block in blocks for b in (block.bn1, block.bn2)]
    registrars = [lambda acc: model.relu.register_forward_hook(acc.hook)]

    for block in blocks:
        registrars.append(
            lambda acc, c2=block.conv2: c2.register_forward_pre_hook(acc.pre_hook)
        )
        registrars.append(lambda acc, b=block: b.register_forward_hook(acc.hook))

    print(
        f"\n== computing rank for all {len(convs)} conv layers from the pretrained model =="
    )
    ranks = core.compute_all_ranks(model, train_loader, RANK_BATCHES, registrars)

    core.prune_and_finetune(
        model, convs, bns, ranks, COMPRESS_RATE, train_loader, FINETUNE_EPOCHS
    )

    final_acc = core.evaluate(model, test_loader)
    print(
        f"\nbaseline: {baseline_acc:.2f}%  ->  final ({len(convs)} layers pruned per HRank schedule): {final_acc:.2f}%"
    )


if __name__ == "__main__":
    main()
