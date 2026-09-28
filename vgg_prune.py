"""
HRank reproduction on chenyaofo's pretrained vgg16_bn + CIFAR-10

NOTE:
only 12 of VGG-16's 13 conv layers are pruned
the last conv (right before the classifier) is left untouched in HRank's own vgg.py/mask.py.
"""

import torch
import hrank_core as core

ARCH = "cifar10_vgg16_bn"

VGG16_CFG = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    "M",
]
# from HRank repo's readme
COMPRESS_RATE = ([0.95] + [0.5] * 6 + [0.9] * 4 + [0.8] * 2)[:12]
RANK_BATCHES = 10  # batches of (augmented) train data used to estimate rank
FINETUNE_EPOCHS = 15  # HRank's own default epochs per layer


def conv_bn_relu_indices(cfg):
    """
    Positions of the Conv2d/BatchNorm2d/ReLU triplets inside chenyaofo's
    `features` Sequential, derived from the VGG-D cfg list
    """
    conv_idx, bn_idx, relu_idx = [], [], []
    idx = 0

    for v in cfg:
        if v == "M":
            idx += 1
        else:
            conv_idx.append(idx)
            bn_idx.append(idx + 1)
            relu_idx.append(idx + 2)
            idx += 3

    return conv_idx, bn_idx, relu_idx


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

    n_prune = len(COMPRESS_RATE)  # 12: HRank never prunes VGG-16's 13th/last conv
    conv_idx, bn_idx, relu_idx = (i[:n_prune] for i in conv_bn_relu_indices(VGG16_CFG))

    features = model.features
    convs = [features[i] for i in conv_idx]
    bns = [features[i] for i in bn_idx]
    relus = [features[i] for i in relu_idx]
    registrars = [
        lambda acc, r=relu: r.register_forward_hook(acc.hook) for relu in relus
    ]

    print(
        f"\n== computing rank for all {n_prune} conv layers from the pretrained model =="
    )
    ranks = core.compute_all_ranks(model, train_loader, RANK_BATCHES, registrars)

    core.prune_and_finetune(
        model, convs, bns, ranks, COMPRESS_RATE, train_loader, FINETUNE_EPOCHS
    )

    final_acc = core.evaluate(model, test_loader)
    print(
        f"\nbaseline: {baseline_acc:.2f}%  ->  final ({n_prune} layers pruned per HRank schedule): {final_acc:.2f}%"
    )


if __name__ == "__main__":
    main()
