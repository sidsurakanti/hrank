# Results log

Pretrained checkpoints from [chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models),
pruned per [HRank](https://arxiv.org/abs/2002.10179)'s published compress-rate schedules.

## VGG-16-bn

| stage | accuracy |
|---|---|
| pretrained baseline | 94.15% |
| pruned, 0 epochs/layer (no finetune) | 10.00% |
| pruned, 15 epochs/layer (ours) | **91.25%** |
| pruned (paper) | 92.34% |

Gap (1.1pp): different pretrained checkpoint than HRank's own (different
weights despite same baseline accuracy), different classifier head, no seed
control.

## ResNet-56

| stage | accuracy |
|---|---|
| pretrained baseline | 94.38% |
| pruned, 0 epochs/layer (no finetune) | 11.81% |
| pruned, 1 epoch/layer (quick mechanical check) | 89.08% |
| pruned, 15 epochs/layer (real setting) | **93.11%** |
| HRank paper | 93.17% |

Gap (0.06pp): essentially a match. chenyaofo's resnet56 uses option-B
(learned) downsample shortcuts vs. HRank's option-A (zero-pad); doesn't
matter in practice since neither approach prunes the shortcut.

## No-finetune control
Both nets collapse to chance without finetuning (VGG 10.00%, ResNet 11.81%).
So all recovered accuracy comes from the prune/finetune loop, not the rank
scores alone. Same control is planned for HRel.

## Compression
Structural FLOPs cut of these schedules: VGG 86.1%, ResNet 73.5% (`../hrel/compare.py`).
HRank's README quotes 108.61M (65.3%) for the VGG schedule because its counter keeps
conv inputs dense. Side-by-side with HRel: `../hrel/RESULTS.md`.

## Toy net (mechanism check only)
3-conv net, 50% uniform compression per layer: 57.93% -> 56.71%. Confirms the
rank/mask/prune-finetune loop works.
