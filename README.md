# HRank reproduction

Reproducing [HRank: Filter Pruning using High-Rank Feature Map](https://arxiv.org/abs/2002.10179) (CVPR 2020) on CIFAR-10

## Results

| network | ours | paper | gap |
|---|---|---|---|
| VGG-16-bn | 91.25% | 92.34% | 1.1pp |
| ResNet-56 | 93.11% | 93.17% | 0.06pp |

Full log and notes on the gaps: `RESULTS.md`.

## The method


- For each conv layer (on pretrained model), measure the matrix rank of its feature maps on a few sample batches
- Low-rank channels carry redundant info and get pruned first
- Pruning is sequential: prune one layer, fine-tune the whole network to recover to the next layer

See `hrank_core.py`'s module docstring.

## Files

| file | what it is |
|---|---|
| `hrank_core.py` | shared engine: data, mat rank scoring, masking, prune->finetune loop |
| `vgg_prune.py` | reproduction on pretrained `vgg16_bn` |
| `resnet_prune.py` | reproduction on pretrained `resnet56` |
| `baseline_eval.py` | load a pretrained checkpoint, confirm reported accuracy |
| `toy.py` | toy 3-conv net, proves the mechanism before real networks |
| `RESULTS.md` | every run, its accuracy, and any gap to the paper |

`vgg_prune.py` / `resnet_prune.py` both:
1. load the pretrained model via `torch.hub`
2. locate each target conv's activation (for architecture specific hooks)
3. `core.compute_all_ranks` - one pass, rank every layer at once
4. `core.prune_and_finetune` - the sequential per-layer loop to prune finetune

## Known deviations from HRank's exact setup

- **Different pretrained checkpoints** (chenyaofo's, not HRank's own). Same
  reported baseline accuracy, different weights
- **resnet56 shortcut type.** chenyaofo uses learned (option-B) downsample
  shortcuts; HRank uses parameter-free (option-A). Neither prunes the
  shortcut, so this only touches 2 blocks and needed no code changes.
- **Published schedule vs. script defaults.** HRank's scripts ship a
  `default_cprate` fallback that isn't what they actually ran; the real
  schedules are only in their README's example commands. We use those.

## How to run

```bash
uv run python baseline_eval.py   # sanity check: checkpoint loads correctly
uv run python toy.py             # sanity check: the mechanism works
uv run python vgg_prune.py       # ~30 min on an RTX 4060
uv run python resnet_prune.py    # ~2-3 hours, 55 layers x 15 epochs
```
