# Competition Notes

Last reviewed: 2026-05-22

## What We Know

- Host: Kaggle competition by NVIDIA.
- Objective: improve reasoning accuracy for Nemotron-3-Nano-30B on a novel benchmark.
- Expected deliverable: a LoRA adapter compatible with the provided base model.
- Public commentary says rank is capped at 32, so our first training design assumes `r <= 32`.
- Public examples and community artifacts indicate task families:
  - bit manipulation
  - cipher/encryption
  - numeral/base conversion
  - unit conversion
  - modified gravity/physics
  - symbolic equation solving
  - numeric equation solving
- Publicly referenced prize/timeline details:
  - entry deadline reported as 2026-06-08
  - final deadline reported as 2026-06-15
  - public prize pool reported as about $106K plus DGX Spark awards

## Useful Public References

- Competition: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
- Progress-prize repo: https://github.com/tonghuikang/nemotron
- Public trajectories dataset: https://www.kaggle.com/datasets/kishanvavdara/nemotron-reasoning-traj
- Base model card: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

## Assumptions To Verify From Kaggle Files

- Official train/test schema.
- Exact submission packaging format.
- Exact model/license constraints.
- Whether external data and synthetic generation are unrestricted.
- Whether final answers must be LaTeX-wrapped or follow another exact extractor format.
- Runtime limits for Kaggle evaluation and adapter upload.

## Initial Strategy

The challenge appears less like broad knowledge fine-tuning and more like teaching exact transformations. The fastest path is to build category-specific synthetic generators, train on clear solution traces and final-answer formatting, and use validation by problem family to keep one family from hiding another.
