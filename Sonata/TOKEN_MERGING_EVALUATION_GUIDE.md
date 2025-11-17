# Token Merging Evaluation and GFLOPs Measurement Guide

This guide explains how to evaluate models with token merging enabled and measure computational efficiency (GFLOPs) in the Sonata codebase.

## Table of Contents
- [Overview](#overview)
- [Token Merging Variants](#token-merging-variants)
- [Configuration Parameters](#configuration-parameters)
- [Evaluating Models with Token Merging](#evaluating-models-with-token-merging)
- [Measuring GFLOPs](#measuring-gflops)
- [Multi-Rate Testing](#multi-rate-testing)
- [Expected Performance](#expected-performance)
- [Troubleshooting](#troubleshooting)

---

## Overview

Token merging reduces computational cost in point cloud transformers by merging similar tokens during inference. This codebase implements several token merging algorithms, with the primary variants being:

- **`patch`**: Spatial-aware patch-based merging (used for training)
- **`weighted_patch`**: Global-informed graph merging (used for evaluation, more aggressive)

Key insight: Models trained with `patch` variant can be evaluated with either `patch` or `weighted_patch` for different efficiency-accuracy trade-offs.

---

## Token Merging Variants

### Patch-Based Merging (`patch`)
- Divides tokens into spatial bins based on stride
- Randomly or deterministically selects destination tokens
- Merges similar tokens within each bin using cosine similarity
- Used during training for stability

### Weighted Patch Merging (`weighted_patch`)
- Global-informed graph-based merging
- More aggressive token reduction
- Better for inference efficiency
- Can be used with models trained on `patch` variant

---

## Configuration Parameters

Token merging behavior is controlled by the `additional_info` dictionary in config files:

```python
additional_info={
    "tome": "patch",           # Merging variant: "patch" or "weighted_patch"
    "r": 0.90,                 # Retention ratio (0.5-0.95)
    "stride": 10,              # Bin size, calculated as ceil(1/(1-r))
    "no_rand": True,           # True: deterministic, False: random DST selection
    "tome_mlp": False,         # Apply merging to MLP layers
    "tome_attention": True,    # Apply merging to attention layers
}
```

### Retention Ratio (`r`) and Stride Relationship

| Retention Ratio (r) | Tokens Retained | Stride | Config Example |
|---------------------|----------------|--------|----------------|
| 0.50 | 50% | 2 | `r0.5-s2` |
| 0.60 | 60% | 3 | `r0.6-s3` |
| 0.70 | 70% | 3 | `r0.7-s3` |
| 0.80 | 80% | 5 | `r0.8-s5` |
| 0.90 | 90% | 10 | `r0.9-s10` |
| 0.95 | 95% | 20 | `r0.95-s20` |

**Formula**: `stride = ceil(1.0 / (1.0 - r))`

---

## Evaluating Models with Token Merging

### Prerequisites

1. **Trained model checkpoint**: A model fine-tuned with token merging enabled
2. **Config file**: Matching the training configuration
3. **Dataset**: Properly prepared validation/test set (e.g., ScanNet, S3DIS)

### Method 1: Single Model Evaluation

#### Step 1: Choose Your Configuration

For models trained with `patch` variant, you can evaluate with either:

**Option A: Same variant (patch)**
```bash
CONFIG="configs/sonata/gitmerge3d-patch-0c-scannet-ft-finetune-100epochs-r0.9-s10.py"
```

**Option B: Weighted variant (more efficient)**
```bash
CONFIG="configs/sonata/gitmerge3d-wpatch-0c-scannet-ft-finetune-100epochs-r0.9-s10.py"
```

#### Step 2: Run Evaluation

```bash
# Basic evaluation
python tools/test.py \
    --config-file ${CONFIG} \
    --options weight=exp/sonata/gitmerge3d-patch-r0.9-scannet-ft/model/model_best.pth

# With specific GPU
export CUDA_VISIBLE_DEVICES=0
python tools/test.py \
    --config-file ${CONFIG} \
    --options weight=path/to/checkpoint.pth
```

#### Step 3: View Results

The script will output:
- **mIoU** (mean Intersection over Union): Primary semantic segmentation metric
- **mAcc** (mean Accuracy): Per-class accuracy
- **allAcc**: Overall pixel accuracy
- **Per-class IoU scores**


### Method 2: Using Evaluation Shell Scripts

Pre-configured scripts are available in `token_merging_evaluation/`:

```bash
# Edit the script to set your config and checkpoint path
vim token_merging_evaluation/eval_scannet.sh

# Run evaluation
bash token_merging_evaluation/eval_scannet.sh
```

Example script content:
```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python tools/test.py \
    --config-file configs/sonata/gitmerge3d-wpatch-0c-scannet-ft-finetune-100epochs-r0.9-s10.py \
    --options weight=exp/sonata/gitmerge3d-patch-r0.9-scannet-ft/model/model_best.pth
```

### Method 3: Evaluate Multiple Checkpoints

To evaluate different epochs or models:

```bash

# Evaluate best model
python tools/test.py \
    --config-file ${CONFIG} \
    --options weight=exp/sonata/your-experiment/model/model_best.pth
```

---

## Measuring GFLOPs

### Method 1: Single Configuration GFLOPs Measurement

If you have a standalone `cal_flops.py` script (typically in `tools/`):

```bash
# Calculate GFLOPs for a specific configuration
python tools/cal_flops.py \
    --config-file configs/sonata/gitmerge3d-wpatch-0c-scannet-ft-finetune-100epochs-r0.9-s10.py \
    --options additional_info.r=0.9 additional_info.stride=10
```

### Method 2: Multi-Rate GFLOPs Sweep

Use the automated sweep script to measure GFLOPs across multiple retention ratios:

```bash
cd token_merging_evaluation

# Run GFLOPs measurement sweep
python run_cal_flops_sweep.py \
    --config ../configs/sonata/gitmerge3d-wpatch-0c-scannet-ft-finetune-100epochs-r0.9-s10.py \
    --merge-rates 0.5 0.6 0.7 0.8 0.9 0.95 \
    --max-scene-count 10 \
    --seed 42
```

#### Parameters:
- `--config`: Base configuration file (will be modified for each rate)
- `--merge-rates`: List of retention ratios to test
- `--max-scene-count`: Number of scenes to process (fewer = faster, less accurate)
- `--seed`: Random seed for reproducibility

#### Output:

The script generates:

1. **Individual log files**: `flops_r{rate:.2f}_s{stride}.log`
   ```
   flops_r0.90_s10.log
   flops_r0.80_s5.log
   flops_r0.70_s3.log
   ```

2. **Summary CSV**: `flops_summary.csv`
   ```csv
   r,stride,gflops,peak_memory_gb
   0.95,20,57.5,12.3
   0.90,10,62.0,11.8
   0.80,5,72.5,10.9
   0.70,3,86.1,10.2
   ```

#### Example Usage:

```bash
# Quick test with fewer scenes
python run_cal_flops_sweep.py \
    --config ../configs/sonata/gitmerge3d-wpatch-0c-scannet-ft-finetune-100epochs-r0.9-s10.py \
    --merge-rates 0.7 0.8 0.9 0.95 \
    --max-scene-count 5

# Full measurement with many scenes
python run_cal_flops_sweep.py \
    --config ../configs/sonata/gitmerge3d-wpatch-0c-scannet-ft-finetune-100epochs-r0.9-s10.py \
    --merge-rates 0.5 0.6 0.7 0.8 0.9 0.95 \
    --max-scene-count 50
```

### Understanding GFLOPs Output

GFLOPs (Giga Floating Point Operations) indicates computational cost:
- **Lower GFLOPs** = More efficient, faster inference
- **Higher retention ratio** = More tokens kept = Higher GFLOPs
- Token merging reduces GFLOPs while maintaining accuracy

---

## Multi-Rate Testing

To test accuracy across multiple retention ratios simultaneously:

```bash
python tools/test_merge_rates.py \
    --config-file configs/sonata/gitmerge3d-wpatch-0c-scannet-ft-finetune-100epochs-r0.9-s10.py \
    --checkpoint exp/sonata/gitmerge3d-patch-r0.9-scannet-ft/model/model_best.pth \
    --merge-rates 0.5 0.6 0.7 0.8 0.9 0.95
```

This script:
1. Loads the checkpoint once
2. Tests each retention ratio
3. Automatically calculates correct stride for each ratio
4. Outputs mIoU and mAcc for each configuration

Example output:
```
Testing r=0.50 (stride=2): mIoU=76.2%, mAcc=84.1%
Testing r=0.60 (stride=3): mIoU=76.8%, mAcc=84.5%
Testing r=0.70 (stride=3): mIoU=77.3%, mAcc=84.9%
Testing r=0.80 (stride=5): mIoU=77.6%, mAcc=85.1%
Testing r=0.90 (stride=10): mIoU=77.5%, mAcc=85.2%
Testing r=0.95 (stride=20): mIoU=77.5%, mAcc=85.2%
```

---

## Expected Performance

Based on ScanNet fine-tuning experiments:

| Retention Ratio | Tokens Retained | mIoU (%) | GFLOPs |
|----------------|----------------|----------|--------|
| Baseline (r=0.0) | 100% | 78.9 | 105.0 |
| r=0.95 | 95% | 77.5 | 57.5 | 
| r=0.90 | 90% | 77.5 | 62.0 | 
| r=0.80 | 80% | 77.6 | 72.5 | 
| r=0.70 | 70% | 77.7 | 86.1 | 
| r=0.95 (Fine-tuned) | 95% | 78.5 | 57.5 | 
| r=0.90 (Fine-tuned) | 90% | 78.8 | 62.0 | 
| r=0.80 (Fine-tuned) | 80% | 79.2 | 72.5 | 
| r=0.70 (Fine-tuned) | 70% | 79.5 | 86.1 | 


---


### Issue: Wrong stride for retention ratio

**Error**: Suboptimal performance or warnings

**Solution**: Use the correct stride formula:
```python
import math
stride = int(math.ceil(1.0 / (1.0 - r)))
```

Common pairs:
- r=0.7 → stride=3
- r=0.8 → stride=5
- r=0.9 → stride=10
- r=0.95 → stride=20

### Issue: Different results between patch and weighted_patch

**Expected behavior**: `weighted_patch` typically shows slightly different accuracy than `patch` due to more global information usage and agressive merging scheme.

**Recommendation**: For inference efficiency, use `weighted_patch`. For training stability, use `patch`.

---

## Quick Reference Commands

### Evaluate model with token merging (weighted_patch):
```bash
python tools/test.py \
    --config-file configs/sonata/gitmerge3d-wpatch-0c-scannet-ft-finetune-100epochs-r0.9-s10.py \
    --options weight=exp/sonata/your-checkpoint/model/model_best.pth
```

### Measure GFLOPs for multiple rates:
```bash
cd token_merging_evaluation
python run_cal_flops_sweep.py \
    --config ../configs/sonata/gitmerge3d-wpatch-0c-scannet-ft-finetune-100epochs-r0.9-s10.py \
    --merge-rates 0.7 0.8 0.9 0.95 \
    --max-scene-count 10
```

### Test multiple retention ratios:
```bash
python tools/test_merge_rates.py \
    --config-file configs/sonata/gitmerge3d-wpatch-0c-scannet-ft-finetune-100epochs-r0.9-s10.py \
    --checkpoint exp/sonata/your-checkpoint/model/model_best.pth \
    --merge-rates 0.7 0.8 0.9 0.95
```

---

## Additional Resources

- **Training Guide**: See `FINETUNING_TOKEN_MERGING_GUIDE.md` for training models with token merging
- **Algorithm Details**: Check `pointcept/models/point_transformer_v3/token_merging_algos.py` for implementation
- **Config Examples**: Browse `configs/sonata/` for various token merging configurations
- **Training Scripts**: Ready-to-use scripts in `token_merging_scripts/`

---

## Citation

If you use token merging in your research, please cite the relevant papers and this implementation.
