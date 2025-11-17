# Token Merging Fine-Tuning Guide

This guide explains how to fine-tune pre-trained models with **GitMerge3D** enabled for efficient 3D semantic segmentation.

## Overview

Token Merging is a technique that reduces computational cost by merging similar tokens during inference and training. This guide shows how to:

1. Use pre-trained downstream models (from `TRAINING_GUIDE.md`)
2. Fine-tune them with token merging enabled

### Token Merging Variants

This codebase supports two token merging variants:

1. **`patch`**: Standard patch-based token merging (use for training)
2. **`weighted_patch`**: Weighted patch-based token merging (use for evaluation only - loads models trained with `patch`)

---

## Key Parameters

### Merging Ratio (`r`)

The retention ratio determines how many tokens are kept after merging:

| Ratio | Tokens Retained | mIoU | GFLOPs |
|-------|----------------|-------------|----------|
| `r=0.95` | 95% | 77.5 | 57.5 |
| `r=0.90` | 90% | 77.5 | 62.0 |
| `r=0.80` | 80% | 77.6 | 72.5 |
| `r=0.70` | 70% | 77.7 | 86.1 |

**Rule of thumb**: Start with `r=0.80` for a good balance, then adjust based on your speed/accuracy requirements.

### Stride Parameter
**stride** is the bin size when sampling DST set (it is set at ```int(math.ceil(1.0 / (1.0 - r))```)
---

## Example Configurations

### ScanNet Token Merging

#### 1. Linear Probing with Token Merging

Train only the segmentation head with token merging enabled:

```bash
# Standard patch-based merging (r=0.90)
sh scripts/train.sh \
    -m 1 -g 1 -d sonata \
    -c gitmerge3d-patch-0a-scannet-lin \
    -n gitmerge3d-patch-0a-scannet-lin \
    -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
```

**Config**: `configs/sonata/gitmerge3d-patch-0a-scannet-lin.py`
**Merging**: `r=0.90`, `stride=10`, type=`patch`
**Backbone**: Frozen

**Checkpoints saved to**:
```
exp/sonata/gitmerge3d-patch-0a-scannet-lin/
├── model/
│   ├── model_best.pth
│   └── model_last.pth
├── log/
└── config/
```

#### 2. Decoder Fine-tuning with Token Merging

```bash
# Patch-based merging with decoder training
sh scripts/train.sh \
    -m 1 -g 1 -d sonata \
    -c gitmerge3d-patch-0b-scannet-dec \
    -n gitmerge3d-patch-0b-scannet-dec \
    -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
```

**Config**: `configs/sonata/gitmerge3d-patch-0b-scannet-dec.py`
**Merging**: `r=0.90`, `stride=10`, type=`patch`
**Backbone**: Frozen (decoder unfrozen)

**Checkpoints saved to**:
```
exp/sonata/gitmerge3d-patch-0b-scannet-dec/
├── model/
│   ├── model_best.pth
│   └── model_last.pth
├── log/
└── config/
```

#### 3. Full Fine-tuning with Token Merging

Full end-to-end training with various merging ratios:

##### Standard Fine-tuning
```bash
# Basic token merging fine-tuning
sh scripts/train.sh \
    -m 1 -g 1 -d sonata \
    -c gitmerge3d-patch-0c-scannet-ft \
    -n gitmerge3d-patch-0c-scannet-ft \
    -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
```

**Config**: `configs/sonata/gitmerge3d-patch-0c-scannet-ft.py`
**Checkpoints**: `exp/sonata/gitmerge3d-patch-0c-scannet-ft/model/`

##### Extended Fine-tuning (100 epochs) with Different Merging Rates

**Example 1: r=0.70**
```bash
sh scripts/train.sh \
    -m 1 -g 1 -d sonata \
    -c gitmerge3d-patch-0c-scannet-ft-finetune-100epochs-r0.7-s3 \
    -n gitmerge3d-patch-0c-scannet-ft-finetune-100epochs-r0.7-s3 \
    -w exp/sonata/semseg-sonata-v1m1-0-base-0c-scannet-ft/model/model_best.pth
```

**Merging**: `r=0.70`, `stride=3`, type=`patch`
**Checkpoints**: `exp/sonata/gitmerge3d-patch-0c-scannet-ft-finetune-100epochs-r0.7-s3/model/`

**Example 2: r=0.80**
```bash
sh scripts/train.sh \
    -m 1 -g 1 -d sonata \
    -c gitmerge3d-patch-0c-scannet-ft-finetune-100epochs-r0.8-s5 \
    -n gitmerge3d-patch-0c-scannet-ft-finetune-100epochs-r0.8-s5 \
    -w exp/sonata/semseg-sonata-v1m1-0-base-0c-scannet-ft/model/model_best.pth
```

**Merging**: `r=0.80`, `stride=5`, type=`patch`
**Checkpoints**: `exp/sonata/gitmerge3d-patch-0c-scannet-ft-finetune-100epochs-r0.8-s5/model/`

**Example 3: r=0.90**
```bash
sh scripts/train.sh \
    -m 1 -g 1 -d sonata \
    -c gitmerge3d-patch-0c-scannet-ft-finetune-100epochs-r0.9-s10 \
    -n gitmerge3d-patch-0c-scannet-ft-finetune-100epochs-r0.9-s10 \
    -w exp/sonata/semseg-sonata-v1m1-0-base-0c-scannet-ft/model/model_best.pth
```

**Merging**: `r=0.90`, `stride=10`, type=`patch`
**Checkpoints**: `exp/sonata/gitmerge3d-patch-0c-scannet-ft-finetune-100epochs-r0.9-s10/model/`

**Example 4: r=0.95**
```bash
sh scripts/train.sh \
    -m 1 -g 1 -d sonata \
    -c gitmerge3d-patch-0c-scannet-ft-finetune-100epochs-r0.95-s20 \
    -n gitmerge3d-patch-0c-scannet-ft-finetune-100epochs-r0.95-s20 \
    -w exp/sonata/semseg-sonata-v1m1-0-base-0c-scannet-ft/model/model_best.pth
```

**Merging**: `r=0.95`, `stride=20`, type=`patch`
**Checkpoints**: `exp/sonata/gitmerge3d-patch-0c-scannet-ft-finetune-100epochs-r0.95-s20/model/`

---

### S3DIS Token Merging

#### Linear Probing with Token Merging

**Example 1: r=0.70 (patch-based)**
```bash
sh scripts/train.sh \
    -m 1 -g 1 -d sonata \
    -c gitmerge3d-patch-r70-s3dis-lin \
    -n gitmerge3d-patch-r70-s3dis-lin \
    -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
```

**Merging**: `r=0.70`, type=`patch`
**Checkpoints**: `exp/sonata/gitmerge3d-patch-r70-s3dis-lin/model/`

**Example 2: r=0.80 (patch-based)**
```bash
sh scripts/train.sh \
    -m 1 -g 1 -d sonata \
    -c gitmerge3d-patch-r80-s3dis-lin \
    -n gitmerge3d-patch-r80-s3dis-lin \
    -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
```

**Merging**: `r=0.80`, type=`patch`
**Checkpoints**: `exp/sonata/gitmerge3d-patch-r80-s3dis-lin/model/`

---

### ScanNet200 Token Merging

#### Linear Probing with Token Merging

**Option 1: r=0.80 (patch-based)**
```bash
sh scripts/train.sh \
    -m 1 -g 1 -d sonata \
    -c gitmerge3d-patch-r80-scannet200-lin \
    -n gitmerge3d-patch-r80-scannet200-lin \
    -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
```

**Merging**: `r=0.80`, type=`patch`
**Checkpoints**: `exp/sonata/gitmerge3d-patch-r80-scannet200-lin/model/`

---

## Quick Start Scripts

For convenience, ready-to-use scripts are available in `token_merging_scripts/`:

### ScanNet Full Fine-tuning (100 epochs)
- `token_merging_scripts/scannet_patch_finetune_100epochs_r0.7.sh` - Merges 70% tokens (retains 30%)
- `token_merging_scripts/scannet_patch_finetune_100epochs_r0.8.sh` - Merges 80% tokens (retains 20%)
- `token_merging_scripts/scannet_patch_finetune_100epochs_r0.9.sh` - Merges 90% tokens (retains 10%) 

### S3DIS Linear Probing
- `token_merging_scripts/s3dis_patch_linear_r0.8.sh` - Merges 80% tokens (retains 20%)

### ScanNet200 Linear Probing
- `token_merging_scripts/scannet200_patch_linear_r0.8.sh` - Merges 80% tokens (retains 20%)

**Usage**:
```bash
# Example: Train ScanNet with token merging 
bash token_merging_scripts/scannet_patch_finetune_100epochs_r0.9.sh

# Example: Train with maximum speedup
bash token_merging_scripts/scannet_patch_finetune_100epochs_r0.7.sh
```

---

## Workflow

### Step 1: Train Baseline Model

First, train a baseline model without token merging using the standard downstream training scripts:

```bash
# Example: Train ScanNet full fine-tuning
bash downstream_training_scripts/scannet_finetune.sh
```

This will save checkpoints to:
```
exp/sonata/semseg-sonata-v1m1-0-base-0c-scannet-ft/model/model_best.pth
```

### Step 2: Fine-tune with Token Merging

**Option A: Use convenient scripts**
```bash
# Quick start with balanced performance
bash token_merging_scripts/scannet_patch_finetune_100epochs_r0.9.sh
```

**Option B: Use manual command**
```bash
# fine-tuning the model with a merging rate of 90%
sh scripts/train.sh \
    -m 1 -g 1 -d sonata \
    -c gitmerge3d-patch-0c-scannet-ft-finetune-100epochs-r0.9-s10 \
    -n gitmerge3d-patch-0c-scannet-ft-finetune-100epochs-r0.9-s10 \
    -w exp/sonata/semseg-sonata-v1m1-0-base-0c-scannet-ft/model/model_best.pth
```

---

## Evaluation

### Evaluate with Standard Patch Merging

To evaluate a model trained with token merging using the same `patch` config:

```bash
python tools/test.py \
    --config-file configs/sonata/gitmerge3d-patch-0c-scannet-ft-finetune-100epochs-r0.9-s10.py \
    --options weight=exp/sonata/gitmerge3d-patch-r0.9-scannet-ft/model/model_best.pth
```

### Evaluate with Global Informed Graph Merging (wpatch)

To evaluate a model trained with `patch` using `weighted_patch` for improved inference accuracy:

```bash
# Use weighted_patch config to evaluate a model trained with patch
python tools/test.py \
    --config-file configs/sonata/gitmerge3d-wpatch-0c-scannet-ft-finetune-100epochs-r0.9-s10.py \
    --options weight=exp/sonata/gitmerge3d-patch-r0.9-scannet-ft/model/model_best.pth
```

**Note**: The `weighted_patch` config loads the model trained with `patch` and applies weighted merging during evaluation for better results.

---


### Patch vs Weighted Patch

- **`patch`**: Spatial-aware token merging.
- **`weighted_patch`**: Use Global Informed Graph for adaptively merging more aggressively

### Fine-tuning Duration

- Standard fine-tuning: Uses default epochs from base config
- For the `gitmerge3d_*` configs, 100 epochs is recommended (original down-stream training required 800 epochs)

---

## Configuration Reference

All token merging configurations are located in:
```
configs/sonata/
├── gitmerge3d-patch-0a-scannet-lin.py                                    # ScanNet linear, patch, r=0.90
├── gitmerge3d-patch-0b-scannet-dec.py                                    # ScanNet decoder, patch, r=0.90
├── gitmerge3d-patch-0c-scannet-ft.py                                     # ScanNet full FT, patch
├── gitmerge3d-patch-0c-scannet-ft-finetune-100epochs-r0.7-s3.py         # ScanNet, r=0.70, stride=3
├── gitmerge3d-patch-0c-scannet-ft-finetune-100epochs-r0.8-s5.py         # ScanNet, r=0.80, stride=5
├── gitmerge3d-patch-0c-scannet-ft-finetune-100epochs-r0.9-s10.py        # ScanNet, r=0.90, stride=10
├── gitmerge3d-patch-0c-scannet-ft-finetune-100epochs-r0.95-s20.py       # ScanNet, r=0.95, stride=20
├── gitmerge3d-wpatch-0c-scannet-ft-finetune-100epochs-r0.7-s3.py   # Weighted patch, r=0.70
├── gitmerge3d-wpatch-0c-scannet-ft-finetune-100epochs-r0.8-s5.py   # Weighted patch, r=0.80
├── gitmerge3d-wpatch-0c-scannet-ft-finetune-100epochs-r0.9-s10.py  # Weighted patch, r=0.90
├── gitmerge3d-wpatch-0c-scannet-ft-finetune-100epochs-r0.95-s20.py # Weighted patch, r=0.95
├── gitmerge3d-patch-r70-s3dis-lin.py                                     # S3DIS, patch, r=0.70
├── gitmerge3d-patch-r80-s3dis-lin.py                                     # S3DIS, patch, r=0.80
├── gitmerge3d-wpatch-r70-s3dis-lin.py                                    # S3DIS, weighted patch, r=0.70
├── gitmerge3d-wpatch-r80-s3dis-lin.py                                    # S3DIS, weighted patch, r=0.80
├── gitmerge3d-patch-r80-scannet200-lin.py                                # ScanNet200, patch, r=0.80
└── gitmerge3d-wpatch-r80-scannet200-lin.py                               # ScanNet200, weighted patch, r=0.80
```

---

