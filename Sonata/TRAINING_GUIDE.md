# Sonata Downstream Training Guide

This guide provides scripts for training downstream semantic segmentation tasks using pretrained Sonata encoders.

## Prerequisites

- Pretrained Sonata encoder weights should be available at: `exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth`
- Ensure the datasets are properly set up in the `data/` directory
- Configure WANDB_API_KEY if using Weights & Biases for logging

## Training Commands

All training commands follow this pattern:
```bash
sh scripts/train.sh -m <num_machines> -g <num_gpus> -d <dataset_dir> -c <config_name> -n <experiment_name> -w <pretrained_weights>
```

Parameters:
- `-m`: Number of machines (typically 1 for single-node training)
- `-g`: Number of GPUs (1 for single GPU)
- `-d`: Dataset directory (use `sonata` for all Sonata configs)
- `-c`: Config file name (without `.py` extension)
- `-n`: Experiment name (for saving results)
- `-w`: Path to pretrained weights

---

## ScanNet Semantic Segmentation

### 1. Linear Probing (lin)
Trains only the segmentation head while freezing the backbone encoder.

```bash
sh scripts/train.sh \
    -m 1 -g 1 -d sonata \
    -c semseg-sonata-v1m1-0a-scannet-lin \
    -n semseg-sonata-v1m1-0-base-0a-scannet-lin \
    -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
```

**Config file**: `configs/sonata/semseg-sonata-v1m1-0a-scannet-lin.py`
**Training epochs**: 800
**Backbone**: Frozen

**Checkpoints saved to**:
```
exp/sonata/semseg-sonata-v1m1-0-base-0a-scannet-lin/
├── model/
│   ├── model_best.pth      # Best model based on validation performance
│   └── model_last.pth      # Model from the last epoch
├── log/                     # Training logs
└── config/                  # Saved configuration files
```

### 2. Decoder Fine-tuning (dec)
Trains the decoder layers while keeping the encoder frozen.

```bash
sh scripts/train.sh \
    -m 1 -g 1 -d sonata \
    -c semseg-sonata-v1m1-0b-scannet-dec \
    -n semseg-sonata-v1m1-0-base-0b-scannet-dec \
    -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
```

**Config file**: `configs/sonata/semseg-sonata-v1m1-0b-scannet-dec.py`
**Backbone**: Frozen (decoder unfrozen)

**Checkpoints saved to**:
```
exp/sonata/semseg-sonata-v1m1-0-base-0b-scannet-dec/
├── model/
│   ├── model_best.pth      # Best model based on validation performance
│   └── model_last.pth      # Model from the last epoch
├── log/
└── config/
```

### 3. Full Fine-tuning (ft)
Trains the entire network end-to-end.

```bash
sh scripts/train.sh \
    -m 1 -g 1 -d sonata \
    -c semseg-sonata-v1m1-0c-scannet-ft \
    -n semseg-sonata-v1m1-0-base-0c-scannet-ft \
    -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
```

**Config file**: `configs/sonata/semseg-sonata-v1m1-0c-scannet-ft.py`
**Backbone**: Unfrozen (full model training)

**Checkpoints saved to**:
```
exp/sonata/semseg-sonata-v1m1-0-base-0c-scannet-ft/
├── model/
│   ├── model_best.pth      # Best model based on validation performance
│   └── model_last.pth      # Model from the last epoch
├── log/
└── config/
```

---

## S3DIS Semantic Segmentation

### Linear Probing (lin)
6-fold cross-validation on S3DIS dataset.

```bash
sh scripts/train.sh \
    -m 1 -g 1 -d sonata \
    -c semseg-sonata-v1m1-3a-s3dis-lin \
    -n semseg-sonata-v1m1-0-base-3a-s3dis-lin \
    -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
```

**Config file**: `configs/sonata/semseg-sonata-v1m1-3a-s3dis-lin.py`
**Training epochs**: 800
**Backbone**: Frozen

**Checkpoints saved to**:
```
exp/sonata/semseg-sonata-v1m1-0-base-3a-s3dis-lin/
├── model/
│   ├── model_best.pth      # Best model based on validation performance
│   └── model_last.pth      # Model from the last epoch
├── log/
└── config/
```

---

## ScanNet200 Semantic Segmentation

### Linear Probing (lin)
Fine-grained semantic segmentation with 200 classes.

```bash
sh scripts/train.sh \
    -m 1 -g 1 -d sonata \
    -c semseg-sonata-v1m1-1a-scannet200-lin \
    -n semseg-sonata-v1m1-0-base-1a-scannet200-lin \
    -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
```

**Config file**: `configs/sonata/semseg-sonata-v1m1-1a-scannet200-lin.py`
**Training epochs**: 800
**Backbone**: Frozen
**Number of classes**: 200

**Checkpoints saved to**:
```
exp/sonata/semseg-sonata-v1m1-0-base-1a-scannet200-lin/
├── model/
│   ├── model_best.pth      # Best model based on validation performance
│   └── model_last.pth      # Model from the last epoch
├── log/
└── config/
```

---

## Quick Reference Scripts

All the above commands are also available as ready-to-use shell scripts in `downstream_training_scripts/`:

- **ScanNet**:
  - `downstream_training_scripts/scannet_linear.sh` (linear probing)
  - `downstream_training_scripts/scannet_decoder.sh` (decoder fine-tuning)
  - `downstream_training_scripts/scannet_finetune.sh` (full fine-tuning)

- **S3DIS**:
  - `downstream_training_scripts/s3dis_linear.sh` (linear probing)

- **ScanNet200**:
  - `downstream_training_scripts/scannet200_linear.sh` (linear probing)

### Using the Scripts

Simply run the scripts directly from the repository root:

```bash
# Example: Train ScanNet with linear probing
bash downstream_training_scripts/scannet_linear.sh

# Example: Train ScanNet with full fine-tuning
bash downstream_training_scripts/scannet_finetune.sh

# Example: Train S3DIS with linear probing
bash downstream_training_scripts/s3dis_linear.sh
```

---

## Multi-GPU Training

For multi-GPU training, adjust the `-g` parameter:

```bash
# Example: 4 GPUs
sh scripts/train.sh \
    -m 1 -g 4 -d sonata \
    -c semseg-sonata-v1m1-0c-scannet-ft \
    -n semseg-sonata-v1m1-0-base-0c-scannet-ft \
    -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
```

---

## Checkpoint Information

### Checkpoint Types

Each training run saves two types of checkpoints:

- **`model_best.pth`**: Model with the best validation performance (recommended for evaluation)
- **`model_last.pth`**: Model from the final training epoch

### Output Directory Structure

All training results follow this structure:
```
exp/sonata/<experiment_name>/
├── model/
│   ├── model_best.pth      # Use this for evaluation/inference
│   └── model_last.pth      # Use this for resuming training
├── log/                     # Training logs and metrics
└── config/                  # Copy of the configuration used
```

### Using Trained Checkpoints

To evaluate a trained model:
```bash
# Example: Evaluate ScanNet linear probing
python tools/test.py \
    --config-file configs/sonata/semseg-sonata-v1m1-0a-scannet-lin.py \
    --options weight=exp/sonata/semseg-sonata-v1m1-0-base-0a-scannet-lin/model/model_best.pth
```

To use as pretrained weights for another task:
```bash
# Example: Use fine-tuned ScanNet model as initialization
sh scripts/train.sh \
    -m 1 -g 1 -d sonata \
    -c <your-config> \
    -n <your-experiment-name> \
    -w exp/sonata/semseg-sonata-v1m1-0-base-0c-scannet-ft/model/model_best.pth
```

---

## Notes

- **Linear probing (lin)**: Fastest training, only trains the segmentation head
- **Decoder fine-tuning (dec)**: Moderate training time, trains decoder while keeping encoder frozen
- **Full fine-tuning (ft)**: Longest training time, trains the entire model for best performance
- Default batch size is typically 24 (check individual config files for specifics)
- Mixed precision training (AMP) is enabled by default for faster training
- Checkpoints are automatically saved during training; no additional configuration needed
