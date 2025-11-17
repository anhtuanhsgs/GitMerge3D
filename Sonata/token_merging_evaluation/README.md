# Token Merging Evaluation Scripts

This folder contains scripts for evaluating models with token merging enabled and measuring computational efficiency.

## Scripts Overview

### 1. `eval_scannet.sh`
**Purpose**: Evaluate a model on ScanNet validation set

**Usage**:
```bash
bash eval_scannet.sh
```

**What it does**:
- Tests a pre-trained model on ScanNet dataset
- Outputs mIoU, mAcc, and per-class accuracy
- Default config: `semseg-sonata-v1m1-0c-scannet-ft`

**Customization**: Edit the script to change:
- `CONFIG_NAME`: Your config file name (without .py extension)
- `EXP_ROOT_FOLDER`: Folder containing your trained model
- `PTV3_WEIGHTS_PATH`: Path to your checkpoint

---

### 2. `eval_s3dis.sh`
**Purpose**: Evaluate a model on S3DIS dataset

**Usage**:
```bash
bash eval_s3dis.sh <config-name>
```

**Example**:
```bash
bash eval_s3dis.sh gitmerge3d-wpatch-3a-s3dis-lin-r0.9-s10
```

**What it does**:
- Tests a pre-trained model on S3DIS dataset
- Accepts config name as command-line argument
- Default weights folder: `semseg-sonata-v1m1-0-base-3a-s3dis-lin`

**Customization**: Edit the script to change:
- `EXP_ROOT_FOLDER`: Folder containing your trained model

---

### 3. `run_cal_flops_sweep.py`
**Purpose**: Measure GFLOPs across multiple token retention ratios

**Usage**:
```bash
python run_cal_flops_sweep.py \
    --config <config-file> \
    --merge-rates 0.7 0.8 0.9 0.95 \
    --max-scene-count 10 \
    --gpu 0
```

**Full Example**:
```bash
python run_cal_flops_sweep.py \
    --config ../configs/sonata/gitmerge3d-wpatch-0c-scannet-ft-finetune-100epochs-r0.9-s10.py \
    --merge-rates 0.5 0.6 0.7 0.8 0.9 0.95 \
    --max-scene-count 10 \
    --seed 42 \
    --gpu 0
```

**Arguments**:
- `--config`: Path to config file
- `--merge-rates`: List of retention ratios to test (default: 0.5 0.6 0.7 0.8 0.9 0.95)
- `--max-scene-count`: Number of scenes to process (default: 1000)
- `--seed`: Random seed for reproducibility (default: 0)
- `--gpu`: GPU device to use (default: 0)

**Output**:
- Individual logs: `exp/sonata/<config-name>/flops_r{rate:.2f}_s{stride}.log`
- Summary CSV: `exp/sonata/<config-name>/flops_summary.csv`

**Note**: This script calls `cal_flops.py` which must exist in your tools directory.

---

### 4. `gitmerge3d_wpatch_scannet_ft.py`
**Purpose**: Convenient script to evaluate weighted_patch models on ScanNet

**Usage**:
```bash
python gitmerge3d_wpatch_scannet_ft.py --config <0-3> --gpu <0-1>
```

**Example**:
```bash
# Test r=0.9 model on GPU 0
python gitmerge3d_wpatch_scannet_ft.py --config 2 --gpu 0

# Test r=0.95 model on GPU 1
python gitmerge3d_wpatch_scannet_ft.py --config 3 --gpu 1
```

**Config Options**:
- `0`: r=0.7 (70% token retention)
- `1`: r=0.8 (80% token retention)
- `2`: r=0.9 (90% token retention)
- `3`: r=0.95 (95% token retention)

**Arguments**:
- `--config`: Which retention ratio to test (0-3)
- `--gpu`: GPU to use (0 or 1)
- `--weight`: Weight file name (default: model_best.pth)

**What it does**:
- Automatically sets up paths for neurips wpatch configs
- Uses weights from corresponding patch training experiments
- Runs evaluation on ScanNet

**Customization**: Edit the `CONFIGS` list in the script to add/modify configurations

---

### 5. `test_merging_rates_0c_scannet_ft.sh`
**Purpose**: Test multiple retention ratios with a single checkpoint

**Usage**:
```bash
bash test_merging_rates_0c_scannet_ft.sh <config-name>
```

**Example**:
```bash
bash test_merging_rates_0c_scannet_ft.sh gitmerge3d-wpatch-0c-scannet-ft-finetune-100epochs-r0.9-s10
```

**What it does**:
- Loads a checkpoint once
- Tests it with multiple retention ratios
- Automatically calculates correct stride for each ratio
- Uses `tools/test_merge_rates.py` under the hood

**Customization**: Edit the script to change:
- `EXP_ROOT_FOLDER`: Folder containing your trained model
- The script passes the config name as an argument

---

## Quick Start Guide

### Evaluate a Single Model

```bash
# For ScanNet
cd token_merging_evaluation
bash eval_scannet.sh

# For S3DIS
cd token_merging_evaluation
bash eval_s3dis.sh your-config-name
```

### Measure GFLOPs

```bash
cd token_merging_evaluation
python run_cal_flops_sweep.py \
    --config ../configs/sonata/your-config.py \
    --merge-rates 0.7 0.8 0.9 0.95 \
    --max-scene-count 10 \
    --gpu 0
```

### Test Multiple Retention Ratios

```bash
cd token_merging_evaluation
bash test_merging_rates_0c_scannet_ft.sh your-config-name
```

---

## Common Workflows

### Workflow 1: Evaluate Trained Model
```bash
# 1. Edit eval_scannet.sh with your paths
vim eval_scannet.sh

# 2. Run evaluation
bash eval_scannet.sh

# 3. Check results in terminal output
```

### Workflow 2: Measure Efficiency
```bash
# 1. Run GFLOPs sweep
python run_cal_flops_sweep.py \
    --config ../configs/sonata/your-config.py \
    --merge-rates 0.7 0.8 0.9 0.95 \
    --max-scene-count 10

# 2. View summary
cat ../exp/sonata/<config-name>/flops_summary.csv
```

### Workflow 3: Compare Multiple Retention Ratios
```bash
# 1. Run multi-rate testing
bash test_merging_rates_0c_scannet_ft.sh your-config-name

# 2. Compare results in terminal output
```

---

## Tips

1. **Start with fewer scenes**: Use `--max-scene-count 1` for quick testing
2. **Check GPU usage**: Monitor with `nvidia-smi` to ensure no conflicts
3. **Use weighted_patch for evaluation**: More efficient than patch variant
4. **Match stride to retention ratio**:
   - r=0.7 → stride=3
   - r=0.8 → stride=5
   - r=0.9 → stride=10
   - r=0.95 → stride=20

---

## Troubleshooting

**Script not found**: Make sure you're running from the Sonata root directory:
```bash
cd /path/to/Sonata
bash token_merging_evaluation/eval_scannet.sh
```

**Wrong paths**: All scripts use relative paths from Sonata root. If you get path errors, check your current directory.

**GPU memory issues**: Reduce `--max-scene-count` or test fewer retention ratios at once.

---

## Related Documentation

- Main evaluation guide: `../TOKEN_MERGING_EVALUATION_GUIDE.md`
- Training guide: `../FINETUNING_TOKEN_MERGING_GUIDE.md`
- General training: `../TRAINING_GUIDE.md`
