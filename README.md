# How Many Tokens Do 3D Point Cloud Transformer Architectures Really Need?

<p align="center">
  <a href="https://gitmerge3d.github.io/">Project Page</a> •
  <a href="https://openreview.net/">Paper</a> •
  <a href="https://github.com/anhtuanhsgs/GitMerge3D">Code</a>
</p>

**NeurIPS 2025**

**Tuan Anh Tran**<sup>1</sup>,
**Duy Minh Ho Nguyen**<sup>2,3</sup>,
**Hoai-Chau Tran**<sup>4</sup>,
**Michael Barz**<sup>1</sup>,
**Khoa D. Doan**<sup>4</sup>,
**Roger Wattenhofer**<sup>5</sup>,
**Vien Anh Ngo**<sup>6</sup>,
**Mathias Niepert**<sup>2,3</sup>,
**Daniel Sonntag**<sup>1,7</sup>,
**Paul Swoboda**<sup>8</sup>

<sup>1</sup>DFKI,
<sup>2</sup>IMPRS-IS,
<sup>3</sup>University of Stuttgart,
<sup>4</sup>VinUniversity,
<sup>5</sup>ETH Zurich,
<sup>6</sup>VinRobotics, Hanoi,
<sup>7</sup>University of Oldenburg,
<sup>8</sup>Heinrich Heine University Düsseldorf

---

<p align="center">
  <img src="assets/Teaser_Figure.png" alt="GitMerge3D Teaser" width="800">
</p>

GitMerge3D enables merging of up to 80-95% of tokens, substantially reducing computational and memory costs while preserving model performance.

<p align="center">
  <img src="assets/feature_pca_merging.gif" alt="Feature PCA at different merging rates" width="600">
  <br>
  <em>Feature PCA visualization at different merging rates (0.15 to 0.95) showing minimal change in learned representations</em>
</p>

## Installation

### Requirements
- Ubuntu: 18.04 and above
- CUDA: 11.3 and above
- PyTorch: 1.10.0 and above

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/anhtuanhsgs/GitMerge3D.git
   cd GitMerge3D/Sonata
   ```

2. **Create conda environment**

   Follow the installation instructions in [Sonata/README.md](Sonata/README.md):

   ```bash
   # Option 1: Using environment.yml
   conda env create -f environment.yml --verbose
   conda activate pointcept-torch2.5.0-cu12.4

   # Option 2: Manual installation
   # See Sonata/README.md for detailed manual setup
   ```

3. **Install additional dependencies**
   ```bash
   # PTv3 dependencies
   cd libs/pointops
   python setup.py install
   cd ../..
   ```

4. **Prepare datasets**

   Follow the data preparation instructions in [Sonata/README.md](Sonata/README.md#data-preparation) for:
   - ScanNet v2
   - S3DIS
   - ScanNet200

## Quick Start

### Training with Token Merging

Fine-tune a pre-trained model with token merging enabled:

```bash
cd Sonata

# ScanNet with r=0.9 (90% token retention)
sh scripts/train.sh -g 4 -d scannet \
    -c gitmerge3d-patch-0c-scannet-ft-finetune-100epochs-r0.9-s10 \
    -n gitmerge3d-patch-r0.9-scannet-ft \
    -w exp/sonata/your-pretrained-model/model/model_best.pth
```

### Evaluation

Evaluate a trained model:

```bash
# Using evaluation script
cd Sonata
bash token_merging_evaluation/eval_scannet.sh
```

Or use the Python script directly:

```bash
python tools/test.py \
    --config-file configs/sonata/gitmerge3d-wpatch-0c-scannet-ft-finetune-100epochs-r0.9-s10.py \
    --options weight=exp/sonata/your-checkpoint/model/model_best.pth
```

### Measuring GFLOPs

Measure computational efficiency across multiple retention ratios:

```bash
cd Sonata/token_merging_evaluation
python run_cal_flops_sweep.py \
    --config ../configs/sonata/gitmerge3d-wpatch-0c-scannet-ft-finetune-100epochs-r0.9-s10.py \
    --merge-rates 0.7 0.8 0.9 0.95 \
    --max-scene-count 10 \
    --gpu 0
```

## Performance

Results on ScanNet semantic segmentation:

| Retention Ratio | Tokens Retained | mIoU (%) | GFLOPs | Checkpoint |
|----------------|----------------|----------|--------|------------|
| Baseline (r=1.0) | 100% | 78.9 | 105.0 | To be uploaded |
| r=0.95 | 95% | 78.5 | 57.5 | To be uploaded |
| r=0.90 | 90% | 78.8 | 62.0 | To be uploaded |
| r=0.80 | 80% | 79.2 | 72.5 | To be uploaded |
| r=0.70 | 70% | 79.5 | 86.1 | To be uploaded |

*Up to 21% reduction in computational cost with minimal accuracy loss*

## Documentation

- **[Token Merging Evaluation Guide](Sonata/TOKEN_MERGING_EVALUATION_GUIDE.md)**: How to evaluate models and measure GFLOPs
- **[Token Merging Training Guide](Sonata/FINETUNING_TOKEN_MERGING_GUIDE.md)**: Fine-tuning with token merging
- **[Training Guide](Sonata/TRAINING_GUIDE.md)**: Baseline training without token merging
- **[Evaluation Scripts](Sonata/token_merging_evaluation/)**: Ready-to-use evaluation tools

## Project Structure

```
GitMerge3D/
├── Sonata/                                    # Main codebase
│   ├── configs/sonata/                       # Token merging configurations
│   ├── pointcept/
│   │   └── models/point_transformer_v3/
│   │       └── token_merging_algos.py        # Core merging algorithms
│   ├── tools/                                # Training and testing scripts
│   ├── token_merging_evaluation/             # Evaluation scripts
│   ├── TOKEN_MERGING_EVALUATION_GUIDE.md
│   ├── FINETUNING_TOKEN_MERGING_GUIDE.md
│   └── README.md                             # Base installation guide
└── README.md                                  # This file
```

## Roadmap

### Current Release
- [x] **Code for Sonata (Pointcept v1.6.0)** - Token merging implementation with Sonata backbone
- [x] Training scripts and configurations
- [x] Evaluation tools and documentation

### Coming Soon
- [ ] **Model Checkpoints** - Pre-trained weights for all retention ratios (r=0.7, 0.8, 0.9, 0.95)
- [ ] **SpatialLM Integration** - Code and checkpoints for SpatialLM with token merging
- [ ] **PTv3 (Pointcept v1.5.1)** - Code and checkpoints for Point Transformer V3 baseline

Stay tuned for updates!

## Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{gitmerge3d2025,
    title={How Many Tokens Do 3D Point Cloud Transformer Architectures Really Need?},
    author={Tuan Anh Tran and Duy Minh Ho Nguyen and Hoai-Chau Tran and Michael Barz and Khoa D. Doan and Roger Wattenhofer and Vien Anh Ngo and Mathias Niepert and Daniel Sonntag and Paul Swoboda},
    booktitle={Advances in Neural Information Processing Systems},
    year={2025}
}
```

Also cite the base Pointcept and Sonata work:

```bibtex
@misc{pointcept2023,
    title={Pointcept: A Codebase for Point Cloud Perception Research},
    author={Pointcept Contributors},
    howpublished = {\url{https://github.com/Pointcept/Pointcept}},
    year={2023}
}
```

## Acknowledgements

This project is built upon:
- [Pointcept](https://github.com/Pointcept/Pointcept) - Point cloud perception codebase
- [Point Transformer V3](https://github.com/Pointcept/PointTransformerV3) - Efficient point cloud backbone
- [Sonata](https://github.com/facebookresearch/sonata) - Self-supervised learning for point clouds

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
