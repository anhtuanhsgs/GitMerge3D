#!/usr/bin/env python3
"""
Simple script to run neurips wpatch configs on ScanNet fine-tuning.
Copies weights from corresponding patch experiments.
"""

import argparse
import os
import subprocess

# Available configs mapping to their source weight paths
CONFIGS = [
    {
        "config": "neurips_tome-wpatch-0c-scannet-ft-finetune-100epochs-r0.7-s3.py",
        "weights_path": "exp/sonata/tome-patch-0c-scannet-ft-finetune-100epochs-r0.7-s3/model",
    },
    {
        "config": "neurips_tome-wpatch-0c-scannet-ft-finetune-100epochs-r0.8-s5.py",
        "weights_path": "exp/sonata/tome-patch-0c-scannet-ft-finetune-100epochs-r0.8-s5/model",
    },
    {
        "config": "neurips_tome-wpatch-0c-scannet-ft-finetune-100epochs-r0.9-s10.py",
        "weights_path": "exp/sonata/tome-patch-0c-scannet-ft-finetune-100epochs-r0.9-s10/model",
    },
    {
        "config": "neurips_tome-wpatch-0c-scannet-ft-finetune-100epochs-r0.95-s20.py",
        "weights_path": "exp/sonata/tome-patch-0c-scannet-ft-finetune-100epochs-r0.95-s20/model",
    },
]


def main():
    parser = argparse.ArgumentParser(description="Run neurips wpatch ScanNet fine-tuning")
    parser.add_argument(
        "--config",
        type=int,
        choices=[0, 1, 2, 3],
        required=True,
        help="Which config to run: 0=r0.7, 1=r0.8, 2=r0.9, 3=r0.95",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        choices=[0, 1],
        required=True,
        help="Which GPU to use (0 or 1)",
    )
    parser.add_argument(
        "--weight",
        default="model_best.pth",
        help="Weight file name (default: model_best.pth)",
    )

    args = parser.parse_args()

    # Get config info
    config_info = CONFIGS[args.config]
    config_name = config_info["config"].replace(".py", "")
    config_path = f"./configs/sonata/{config_info['config']}"
    weights_path = f"./{config_info['weights_path']}/{args.weight}"
    save_path = f"./exp/scannet/{config_name}"

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["PYTHONPATH"] = "./"

    # Print command
    cmd = [
        "python",
        "tools/test.py",
        "--config-file",
        config_path,
        "--options",
        f"save_path={save_path}",
        f"weight={weights_path}",
    ]

    print(f"Running on GPU {args.gpu}:")
    print(f"Config: {config_name}")
    print(f"Weights: {weights_path}")
    print(f"Command: {' '.join(cmd)}")
    print()

    # Run command
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
