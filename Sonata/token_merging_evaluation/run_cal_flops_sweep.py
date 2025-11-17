#!/usr/bin/env python3
"""
Script to run cal_flops.py with multiple merge rates and generate summary CSV.
"""

import argparse
import os
import subprocess
import pandas as pd
from pathlib import Path

import math


def run_cal_flops(config_path, merge_rate, max_scene_count, seed, log_file):
    """Run cal_flops.py for a single merge rate."""
    stride = int(1 / (1.0 - merge_rate))

    cmd = [
        "python",
        "cal_flops.py",
        "--config", config_path,
        "--merge_rates", str(merge_rate),
        "--max_scene_count", str(max_scene_count),
        "--seed", str(seed),
    ]

    print(f"\nRunning: r={merge_rate}, stride={stride}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_file}")

    with open(log_file, "w") as f:
        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if result.returncode != 0:
        print(f"WARNING: Command failed with return code {result.returncode}")
        return False

    return True


def parse_log_file(log_file):
    """Parse a single log file to extract r, stride, gflops, and peak_memory."""
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()

        # Find lines with the pattern "r\tstride\tfinal_gflops\tpeak_memory"
        for i, line in enumerate(lines):
            if "r\tstride\tfinal_gflops\tpeak_memory" in line:
                # Next line should have the values
                if i + 1 < len(lines):
                    data_line = lines[i + 1].strip()
                    parts = data_line.split('\t')
                    if len(parts) >= 4:
                        r = float(parts[0])
                        stride = int(parts[1])
                        gflops = float(parts[2])
                        peak_memory = float(parts[3])
                        return {
                            "r": r,
                            "stride": stride,
                            "gflops": gflops,
                            "peak_memory_gb": peak_memory,
                        }

        print(f"WARNING: Could not parse results from {log_file}")
        return None

    except Exception as e:
        print(f"ERROR parsing {log_file}: {e}")
        return None


def create_summary_csv(exp_dir, merge_rates, config_name):
    """Create summary CSV from all log files."""
    results = []

    for r in merge_rates:
        stride = int(1 / (1.0 - r))
        log_file = exp_dir / f"flops_r{r:.2f}_s{stride}.log"

        if log_file.exists():
            data = parse_log_file(log_file)
            if data:
                results.append(data)

    if not results:
        print("WARNING: No results to summarize")
        return

    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df = df.sort_values("r")

    csv_path = exp_dir / "flops_summary.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print("Summary:")
    print(df.to_string(index=False))
    print(f"\nSummary saved to: {csv_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Run cal_flops.py with multiple merge rates and generate summary"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--merge-rates",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        help="List of merge rates to test (default: 0.5 0.6 0.7 0.8 0.9)",
    )
    parser.add_argument(
        "--max-scene-count",
        type=int,
        default=1000,
        help="Maximum number of scenes to process (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU to use (default: 0)",
    )

    args = parser.parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["PYTHONPATH"] = "./"

    # Extract config name and create exp directory
    config_path = Path(args.config)
    config_name = config_path.stem  # filename without extension
    exp_dir = Path(f"exp/sonata/{config_name}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config: {config_name}")
    print(f"Experiment directory: {exp_dir}")
    print(f"Merge rates: {args.merge_rates}")
    print(f"Using GPU: {args.gpu}")

    # Run cal_flops for each merge rate
    successful_runs = []
    for r in args.merge_rates:
        stride = int(math.ceil(1 / (1.0 - r) - 1e-9))
        log_file = exp_dir / f"flops_r{r:.2f}_s{stride}.log"

        success = run_cal_flops(
            config_path=str(config_path),
            merge_rate=r,
            max_scene_count=args.max_scene_count,
            seed=args.seed,
            log_file=str(log_file),
        )

        if success:
            successful_runs.append(r)

    # Create summary CSV
    if successful_runs:
        print(f"\nSuccessfully completed {len(successful_runs)}/{len(args.merge_rates)} runs")
        create_summary_csv(exp_dir, successful_runs, config_name)
    else:
        print("\nNo successful runs to summarize")


if __name__ == "__main__":
    main()
