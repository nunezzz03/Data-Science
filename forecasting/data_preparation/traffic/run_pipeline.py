#!/usr/bin/env python3
"""
Lab 5 - Time Series Data Preparation Pipeline
Dataset: TrafficTwoMonth.csv

This script runs all data preparation steps in order:
1. Scaling - Compare scaling approaches, select best
2. Aggregation - Compare aggregation levels, select best
3. Differentiation - Compare differencing approaches, select best
4. Smoothing - Compare smoothing methods, select best

Each step reads from the previous step's output and saves its own output.
"""

import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PIPELINE_STEPS = [
    ("01_scaling.py", "Scaling Analysis"),
    ("02_aggregation.py", "Aggregation Analysis"),
    ("03_differentiation.py", "Differentiation Analysis"),
    ("04_smoothing.py", "Smoothing Analysis"),
]


def run_step(script_name: str, step_name: str) -> bool:
    """Run a single pipeline step."""
    script_path = os.path.join(SCRIPT_DIR, script_name)

    if not os.path.exists(script_path):
        print(f"ERROR: Script not found: {script_path}")
        return False

    print(f"\n{'#' * 80}")
    print(f"# STEP: {step_name}")
    print(f"# Script: {script_name}")
    print(f"{'#' * 80}\n")

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=SCRIPT_DIR,
    )

    if result.returncode != 0:
        print(f"\nERROR: {step_name} failed with return code {result.returncode}")
        return False

    print(f"\n✓ {step_name} completed successfully")
    return True


def main():
    print("=" * 80)
    print("TRAFFIC DATA PREPARATION PIPELINE")
    print("=" * 80)
    print("\nPipeline Order:")
    for i, (script, name) in enumerate(PIPELINE_STEPS, 1):
        print(f"  {i}. {name} ({script})")

    # Run each step
    for script, name in PIPELINE_STEPS:
        success = run_step(script, name)
        if not success:
            print(f"\nPipeline stopped due to error in {name}")
            sys.exit(1)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)

    # Summary of output files
    print("\nOutput files created:")
    output_dirs = [
        "processed_data/scaling",
        "processed_data/aggregation",
        "processed_data/differentiation",
        "processed_data/smoothing",
    ]
    for output_dir in output_dirs:
        full_path = os.path.join(SCRIPT_DIR, output_dir)
        if os.path.exists(full_path):
            files = os.listdir(full_path)
            for f in files:
                print(f"  - {output_dir}/{f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
