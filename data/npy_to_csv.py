#!/usr/bin/env python3
"""
Convert .npy dataset files to CSV format for inspection and analysis.
"""

import argparse
import numpy as np
import pandas as pd
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert SBA .npy files to CSV format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/sba",
        help="Directory containing .npy files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for CSV files (defaults to same as data-dir)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="sba",
        help="Prefix for output CSV files"
    )

    return parser.parse_args()


def npy_to_csv(data_dir: str, npy_file: str, output_file: str, feature_names: np.ndarray):
    """
    Convert a single .npy file to CSV.

    Args:
        data_dir: Directory containing .npy files
        npy_file: Name of .npy file to convert
        output_file: Output CSV filename
        feature_names: Array of feature names
    """
    npy_path = os.path.join(data_dir, npy_file)

    if not os.path.exists(npy_path):
        print(f"⚠️  Skipping {npy_file} (file not found)")
        return

    # Load array: first column is target, rest are features
    arr = np.load(npy_path, allow_pickle=False)

    # Create DataFrame with features
    df = pd.DataFrame(arr[:, 1:], columns=feature_names)

    # Add target column
    df.insert(0, "NoDefault", arr[:, 0].astype(int))

    # Save to CSV
    output_path = os.path.join(data_dir, output_file)
    df.to_csv(output_path, index=False)

    print(f"✅ {output_file:<30} (shape: {df.shape[0]:>6,} rows × {df.shape[1]:>3} cols)")


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("..")
    args = parse_args()

    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.data_dir

    print("\n" + "=" * 60)
    print("NPY TO CSV CONVERTER")
    print("=" * 60)
    print(f"Data directory:   {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60 + "\n")

    # Check if directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Data directory not found: {args.data_dir}")
        return

    # Load feature names
    feature_names_path = os.path.join(args.data_dir, "sba_feature_names.npy")
    if not os.path.exists(feature_names_path):
        print(f"❌ Error: Feature names file not found: {feature_names_path}")
        return

    feature_names = np.load(feature_names_path, allow_pickle=True)
    print(f"📋 Loaded {len(feature_names)} feature names\n")

    # Convert each dataset
    datasets = [
        ("sba_train.npy", f"{args.prefix}_train.csv"),
        ("sba_test.npy", f"{args.prefix}_test.csv"),
        ("sba_query.npy", f"{args.prefix}_query.csv")
    ]

    for npy_file, csv_file in datasets:
        npy_to_csv(args.data_dir, npy_file, csv_file, feature_names)

    print("\n✅ Conversion complete!")


if __name__ == "__main__":
    main()