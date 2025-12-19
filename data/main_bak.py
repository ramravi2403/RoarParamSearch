#!/usr/bin/env python3
"""
SBA Dataset Preparation Script

Loads, processes, and splits SBA loan data with configurable options.
Supports minimal/full features, stratified/chronological splits, and custom ratios.
"""

import argparse
import os
import sys
from pathlib import Path
from SBADataset import SBADataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare SBA loan dataset with configurable options",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output
    parser.add_argument(
        "--input",
        type=str,
        default="data/SBAcase.11.13.17.csv",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sba",
        help="Directory to save processed .npy files"
    )

    # Feature configuration
    parser.add_argument(
        "--feature-mode",
        type=str,
        choices=["minimal", "full"],
        default="full",
        help="Feature set: 'minimal' (7 features) or 'full' (all features with OHE)"
    )

    # Split configuration
    parser.add_argument(
        "--split-strategy",
        type=str,
        choices=["stratified", "chronological"],
        default="chronological",
        help="Data splitting strategy"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Proportion of data for training (0-1)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Proportion of data for testing (0-1)"
    )
    parser.add_argument(
        "--query-ratio",
        type=float,
        default=0.2,
        help="Proportion of data for query set (0-1)"
    )

    # Experimental options
    parser.add_argument(
        "--introduce-leakage",
        action="store_true",
        help="Scale features before splitting (introduces data leakage, for experiments only)"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold number for stratified splits (affects random seed)"
    )

    # Convenience presets
    parser.add_argument(
        "--preset",
        type=str,
        choices=["default", "minimal", "roar", "leakage"],
        help="Use a preset configuration (overrides other args)"
    )

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.test_ratio + args.query_ratio
    if not (0.99 <= total_ratio <= 1.01):  # Allow small floating point errors
        parser.error(f"Ratios must sum to 1.0, got {total_ratio:.4f}")

    # Apply presets if specified
    if args.preset:
        args = apply_preset(args)

    return args


def apply_preset(args):
    """Apply configuration presets."""
    presets = {
        "default": {
            "feature_mode": "full",
            "split_strategy": "chronological",
            "train_ratio": 0.6,
            "test_ratio": 0.2,
            "query_ratio": 0.2,
            "introduce_leakage": False,
            "output_dir": "data/sba"
        },
        "minimal": {
            "feature_mode": "minimal",
            "split_strategy": "stratified",
            "train_ratio": 0.6,
            "test_ratio": 0.2,
            "query_ratio": 0.2,
            "introduce_leakage": False,
            "output_dir": "data/sba_minimal"
        },
        "roar": {
            "feature_mode": "full",
            "split_strategy": "chronological",
            "train_ratio": 0.8,
            "test_ratio": 0.1,
            "query_ratio": 0.1,
            "introduce_leakage": False,
            "output_dir": "data/sba_roar"
        },
        "leakage": {
            "feature_mode": "full",
            "split_strategy": "chronological",
            "train_ratio": 0.6,
            "test_ratio": 0.2,
            "query_ratio": 0.2,
            "introduce_leakage": True,
            "output_dir": "data/sba_leakage"
        }
    }

    preset_config = presets.get(args.preset, {})
    for key, value in preset_config.items():
        setattr(args, key, value)

    print(f"📋 Applied preset: '{args.preset}'")
    return args


def print_config(args):
    print("\n" + "=" * 60)
    print("SBA DATASET PREPARATION")
    print("=" * 60)
    print(f"Input file:       {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nFeature mode:     {args.feature_mode}")
    print(f"Split strategy:   {args.split_strategy}")
    print(f"Split ratios:     Train={args.train_ratio:.1%}, Test={args.test_ratio:.1%}, Query={args.query_ratio:.1%}")
    print(f"Data leakage:     {'YES (EXPERIMENTAL)' if args.introduce_leakage else 'No'}")
    print(f"Fold:             {args.fold}")
    print("=" * 60 + "\n")


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("..")
    args = parse_args()
    print_config(args)
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    print("🔧 Initializing SBA Dataset processor...")
    dataset = SBADataset(
        fold=args.fold,
        feature_mode=args.feature_mode,
        split_strategy=args.split_strategy
    )

    print("Loading and processing data...")
    try:
        X_train, y_train, X_test, y_test, X_query, y_query = dataset.get_data(
            file_name=args.input,
            save_dir=args.output_dir,
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio,
            query_ratio=args.query_ratio,
            introduce_leakage=args.introduce_leakage
        )

        # Print summary
        print("\nDataset preparation complete!")
        print(f"\nDataset sizes:")
        print(f"  Training:   {len(X_train):,} samples")
        print(f"  Testing:    {len(X_test):,} samples")
        print(f"  Query:      {len(X_query):,} samples")
        print(f"  Total:      {len(X_train) + len(X_test) + len(X_query):,} samples")
        print(f"\nFeatures:     {X_train.shape[1]} columns")
        print(f"Class balance (train): {y_train.mean():.1%} positive class")

        if args.introduce_leakage:
            print("WARNING: Data leakage was introduced (for experimental purposes)")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()