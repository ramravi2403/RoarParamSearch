import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from CombinedEvaluator import CombinedEvaluator
from ValueObject import ValueObject


def set_seed(seed_value=42):
    import random
    import numpy as np
    import torch
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
def load_data(data_dir: Path, train_file: str, query_file: str,
              test_file: str, label_column: str = "NoDefault"):
    """Load training, query, and test data."""
    train_df = pd.read_csv(data_dir / train_file)
    query_df = pd.read_csv(data_dir / query_file)
    test_df = pd.read_csv(data_dir / test_file)

    y_train = train_df[label_column].values.astype(np.int64)
    X_train = train_df.drop(columns=[label_column]).values.astype(np.float32)

    y_query = query_df[label_column].values.astype(np.int64)
    X_query = query_df.drop(columns=[label_column]).values.astype(np.float32)

    y_test = test_df[label_column].values.astype(np.int64)
    X_test = test_df.drop(columns=[label_column]).values.astype(np.float32)

    feature_names_path = data_dir / "sba_feature_names.npy"
    if feature_names_path.exists():
        feature_names = np.load(feature_names_path, allow_pickle=True).tolist()
    else:
        feature_names = train_df.drop(columns=[label_column]).columns.tolist()

    return X_train, y_train, X_query, y_query, X_test, y_test, feature_names


def main():
    def parse_norm(value):
        if value.lower() == 'inf':
            return float('inf')
        return int(value)
    set_seed(42)
    parser = argparse.ArgumentParser(
        description="Combined parameter evaluation: CF/CCF generation + extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--data-dir', type=str, default='data/sba')
    parser.add_argument('--train-file', type=str, default='sba_train.csv')
    parser.add_argument('--query-file', type=str, default='sba_query.csv')
    parser.add_argument('--test-file', type=str, default='sba_test.csv')
    parser.add_argument('--label-column', type=str, default='NoDefault')
    parser.add_argument('--model-type', type=str, choices=['simple', 'deep'],
                        default='simple', help='Baseline model architecture')

    # [0.0, 0.05, 0.1, 0.15, 0.2,]
    parser.add_argument('--delta-max-values', type=float, nargs='+',
                        default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0,2.0],
                        help='Robustness parameters (0.0 = non-robust)')
    # [0.1, 0.2, 0.3]
    parser.add_argument('--lamb-values', type=float, nargs='+',
                        default=[0.2],
                        help='Lambda cost tradeoff parameters')
    # [0.001, 0.005, 0.01, 0.05, 0.1]
    parser.add_argument('--alpha-values', type=float, nargs='+',
                        default=[0.005,0.01,0.1],
                        help='Learning rates for extraction')
    parser.add_argument('--size-values', type=float, nargs='+',
                        default=[0.2, 0.3, 0.5, 1.0],
                        help='Percentages of query set to use')
    parser.add_argument('--norm-values', type=parse_norm, nargs='+', default=[1],
                        help="List of norms to test (e.g., 1 2 inf)")

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--output', type=str, default='combined_evaluation_results')
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()
    print("Loading data...")
    data_dir = Path(args.data_dir)
    X_train, y_train, X_query, y_query, X_test, y_test, feature_names = load_data(
        data_dir, args.train_file, args.query_file, args.test_file, args.label_column
    )
    print(f"Train: {len(X_train)}, Query: {len(X_query)}, Test: {len(X_test)}")

    evaluator = CombinedEvaluator(verbose=not args.quiet)
    value_object = ValueObject(delta_max_values=args.delta_max_values,
                               lambda_values=args.lamb_values,
                               alpha_values=args.alpha_values,
                               norm_values=args.norm_values,
                               X_train=X_train,
                               y_train=y_train,
                               X_query=X_query,
                               X_test=X_test,
                               y_test=y_test,
                               feature_names=feature_names)
    results = evaluator.evaluate(
        value_object,
        query_size_pcts=args.size_values,
        num_epochs=args.epochs,
        model_type=args.model_type
    )

    evaluator.print_summary()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('combined_evaluation_runs') / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator.save_results(output_dir / args.output)

    print(f"\n✅ Evaluation complete! Results in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
