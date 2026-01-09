import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from Metrics import Metrics
from ValueObject import ValueObject
from counterfactual_metrics import generate_cfs_ccfs, evaluate_extraction, calculate_quality_score
from models.ModelWrapper import ModelWrapper


class CombinedEvaluator:
    """Evaluates CF/CCF generation + extraction across parameter combinations."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[Metrics] = []

    def __print(self, message: str):
        if self.verbose:
            print(message)

    def run_single_combination(
        self,
            delta_max: float,
            lamb: float,
            alpha: float,
            norm,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_query: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray,
            feature_names: List[str],
            query_size_pct: float,
            num_epochs: int = 100,
            model_type: str = 'simple',
    ) -> Optional[Metrics]:

        self.__print(f"\n{'─' * 70}")
        self.__print(f"  δ_max={delta_max:.3f}, λ={lamb:.3f}, α={alpha:.5f}")
        self.__print(f"{'─' * 70}")

        self.__print("  [1] Training baseline model...")
        baseline_model = ModelWrapper(input_dim=X_train.shape[1],model_type=model_type)
        model = baseline_model.train(X_train, y_train, num_epochs=100, lr=0.01, verbose=False)
        if model_type == 'simple':
            self.__print("  Identified simple model...Extracting weights")
            W, W0 = model.extract_weights()
        else:
            self.__print("  Identified Deep model...Extracting weights")
            W, W0 = None, None

        self.__print(f"  [2] Generating CFs/CCFs...")
        cfs_df, ccfs_df, gen_metrics = generate_cfs_ccfs(
            X_query, model, W, W0, delta_max, lamb,
            feature_names,
            X_train=X_train,
            query_size_pct=query_size_pct,
            norm=norm,
            random_seed=42,
            verbose=False,
            model_type = model_type
        )

        if len(cfs_df) == 0 or len(ccfs_df) == 0:
            self.__print(f"  WARNING:  Skipping: Insufficient CFs/CCFs generated")
            return None

        cf_dists = np.array(gen_metrics['cf_distances'])
        ccf_dists = np.array(gen_metrics['ccf_distances'])

        n_queries = len(X_query)
        cf_success_rate = gen_metrics['cf_success'] / n_queries if n_queries > 0 else 0.0
        ccf_success_rate = gen_metrics['ccf_success'] / len(cfs_df) if len(cfs_df) > 0 else 0.0

        self.__print(f"  [3] Evaluating extraction (alpha={alpha})...")
        extraction_metrics = evaluate_extraction(
            cfs_df, ccfs_df, X_test, y_test,
            model, feature_names, alpha,
            X_train=X_train,
            y_train=y_train,
            num_epochs=num_epochs, verbose=False,
            model_type= model_type
        )

        quality = calculate_quality_score(
            extracted_acc=extraction_metrics['extracted_accuracy'],
            extracted_auc=extraction_metrics['extracted_auc'],
            extracted_agreement=extraction_metrics['extracted_agreement'],
            augmented_acc=extraction_metrics['augmented_accuracy'],
            augmented_auc=extraction_metrics['augmented_auc'],
            cf_mean_l1=float(np.mean(cf_dists))
        )

        result = Metrics(
            model_type=model_type,
            delta_max=delta_max,
            lamb=lamb,
            alpha=alpha,
            norm=norm,
            n_cfs_generated=len(cfs_df),
            n_ccfs_generated=len(ccfs_df),
            cf_success_rate=cf_success_rate,
            ccf_success_rate=ccf_success_rate,
            cf_mean_distance=float(np.mean(cf_dists)),
            cf_std_distance=float(np.std(cf_dists)),
            cf_min_distance=float(np.min(cf_dists)),
            cf_max_distance=float(np.max(cf_dists)),
            ccf_mean_distance=float(np.mean(ccf_dists)),
            ccf_std_distance=float(np.std(ccf_dists)),
            extracted_accuracy=extraction_metrics['extracted_accuracy'],
            extracted_auc=extraction_metrics['extracted_auc'],
            extracted_agreement=extraction_metrics['extracted_agreement'],
            extracted_mean_prob_shift=extraction_metrics['extracted_mean_prob_shift'],
            augmented_accuracy=extraction_metrics['augmented_accuracy'],
            augmented_auc=extraction_metrics['augmented_auc'],
            query_size_pct=query_size_pct,
            quality_score=quality
        )

        self.__print(f"  ✓ CF Distance L - {result.norm}: {result.cf_mean_distance:.4f} ± {result.cf_std_distance:.4f}")
        self.__print(f"  ✓ Extracted Acc: {result.extracted_accuracy:.4f}, Agreement: {result.extracted_agreement:.4f}")
        self.__print(f"  ✓ Quality Score: {result.quality_score:.4f}")

        return result

    def evaluate(
        self,
        value_object:ValueObject,
        query_size_pcts: List[float],
        num_epochs: int = 100,
        model_type:str = 'simple'
    ) -> List[Metrics]:

        self.__print("\n" + "=" * 70)
        self.__print("   COMBINED PARAMETER EVALUATION")
        self.__print("=" * 70)
        self.__print(f"\nδ_max values: {value_object.delta_max_values}")
        self.__print(f"λ values: {value_object.lambda_values}")
        self.__print(f"α values: {value_object.alpha_values}")
        self.__print(f"Query size %: {query_size_pcts}")

        total = len(value_object.delta_max_values) * len(value_object.lambda_values) * len(value_object.alpha_values) * len(query_size_pcts) * len(value_object.norm_values)
        self.__print(f"\nTotal combinations: {total}")

        self.results = []
        current = 0
        for norm in value_object.norm_values:
            for delta_max in value_object.delta_max_values:
                for lamb in value_object.lambda_values:
                    for alpha in value_object.alpha_values:
                        for query_pct in query_size_pcts:
                            current += 1
                            self.__print(f"\n[{current}/{total}]")

                            result = self.run_single_combination(
                                delta_max, lamb, alpha,norm,
                                value_object.X_train, value_object.y_train, value_object.X_query, value_object.X_test, value_object.y_test,
                                value_object.feature_names, query_pct,num_epochs,model_type = model_type
                            )

                            if result is not None:
                                self.results.append(result)

        return self.results

    def print_summary(self):
        if not self.results:
            print("No results to summarize.")
            return

        df = pd.DataFrame([asdict(r) for r in self.results])

        self.__print("\n" + "=" * 70)
        self.__print("   SUMMARY: BEST RESULTS")
        self.__print("=" * 70)

        best_quality = df.loc[df['quality_score'].idxmax()]
        self.__print("\n🏆 Best Overall Quality Score:")
        self.__print(
            f"  δ_max={best_quality['delta_max']:.3f}, λ={best_quality['lamb']:.3f}, α={best_quality['alpha']:.5f}, Norm={best_quality['norm']}")
        self.__print(f"  Quality: {best_quality['quality_score']:.4f}")
        self.__print(
            f"  Dist ({best_quality['norm']}): {best_quality['cf_mean_distance']:.4f}, Extracted Acc: {best_quality['extracted_accuracy']:.4f}")

        best_dist = df.loc[df['cf_mean_distance'].idxmin()]
        self.__print(f"\n Lowest CF Distance (Relative to Norm):")
        self.__print(f"  δ_max={best_dist['delta_max']:.3f}, λ={best_dist['lamb']:.3f}, Norm={best_dist['norm']}")
        self.__print(f"  Distance ({best_dist['norm']}): {best_dist['cf_mean_distance']:.4f}")
        self.__print(
            f"  Extracted Acc: {best_dist['extracted_accuracy']:.4f}, Quality: {best_dist['quality_score']:.4f}")

        best_acc = df.loc[df['extracted_accuracy'].idxmax()]
        self.__print("\n Highest Extraction Accuracy:")
        self.__print(f"  δ_max={best_acc['delta_max']:.3f}, α={best_acc['alpha']:.5f}, Norm={best_acc['norm']}")
        self.__print(f"  Extracted Acc: {best_acc['extracted_accuracy']:.4f}")
        self.__print(
            f"  Dist ({best_acc['norm']}): {best_acc['cf_mean_distance']:.4f}, Quality: {best_acc['quality_score']:.4f}")

        best_agree = df.loc[df['extracted_agreement'].idxmax()]
        self.__print("\n Highest Agreement with Baseline:")
        self.__print(f"  δ_max={best_agree['delta_max']:.3f}, α={best_agree['alpha']:.5f}, Norm={best_agree['norm']}")
        self.__print(f"  Agreement: {best_agree['extracted_agreement']:.4f}")
        self.__print(
            f"  Dist ({best_agree['norm']}): {best_agree['cf_mean_distance']:.4f}, Quality: {best_agree['quality_score']:.4f}")

        self.__print("\n" + "=" * 70)
        self.__print("   ANALYSIS: Impact of Norm Type")
        self.__print("=" * 70)
        for norm in sorted(df['norm'].unique(), key=lambda x: float(x)):
            subset = df[df['norm'] == norm]
            self.__print(f"\nNorm = {norm}:")
            self.__print(f"  Mean Distance: {subset['cf_mean_distance'].mean():.4f}")
            self.__print(f"  Mean Extracted Acc: {subset['extracted_accuracy'].mean():.4f}")
            self.__print(f"  Mean Agreement: {subset['extracted_agreement'].mean():.4f}")

        self.__print("\n" + "=" * 70)
        self.__print("   ANALYSIS: Impact of Robustness (δ_max)")
        self.__print("=" * 70)
        for delta in sorted(df['delta_max'].unique()):
            subset = df[df['delta_max'] == delta]
            self.__print(f"\nδ_max = {delta:.3f}:")
            self.__print(
                f"  Mean Distance: {subset['cf_mean_distance'].mean():.4f} ± {subset['cf_mean_distance'].std():.4f}")
            self.__print(f"  Mean Extracted Acc: {subset['extracted_accuracy'].mean():.4f}")
            self.__print(f"  Mean Quality: {subset['quality_score'].mean():.4f}")

        self.__print("\n" + "=" * 70)
        self.__print("   ANALYSIS: Impact of Query Set Size")
        self.__print("=" * 70)
        for size_pct in sorted(df['query_size_pct'].unique()):
            subset = df[df['query_size_pct'] == size_pct]
            n_cfs_avg = subset['n_cfs_generated'].mean()
            n_ccfs_avg = subset['n_ccfs_generated'].mean()

            self.__print(f"\nQuery Size = {size_pct * 100:.0f}% (avg {n_cfs_avg:.0f} CFs):")
            self.__print(f"  Mean Distance: {subset['cf_mean_distance'].mean():.4f}")
            self.__print(f"  Mean Extracted Acc: {subset['extracted_accuracy'].mean():.4f}")
            self.__print(f"  Mean Agreement: {subset['extracted_agreement'].mean():.4f}")
            self.__print(f"  CF Success Rate: {subset['cf_success_rate'].mean():.2%}")

    def save_results(self, output_path: Path):
        output_path = Path(output_path)
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            results_dict = [asdict(r) for r in self.results]
            for result in results_dict:
                for key, value in result.items():
                    if isinstance(value, (np.floating, np.float32, np.float64)):
                        result[key] = float(value)
                    elif isinstance(value, (np.integer, np.int32, np.int64)):
                        result[key] = int(value)
            json.dump(results_dict, f, indent=2)

        csv_path = output_path.with_suffix('.csv')
        df = pd.DataFrame([asdict(r) for r in self.results])
        df.to_csv(csv_path, index=False)

        self.__print(f"\n Results saved:")
        self.__print(f"   JSON: {json_path}")
        self.__print(f"   CSV: {csv_path}")