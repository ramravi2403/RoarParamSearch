import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from Metrics import Metrics
from ValueObject import ValueObject
from counterfactual_metrics import evaluate_extraction, calculate_quality_score, generate_cfs_ccfs
from logger.RunLogger import RunLogger
from models.ModelWrapper import ModelWrapper


class CombinedEvaluator:
    """Evaluates CF/CCF generation + extraction across parameter combinations."""

    def __init__(self, verbose: bool = True, logger: Optional[RunLogger] = None):
        self.verbose = verbose
        self.results: List[Metrics] = []
        self.__logger = logger

    def __log(self, message: str, level: str = "info"):
        if self.__logger is not None:
            getattr(self.__logger, level)(message)
        elif self.verbose:
            print(message)

    def run_single_combination(
            self,
            vo: ValueObject,
            recourse_method: str,
            query_size_pct: float,
            num_epochs: int = 100,
            model_type: str = 'simple',
    ) -> Optional[Metrics]:

        baseline_model = ModelWrapper(input_dim=vo.X_train.shape[1], model_type=model_type)
        baseline_model.train(vo.X_train, vo.y_train, num_epochs=num_epochs, lr=vo.alpha_values[0], verbose=False)
        W, W0 = baseline_model.extract_weights() if model_type == 'simple' else (None, None)

        cfs_df, ccfs_df, gen_metrics = generate_cfs_ccfs(self.__logger,
                                                         vo, baseline_model, W, W0, query_size_pct, recourse_method
                                                         )

        if len(cfs_df) == 0 or len(ccfs_df) == 0:
            return None

        ext_metrics = evaluate_extraction(self.__logger,
                                          cfs_df, ccfs_df, vo.X_test, vo.y_test, baseline_model,
                                          vo.feature_names, vo.alpha_values[0], vo.X_train, vo.y_train,
                                          num_epochs, model_type=model_type
                                          )

        cf_dists = np.array(gen_metrics['cf_distances'])
        ccf_dists = np.array(gen_metrics['ccf_distances'])

        return Metrics(
            model_type=model_type,
            recourse_method=recourse_method,
            delta_max=vo.delta_max_values[0],
            lamb=vo.lambda_values[0],
            alpha=vo.alpha_values[0],
            norm=vo.norm_values[0],
            n_cfs_generated=len(cfs_df),
            n_ccfs_generated=len(ccfs_df),
            cf_success_rate=gen_metrics['cf_success'] / len(vo.X_query),
            ccf_success_rate=gen_metrics['ccf_success'] / len(cfs_df),
            cf_mean_distance=float(np.mean(cf_dists)),
            cf_std_distance=float(np.std(cf_dists)),
            cf_min_distance=float(np.min(cf_dists)),
            cf_max_distance=float(np.max(cf_dists)),
            ccf_mean_distance=float(np.mean(ccf_dists)),
            ccf_std_distance=float(np.std(ccf_dists)),
            extracted_accuracy=ext_metrics['extracted_accuracy'],
            extracted_auc=ext_metrics['extracted_auc'],
            extracted_agreement=ext_metrics['extracted_agreement'],
            extracted_mean_prob_shift=ext_metrics['extracted_mean_prob_shift'],
            augmented_accuracy=ext_metrics['augmented_accuracy'],
            augmented_auc=ext_metrics['augmented_auc'],
            query_size_pct=query_size_pct,
            quality_score=calculate_quality_score(
                ext_metrics['extracted_accuracy'], ext_metrics['extracted_auc'],
                ext_metrics['extracted_agreement'], ext_metrics['augmented_accuracy'],
                ext_metrics['augmented_auc'], float(np.mean(cf_dists))
            )
        )

    def evaluate(
            self,
            value_object: ValueObject,
            query_size_pcts: List[float],
            recourse_methods: List[str],
            num_epochs: int = 100,
            model_type: str = 'simple',
    ) -> List[Metrics]:

        self.__logger.log("   COMBINED PARAMETER EVALUATION", "info")
        total = (len(value_object.delta_max_values) * len(value_object.lambda_values) * len(
            value_object.alpha_values) * len(query_size_pcts) * len(value_object.norm_values) * len(recourse_methods))

        self.results = []
        current = 0
        for recourse_method in recourse_methods:
            for norm in value_object.norm_values:
                for delta_max in value_object.delta_max_values:
                    for lamb in value_object.lambda_values:
                        for alpha in value_object.alpha_values:
                            for query_pct in query_size_pcts:
                                current += 1
                                self.__logger.log(f"\n[{current}/{total}]")
                                single_context_vo = ValueObject(
                                    delta_max_values=[delta_max],
                                    lambda_values=[lamb],
                                    alpha_values=[alpha],
                                    norm_values=[norm],
                                    X_train=value_object.X_train,
                                    y_train=value_object.y_train,
                                    X_query=value_object.X_query,
                                    X_test=value_object.X_test,
                                    y_test=value_object.y_test,
                                    feature_names=value_object.feature_names
                                )

                                result = self.run_single_combination(
                                    vo=single_context_vo,
                                    recourse_method=recourse_method,
                                    query_size_pct=query_pct,
                                    num_epochs=num_epochs,
                                    model_type=model_type
                                )

                                if result is not None:
                                    self.results.append(result)

        return self.results

    def log_summary(self):
        if not self.results:
            print("No results to summarize.")
            return

        df = pd.DataFrame([asdict(r) for r in self.results])

        self.__logger.log("\n" + "=" * 70)
        self.__logger.log("   SUMMARY: BEST RESULTS")
        self.__logger.log("=" * 70)

        best_quality = df.loc[df['quality_score'].idxmax()]
        self.__logger.log("\n🏆 Best Overall Quality Score:")
        self.__logger.log(
            f"  δ_max={best_quality['delta_max']:.3f}, λ={best_quality['lamb']:.3f}, α={best_quality['alpha']:.5f}, Norm={best_quality['norm']}")
        self.__logger.log(f"  Quality: {best_quality['quality_score']:.4f}")
        self.__logger.log(
            f"  Dist ({best_quality['norm']}): {best_quality['cf_mean_distance']:.4f}, Extracted Acc: {best_quality['extracted_accuracy']:.4f}")

        best_dist = df.loc[df['cf_mean_distance'].idxmin()]
        self.__logger.log(f"\n Lowest CF Distance (Relative to Norm):")
        self.__logger.log(f"  δ_max={best_dist['delta_max']:.3f}, λ={best_dist['lamb']:.3f}, Norm={best_dist['norm']}")
        self.__logger.log(f"  Distance ({best_dist['norm']}): {best_dist['cf_mean_distance']:.4f}")
        self.__logger.log(
            f"  Extracted Acc: {best_dist['extracted_accuracy']:.4f}, Quality: {best_dist['quality_score']:.4f}")

        best_acc = df.loc[df['extracted_accuracy'].idxmax()]
        self.__logger.log("\n Highest Extraction Accuracy:")
        self.__logger.log(f"  δ_max={best_acc['delta_max']:.3f}, α={best_acc['alpha']:.5f}, Norm={best_acc['norm']}")
        self.__logger.log(f"  Extracted Acc: {best_acc['extracted_accuracy']:.4f}")
        self.__logger.log(
            f"  Dist ({best_acc['norm']}): {best_acc['cf_mean_distance']:.4f}, Quality: {best_acc['quality_score']:.4f}")

        best_agree = df.loc[df['extracted_agreement'].idxmax()]
        self.__logger.log("\n Highest Agreement with Baseline:")
        self.__logger.log(
            f"  δ_max={best_agree['delta_max']:.3f}, α={best_agree['alpha']:.5f}, Norm={best_agree['norm']}")
        self.__logger.log(f"  Agreement: {best_agree['extracted_agreement']:.4f}")
        self.__logger.log(
            f"  Dist ({best_agree['norm']}): {best_agree['cf_mean_distance']:.4f}, Quality: {best_agree['quality_score']:.4f}")

        self.__logger.log("\n" + "=" * 70)
        self.__logger.log("   ANALYSIS: Impact of Norm Type")
        self.__logger.log("=" * 70)
        for norm in sorted(df['norm'].unique(), key=lambda x: float(x)):
            subset = df[df['norm'] == norm]
            self.__logger.log(f"\nNorm = {norm}:")
            self.__logger.log(f"  Mean Distance: {subset['cf_mean_distance'].mean():.4f}")
            self.__logger.log(f"  Mean Extracted Acc: {subset['extracted_accuracy'].mean():.4f}")
            self.__logger.log(f"  Mean Agreement: {subset['extracted_agreement'].mean():.4f}")

        self.__logger.log("\n" + "=" * 70)
        self.__logger.log("   ANALYSIS: Impact of Robustness (δ_max)")
        self.__logger.log("=" * 70)
        for delta in sorted(df['delta_max'].unique()):
            subset = df[df['delta_max'] == delta]
            self.__logger.log(f"\nδ_max = {delta:.3f}:")
            self.__logger.log(
                f"  Mean Distance: {subset['cf_mean_distance'].mean():.4f} ± {subset['cf_mean_distance'].std():.4f}")
            self.__logger.log(f"  Mean Extracted Acc: {subset['extracted_accuracy'].mean():.4f}")
            self.__logger.log(f"  Mean Quality: {subset['quality_score'].mean():.4f}")

        self.__logger.log("\n" + "=" * 70)
        self.__logger.log("   ANALYSIS: Impact of Query Set Size")
        self.__logger.log("=" * 70)
        for size_pct in sorted(df['query_size_pct'].unique()):
            subset = df[df['query_size_pct'] == size_pct]
            n_cfs_avg = subset['n_cfs_generated'].mean()
            n_ccfs_avg = subset['n_ccfs_generated'].mean()

            self.__logger.log(f"\nQuery Size = {size_pct * 100:.0f}% (avg {n_cfs_avg:.0f} CFs):")
            self.__logger.log(f"  Mean Distance: {subset['cf_mean_distance'].mean():.4f}")
            self.__logger.log(f"  Mean Extracted Acc: {subset['extracted_accuracy'].mean():.4f}")
            self.__logger.log(f"  Mean Agreement: {subset['extracted_agreement'].mean():.4f}")
            self.__logger.log(f"  CF Success Rate: {subset['cf_success_rate'].mean():.2%}")

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

        self.__logger.log(f"\n Results saved:")
        self.__logger.log(f"   JSON: {json_path}")
        self.__logger.log(f"   CSV: {csv_path}")
