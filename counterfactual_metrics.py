
import traceback
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

import ValueObject
from logger.RunLogger import RunLogger
from models.ModelWrapper import ModelWrapper
from recourse_solvers.RecourseSolverFactory import RecourseSolverFactory
from recourse_utils import recourse_needed, lime_explanation


def _get_local_explanation(
        x: np.ndarray,
        baseline_model: ModelWrapper,
        W: Optional[np.ndarray],
        W0: Optional[np.ndarray],
        X_train: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Get local linear explanation via LIME or return global weights."""
    use_lime = (W is None or W0 is None)

    if use_lime:
        def predict_proba_fn(x_input):
            if len(x_input.shape) == 1:
                x_input = x_input.reshape(1, -1)
            probs, _ = baseline_model.predict(x_input)
            probs = probs.flatten()
            return np.vstack([1 - probs, probs]).T

        return lime_explanation(predict_proba_fn, X_train, x)

    return W, W0


def generate_cfs_ccfs(
        logger:RunLogger,
        vo: ValueObject,
        baseline_model: ModelWrapper,
        W: Optional[np.ndarray],
        W0: Optional[np.ndarray],
        query_size_pct: float,
        recourse_method: str = 'roar',
        verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    use_lime = (W is None or W0 is None)
    factory = RecourseSolverFactory(vo, recourse_method, W, W0)

    metrics = {'cf_success': 0, 'cf_failures': 0, 'ccf_success': 0, 'cf_success_rate': 0 ,'ccf_success_rate': 0, 'ccf_failures': 0,
               'cf_distances': [], 'ccf_distances': []}

    predict_fn = lambda x: baseline_model.predict(x)[1]
    denied_indices = recourse_needed(predict_fn, vo.X_query, target=1)

    np.random.seed(42)
    n_samples = max(1, int(len(denied_indices) * query_size_pct))
    sampled_indices = np.random.choice(denied_indices, size=n_samples,
                                       replace=False) if query_size_pct < 1.0 else denied_indices
    cf_list = []
    for i in sampled_indices:
        x_i = vo.X_query[i].astype(np.float32)
        try:
            local_W, local_W0 = _get_local_explanation(x_i, baseline_model, W, W0, vo.X_train)
            solver = factory.create(local_W, local_W0, target=1, use_lime=use_lime)

            cf = solver.get_recourse(x_i, lamb=vo.lambda_values[0], norm=vo.norm_values[0], lime=use_lime)[
                0] if recourse_method == 'roar' else solver.get_recourse(x_i)

            dist = np.linalg.norm(cf.flatten() - x_i, ord=vo.norm_values[0])
            metrics['cf_distances'].append(dist)
            metrics['cf_success'] += 1
            cf_list.append(
                pd.DataFrame(cf.reshape(1, -1), columns=vo.feature_names).assign(original_query_idx=i, distance=dist))
        except Exception as e:
            metrics['cf_failures'] += 1
            message = f"[DEBUG CF {i}] EXCEPTION during generation: {str(e)} \n {traceback.print_exc()}"
            logger.error(message)

    if not cf_list: return pd.DataFrame(), pd.DataFrame(), metrics
    cfs_df = pd.concat(cf_list, ignore_index=True)

    ccf_list = []
    for _, row in cfs_df.iterrows():
        x_cf = row[vo.feature_names].values.astype(np.float32)
        try:
            local_W, local_W0 = _get_local_explanation(x_cf, baseline_model, W, W0, vo.X_train)
            solver_ccf = factory.create(local_W, local_W0, target=0, use_lime=use_lime)
            ccf = solver_ccf.get_recourse(x_cf, lamb=vo.lambda_values[0], norm=vo.norm_values[0], lime=use_lime)[
                0] if recourse_method == 'roar' else solver_ccf.get_recourse(x_cf)

            dist = np.linalg.norm(ccf.flatten() - vo.X_query[int(row['original_query_idx'])], ord=vo.norm_values[0])
            metrics['ccf_distances'].append(dist)
            metrics['ccf_success'] += 1
            ccf_list.append(pd.DataFrame(ccf.reshape(1, -1), columns=vo.feature_names).assign(
                original_query_idx=row['original_query_idx'], distance=dist))
            metrics['cf_success_rate'] = metrics['cf_success'] / len(denied_indices)
            metrics['ccf_success_rate'] = metrics['ccf_success'] / len(cfs_df)
        except Exception as e:
            metrics['ccf_failures'] += 1
            message = f"[DEBUG CCF {i}] EXCEPTION during generation: {str(e)} \n {traceback.print_exc()}"
            logger.exception(message)

    return cfs_df, (pd.concat(ccf_list, ignore_index=True) if ccf_list else pd.DataFrame()), metrics


def evaluate_extraction(
        logger:RunLogger,
        cfs_df: pd.DataFrame,
        ccfs_df: pd.DataFrame,
        X_test: np.ndarray,
        y_test: np.ndarray,
        baseline_model: ModelWrapper,
        feature_names: List[str],
        alpha: float,
        recourse_method:str,
        X_train=None,
        y_train=None,
        num_epochs: int = 50,
        verbose: bool = False,
        model_type:str ="simple"


) -> Dict:
    """Train extracted and augmented models, evaluate on test set."""

    X_cfs = cfs_df[feature_names].values.astype(np.float32)
    X_ccfs = ccfs_df[feature_names].values.astype(np.float32)

    _, y_cfs = baseline_model.predict(X_cfs)
    _, y_ccfs = baseline_model.predict(X_ccfs)

    X_extracted = np.vstack([X_cfs, X_ccfs])
    y_extracted = np.concatenate([y_cfs, y_ccfs])

    if verbose:
        logger.info(f"    Training extracted model (learning_rate={alpha})...")
    extracted_model = ModelWrapper(input_dim=X_extracted.shape[1], model_type=model_type)
    extracted_model.train(
        X_extracted, y_extracted,
        num_epochs=num_epochs,
        lr=alpha,
        verbose=verbose,
    )

    baseline_probs, baseline_preds = baseline_model.predict(X_test)
    extracted_probs, extracted_preds = extracted_model.predict(X_test)
    if verbose:
        logger.info(f">>> Raw Prediction Audit [Method: {recourse_method}] | Model {model_type}")
        n_pos = np.sum(y_test == 1)
        n_neg = np.sum(y_test == 0)
        logger.info(f"Test Set Balance: {n_pos} Positives (Approvals), {n_neg} Negatives (Defaults)")
        for i in range(len(y_test)):
            log_msg = (
                f"Sample {i} | True: {y_test[i]} | "
                f"Base Prob: {baseline_probs[i]:.4f} (Pred: {baseline_preds[i]}) | "
                f"Extracted Prob: {extracted_probs[i]:.4f} (Pred: {extracted_preds[i]})"
            )
            logger.info(log_msg)

        logger.info(f"Extracted Probs - Mean: {np.mean(extracted_probs):.4f}, Std: {np.std(extracted_probs):.4f}")

    extracted_acc = accuracy_score(y_test, extracted_preds)
    extracted_auc = roc_auc_score(y_test, extracted_probs) if len(np.unique(y_test)) > 1 else 0.0
    extracted_agreement = accuracy_score(baseline_preds, extracted_preds)
    extracted_prob_shift = np.mean(np.abs(extracted_probs - baseline_probs))

    metrics = {
        'extracted_accuracy': extracted_acc,
        'extracted_auc': extracted_auc,
        'extracted_agreement': extracted_agreement,
        'extracted_mean_prob_shift': extracted_prob_shift,
        'augmented_accuracy': 0,
        'augmented_auc': 0
    }

    if X_train is not None:
        X_aug = np.vstack([X_train, X_extracted])
        y_aug = np.concatenate([y_train, y_extracted])
        aug_model = ModelWrapper(input_dim=X_aug.shape[1], model_type=model_type)
        aug_model.train(X_aug, y_aug, num_epochs=num_epochs, lr=alpha, verbose=verbose)
        aug_probs, aug_preds = aug_model.predict(X_test)
        metrics['augmented_accuracy'] = accuracy_score(y_test, aug_preds)
        metrics['augmented_auc'] = roc_auc_score(y_test, aug_probs)

    return metrics

def calculate_quality_score(
        extracted_acc: float,
        extracted_auc: float,
        extracted_agreement: float,
        augmented_acc: float,
        augmented_auc: float,
        cf_mean_l1: float,
        normalization_factor: float = 10.0
) -> float:
    """
    Calculate composite quality score - Linear ocombination of metrics
    """
    normalized_l1 = max(0, 1 - (cf_mean_l1 / normalization_factor))

    quality = (
        # 0.25 * extracted_acc +
        # 0.20 * extracted_auc +
        # 0.20 * extracted_agreement +
        # 0.10 * augmented_acc +
        # 0.10 * augmented_auc +
        # 0.15 * normalized_l1
            0.40 * extracted_acc +
            0.40 * extracted_auc +
            0.10 * extracted_agreement +
            0.0 * augmented_acc +
            0.0 * augmented_auc +
            0.10 * normalized_l1
    )
    return quality
