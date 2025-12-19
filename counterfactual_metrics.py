from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from torch import nn

from Model import train_classifier, predict_with_model
from recourse_methods import RobustRecourse
from recourse_utils import recourse_needed


def generate_cfs_ccfs(
        X_query: np.ndarray,
        baseline_model: nn.Module,
        W: np.ndarray,
        W0: np.ndarray,
        delta_max: float,
        lamb: float,
        feature_names: List[str],
        query_size_pct: float = 1.0,  # NEW
        random_seed: int = 42,
        verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Generate CFs and CCFs with given parameters."""
    metrics = {
        'cf_success': 0,
        'cf_failures': 0,
        'ccf_success': 0,
        'ccf_failures': 0,
        'cf_l1_distances': [],
        'ccf_l1_distances': []
    }


    predict_fn = lambda x: predict_with_model(baseline_model, x)[1]
    denied_indices = recourse_needed(predict_fn, X_query, target=1)

    if verbose:
        print(f"    Found {len(denied_indices)} denied queries out of {len(X_query)} total")

    # NEW: Subsample denied queries
    if query_size_pct < 1.0:
        np.random.seed(random_seed)
        n_samples = max(1, int(len(denied_indices) * query_size_pct))
        sampled_denied_indices = np.random.choice(
            denied_indices,
            size=n_samples,
            replace=False
        )
        if verbose:
            print(f"    Using {len(sampled_denied_indices)} denied queries ({query_size_pct * 100:.0f}%)")
    else:
        sampled_denied_indices = denied_indices

    rr_cf = RobustRecourse(W=W, W0=W0, y_target=1, delta_max=delta_max)
    cf_list = []

    for i in sampled_denied_indices:  # CHANGED from range(len(X_query))
        x_i = np.array(X_query[i], dtype=np.float32)
        try:
            cf, _ = rr_cf.get_recourse(x_i, lamb=lamb)
            if cf.ndim == 1:
                cf = cf.reshape(1, -1)

            l1_dist = l1_distance(x_i, cf.flatten())
            metrics['cf_l1_distances'].append(l1_dist)
            metrics['cf_success'] += 1

            cf_df = pd.DataFrame(cf, columns=feature_names)
            cf_df["original_query_idx"] = i
            cf_df["l1_distance"] = l1_dist
            cf_list.append(cf_df)
        except Exception as e:
            metrics['cf_failures'] += 1
            if verbose:
                print(f"    [CF Error] idx={i}: {e}")

    if not cf_list:
        return pd.DataFrame(), pd.DataFrame(), metrics

    cfs_df = pd.concat(cf_list, ignore_index=True)

    rr_ccf = RobustRecourse(W=W, W0=W0, y_target=0, delta_max=delta_max)
    ccf_list = []

    for i in range(len(cfs_df)):
        x_cf = cfs_df[feature_names].iloc[i].values.astype(np.float32)
        original_idx = int(cfs_df.iloc[i]["original_query_idx"])
        x_original = X_query[original_idx]

        try:
            ccf, _ = rr_ccf.get_recourse(x_cf, lamb=lamb)
            if ccf.ndim == 1:
                ccf = ccf.reshape(1, -1)

            l1_dist = l1_distance(x_original, ccf.flatten())
            metrics['ccf_l1_distances'].append(l1_dist)
            metrics['ccf_success'] += 1

            ccf_df = pd.DataFrame(ccf, columns=feature_names)
            ccf_df["original_query_idx"] = original_idx
            ccf_df["l1_distance"] = l1_dist
            ccf_list.append(ccf_df)
        except Exception as e:
            metrics['ccf_failures'] += 1
            if verbose:
                print(f"    [CCF Error] idx={i}: {e}")

    if ccf_list:
        ccfs_df = pd.concat(ccf_list, ignore_index=True)
    else:
        ccfs_df = pd.DataFrame()

    return cfs_df, ccfs_df, metrics


def evaluate_extraction(
        cfs_df: pd.DataFrame,
        ccfs_df: pd.DataFrame,
        X_test: np.ndarray,
        y_test: np.ndarray,
        baseline_model: nn.Module,
        feature_names: List[str],
        alpha: float,
        X_train=None,
        y_train=None,
        num_epochs: int = 50,
        verbose: bool = False,

) -> Dict:
    """Train extracted and augmented models, evaluate on test set."""

    # Prepare CF/CCF training data
    X_cfs = cfs_df[feature_names].values.astype(np.float32)
    X_ccfs = ccfs_df[feature_names].values.astype(np.float32)

    _, y_cfs = predict_with_model(baseline_model, X_cfs)
    _, y_ccfs = predict_with_model(baseline_model, X_ccfs)

    X_extracted = np.vstack([X_cfs, X_ccfs])
    y_extracted = np.concatenate([y_cfs, y_ccfs])

    if verbose:
        print(f"    Training extracted model (alpha={alpha})...")
    extracted_model = train_classifier(
        X_extracted, y_extracted,
        input_dim=X_extracted.shape[1],
        num_epochs=num_epochs,
        lr=alpha,
        verbose=verbose
    )

    baseline_probs, baseline_preds = predict_with_model(baseline_model, X_test)
    extracted_probs, extracted_preds = predict_with_model(extracted_model, X_test)

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
        # Combine original data with CFs/CCFs
        X_aug = np.vstack([X_train, X_extracted])
        y_aug = np.concatenate([y_train, y_extracted])

        aug_model = train_classifier(X_aug, y_aug, input_dim=X_aug.shape[1], lr=alpha)
        aug_probs, aug_preds = predict_with_model(aug_model, X_test)

        metrics['augmented_accuracy'] = accuracy_score(y_test, aug_preds)
        metrics['augmented_auc'] = roc_auc_score(y_test, aug_probs)

    return metrics

def l1_distance(x_original, x_counterfactual):
    """Calculate L1 distance between original and counterfactual."""
    return np.sum(np.abs(x_counterfactual - x_original))

def calculate_quality_score(
    extracted_acc: float,
    extracted_auc: float,
    extracted_agreement: float,
    augmented_acc: float,
    augmented_auc: float,
    cf_mean_l1: float,
    normalization_factor: float = 10.0  # Normalization factor for L1
) -> float:
    """
    Calculate composite quality score.

    Higher extraction metrics = better
    Lower L1 distance = better (more minimal changes)
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