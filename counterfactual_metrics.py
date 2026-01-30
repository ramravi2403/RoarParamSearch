
import traceback
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

import ValueObject
from models.ModelWrapper import ModelWrapper
from optimal_recourse.src.recourse import Recourse, LARRecourse, ROARLInf
from recourse_methods import RobustRecourse
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


def generate_cfs_ccfs2(
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

    metrics = {'cf_success': 0, 'cf_failures': 0, 'ccf_success': 0, 'ccf_failures': 0,
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
            print(f"[DEBUG CF {i}] EXCEPTION during generation: {str(e)}")
            traceback.print_exc()

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
        except Exception as e:
            metrics['ccf_failures'] += 1
            print(f"[DEBUG CCF {i}] EXCEPTION during generation: {str(e)}")
            traceback.print_exc()

    return cfs_df, (pd.concat(ccf_list, ignore_index=True) if ccf_list else pd.DataFrame()), metrics
def generate_cfs_ccfs(
        X_query: np.ndarray,
        baseline_model: ModelWrapper,
        W: np.ndarray,
        W0: np.ndarray,
        delta_max: float,
        lamb: float,
        feature_names: List[str],
        X_train: np.ndarray = None,
        norm=1,
        query_size_pct: float = 1.0,
        random_seed: int = 42,
        verbose: bool = False,
        recourse_method: str = 'roar',
        model_type: str = 'simple'
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    if recourse_method not in ['roar', 'optimal']:
        raise ValueError('recourse method needs to be roar or optimal')
    use_lime = (W is None or W0 is None)
    if use_lime and X_train is None:
        raise ValueError("X_train must be provided when using LIME (deep model)")
    """Generate CFs and CCFs with given parameters."""
    metrics = {
        'cf_success': 0,
        'cf_failures': 0,
        'ccf_success': 0,
        'ccf_failures': 0,
        'cf_distances': [],
        'ccf_distances': []
    }

    predict_fn = lambda x: baseline_model.predict(x)[1]
    denied_indices = recourse_needed(predict_fn, X_query, target=1)

    if verbose:
        print(f"    Found {len(denied_indices)} denied queries out of {len(X_query)} total")

    if query_size_pct < 1.0:
        np.random.seed(random_seed)
        n_samples = max(1, int(len(denied_indices) * query_size_pct))
        sampled_denied_indices = np.random.choice(
            denied_indices,
            size=n_samples,
            replace=False
        )
    else:
        sampled_denied_indices = denied_indices

    rr_cf_persistent = None
    rr_ccf_persistent = None

    if recourse_method == 'roar':
        rr_cf_persistent = RobustRecourse(W=W, W0=W0, y_target=1, delta_max=delta_max)
        rr_ccf_persistent = RobustRecourse(W=W, W0=W0, y_target=0, delta_max=delta_max)
    cf_list = []

    for i in sampled_denied_indices:
        x_i = np.array(X_query[i], dtype=np.float32)
        if use_lime:
            def predict_proba_fn(x):
                if len(x.shape) == 1:
                    x = x.reshape(1, -1)
                probs, _ = baseline_model.predict(x)
                probs = probs.flatten()
                return np.vstack([1 - probs, probs]).T
            local_W, local_W0 = lime_explanation(predict_proba_fn, X_train, x_i)
        else:
            local_W, local_W0 = W, W0
        try:
            if recourse_method == 'roar':
                rr_cf_persistent.set_W_lime(local_W) if use_lime else rr_cf_persistent.set_W(local_W)
                rr_cf_persistent.set_W0_lime(local_W0) if use_lime else rr_cf_persistent.set_W0(local_W0)
                solver = rr_cf_persistent
            else:
                if norm == 1:
                    solver = LARRecourse(
                        weights=local_W.flatten(),
                        bias=local_W0.flatten(),
                        alpha=delta_max,
                        lamb=lamb
                    )
                elif norm in [float('inf'), 'inf']:
                    solver = ROARLInf(
                        weights=local_W.flatten(),
                        bias=local_W0.flatten(),
                        alpha=delta_max,
                        lamb=lamb
                    )
            if recourse_method == 'roar':
                cf, _ = solver.get_recourse(x_i, lamb=lamb, norm=norm, lime=use_lime)
            else:
                cf = solver.get_recourse(x_i)
            if cf.ndim == 1:
                cf = cf.reshape(1, -1)
            dist = np.linalg.norm(cf.flatten() - x_i, ord=norm)
            metrics['cf_distances'].append(dist)
            metrics['cf_success'] += 1

            cf_df = pd.DataFrame(cf, columns=feature_names)
            cf_df["original_query_idx"] = i
            cf_df["distance"] = dist
            cf_list.append(cf_df)
        except Exception as e:
            metrics['cf_failures'] += 1
            print(f"[DEBUG CF {i}] EXCEPTION during generation: {str(e)}")
            traceback.print_exc()
            if verbose:
                print(f"    [CF Error] idx={i}: {e}")

    if not cf_list:
        return pd.DataFrame(), pd.DataFrame(), metrics

    cfs_df = pd.concat(cf_list, ignore_index=True)
    ccf_list = []

    for i in range(len(cfs_df)):
        x_cf = cfs_df[feature_names].iloc[i].values.astype(np.float32)
        original_idx = int(cfs_df.iloc[i]["original_query_idx"])
        x_original = X_query[original_idx]

        if use_lime:
            def predict_proba_fn(x):
                if len(x.shape) == 1:
                    x = x.reshape(1, -1)
                probs, _ = baseline_model.predict(x)
                probs = probs.flatten()
                return np.vstack([1 - probs, probs]).T

            local_W, local_W0 = lime_explanation(predict_proba_fn, X_train, x_cf)
        else:
            local_W, local_W0 = W, W0

        try:
            if recourse_method == 'roar':
                rr_ccf_persistent.set_W_lime(local_W) if use_lime else rr_ccf_persistent.set_W(local_W)
                rr_ccf_persistent.set_W0_lime(local_W0) if use_lime else rr_ccf_persistent.set_W0(local_W0)
                solver_ccf = rr_ccf_persistent
            else:
                if norm == 1:
                    solver_ccf = LARRecourse(
                        weights=-local_W.flatten(),
                        bias=-local_W0.flatten(),
                        alpha=delta_max,
                        lamb=lamb
                    )
                elif norm in [float('inf'), 'inf']:
                    solver_ccf = ROARLInf(
                        weights=-local_W.flatten(),
                        bias=-local_W0.flatten(),
                        alpha=delta_max,
                        lamb=lamb
                    )

            if recourse_method == 'roar':
                ccf, _ = solver_ccf.get_recourse(x_cf, lamb=lamb, norm=norm, lime=use_lime)
            else:
                ccf = solver_ccf.get_recourse(x_cf)

            if ccf.ndim == 1:
                ccf = ccf.reshape(1, -1)

            dist = np.linalg.norm(ccf.flatten() - x_original, ord=norm)
            metrics['ccf_distances'].append(dist)
            metrics['ccf_success'] += 1

            ccf_df = pd.DataFrame(ccf, columns=feature_names)
            ccf_df["original_query_idx"] = original_idx
            ccf_df["distance"] = dist
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
        baseline_model: ModelWrapper,
        feature_names: List[str],
        alpha: float,
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
        print(f"    Training extracted model (alpha={alpha})...")
    extracted_model = ModelWrapper(input_dim=X_extracted.shape[1], model_type=model_type)
    extracted_model.train(
        X_extracted, y_extracted,
        num_epochs=num_epochs,
        lr=alpha,
        verbose=verbose,
    )

    baseline_probs, baseline_preds = baseline_model.predict(X_test)
    extracted_probs, extracted_preds = extracted_model.predict(X_test)

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
        normalization_factor: float = 10.0  # Normalization factor for L1
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
