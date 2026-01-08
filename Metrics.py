from dataclasses import dataclass
from typing import Union


@dataclass
class Metrics:
    """Results for a single (delta_max, lamb, alpha) combination."""
    # Parameters
    model_type: str
    delta_max: float
    lamb: float
    alpha: float
    norm: Union[int, float]
    n_cfs_generated: int
    n_ccfs_generated: int
    cf_success_rate: float
    ccf_success_rate: float
    cf_mean_distance: float
    cf_std_distance: float
    cf_min_distance: float
    cf_max_distance: float
    ccf_mean_distance: float
    ccf_std_distance: float
    extracted_accuracy: float
    extracted_auc: float
    extracted_agreement: float
    extracted_mean_prob_shift: float
    augmented_accuracy: float
    augmented_auc: float
    query_size_pct: float
    quality_score: float