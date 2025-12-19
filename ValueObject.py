from dataclasses import dataclass
from typing import List, Union
import numpy as np


@dataclass
class ValueObject:
    delta_max_values: List[float]
    lambda_values: List[float]
    alpha_values: List[float]
    norm_values:List[Union[int, float]]
    X_train: np.ndarray
    y_train: np.ndarray
    X_query: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]

