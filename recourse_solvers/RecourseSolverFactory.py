from typing import Optional
import numpy as np
from ValueObject import ValueObject
from optimal_recourse.src.recourse import LARRecourse, ROARLInf
from recourse_methods import RobustRecourse

class RecourseSolverFactory:
    def __init__(
            self,
            vo: ValueObject,
            recourse_method: str,
            W: Optional[np.ndarray] = None,
            W0: Optional[np.ndarray] = None
    ):
        self.recourse_method = recourse_method
        self.delta_max = vo.delta_max_values[0]
        self.lamb = vo.lambda_values[0]
        self.norm = vo.norm_values[0]

        self.persistent_solvers = {}
        if recourse_method == 'roar':
            self.persistent_solvers[1] = RobustRecourse(W=W, W0=W0, y_target=1, delta_max=self.delta_max)
            self.persistent_solvers[0] = RobustRecourse(W=W, W0=W0, y_target=0, delta_max=self.delta_max)

    def create(
            self,
            local_W: np.ndarray,
            local_W0: np.ndarray,
            target: int,
            use_lime: bool
    ):
        if self.recourse_method == 'roar':
            solver = self.persistent_solvers[target]
            if use_lime:
                solver.set_W_lime(local_W)
                solver.set_W0_lime(local_W0)
            else:
                solver.set_W(local_W)
                solver.set_W0(local_W0)
            return solver

        weights = -local_W.flatten() if target == 0 else local_W.flatten()
        bias = -local_W0.flatten() if target == 0 else local_W0.flatten()

        if self.norm == 1:
            return LARRecourse(weights=weights, bias=bias, alpha=self.delta_max, lamb=self.lamb)
        elif self.norm in [float('inf'), 'inf']:
            return ROARLInf(weights=weights, bias=bias, alpha=self.delta_max, lamb=self.lamb)
        raise ValueError(f"Unsupported norm: {self.norm}")