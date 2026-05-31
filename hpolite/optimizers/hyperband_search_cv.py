from __future__ import annotations

from typing import Dict, List, Callable, Tuple
import copy
import math

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

import numpy as np

from hpolite.optimizers.base import BaseOptimizer
from hpolite.optimizers.halving_random_search_cv import HalvingRandomSearchCV
from hpolite.space import Categorical, Integer, Real


class HyperbandSearchCV(BaseOptimizer):
    """
    Hyperband algorithm that uses Successive Halving Random Search as a subroutine.

    NOTE: The resource (budget) is the number of training samples used for evalutation.

    Reference:
    ----------
    Non-stochastic Best Arm Identification and Hyperparameter Optimization 
    https://arxiv.org/pdf/1502.07943

    Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization, Jamieson et al., 2016
    https://arxiv.org/abs/1603.06560    

    BOHB: Robust and Efficient Hyperparameter Optimization at Scale, Hutter et al., 2018
    https://arxiv.org/abs/1807.01774 
    """ 
    def __init__(
            self, 
            estimator: BaseEstimator, 
            param_dict: Dict[str, Categorical | Integer | Real],
            factor: int = 2,
            min_resources: int = 50,
            max_resources: int | None = None,
            cv = 5,
            scoring: Callable | None = None,
            random_state: int = 0
    ) -> None:
        super().__init__(estimator, param_dict, cv, scoring)

        if factor <= 1:
            raise ValueError("factor must be > 1")

        if min_resources <= 0:
            raise ValueError("min_resources must be > 0")

        self.factor = factor 
        self.min_resources = min_resources
        self.max_resources = max_resources
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> HyperbandSearchCV:
        self.results_ = []

        eta = self.factor
        min_resources = self.min_resources
        max_resources = self.max_resources
        if max_resources is None:
            max_resources = len(X)

        # Read more here: https://arxiv.org/abs/1807.01774
        ratio = max_resources / min_resources
        s_max = int(math.log(ratio, eta))
        for s in range(s_max, -1, -1):
            n_candidates = math.ceil((s_max + 1) / (s + 1) * eta**s)
            budget = int((eta**(-s)) * max_resources)

            sh = HalvingRandomSearchCV(
                estimator=self.estimator,
                param_dict=self.param_dict,
                n_candidates=n_candidates,
                factor=self.factor,
                min_resources=budget,
                max_resources=max_resources,
                cv=self.cv,
                scoring=self.scoring,
                random_state=self.random_state + s
            )
            sh.fit(X, y)

            self.results_.append(sh.results_)

            if sh.best_score_ >= self.best_score_:
                self.best_estimator_ = sh.best_estimator_
                self.best_params_ = sh.best_params_
                self.best_score_ = sh.best_score_

        return self