from __future__ import annotations

from typing import Dict, Any, Callable
import copy

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

import numpy as np

from hpolite.optimizers.base import BaseOptimizer
from hpolite.space import Categorical, Integer, Real


class RandomSearchCV(BaseOptimizer):
    """
    Evaluates a param grid randomly for a given number of iterations. 

    Reference:
    ---------- 
    Random Search for Hyper-Parameter Optimization, Bergstra and Bengio, 2012 
    https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
    """ 
    def __init__(
            self, 
            estimator: BaseEstimator, 
            param_dict: Dict[str, Categorical | Integer | Real],
            n_iter: int = 10,
            cv = 5, 
            scoring: Callable | None = None,
            random_state: int = 0
    ) -> None:
        super().__init__(estimator, param_dict, cv, scoring)
        self.n_iter = n_iter
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state) 

    def _sample_candidate(self) -> dict[str, Any]:
        params = {}
        for key, value in self.param_dict.items():
            params[key] = value.sample(self.rng)
        return params

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> RandomSearchCV:
        for _ in range(self.n_iter):
            params = self._sample_candidate()
            
            model = copy.deepcopy(self.estimator) 
            model.set_params(**params)

            scores = cross_val_score(model, X, y, scoring=self.scoring, cv=self.cv)
            mean_score = np.mean(scores)

            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params

        model = copy.deepcopy(self.estimator) 
        model.set_params(**self.best_params_)
        self.best_estimator_ = model.fit(X, y)

        return self