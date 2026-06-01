import copy
from itertools import product
from typing import Dict, Callable, Generator

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

from hpolite.optimizers.base import BaseOptimizer
from hpolite.space.categorical import Categorical


class GridSearchCV(BaseOptimizer):
    """
    Evaluates a param grid to maximize the performance of an estimator.
    """
    def __init__(
            self,
            estimator: BaseEstimator,
            param_dict: Dict[str, Categorical],
            cv: int = 5,
            scoring: Callable | None = None
    ) -> None:
        """
        Parameters:
        ----------- 
        estimator : BaseEstimator (sklearn)
            Estimator to be evaluated.

        param_dict : Dict[str, Categorical] 
            Parameter dictionary with categorical values
        """ 
        super().__init__(estimator, param_dict, cv, scoring)

    def _param_combinations(self) -> Generator:
        keys = list(self.param_dict.keys())
        values = list(cat.grid() for cat in self.param_dict.values())
        for combination in product(*values):
            yield dict(zip(keys, combination))

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "GridSearchCV":
        for params in self._param_combinations():
            model = copy.deepcopy(self.estimator)
            model.set_params(**params) 

            scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring)
            score = np.mean(scores)
            
            if score > self.best_score_:
                self.best_params_ = params
                self.best_score_ = score

        model = copy.deepcopy(self.estimator)
        model.set_params(**self.best_params_)
        self.best_estimator_ = model.fit(X, y)

        return self