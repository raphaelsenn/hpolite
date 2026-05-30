from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable

from sklearn.base import BaseEstimator
import numpy as np

from hpolite.space.base import BaseParameter

class BaseOptimizer(ABC):
    """Base optimizer class (maximization).""" 
    def __init__(
            self, 
            estimator: BaseEstimator,
            param_dict: Dict[str, BaseParameter],
            cv: int = 5,
            scoring: Callable | None = None
    ) -> None:
        """
        Parameters:
        ----------- 
        estimator : BaseEstimator (sklearn)
            Estimator to be evaluated.

        param_dict : Dict[str, BaseParameter] 
            Dictionary where keys are base parameters.

        cv : int, defalt = 5
            Determines the cross-validation splitting strategy.

        scoring : None | Callable, default = None 
            The score method for the estimator. 
        """ 
        super().__init__()

        self.estimator = estimator
        self.param_dict = param_dict
        self.cv = cv
        self.scoring = scoring

        self.best_estimator_ = None
        self.best_params_ = None
        self.best_index_ = None 
        self.best_score_ = float("-inf")

    @abstractmethod
    def fit(X: np.ndarray, y: np.ndarray | None = None) -> BaseEstimator:
        raise NotImplementedError