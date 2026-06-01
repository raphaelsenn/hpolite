from __future__ import annotations

import copy
from typing import Dict, Any, Callable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

from hpolite.optimizers.base import BaseOptimizer
from hpolite.surrogates.base import BaseSurrogateModel
from hpolite.acquisition.base import BaseAquisitionFunction
from hpolite.space import Categorical, Integer, Real


class BayesianSearchCV(BaseOptimizer):
    """
    Bayesian optimization over hyperparameters. 
    """
    def __init__(
            self,
            estimator: BaseEstimator,
            surrogate_model: BaseSurrogateModel,
            aquisition_func: BaseAquisitionFunction,
            param_dict: Dict[str, Categorical | Integer | Real],
            cv: int = 5,
            n_iter: int = 10,
            n_aquisition_samples: int = 10,
            n_initial_points: int = 5,
            scoring: Callable | None = None,
            random_state: int = 0
    ) -> None:
        super().__init__(estimator, param_dict, cv, scoring)

        self.surrogate_model = surrogate_model
        self.aquisition_func = aquisition_func

        self.n_iter = n_iter
        self.n_aquisition_samples = n_aquisition_samples
        self.n_initial_points = n_initial_points
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def _encode_candidate(self, params: Dict[str, Any]) -> np.ndarray:
        encoded = []
        for key in self.param_dict:
            parameter_space = self.param_dict[key]  # Categorical, Integer or Real
            value = params[key]
            encoded.append(parameter_space.encode(value))
        encoded = np.concatenate(encoded, dtype=np.float32)
        return encoded

    def _select_next_candidate(self) -> Dict[str, Any]:
        """
        Random Search that simply maximizes the utility score of an aquisition function 
        using random hyperparameter configurations.
        """
        best_params = None
        best_score = float("-inf")

        for _ in range(self.n_aquisition_samples):
            # Sample random candidate and encode
            params = self._sample_candidate()                             # Param dict
            params_enc = self._encode_candidate(params).reshape(1, -1)    # [1, D], np.ndarray

            # Compute utility score of candidate using aquistion function and surrogate
            score = self.aquisition_func.compute(params_enc)               # [1]
            score = float(score.item())

            if score >= best_score:
                best_score = score
                best_params = params
        
        return best_params

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> BayesianSearchCV:
        X_candidates = [] 
        y_scores = []

        # Sample initial observations
        for _ in range(self.n_initial_points): 
            # Sample initial candidate
            params = self._sample_candidate()               # Param dict
            params_enc = self._encode_candidate(params)     # [1, D], (np.ndarray)

            # Evaluate inital candidate
            score = self._evaluate_candidate(params)

            # Update observations
            X_candidates.append(params_enc)
            y_scores.append(score)

        # Initial candidate/score history
        X_candidates = np.vstack(X_candidates, dtype=np.float32)    # [n_initial_points, D]
        y_scores = np.asarray(y_scores, dtype=np.float32)           # [n_initial_points]

        for _ in range(self.n_iter):
            # Fit predictive model 
            self.surrogate_model.fit(X_candidates, y_scores)
            self.aquisition_func.update(self.surrogate_model)

            # Select next hyperparameter configuration (greedy w.r.t aquisition function)
            params = self._select_next_candidate()
            params_enc = self._encode_candidate(params)

            # Evaluate hyperparameter configuration
            score = self._evaluate_candidate(params)

            # Update observations
            X_candidates = np.concatenate([X_candidates, np.asarray([params_enc])]) # [N, D]
            y_scores = np.concatenate([y_scores, np.asarray([score])])              # [N]

            if score >= self.best_score_:
                self.best_score_ = score
                self.best_params_ = params
    
        model = copy.deepcopy(self.estimator)
        model.set_params(**self.best_params_)
        model.fit(X, y)
        self.best_estimator_ = model

        return self