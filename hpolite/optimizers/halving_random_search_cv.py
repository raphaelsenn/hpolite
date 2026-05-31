from __future__ import annotations

from typing import Dict, List, Callable, Tuple
import copy

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

import numpy as np

from hpolite.optimizers.base import BaseOptimizer
from hpolite.space import Categorical, Integer, Real


class HalvingRandomSearchCV(BaseOptimizer):
    """
    Successive Halving Random Search.

    NOTE: The resource (budget) is the number of training samples used for evalutation.

    Reference:
    ----------
    Non-stochastic Best Arm Identification and Hyperparameter Optimization 
    https://arxiv.org/pdf/1502.07943
    """ 
    def __init__(
            self, 
            estimator: BaseEstimator, 
            param_dict: Dict[str, Categorical | Integer | Real],
            n_candidates: int = 32,
            factor: int = 2,
            min_resources: int = 50,
            max_resources: int | None = None,
            cv = 5,
            scoring: Callable | None = None,
            random_state: int = 0
    ) -> None:
        super().__init__(estimator, param_dict, cv, scoring)

        if n_candidates <= 0:
            raise ValueError("n_candidates must be > 0") 

        if factor <= 1:
            raise ValueError("factor must be > 1")

        if min_resources <= 0:
            raise ValueError("min_resources must be > 0")

        self.n_candidates = n_candidates
        self.factor = factor 
        self.min_resources = min_resources
        self.max_resources = max_resources
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state) 

    def _sample_candidate(self) -> Dict[str, Categorical | Integer | Real]:
        params = {}
        for key, value in self.param_dict.items():
            params[key] = value.sample(self.rng)
        return params

    def _sample_candidates(self) -> List[Dict[str, Categorical | Integer | Real]]:
        return [self._sample_candidate() for _ in range(self.n_candidates)]

    def _sample_data(
            self, 
            X: np.ndarray, 
            y: np.ndarray | None, 
            n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        n_total = len(X)
        n_samples = min(n_samples, n_total)
        indices = self.rng.choice(n_total, size=n_samples, replace=False)

        X_sub = X[indices]
        y_sub = y[indices] if y is not None else None

        return X_sub, y_sub

    def _evaluate_candidate(
            self, 
            params: Dict[str, Categorical | Integer | Real],
            X: np.ndarray,
            y: np.ndarray | None,
            n_resources: int
    ) -> float:
        X_sub, y_sub = self._sample_data(X, y, n_resources)
        
        model = copy.deepcopy(self.estimator)
        model.set_params(**params)

        scores = cross_val_score(model, X_sub, y_sub, scoring=self.scoring, cv=self.cv)
        score = float(np.mean(scores))

        return score

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> HalvingRandomSearchCV:
        self.results_ = []

        max_resources = self.max_resources 
        if max_resources is None:
            max_resources = len(X)

        configs = {id: candidate for id, candidate in enumerate(self._sample_candidates())}
        configs_to_eval = list(range(len(configs)))

        n_resources = self.min_resources
        round_idx = 1
        
        while n_resources <= max_resources:
            round_results = []

            for id in configs_to_eval: 
                params = configs[id] 
                score = self._evaluate_candidate(params, X, y, n_resources)
                round_results.append((score, id))

                if score >= self.best_score_:
                    self.best_score_ = score
                    self.best_params_ = params

            # Sort from best to worst
            round_results.sort(key=lambda pair: pair[0], reverse=True)

            for score, id in round_results: 
                self.results_.append({
                    "round" : round_idx,
                    "n_resources": n_resources,
                    "score" : float(score),
                    "candidate": id,
                })

            n_keep = max(1, len(configs_to_eval) // self.factor)
            top_k = round_results[:n_keep]
            configs_to_eval = [id for _, id in top_k]
            
            # Increasing the resources for the next iteration
            n_resources = int(n_resources * self.factor)
            round_idx += 1
        
        model = copy.deepcopy(self.estimator)
        print() 
        model.set_params(**self.best_params_)
        model.fit(X, y)
        
        self.best_estimator_ = model

        return self