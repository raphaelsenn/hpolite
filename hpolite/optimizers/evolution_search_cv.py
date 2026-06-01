from __future__ import annotations

from typing import Dict, List, Callable, Any
import copy

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

import numpy as np

from hpolite.optimizers.base import BaseOptimizer
from hpolite.space import BaseParameter, Categorical, Integer, Real


class EvolutionSearchCV(BaseOptimizer):
    """
    Evolutionary algorithm applied to sklearn hypeparameter optimization.
    """ 
    def __init__(
            self, 
            estimator: BaseEstimator, 
            param_dict: Dict[str, BaseParameter],
            population_size: int = 10,
            n_generations: int = 40, 
            n_children_per_step: int = 5,
            mutation_probability: float = 0.1,
            crossover_probability: float = 0.8,
            selection_type: str = "neutral",
            cv = 5, 
            scoring: Callable | None = None,
            random_state: int = 0
    ) -> None:
        super().__init__(estimator, param_dict, cv, scoring)
        self.population_size = population_size
        self.n_generations = n_generations
        self.n_children_per_step = n_children_per_step
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.selection_type = selection_type
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state) 

    def _sample_candidate(self) -> Dict[str, Any]:
        params = {}
        for key, value in self.param_dict.items():
            params[key] = value.sample(self.rng)
        return params

    def _sample_population(self) -> List[Dict[str, Any]]:
        return [self._sample_candidate() for _ in range(self.population_size)]

    def _evaluate_candidate(
            self, 
            params: Dict[str, Any],
            X: np.ndarray,
            y: np.ndarray | None
    ) -> float:
        model = copy.deepcopy(self.estimator) 
        model.set_params(**params)

        scores = cross_val_score(model, X, y, scoring=self.scoring, cv=self.cv)
        score = float(np.mean(scores))
        
        return score

    def _evaluate_population(
            self, 
            population: List[Dict[str, Any]],
            X: np.ndarray,
            y: np.ndarray | None
    ) -> List[float]:
        fitness = []
        
        for params in population:
            score = self._evaluate_candidate(params, X, y)
            fitness.append(score)

        return fitness

    def _mutate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Uniform mutation.""" 
        params = copy.deepcopy(params) 

        for key, space in self.param_dict.items():
            if self.rng.random() < self.mutation_probability:
                params[key] = space.sample(self.rng)

        return params

    def _crossover(self, params: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
        """Uniform crossover.""" 
        params = copy.deepcopy(params)
        other = copy.deepcopy(other)

        for key, _ in self.param_dict.items():
            if self.rng.random() < self.crossover_probability:
                params[key] = other[key]

        return params

    def _select_parents(self, population: List[Dict[str, Any]], fitness: List) -> None:
        if self.selection_type == "neutral":
            parents = self.rng.choice(len(population), size=self.n_children_per_step, replace=False)
        elif self.selection_type == "fitness":
            # Stable softmax: https://stackoverflow.com/questions/42599498/numerically-stable-softmax
            fitness_shifted = np.array(fitness) - max(fitness) 
            probs = np.exp(fitness_shifted) / np.exp(fitness_shifted).sum()
            parents = self.rng.choice(len(population), p=probs, size=self.n_children_per_step, replace=True)
        else:
            raise NotImplementedError() 

        parents = [population[id] for id in parents]

        return parents

    def _surival_selection(
            self, 
            population: List[Dict[str, Any]], 
            fitness: List[float]
    ) -> List[Dict[str, Any]]:
        order = np.argsort(fitness)[::-1]
        keep = order[:self.population_size] 
        
        population = [population[id] for id in keep]
        fitness = [fitness[id] for id in keep]

        return population, fitness

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> EvolutionSearchCV:
        population = self._sample_population()
        fitness = self._evaluate_population(population, X, y)

        for params, score in zip(population, fitness):
            if score >= self.best_score_:
                self.best_score_ = score
                self.best_params_ = copy.deepcopy(params)

        for _ in range(self.n_generations):
            parents = self._select_parents(population, fitness)
            for params in parents:

                # Crossover 
                other_id = self.rng.choice(len(parents))
                other = parents[other_id]
                params = self._crossover(params, other)

                # Mutation 
                params = self._mutate(params)

                # Evaluate
                score = self._evaluate_candidate(params, X, y)
                population.append(params)
                fitness.append(score)

                if score >= self.best_score_:
                    self.best_score_ = score
                    self.best_params_ = params

            population, fitness = self._surival_selection(population, fitness)

        model = copy.deepcopy(self.estimator)
        model.set_params(**self.best_params_)
        model.fit(X, y)
        self.best_estimator_ = model

        return self