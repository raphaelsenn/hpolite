from abc import ABC, abstractmethod

import numpy as np


class BaseSurrogateModel(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.X_ = []
        self.y_ = []

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(self, x: float, y: float) -> None:
        self.X_.append(x)
        self.y_.append(y)

        X = np.asarray(self.X_, dtype=np.float32)
        y = np.asarray(self.y_, dtype=np.float32)

        self.fit(X, y)

    def get_incumbent(self) -> float:
        assert len(self.y_) > 0, f"Fit surrogate model first." 
        return min(self.y_)