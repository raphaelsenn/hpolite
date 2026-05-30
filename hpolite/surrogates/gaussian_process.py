from typing import Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel

from hpolite.surrogates.base import BaseSurrogateModel


class GaussianProcess(BaseSurrogateModel):
    def __init__(self, kernel: Kernel | None = None, random_state: int = 0) -> None:
        super().__init__()
        self.surrogate_model = GaussianProcessRegressor(kernel, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X_ = X.tolist()
        self.y_ = y.tolist() 
        self.surrogate_model.fit(X, y)

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.surrogate_model.predict(X_test, return_std=True)
    
    def sample(self, X: np.ndarray) -> np.ndarray:
        return self.surrogate_model.sample_y(X)