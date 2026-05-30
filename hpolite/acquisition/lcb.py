import numpy as np
from scipy.stats import norm as Normal

from hpolite.acquisition.base import BaseAquisitionFunction
from hpolite.surrogates.base import BaseSurrogateModel


class LCB(BaseAquisitionFunction):
    """
    Lower confidence bound (LCB) aquisition function.
    The LCB aquisition function is given by:

        LCB(x) := -(mu(x) - alpha * std)

    NOTE: The next hyperparameters are selected according to: x_next = max_{x in X} LCB(x)  
    """ 
    def __init__(self, surrogate: BaseSurrogateModel, alpha: float) -> None:
        super().__init__(surrogate)
        self.alpha = alpha

    def compute(self, x: np.ndarray) -> np.ndarray:
        mean, std = self.surrogate.predict(x)
        return -(mean + self.alpha * std)