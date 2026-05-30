import numpy as np
from scipy.stats import norm as Normal

from hpolite.acquisition.base import BaseAquisitionFunction
from hpolite.surrogates.base import BaseSurrogateModel


class ES(BaseAquisitionFunction):
    """
    Entropy Search (ES) aquisition function.

    NOTE: The next hyperparameters are selected according to: x_next = max_{x in X} ES(x)  
    """ 
    def __init__(self, surrogate: BaseSurrogateModel) -> None:
        super().__init__(surrogate)

    def compute(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError