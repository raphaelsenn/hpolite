import numpy as np
from scipy.stats import norm as Normal

from hpolite.acquisition.base import BaseAquisitionFunction
from hpolite.surrogates.base import BaseSurrogateModel


class TS(BaseAquisitionFunction):
    """
    Thompson sampling (TS) aquisition function.
    The TS aquisition function samples from the surrogate model:

        TS(x) := g, where g ~ surrogate(x)

    NOTE: The next hyperparameters are selected according to: x_next = max_{x in X} TS(x)  
    """ 
    def __init__(self, surrogate: BaseSurrogateModel) -> None:
        super().__init__(surrogate)

    def compute(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError