import numpy as np
from scipy.stats import norm as Normal

from hpolite.acquisition.base import BaseAquisitionFunction
from hpolite.surrogates.base import BaseSurrogateModel


class KG(BaseAquisitionFunction):
    """
    Knowledge Gradient (CG) aquisition function.
    The CG aquisition function samples from the surrogate model:

    NOTE: The next hyperparameters are selected according to: x_next = max_{x in X} TS(x)  
    """ 
    def __init__(self, surrogate: BaseSurrogateModel, n_samples: int = 10) -> None:
        super().__init__(surrogate)
        self.n_samples = n_samples

    def compute(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError