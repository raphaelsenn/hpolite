import numpy as np
from scipy.stats import norm as Normal

from hpolite.acquisition.base import BaseAquisitionFunction
from hpolite.surrogates.base import BaseSurrogateModel


class EI(BaseAquisitionFunction):
    """
    Expected improvement (EI) aquisition function. 
    The EI aquisition function is given by:

        EI(x) := E[max(0, x_incumbent - x)]
               = std(x) * [Z * Phi(Z) + phi(Z)], where

        Z = (x_incumbent - mu(x)) / (std(x)), and
        Phi(Z) = Normal.cdf(Z), phi(Z) = Normal.pdf(Z)
    
    NOTE: The next hyperparameters are selected according to: x_next = max_{x in X} EI(x) 
    """ 
    def __init__(self, surrogate: BaseSurrogateModel) -> None:
        super().__init__(surrogate)

    def compute(self, x: np.ndarray) -> np.ndarray:
        x_inc = self.surrogate.get_incumbent()
        mean, std = self.surrogate.predict(x)
        z = (x_inc - mean) / std
        return std * (z * Normal.cdf(z) + Normal.pdf(z))