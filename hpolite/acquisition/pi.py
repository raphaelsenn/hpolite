import numpy as np
from scipy.stats import norm as Normal

from hpolite.acquisition.base import BaseAquisitionFunction
from hpolite.surrogates.base import BaseSurrogateModel


class PI(BaseAquisitionFunction):
    """
    Probability of improvement (PI) aquisition function.
    The PI aquisition function is given by:

        PI(x) := P(x >= x_incumbent) 
               = Phi(Z), where

        Z = (x_incumbent - mu(x)) / (std(x)), and
        Phi(Z) = Normal.cdf(Z)
    
    NOTE: The next hyperparameters are selected according to: x_next = max_{x in X} PI(x)
    """ 
    def __init__(self, surrogate: BaseSurrogateModel) -> None:
        super().__init__(surrogate)

    def compute(self, x: np.ndarray) -> np.ndarray:
        x_inc = self.surrogate.get_incumbent()
        mean, std = self.surrogate.predict(x)
        z = (x_inc - mean) / std 
        return Normal.cdf(z)