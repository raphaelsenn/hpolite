from abc import ABC, abstractmethod

import numpy as np
from hpolite.surrogates.base import BaseSurrogateModel


class BaseAquisitionFunction(ABC):
    """Base class for aquisition functions."""
    def __init__(self, surrogate: BaseSurrogateModel) -> None:
        """
        Parameters:
        ----------- 
        surrogate : BaseSurrogateModel
            Surrogate model (i.e. Gaussian Process or Random Forest) to model the black box objective. 
        """ 
        super().__init__()
        self.surrogate = surrogate

    @abstractmethod
    def compute(self, x: np.ndarray) -> np.ndarray: 
        """
        Computes the aquisition function for a given point x.
        
        Parameters:
        ----------- 
        x : np.ndarray, shape [1, D]
            Hyperparameter (encoded) that should be evaluated.
        """
        raise NotImplementedError

    def update(self, surrogate: BaseSurrogateModel) -> None:
        """
        Updates the surrogate model.
        
        Parameters:
        ----------- 
        surrogate : BaseSurrogateModel
            Surrogate model (i.e. Gaussian Process or Random Forest) to model the black box objective.  
        """ 
        self.surrogate = surrogate

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.compute(x)