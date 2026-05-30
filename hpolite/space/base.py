from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator


class BaseParameter(ABC):
    """Base parameter calss."""
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def sample(self, rng: Generator) -> int | float | str:
        raise NotImplementedError
    
    @abstractmethod 
    def encode(self, value: int | float | str) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod 
    def decode(self, encoded: np.ndarray) -> int | float | str:
        raise NotImplementedError