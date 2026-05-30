import numpy as np
from numpy.random import Generator

from hpolite.space.base import BaseParameter


class Integer(BaseParameter):
    def __init__(self, lower: int, upper: int) -> None:
        super().__init__()
        self.lower = lower
        self.upper = upper

    def sample(self, rng: Generator) -> int:
        return int(rng.integers(self.lower, self.upper + 1))
    
    def encode(self, value: int) -> float:
        if self.upper == self.lower:
            return np.asarray([0.0], dtype=np.float32) 

        encoded = (value - self.lower) / (self.upper - self.lower)
        return np.asarray([encoded], dtype=np.float32)
    
    def decode(self, encoded: np.ndarray) -> int:
        if self.upper == self.lower:
            return self.lower

        encoded = float(np.asarray(encoded).item())
        encoded = np.clip(encoded, 0.0, 1.0)

        decoded = round((self.upper - self.lower) * encoded + self.lower)
        return decoded