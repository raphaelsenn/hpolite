import numpy as np
from numpy.random import Generator

from hpolite.space.base import BaseParameter


class Real(BaseParameter):
    def __init__(
            self, 
            lower: float, 
            upper: float, 
            prior: str = "log-uniform"
    ) -> None:
        super().__init__()
        self.lower = lower
        self.upper = upper
        
        assert prior in {"uniform", "log-uniform"}, (
            f"prior needs to be `uniform` or `log-uniform`, got: {prior}" 
        )
        if prior == "log-uniform" and lower <= 0:
            raise ValueError("log-uniform requires lower > 0.")

        self.prior = prior

    def sample(self, rng: Generator) -> float:
        # logU(a, b) ~ exp(U(log(a), log(b))
        # https://stackoverflow.com/questions/43977717/how-do-i-generate-log-uniform-distribution-in-python
        if self.prior == "log-uniform":
            value = rng.uniform(np.log(self.lower), np.log(self.upper)) # in parameter space
            return float(np.exp(value))
        return float(rng.uniform(self.lower, self.upper))               # in parameter space

    def encode(self, value: float) -> float:
        # value is in parameter space 

        if self.prior == "log-uniform":
            lower = np.log(self.lower)
            upper = np.log(self.upper)
            value = np.log(value)   # ~ U[log(a), log(b)]
        else:
            lower = self.lower
            upper = self.upper
        encoded = (value - lower) / (upper - lower)                     # map to [0, 1]
        return np.asarray([encoded], dtype=np.float32)
    
    def decode(self, encoded: np.ndarray) -> float:
        # encoded is in [0, 1] space 
        encoded = float(encoded.item())
        encoded = np.clip(encoded, 0.0, 1.0)

        if self.prior == "log-uniform":
            lower = np.log(self.lower)
            upper = np.log(self.upper)
            value = (upper - lower) * encoded + lower                     # U[log(a), log(b)] space
            return np.exp(value)                                          # parameter space

        value = (self.upper - self.lower) * encoded + self.lower          # parameter space
        return value