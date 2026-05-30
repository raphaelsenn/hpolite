from typing import List

import numpy as np
from numpy.random import Generator

from hpolite.space.base import BaseParameter


class Categorical(BaseParameter):
    def __init__(self, categories: List[str | int | float]) -> None:
        super().__init__()

        if len(categories) == 0:
            raise ValueError("Got an empty category list.")

        first_type = type(categories[0])

        self.categories = list(categories)
        self.n_categories = len(categories)
        self.index = {v: i for i, v in enumerate(categories)} 
        self.type = first_type

    def sample(self, rng: Generator) -> str | int | float:
        return rng.choice(self.categories)

    def grid(self) -> List[str | int | float]:
        return self.categories

    def encode(self, value: str | int | float) -> np.ndarray:
        encoded = np.zeros(self.n_categories, dtype=np.float32)
        id = self.index[value]
        encoded[id] = 1.0
        return encoded

    def decode(self, encoded: np.ndarray) -> str | int | float:
        id = int(np.argmax(encoded))
        return self.categories[id]