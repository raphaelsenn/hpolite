import numpy as np
from scipy.stats import norm as Normal

from hpolite.acquisition.pi import PI


class DummySurrogate:
    def get_incumbent(self) -> float:
        return 1.0

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.array([0.5, 1.0, 1.5])
        std = np.array([0.5, 0.5, 0.5])
        return mean, std


class TestPI:
    def test_compute_matches_closed_form_for_minimization(self) -> None:
        surrogate = DummySurrogate()
        pi = PI(surrogate)

        X = np.zeros((3, 1))
        actual = pi.compute(X)

        incumbent = 1.0
        mean = np.array([0.5, 1.0, 1.5])
        std = np.array([0.5, 0.5, 0.5])

        z = (incumbent - mean) / std
        expected = Normal.cdf(z)

        np.testing.assert_allclose(actual, expected)
        assert actual.shape == (3,)