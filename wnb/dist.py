from typing import Any, Mapping, Sequence

import numpy as np

from ._enums import Distribution


__all__ = [
    'NormalDist',
    'LognormalDist',
    'ExponentialDist',
    'UniformDist',
    'CategoricalDist',
    'MultinomialDist',
    'PoissonDist'
]


class NormalDist:
    name = Distribution.NORMAL

    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    @classmethod
    def from_data(cls, data: np.ndarray):
        return cls(mu=np.mean(data), sigma=np.std(data, ddof=1))

    def pdf(self, x: float) -> float:
        return (1.0 / np.sqrt(2 * np.pi * self.sigma**2)) * np.exp(-0.5 * (((x - self.mu) / self.sigma)**2))

    def __call__(self, x: float) -> float:
        return self.pdf(x)


class LognormalDist:
    name = Distribution.LOGNORMAL

    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    @classmethod
    def from_data(cls, data: np.ndarray):
        mu_hat = np.sum(np.log(data)) / len(data)
        sigma_hat = np.sum((np.log(data) - mu_hat)**2) / len(data)
        return cls(mu=mu_hat, sigma=sigma_hat)

    def pdf(self, x: float) -> float:
        return (1.0 / (x * np.sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(x - self.mu) / self.sigma)**2))

    def __call__(self, x: float) -> float:
        return self.pdf(x)


class ExponentialDist:
    name = Distribution.EXPONENTIAL

    def __init__(self, rate: float):
        self.rate = rate

    @classmethod
    def from_data(cls, data: np.ndarray):
        return cls(rate=(len(data)-2) / np.sum(data))

    def pdf(self, x: float) -> float:
        return self.rate * np.exp(-self.rate * x)

    def __call__(self, x: float) -> float:
        return self.pdf(x)


class UniformDist:
    name = Distribution.UNIFORM

    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    @classmethod
    def from_data(cls, data):
        return cls(a=np.min(data), b=np.max(data))

    def pdf(self, x: float) -> float:
        return 1 / (self.b - self.a) if self.a <= x <= self.b else 0.0

    def __call__(self, x: float) -> float:
        return self.pdf(x)


class CategoricalDist:
    name = Distribution.CATEGORICAL

    def __init__(self, prob: Mapping[Any, float]):
        self.prob = prob

    @classmethod
    def from_data(cls, data):
        values, counts = np.unique(data, return_counts=True)
        return cls(prob={v: c/len(data) for v, c in zip(values, counts)})

    def pmf(self, x: Any) -> float:
        return self.prob.get(x)

    def __call__(self, x: Any) -> float:
        return self.pmf(x)


class MultinomialDist(CategoricalDist):
    name = Distribution.MULTINOMIAL

    def __init__(self, n: int, prob: Mapping[Any, float]):
        self.n = n
        super().__init__(prob)

    def pmf(self, x: Sequence[int]) -> float:
        if sum(x) != self.n:
            return 0.0
        else:
            return np.math.factorial(self.n) * np.product([p**v for v, p in self.prob.items()]) / \
                np.product([np.math.factorial(v) for v in self.prob.keys()])


class PoissonDist:
    name = Distribution.POISSON

    def __init__(self, rate: float):
        self.rate = rate

    @classmethod
    def from_data(cls, data):
        return cls(rate=np.sum(data)/len(data))

    def pmf(self, x: int) -> float:
        return (np.exp(-self.rate) * self.rate**x) / np.math.factorial(x)

    def __call__(self, x: int) -> float:
        return self.pmf(x)
