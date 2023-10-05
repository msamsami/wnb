from typing import Any, Mapping

import numpy as np
from scipy.special import gamma

from ._base import ContinuousDistMixin, DiscreteDistMixin
from ._enums import Distribution as D

__all__ = [
    "NormalDist",
    "LognormalDist",
    "ExponentialDist",
    "UniformDist",
    "ParetoDist",
    "GammaDist",
    "BernoulliDist",
    "CategoricalDist",
    "GeometricDist",
    "PoissonDist",
]


class NormalDist(ContinuousDistMixin):
    name = D.NORMAL
    _support = (-np.inf, np.inf)

    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma
        super().__init__()

    @classmethod
    def from_data(cls, data: np.ndarray, **kwargs):
        return cls(mu=np.average(data), sigma=np.std(data))

    def pdf(self, x: float) -> float:
        return (1.0 / np.sqrt(2 * np.pi * self.sigma**2)) * np.exp(
            -0.5 * (((x - self.mu) / self.sigma) ** 2)
        )


class LognormalDist(ContinuousDistMixin):
    name = D.LOGNORMAL
    _support = (0, np.inf)

    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma
        super().__init__()

    @classmethod
    def from_data(cls, data: np.ndarray, **kwargs):
        mu_hat = np.sum(np.log(data)) / len(data)
        sigma_hat = np.sum((np.log(data) - mu_hat) ** 2) / len(data)
        return cls(mu=mu_hat, sigma=sigma_hat)

    def pdf(self, x: float) -> float:
        return (1.0 / (x * np.sigma * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((np.log(x - self.mu) / self.sigma) ** 2)
        )


class ExponentialDist(ContinuousDistMixin):
    name = D.EXPONENTIAL
    _support = (0, np.inf)

    def __init__(self, rate: float):
        self.rate = rate
        super().__init__()

    @classmethod
    def from_data(cls, data: np.ndarray, **kwargs):
        return cls(rate=(len(data) - 2) / np.sum(data))

    def pdf(self, x: float) -> float:
        return self.rate * np.exp(-self.rate * x) if x >= 0 else 0.0


class UniformDist(ContinuousDistMixin):
    name = D.UNIFORM

    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b
        self._support = (a, b)
        super().__init__()

    @classmethod
    def from_data(cls, data, **kwargs):
        return cls(a=np.min(data), b=np.max(data))

    def pdf(self, x: float) -> float:
        return 1 / (self.b - self.a) if self.a <= x <= self.b else 0.0


class ParetoDist(ContinuousDistMixin):
    name = D.PARETO

    def __init__(self, x_m: float, alpha: float):
        self.x_m = x_m
        self.alpha = alpha
        self._support = (self.x_m, np.inf)
        super().__init__()

    @classmethod
    def from_data(cls, data, **kwargs):
        x_m = np.min(data)
        return cls(x_m=x_m, alpha=len(data) / np.sum(np.log(data / x_m)))

    def pdf(self, x: float) -> float:
        return (
            (self.alpha * self.x_m**self.alpha) / x ** (self.alpha + 1)
            if x >= self.x_m
            else 0.0
        )


class GammaDist(ContinuousDistMixin):
    name = D.GAMMA
    _support = (0, np.inf)

    def __init__(self, k: float, theta: float):
        self.k = k
        self.theta = theta
        super().__init__()

    @classmethod
    def from_data(cls, data, **kwargs):
        n = len(data)
        return cls(
            k=n
            * np.sum(data)
            / (n * np.sum(data * np.log(data)) - np.sum(data * np.sum(np.log(data)))),
            theta=(
                n * np.sum(data * np.log(data)) - np.sum(data * np.sum(np.log(data)))
            )
            / n**2,
        )

    def pdf(self, x: float) -> float:
        return (x ** (self.k - 1) * np.exp(-x / self.theta)) / (
            gamma(self.k) * self.theta**self.k
        )


class BernoulliDist(DiscreteDistMixin):
    name = D.BERNOULLI
    _support = [0, 1]

    def __init__(self, p: float):
        self.p = p
        super().__init__()

    @classmethod
    def from_data(cls, data, **kwargs):
        alpha = kwargs.get("alpha", 1e-10)
        return cls(p=((np.array(data) == 1).sum() + alpha) / len(data))

    def pmf(self, x: int) -> float:
        if x not in self._support:
            return 0.0
        else:
            if x == 1:
                return self.p
            else:
                return 1 - self.p


class CategoricalDist(DiscreteDistMixin):
    name = D.CATEGORICAL

    def __init__(self, prob: Mapping[Any, float]):
        self.prob = prob
        self._support = list(self.prob.keys())
        super().__init__()

    @classmethod
    def from_data(cls, data, **kwargs):
        alpha = kwargs.get("alpha", 1e-10)
        values, counts = np.unique(data, return_counts=True)
        return cls(prob={v: (c + alpha) / len(data) for v, c in zip(values, counts)})

    def pmf(self, x: Any) -> float:
        return self.prob.get(x, 0.0)


class GeometricDist(DiscreteDistMixin):
    name = D.GEOMETRIC
    _support = (1, np.inf)

    def __init__(self, p: float):
        self.p = p
        super().__init__()

    @classmethod
    def from_data(cls, data, **kwargs):
        return cls(p=len(data) / np.sum(data))

    def pmf(self, x: int) -> float:
        return self.p * (1 - self.p) ** (x - 1) if x >= 1 else 0.0


class PoissonDist(DiscreteDistMixin):
    name = D.POISSON
    _support = (0, np.inf)

    def __init__(self, rate: float):
        self.rate = rate
        super().__init__()

    @classmethod
    def from_data(cls, data, **kwargs):
        return cls(rate=np.sum(data) / len(data))

    def pmf(self, x: int) -> float:
        return (np.exp(-self.rate) * self.rate**x) / np.math.factorial(x)


AllDistributions = {eval(cls).name: eval(cls) for cls in __all__}


NonNumericDistributions = [
    D.CATEGORICAL,
]
