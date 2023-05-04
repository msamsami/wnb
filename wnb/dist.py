from typing import Any, Mapping, Sequence
import warnings

import numpy as np
from scipy.special import gamma

from ._enums import Distribution


__all__ = [
    'NormalDist',
    'LognormalDist',
    'ExponentialDist',
    'UniformDist',
    'ParetoDist',
    'GammaDist',
    'BernoulliDist',
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

    def __repr__(self) -> str:
        return f"<NormalDist(mu={self.mu:.4f}, sigma={self.sigma:.4f})>"


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

    def __repr__(self) -> str:
        return f"<LognormalDist(mu={self.mu:.4f}, sigma={self.sigma:.4f})>"


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

    def __repr__(self) -> str:
        return f"<ExponentialDist(rate={self.rate:.4f})>"


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

    def __repr__(self) -> str:
        return f"<UniformDist(a={self.a:.4f}, b={self.b:.4f})>"


class ParetoDist:
    name = Distribution.PARETO

    def __init__(self, x_m: float, alpha: float):
        self.x_m = x_m
        self.alpha = alpha

    @classmethod
    def from_data(cls, data):
        x_m = np.min(data)
        return cls(x_m=x_m, alpha=len(data) / np.sum(np.log(data / x_m)))

    def pdf(self, x: float) -> float:
        return (self.alpha * self.x_m**self.alpha) / x**(self.alpha + 1) if x >= self.x_m else 0.0

    def __call__(self, x: float) -> float:
        return self.pdf(x)

    def __repr__(self) -> str:
        return f"<ParetoDist(x_m={self.x_m:.4f}, alpha={self.alpha:.4f})>"


class GammaDist:
    name = Distribution.GAMMA

    def __init__(self, k: float, theta: float):
        self.k = k
        self.theta = theta

    @classmethod
    def from_data(cls, data):
        n = len(data)
        return cls(
            k=n * np.sum(data) / (n * np.sum(data * np.log(data)) - np.sum(data * np.sum(np.log(data)))),
            theta=(n * np.sum(data * np.log(data)) - np.sum(data * np.sum(np.log(data)))) / n**2
        )

    def pdf(self, x: float) -> float:
        return (x ** (self.k-1) * np.exp(-x / self.theta)) / (gamma(self.k) * self.theta ** self.k)

    def __call__(self, x: float) -> float:
        return self.pdf(x)

    def __repr__(self) -> str:
        return f"<GammaDist(k={self.k:.4f}, theta={self.theta:.4f})>"


class BernoulliDist:
    name = Distribution.BERNOULLI

    def __init__(self, p: float):
        self.p = p

    @classmethod
    def from_data(cls, data):
        if any(x not in [0, 1] for x in data):
            warnings.warn("Bernoulli data points should be either 0 or 1")

        return cls(p=(np.array(data) == 1).sum() / len(data))

    def pmf(self, x: int) -> float:
        if x not in [0, 1]:
            raise ValueError("Bernoulli data points should be either 0 or 1")

        return self.p if x == 1 else 1 - self.p

    def __call__(self, x: int) -> float:
        return self.pmf(x)

    def __repr__(self) -> str:
        return f"<BernoulliDist(p={self.p:.4f})>"


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

    def __repr__(self) -> str:
        return f"<CategoricalDist(prob={self.prob})>"


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

    def __repr__(self) -> str:
        return f"<MultinomialDist(n={self.n}, prob={self.prob})>"


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

    def __repr__(self) -> str:
        return f"<PoissonDist(rate={self.rate:.4f})>"


AllDistributions = {
    Distribution.NORMAL: NormalDist,
    Distribution.LOGNORMAL: LognormalDist,
    Distribution.EXPONENTIAL: ExponentialDist,
    Distribution.UNIFORM: UniformDist,
    Distribution.PARETO: ParetoDist,
    Distribution.GAMMA: GammaDist,
    Distribution.BERNOULLI: BernoulliDist,
    Distribution.CATEGORICAL: CategoricalDist,
    Distribution.MULTINOMIAL: MultinomialDist,
    Distribution.POISSON: PoissonDist
}
