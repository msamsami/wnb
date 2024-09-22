import numpy as np
from scipy.special import beta, gamma
from scipy.stats import chi2

from .base import ContinuousDistMixin
from .enums import Distribution as D

__all__ = [
    "NormalDist",
    "LognormalDist",
    "ExponentialDist",
    "UniformDist",
    "ParetoDist",
    "GammaDist",
    "BetaDist",
    "ChiSquaredDist",
    "TDist",
    "RayleighDist",
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
        return (1.0 / np.sqrt(2 * np.pi * self.sigma**2)) * np.exp(-0.5 * (((x - self.mu) / self.sigma) ** 2))


class LognormalDist(ContinuousDistMixin):
    name = D.LOGNORMAL
    _support = (0, np.inf)

    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma
        super().__init__()

    @classmethod
    def from_data(cls, data: np.ndarray, **kwargs):
        log_data = np.log(data)
        return cls(mu=np.average(log_data), sigma=np.std(log_data))

    def pdf(self, x: float) -> float:
        return (1.0 / (x * self.sigma * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((np.log(x) - self.mu) / self.sigma) ** 2
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
    _support = None

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
    _support = None

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
        return (self.alpha * self.x_m**self.alpha) / x ** (self.alpha + 1) if x >= self.x_m else 0.0


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
            k=n * np.sum(data) / (n * np.sum(data * np.log(data)) - np.sum(data * np.sum(np.log(data)))),
            theta=(n * np.sum(data * np.log(data)) - np.sum(data * np.sum(np.log(data)))) / n**2,
        )

    def pdf(self, x: float) -> float:
        return (x ** (self.k - 1) * np.exp(-x / self.theta)) / (gamma(self.k) * self.theta**self.k)


class BetaDist(ContinuousDistMixin):
    name = D.BETA
    _support = (0, 1)

    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta
        super().__init__()

    @classmethod
    def from_data(cls, data, **kwargs):
        mu_hat = np.average(data)
        var_hat = np.var(data, ddof=1)
        multiplied_term = (mu_hat * (1 - mu_hat) / var_hat) - 1
        return cls(
            alpha=mu_hat * multiplied_term,
            beta=(1 - mu_hat) * multiplied_term,
        )

    def pdf(self, x: float) -> float:
        return ((x ** (self.alpha - 1)) * (1 - x) ** (self.beta - 1)) / beta(self.alpha, self.beta)


class ChiSquaredDist(ContinuousDistMixin):
    name = D.CHI_SQUARED
    _support = (0, np.inf)

    def __init__(self, k: int):
        self.k = k
        super().__init__()

    @classmethod
    def from_data(cls, data, **kwargs):
        return cls(k=round(np.average(data)))

    def pdf(self, x: float) -> float:
        return chi2.pdf(x, self.k)


class TDist(ContinuousDistMixin):
    name = D.T
    _support = (-np.inf, np.inf)

    def __init__(self, df: float):
        self.df = df
        super().__init__()

    @classmethod
    def from_data(cls, data, **kwargs):
        return cls(df=len(data) - 1)

    def pdf(self, x: float) -> float:
        return (gamma((self.df + 1) / 2) / (np.sqrt(self.df * np.pi) * gamma(self.df / 2))) * (
            1 + (x**2 / self.df)
        ) ** (-(self.df + 1) / 2)


class RayleighDist(ContinuousDistMixin):
    name = D.RAYLEIGH
    _support = (0, np.inf)

    def __init__(self, sigma: float):
        self.sigma = sigma
        super().__init__()

    @classmethod
    def from_data(cls, data, **kwargs):
        sigma = np.sqrt(np.mean(data**2) / 2)
        return cls(sigma=sigma)

    def pdf(self, x: float) -> float:
        return (x / self.sigma**2) * np.exp(-(x**2) / (2 * self.sigma**2)) if x >= 0 else 0.0
