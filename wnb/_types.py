import numpy as np


class GaussianDistribution:
    distribution = "gaussian"

    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def pdf(self, x: float) -> float:
        return (1.0 / np.sqrt(2 * np.pi * self.sigma**2)) * np.exp(-0.5 * (((x - self.mu) / self.sigma)**2))

    def __call__(self, x: float) -> float:
        return self.pdf(x)


class LogNormalDistribution:
    distribution = "lognormal"

    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def pdf(self, x: float) -> float:
        return (1.0 / (x * np.sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(x - self.mu) / self.sigma)**2))

    def __call__(self, x: float) -> float:
        return self.pdf(x)


class ExponentialDistribution:
    distribution = "exponential"

    def __init__(self, rate: float):
        self.rate = rate

    def pdf(self, x: float) -> float:
        return self.rate * np.exp(-self.rate * x)

    def __call__(self, x: float) -> float:
        return self.pdf(x)
