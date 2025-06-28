from collections import Counter
from math import factorial
from typing import Any, Mapping

import numpy as np

from .base import DiscreteDistMixin
from .enums import Distribution as D

__all__ = [
    "BernoulliDist",
    "CategoricalDist",
    "GeometricDist",
    "PoissonDist",
]


class BernoulliDist(DiscreteDistMixin):
    name = D.BERNOULLI
    _support = [0, 1]

    def __init__(self, p: float) -> None:
        self.p = p
        super().__init__()

    @classmethod
    def from_data(cls, data, **kwargs: Any) -> "BernoulliDist":
        alpha = kwargs.get("alpha", 1e-10)
        return cls(p=((np.array(data) == 1).sum() + alpha) / len(data))

    def pmf(self, x: int) -> float:
        if x not in self._support:
            return 0.0
        else:
            return self.p if x == 1 else 1 - self.p


class CategoricalDist(DiscreteDistMixin):
    name = D.CATEGORICAL
    _support = None

    def __init__(self, prob: Mapping[Any, float]) -> None:
        self.prob = prob
        self._support = list(self.prob.keys())
        super().__init__()

    @classmethod
    def from_data(cls, data, **kwargs: Any) -> "CategoricalDist":
        alpha = kwargs.get("alpha", 1e-10)
        counter = Counter(data)
        values = list(counter.keys())
        counts = list(counter.values())
        return cls(prob={v: (c + alpha) / len(data) for v, c in zip(values, counts)})

    def pmf(self, x: Any) -> float:
        return self.prob.get(x, 0.0)


class GeometricDist(DiscreteDistMixin):
    name = D.GEOMETRIC
    _support = (1, np.inf)

    def __init__(self, p: float) -> None:
        self.p = p
        super().__init__()

    @classmethod
    def from_data(cls, data, **kwargs: Any) -> "GeometricDist":
        return cls(p=len(data) / np.sum(data))

    def pmf(self, x: int) -> float:
        return self.p * (1 - self.p) ** (x - 1) if x >= self._support[0] and x - int(x) == 0 else 0.0


class PoissonDist(DiscreteDistMixin):
    name = D.POISSON
    _support = (0, np.inf)

    def __init__(self, rate: float) -> None:
        self.rate = rate
        super().__init__()

    @classmethod
    def from_data(cls, data, **kwargs: Any) -> "PoissonDist":
        return cls(rate=np.sum(data) / len(data))

    def pmf(self, x: int) -> float:
        return (
            (np.exp(-self.rate) * self.rate**x) / factorial(x)
            if x >= self._support[0] and x - int(x) == 0
            else 0.0
        )
