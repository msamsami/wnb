from __future__ import annotations
from abc import ABCMeta
from functools import wraps
import inspect
from numbers import Number
import warnings

import numpy as np

from .enums import Distribution

__all__ = ["ContinuousDistMixin", "DiscreteDistMixin"]


def vectorize(otypes=None, excluded=None, signature=None):
    """
    Numpy vectorization wrapper that works with class methods.
    """

    def decorator(func):
        vectorized = np.vectorize(
            func, otypes=otypes, excluded=excluded, signature=signature
        )

        @wraps(func)
        def wrapper(*args):
            return vectorized(*args)

        return wrapper

    return decorator


class DistMixin(metaclass=ABCMeta):
    """
    Mixin class for probability distributions in wnb.
    """

    name: str | Distribution
    _support: list[float] | tuple[float, float]

    @classmethod
    def from_data(cls, data, **kwargs):
        """Creates an instance of the class from given data. Distribution parameters will be estimated from data.

        Args:
            data: Input data from which distribution parameters will be estimated.

        Returns:
            self: An instance of the class.
        """
        pass

    @classmethod
    def _get_param_names(cls):
        """
        Gets parameter names for the distribution instance.
        """
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            return []

        init_signature = inspect.signature(init)
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]

        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "wnb estimators should always "
                    "specify their parameters in the signature "
                    "of their __init__ (no varargs). "
                    "%s with constructor %s doesn't "
                    "follow this convention." % (cls, init_signature)
                )

        return sorted([p.name for p in parameters])

    def get_params(self) -> dict:
        """Gets parameters for this distribution instance.

        Returns:
            dict: Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            out[key] = value
        return out

    @property
    def support(self) -> list[float] | tuple[float, float]:
        """Returns the support of the probability distribution.

        If support is a list, it represents a limited number of discrete values.
        If it is a tuple, it indicates a limited or unlimited range of continuous values.
        """
        return self._support

    def _check_support(self, x):
        if (isinstance(self.support, list) and x not in self.support) or (
            isinstance(self.support, tuple)
            and (x < self.support[0] or x > self.support[1])
        ):
            warnings.warn(
                "Value doesn't lie within the support of the distribution",
                RuntimeWarning,
            )

    def __repr__(self) -> str:
        return "".join(
            [
                "<",
                self.__class__.__name__,
                "(",
                ", ".join(
                    [
                        f"{k}={v:.4f}" if isinstance(v, Number) else f"{k}={v}"
                        for k, v in self.get_params().items()
                    ]
                ),
                ")>",
            ]
        )


class ContinuousDistMixin(DistMixin, metaclass=ABCMeta):
    """
    Mixin class for all continuous probability distributions in wnb.
    """

    _type = "continuous"

    def __init__(self, **kwargs):
        """
        Initializes an instance of the continuous probability distribution with given parameters.
        """
        pass

    def pdf(self, x: float) -> float:
        """Returns the value of probability density function (PDF) at x.

        Args:
            x (float): Input value.

        Returns:
            float: Probability density.
        """
        pass

    @vectorize(signature="(),()->()")
    def __call__(self, x: float) -> float:
        self._check_support(x)
        return self.pdf(x)


class DiscreteDistMixin(DistMixin, metaclass=ABCMeta):
    """
    Mixin class for all discrete probability distributions in wnb.
    """

    _type = "discrete"

    def __init__(self, **kwargs):
        """
        Initializes an instance of the discrete probability distribution with given parameters.
        """
        pass

    def pmf(self, x: float) -> float:
        """Returns the value of probability mass function (PMF) at x.

        Args:
            x (float): Input value.

        Returns:
            float: Probability mass.
        """
        pass

    @vectorize(signature="(),()->()")
    def __call__(self, x: float) -> float:
        self._check_support(x)
        return self.pmf(x)
