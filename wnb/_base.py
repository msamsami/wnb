from abc import ABCMeta
from functools import wraps

import numpy as np

__all__ = [
    'ContinuousDistMixin',
    'DiscreteDistMixin'
]


def vectorize(otypes=None, excluded=None, signature=None):
    """
    Numpy vectorization wrapper that works with class methods.
    """

    def decorator(func):
        vectorized = np.vectorize(func, otypes=otypes, excluded=excluded, signature=signature)

        @wraps(func)
        def wrapper(*args):
            return vectorized(*args)

        return wrapper

    return decorator


class ContinuousDistMixin(metaclass=ABCMeta):
    """
    Mixin class for all continuous probability distributions in wnb.
    """

    name = None

    def __init__(self, **kwargs):
        """Initializes an instance of the probability distribution with given parameters.

        """
        pass

    @classmethod
    def from_data(cls, data):
        """Creates an instance of the class from given data. Distribution parameters will be estimated from the data.

        Returns:
            self: An instance of the class.
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
        return self.pdf(x)

    def __repr__(self) -> str:
        pass


class DiscreteDistMixin(metaclass=ABCMeta):
    """
    Mixin class for all discrete probability distributions in wnb.
    """

    name = None

    def __init__(self, **kwargs):
        """Initializes an instance of the probability distribution with given parameters.

        """
        pass

    @classmethod
    def from_data(cls, data):
        """Creates an instance of the class from given data. Distribution parameters will be estimated from the data.

        Returns:
            self: An instance of the class.
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
        return self.pmf(x)

    def __repr__(self) -> str:
        pass
