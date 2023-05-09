from abc import ABCMeta

__all__ = [
    'ContinuousDistMixin',
    'DiscreteDistMixin'
]

import numpy as np


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

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Returns the value of probability density function (PDF) at x.

        Args:
            x (np.ndarray): Input values; a flat numpy array.

        Returns:
            np.ndarray: Probability density.
        """
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
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

    def pmf(self, x: np.ndarray) -> np.ndarray:
        """Returns the value of probability mass function (PMF) at x.

        Args:
            x (np.ndarray): Input values; a flat numpy array.

        Returns:
            float: Probability mass.
        """
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.pmf(x)

    def __repr__(self) -> str:
        pass
