"""
Python library for the implementations of general and weighted naive Bayes (WNB) classifiers.
"""

__version__ = "0.2.5"
__author__ = "Mehdi Samsami"


from ._base import ContinuousDistMixin, DiscreteDistMixin
from .enums import Distribution
from .gnb import GeneralNB
from .gwnb import GaussianWNB

__all__ = [
    "GeneralNB",
    "GaussianWNB",
    "Distribution",
    "ContinuousDistMixin",
    "DiscreteDistMixin",
]
