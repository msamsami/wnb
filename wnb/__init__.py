"""
Python library for the implementations of general and weighted naive Bayes (WNB) classifiers.
"""

__version__ = "0.5.1"
__author__ = "Mehdi Samsami"


from wnb.gnb import GeneralNB
from wnb.gwnb import GaussianWNB
from wnb.stats.base import ContinuousDistMixin, DiscreteDistMixin
from wnb.stats.enums import Distribution

__all__ = [
    "GeneralNB",
    "GaussianWNB",
    "Distribution",
    "ContinuousDistMixin",
    "DiscreteDistMixin",
]
