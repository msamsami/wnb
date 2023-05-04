__version__ = "0.1.4"
__author__ = "Mehdi Samsami"

__all__ = [
    'GeneralNB',
    'GaussianWNB',
    'Distribution'
]

from ._enums import Distribution
from .gnb import GeneralNB
from .gwnb import GaussianWNB
