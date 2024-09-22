from typing import Type, Union

from .base import ContinuousDistMixin, DiscreteDistMixin
from .enums import Distribution

__all__ = ["DistributionLike"]


DistributionLike = Union[
    str,
    Distribution,
    Type[ContinuousDistMixin],
    Type[DiscreteDistMixin],
]
