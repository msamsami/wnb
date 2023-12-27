from typing import Union, Type

from sklearn._typing import MatrixLike, ArrayLike, Float

from ._base import ContinuousDistMixin, DiscreteDistMixin
from ._enums import Distribution

__all__ = ["MatrixLike", "ArrayLike", "Float", "DistibutionLike"]


DistibutionLike = Union[
    str,
    Distribution,
    Type[ContinuousDistMixin],
    Type[DiscreteDistMixin],
]
