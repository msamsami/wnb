from typing import Union, Type

import numpy.typing
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix

from ._base import ContinuousDistMixin, DiscreteDistMixin
from ._enums import Distribution

__all__ = ["MatrixLike", "ArrayLike", "Int", "Float", "DistibutionLike"]

ArrayLike = numpy.typing.ArrayLike
MatrixLike = Union[np.ndarray, pd.DataFrame, spmatrix]

Int = Union[int, np.int8, np.int16, np.int32, np.int64]
Float = Union[float, np.float16, np.float32, np.float64]

DistibutionLike = Union[
    str,
    Distribution,
    Type[ContinuousDistMixin],
    Type[DiscreteDistMixin],
]
