import warnings
from typing import Union

import numpy as np
import numpy.typing
import pandas as pd
from scipy.sparse import spmatrix

__all__ = ["MatrixLike", "ArrayLike", "Int", "Float", "DistributionLike"]  # noqa: F822


ArrayLike = numpy.typing.ArrayLike
MatrixLike = Union[np.ndarray, pd.DataFrame, spmatrix]

Int = Union[int, np.int8, np.int16, np.int32, np.int64]
Float = Union[float, np.float16, np.float32, np.float64]


def __getattr__(name):
    if name == "DistributionLike":
        warnings.warn(
            "The `DistributionLike` type has been moved to `wnb.stats.typing`. "
            "Please update your imports accordingly. This import will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        from wnb.stats.typing import DistributionLike

        return DistributionLike
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
