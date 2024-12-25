from typing import Any

import sklearn
from packaging import version
from sklearn.utils import check_array

__all__ = ["SKLEARN_V1_6_OR_LATER", "validate_data"]


SKLEARN_V1_6_OR_LATER = version.parse(sklearn.__version__) >= version.parse("1.6")


if SKLEARN_V1_6_OR_LATER:
    from sklearn.utils.validation import validate_data
else:

    def validate_data(estimator, X, **kwargs: Any):
        kwargs.pop("reset", None)
        return check_array(X, estimator=estimator, **kwargs)
