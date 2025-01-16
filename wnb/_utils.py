from typing import Any

import sklearn
from packaging import version
from sklearn.utils import check_array

__all__ = ["SKLEARN_V1_6_OR_LATER", "validate_data", "_check_n_features", "_check_feature_names"]


SKLEARN_V1_6_OR_LATER = version.parse(sklearn.__version__) >= version.parse("1.6")


if SKLEARN_V1_6_OR_LATER:
    from sklearn.utils.validation import _check_feature_names, _check_n_features
    from sklearn.utils.validation import validate_data as _validate_data

    def validate_data(*args, **kwargs):
        if kwargs.get("force_all_finite"):
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return _validate_data(*args, **kwargs)

else:

    def validate_data(estimator, X, **kwargs: Any):
        kwargs.pop("reset", None)
        return check_array(X, estimator=estimator, **kwargs)

    def _check_n_features(estimator, X, reset):
        return estimator._check_n_features(X, reset)

    def _check_feature_names(estimator, X, reset):
        return estimator._check_feature_names(X, reset)
