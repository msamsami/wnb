from __future__ import annotations

import sys
from collections import defaultdict
from numbers import Real
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np
from sklearn.naive_bayes import _BaseNB
from sklearn.utils.multiclass import check_classification_targets

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from wnb.stats import Distribution, NonNumericDistributions
from wnb.stats._utils import get_dist_class, is_dist_supported
from wnb.stats.base import DistMixin
from wnb.stats.typing import DistributionLike

from ._utils import (
    SKLEARN_V1_6_OR_LATER,
    _check_feature_names,
    _check_n_features,
    _fit_context,
    check_X_y,
    validate_data,
)
from .typing import ArrayLike, ColumnKey, Float, MatrixLike

__all__ = ["GeneralNB"]


def _get_parameter_constraints() -> dict[str, list[Any]]:
    """
    Gets parameter validation constraints for GeneralNB based on Scikit-learn version.
    """
    try:
        # Added in sklearn v1.2.0
        from sklearn.utils._param_validation import Interval

        return {
            "distributions": ["array-like", None],
            "priors": ["array-like", None],
            "alpha": [Interval(Real, 0, None, closed="left")],
            "var_smoothing": [Interval(Real, 0, None, closed="left")],
        }
    except (ImportError, ModuleNotFoundError):
        return {}


class GeneralNB(_BaseNB):
    """A General Naive Bayes classifier that supports distinct likelihood distributions for individual features,
    enabling more tailored modeling beyond the standard single-distribution approaches such as GaussianNB and BernoulliNB.

    Parameters
    ----------
    distributions : sequence of distribution-likes or tuples, default=None
        Probability distributions to be used for feature likelihoods. If not specified,
        all features will use Gaussian (normal) distributions.

        Can be specified in two ways:
            A sequence of length n_features specifying feature distributions in the order of appearance.

            A sequence of `(distribution, columns)` tuples each specifying the distribution of one or more columns.
            Columns can be referenced by position (int or array-like of int) or name (str or array-like of str)
            when using a DataFrame. Any columns not explicitly specified will use a Gaussian distribution.

    priors : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.

    alpha : float, default=1e-10
        Additive (Laplace/Lidstone) smoothing parameter. Set alpha=0 for no smoothing.

    var_smoothing : float, default=1e-9
        Portion of the largest variance of all features that is added to
        variances of normal distributions for calculation stability.

    Attributes
    ----------
    class_count_ : ndarray of shape (n_classes,)
        Number of training samples observed in each class.

    class_prior_ : ndarray of shape (n_classes,)
        Probability of each class.

    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier.

    n_classes_ : int
        Number of classes seen during :term:`fit`.

    epsilon_ : float
        Absolute additive value to variances of normal distributions.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    distributions_ : list of length `n_features_in_`
        List of likelihood distributions used to fit to features.

    likelihood_params_ : dict
        A mapping from class labels to their fitted likelihood distributions.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, 1], [-2, 1], [-3, 2], [1, 1], [2, 1], [3, 2]])
    >>> Y = np.array([1, 1, 1, 2, 2, 2])
    >>> from wnb import GeneralNB, Distribution as D
    >>> clf = GeneralNB(distributions=[D.NORMAL, D.POISSON])
    >>> clf.fit(X, Y)
    GeneralNB(distributions=[<Distribution.NORMAL: 'Normal'>,
                             <Distribution.POISSON: 'Poisson'>])
    >>> print(clf.predict([[-0.8, 1]]))
    [1]
    >>> X = np.array([[-1, 1, 1], [-2, 1, 1], [-3, 2, 2], [1, 1, 1], [2, 1, 1], [3, 2, 2]])
    >>> Y = np.array([-1, -1, -1, 1, 1, 1])
    >>> clf_2 = GeneralNB(distributions=[(D.NORMAL, [0, 2]), (D.POISSON, [1])])
    >>> clf_2.fit(X, Y)
    GeneralNB(distributions=[(<Distribution.NORMAL: 'Normal'>, [0, 2]),
                             (<Distribution.POISSON: 'Poisson'>, [1])])
    >>> print(clf_2.predict([[-0.8, 1, 1]]))
    [-1]
    """

    if parameter_constraints := _get_parameter_constraints():
        _parameter_constraints = parameter_constraints

    def __init__(
        self,
        distributions: Optional[
            Union[Sequence[DistributionLike], Sequence[tuple[DistributionLike, ColumnKey]]]
        ] = None,
        *,
        priors: Optional[ArrayLike] = None,
        alpha: Float = 1e-10,
        var_smoothing: Float = 1e-9,
    ) -> None:
        self.distributions = distributions
        self.priors = priors
        self.alpha = alpha
        self.var_smoothing = var_smoothing

    if SKLEARN_V1_6_OR_LATER:

        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.target_tags.required = True
            return tags

    def _more_tags(self) -> dict[str, bool]:
        return {"requires_y": True}

    def _get_distributions(self) -> Sequence[DistributionLike]:
        try:
            if self.distributions_ is not None:
                return self.distributions_
        except Exception:
            return self.distributions or []

    def _check_X(self, X) -> np.ndarray:
        X = validate_data(
            self,
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype=(
                None if any(d in self._get_distributions() for d in NonNumericDistributions) else "numeric"
            ),
            force_all_finite=True,
            reset=False,
        )
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Expected input with %d features, got %d instead." % (self.n_features_in_, X.shape[1])
            )
        return X

    def _check_X_y(self, X, y) -> tuple[np.ndarray, np.ndarray]:
        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype=(
                None if any(d in self._get_distributions() for d in NonNumericDistributions) else "numeric"
            ),
            force_all_finite=True,
            estimator=self,
        )
        check_classification_targets(y)
        return X, y

    @staticmethod
    def _find_dist(
        feature_idx: int, feature_name: str | None, dist_mapping: dict[DistributionLike, ColumnKey]
    ) -> DistributionLike:
        for dist, cols in dist_mapping.items():
            cols_ = (
                [cols]
                if isinstance(cols, (int, str)) or (isinstance(cols, np.ndarray) and cols.shape == ())
                else cols
            )
            if feature_idx in cols_:
                return dist
            if feature_name is not None and feature_name in cols_:
                return dist
        return Distribution.NORMAL

    def _init_parameters(self) -> None:
        if self.priors is None:
            # Calculate empirical prior probabilities
            self.class_prior_ = self.class_count_ / self.class_count_.sum()
        else:
            priors = np.asarray(self.priors)

            # Check that the provided priors match the number of classes
            if len(priors) != self.n_classes_:
                raise ValueError("Number of priors must match the number of classes.")
            # Check that the sum of priors is 1
            if not np.isclose(priors.sum(), 1.0):
                raise ValueError("The sum of the priors should be 1.")
            # Check that the priors are non-negative
            if (priors < 0).any():
                raise ValueError("Priors must be non-negative.")

            self.class_prior_ = priors

        distributions_error_msg = "distributions parameter must be a sequence of distributions or a sequence of tuples of (distribution, column_key)"
        if self.distributions is None:
            # Set distributions if not specified
            self.distributions_: list[DistributionLike] = [Distribution.NORMAL] * self.n_features_in_
        elif not isinstance(self.distributions, (Sequence, np.ndarray)) or isinstance(
            self.distributions, str
        ):
            raise ValueError(distributions_error_msg)
        else:
            if not any(isinstance(d, tuple) for d in self.distributions):
                # Check if the number of distributions matches the number of features
                if len(self.distributions) != self.n_features_in_:
                    raise ValueError(
                        "Number of specified distributions must match the number of features "
                        f"({len(self.distributions)} != {self.n_features_in_})."
                    )

                # Handle `Sequence[DistributionLike]`
                dist_mapping = defaultdict(list)
                for i, dist in enumerate(self.distributions):
                    dist_mapping[dist].append(i)
            elif all(isinstance(d, tuple) for d in self.distributions):
                # Handle `Sequence[tuple[DistributionLike, ColumnKey]]`
                dist_mapping = {d[0]: d[1] for d in self.distributions}
            else:
                raise ValueError(distributions_error_msg)

            # Check that all specified distributions are supported
            for dist, cols in dist_mapping.items():
                if not is_dist_supported(dist):
                    raise ValueError(f"Distribution '{dist}' is not supported.")
                if isinstance(cols, str) or (
                    isinstance(cols, Iterable)
                    and any(isinstance(c, str) for c in cols)
                    and not hasattr(self, "feature_names_in_")
                ):
                    raise ValueError(
                        "Feature names are only supported when input data X is a DataFrame with named columns."
                    )

            # Initialize self.distributions_
            self.distributions_: list[DistributionLike] = []
            if hasattr(self, "feature_names_in_"):
                feature_names = self.feature_names_in_.tolist()
            else:
                feature_names = [None] * self.n_features_in_
            for i, feature_name in enumerate(feature_names):
                self.distributions_.append(self._find_dist(i, feature_name, dist_mapping))

        # Ensure alpha is a non-negative real number
        if not isinstance(self.alpha, Real) or self.alpha < 0:
            raise ValueError("Alpha must be a non-negative real number; got (alpha=%r) instead." % self.alpha)

        # Ensure variance smoothing is a non-negative real number
        if not isinstance(self.var_smoothing, Real) or self.var_smoothing < 0:
            raise ValueError(
                "Variance smoothing parameter must be a non-negative real number; got (var_smoothing=%r) instead."
                % self.var_smoothing
            )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: MatrixLike, y: ArrayLike) -> Self:
        """Fits general Naive Bayes classifier according to X, y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.n_features_in_: int
        self.feature_names_in_: np.ndarray
        _check_n_features(self, X=X, reset=True)
        _check_feature_names(self, X=X, reset=True)

        X, y = self._check_X_y(X, y)

        self.classes_, y_, self.class_count_ = np.unique(y, return_counts=True, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        self._init_parameters()

        self.epsilon_ = 0.0
        if np.issubdtype(X.dtype, np.number) and np.all(np.isreal(X)):
            self.epsilon_ = self.var_smoothing * np.var(X, axis=0).max()

        self.likelihood_params_: dict[int, list[DistMixin]] = {
            c: [
                get_dist_class(self.distributions_[i]).from_data(
                    X[y_ == c, i], alpha=self.alpha, epsilon=self.epsilon_
                )
                for i in range(self.n_features_in_)
            ]
            for c in range(self.n_classes_)
        }

        return self

    def _joint_log_likelihood(self, X) -> np.ndarray:
        jll = np.zeros((X.shape[0], self.n_classes_))
        for c in range(self.n_classes_):
            jll[:, c] = np.log(self.class_prior_[c]) + np.sum(
                [np.log(likelihood(X[:, i])) for i, likelihood in enumerate(self.likelihood_params_[c])],
                axis=0,
            )
        return jll
