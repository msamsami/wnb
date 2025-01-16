from __future__ import annotations

import sys
import warnings
from abc import ABCMeta
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.utils import as_float_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

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
    validate_data,
)
from .typing import ArrayLike, Float, MatrixLike

__all__ = ["GeneralNB"]


class GeneralNB(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
    """A General Naive Bayes classifier that supports distinct likelihood distributions for individual features,
    enabling more tailored modeling beyond the standard single-distribution approaches such as GaussianNB and BernoulliNB.

    Parameters
    ----------
    priors : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.

    distributions : sequence of distribution-like of length n_features, default=None
        Probability distributions to be used for features' likelihoods. If not specified,
        all likelihoods will be considered Gaussian.

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
    """

    def __init__(
        self,
        *,
        priors: Optional[ArrayLike] = None,
        distributions: Optional[Sequence[DistributionLike]] = None,
        alpha: Float = 1e-10,
        var_smoothing: Float = 1e-9,
    ) -> None:
        self.priors = priors
        self.distributions = distributions
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

    def _check_inputs(self, X, y) -> None:
        # Check if the targets are suitable for classification
        check_classification_targets(y)

        # Check if only one class is present in label vector
        if self.n_classes_ == 1:
            raise ValueError("Classifier can't train when only one class is present")

        X = validate_data(
            self,
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype=(
                None if any(d in self._get_distributions() for d in NonNumericDistributions) else "numeric"
            ),
            force_all_finite=True,
            ensure_2d=True,
            ensure_min_samples=1,
            ensure_min_features=1,
        )

        # Check if X contains complex values
        if np.iscomplex(X).any() or np.iscomplex(y).any():
            raise ValueError("Complex data not supported")

        # Check that the number of samples and labels are compatible
        if X.shape[0] != y.shape[0]:
            raise ValueError("X.shape[0]=%d and y.shape[0]=%d are incompatible." % (X.shape[0], y.shape[0]))

    def _prepare_X_y(self, X=None, y=None, from_fit: bool = False):
        if from_fit and y is None:
            raise ValueError("requires y to be passed, but the target y is None.")

        if X is not None:
            # Convert to NumPy array if X is Pandas DataFrame
            if isinstance(X, pd.DataFrame):
                X = X.values
            X = (
                X
                if any(d in self._get_distributions() for d in NonNumericDistributions)
                else as_float_array(X)
            )

        if y is not None:
            # Convert to a NumPy array
            if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                y = y.values
            else:
                y = np.array(y)

            # Warning in case of y being 2d
            if y.ndim > 1:
                warnings.warn(
                    "A column-vector y was passed when a 1d array was expected.",
                    DataConversionWarning,
                )

            y = y.flatten()

        output = tuple(item for item in [X, y] if item is not None)
        return output[0] if len(output) == 1 else output

    def _prepare_parameters(self) -> None:
        self.class_prior_: np.ndarray

        # Set priors if not specified
        if self.priors is None:
            self.class_prior_ = (
                self.class_count_ / self.class_count_.sum()
            )  # Calculate empirical prior probabilities

        else:
            # Check that the provided priors match the number of classes
            if len(self.priors) != self.n_classes_:
                raise ValueError("Number of priors must match the number of classes.")
            # Check that the sum of priors is 1
            if not np.isclose(self.priors.sum(), 1.0):
                raise ValueError("The sum of the priors should be 1.")
            # Check that the priors are non-negative
            if (self.priors < 0).any():
                raise ValueError("Priors must be non-negative.")

            self.class_prior_ = self.priors

        # Convert to NumPy array if input priors is in a list/tuple/set
        if isinstance(self.class_prior_, (list, tuple, set)):
            self.class_prior_ = np.array(list(self.class_prior_))

        # Set distributions if not specified
        if self.distributions is None:
            self.distributions_: list[DistributionLike] = [Distribution.NORMAL] * self.n_features_in_
        else:
            # Check if the number of distributions matches the number of features
            if len(self.distributions) != self.n_features_in_:
                raise ValueError(
                    "Number of specified distributions must match the number of features."
                    f"({len(self.distributions)} != {self.n_features_in_})"
                )

            # Check that all specified distributions are supported
            for i, dist in enumerate(self.distributions):
                if not is_dist_supported(dist):
                    raise ValueError(f"Distribution '{dist}' at index {i} is not supported.")

            self.distributions_: list[DistributionLike] = list(self.distributions)

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

        X, y = self._prepare_X_y(X, y, from_fit=True)

        self.classes_: np.ndarray
        self.class_count_: np.ndarray
        self.classes_, y_, self.class_count_ = np.unique(
            y, return_counts=True, return_inverse=True
        )  # Unique class labels, their indices, and class counts
        self.n_classes_: int = len(self.classes_)  # Number of classes

        self._check_inputs(X, y)
        y = y_
        self._prepare_parameters()

        if np.all(np.isreal(X)):
            self.epsilon_ = self.var_smoothing * np.var(X, axis=0).max()
        else:
            self.epsilon_ = 0.0

        self.likelihood_params_: dict[int, list[DistMixin]] = {
            c: [
                get_dist_class(self.distributions_[i]).from_data(
                    X[y == c, i], alpha=self.alpha, epsilon=self.epsilon_
                )
                for i in range(self.n_features_in_)
            ]
            for c in range(self.n_classes_)
        }

        return self

    def predict(self, X: MatrixLike) -> np.ndarray:
        """Performs classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : ndarray of shape (n_samples,)
            Predicted target values for X.
        """
        p_hat = self.predict_log_proba(X)
        return self.classes_[np.argmax(p_hat, axis=1)]

    def predict_log_proba(self, X: MatrixLike) -> np.ndarray:
        """Returns log-probability estimates for the array of test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = validate_data(
            self,
            X,
            accept_large_sparse=False,
            force_all_finite=True,
            dtype=(
                None if any(d in self._get_distributions() for d in NonNumericDistributions) else "numeric"
            ),
            reset=False,
        )

        # Check if the number of input features matches the data seen during fit
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Expected input with %d features, got %d instead." % (self.n_features_in_, X.shape[1])
            )

        n_samples = X.shape[0]
        X = self._prepare_X_y(X=X)

        log_joint = np.zeros((n_samples, self.n_classes_))
        for c in range(self.n_classes_):
            log_joint[:, c] = np.log(self.class_prior_[c]) + np.sum(
                [np.log(likelihood(X[:, i])) for i, likelihood in enumerate(self.likelihood_params_[c])],
                axis=0,
            )

        log_proba = log_joint - np.transpose(
            np.repeat(logsumexp(log_joint, axis=1).reshape(1, -1), self.n_classes_, axis=0)
        )
        return log_proba

    def predict_proba(self, X: MatrixLike) -> np.ndarray:
        """Returns probability estimates for the array of test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        return np.exp(self.predict_log_proba(X))
