from abc import ABCMeta
from typing import Optional, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.utils import check_array, as_float_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from ._enums import Distribution
from ._typing import MatrixLike, ArrayLike, Float, DistibutionLike
from .dist import AllDistributions, NonNumericDistributions

__all__ = [
    "GeneralNB",
]


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

    class_count_: np.ndarray
    class_prior_: np.ndarray
    classes_: np.ndarray
    n_classes_: int
    n_features_in_: int
    feature_names_in_: np.ndarray
    distributions_: list
    likelihood_params_: dict

    def __init__(
        self,
        *,
        priors: Optional[ArrayLike] = None,
        distributions: Optional[Sequence[DistibutionLike]] = None,
        alpha: Float = 1e-10,
    ) -> None:
        self.priors = priors
        self.distributions = distributions
        self.alpha = alpha

    def _more_tags(self):
        return {"requires_y": True}

    def _get_distributions(self):
        try:
            if self.distributions_ is not None:
                return self.distributions_
        except Exception:
            return self.distributions or []

    def _check_inputs(self, X, y):
        # Check if the targets are suitable for classification
        check_classification_targets(y)

        # Check if only one class is present in label vector
        if self.n_classes_ == 1:
            raise ValueError("Classifier can't train when only one class is present")

        X = check_array(
            array=X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype=None
            if any(d in self._get_distributions() for d in NonNumericDistributions)
            else "numeric",
            force_all_finite=True,
            ensure_2d=True,
            ensure_min_samples=1,
            ensure_min_features=1,
            estimator=self,
        )

        # Check if X contains complex values
        if np.iscomplex(X).any() or np.iscomplex(y).any():
            raise ValueError("Complex data not supported")

        # Check that the number of samples and labels are compatible
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X.shape[0]=%d and y.shape[0]=%d are incompatible."
                % (X.shape[0], y.shape[0])
            )

    def _prepare_X_y(self, X=None, y=None, from_fit=False):
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

    def _prepare_parameters(self):
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

        # Convert to NumPy array if input priors is in a list
        if type(self.class_prior_) is list:
            self.class_prior_ = np.array(self.class_prior_)

        # Set distributions if not specified
        if self.distributions is None:
            self.distributions_ = [Distribution.NORMAL] * self.n_features_in_

        else:
            # Check if the number of distributions matches the number of features
            if len(self.distributions) != self.n_features_in_:
                raise ValueError(
                    "Number of specified distributions must match the number of features."
                    f"({len(self.distributions)} != {self.n_features_in_})"
                )

            # Check that all specified distributions are supported
            for dist in self.distributions:
                if not (
                    isinstance(dist, Distribution)
                    or dist in Distribution.__members__.values()
                    or (hasattr(dist, "from_data") and hasattr(dist, "__call__"))
                ):
                    raise ValueError(f"Distribution '{dist}' is not supported.")

            self.distributions_ = self.distributions

    def fit(self, X: MatrixLike, y: ArrayLike):
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
        self._check_n_features(X=X, reset=True)
        self._check_feature_names(X=X, reset=True)

        X, y = self._prepare_X_y(X, y, from_fit=True)

        self.classes_, y_, self.class_count_ = np.unique(
            y, return_counts=True, return_inverse=True
        )  # Unique class labels, their indices, and class counts
        self.n_classes_ = len(self.classes_)  # Number of classes

        self._check_inputs(X, y)

        y = y_
        self._prepare_parameters()

        self.likelihood_params_ = {
            c: [
                AllDistributions[self.distributions_[i]].from_data(
                    X[y == c, i], alpha=self.alpha
                )
                if isinstance(self.distributions_[i], Distribution)
                or self.distributions_[i] in Distribution.__members__.values()
                else self.distributions_[i].from_data(X[y == c, i], alpha=self.alpha)
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
        y_hat = self.classes_[np.argmax(p_hat, axis=1)]
        return y_hat

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
        X = check_array(
            array=X,
            accept_large_sparse=False,
            force_all_finite=True,
            dtype=None
            if any(d in self._get_distributions() for d in NonNumericDistributions)
            else "numeric",
            estimator=self,
        )

        # Check if the number of input features matches the data seen during fit
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Expected input with %d features, got %d instead."
                % (self.n_features_in_, X.shape[1])
            )

        n_samples = X.shape[0]

        X = self._prepare_X_y(X=X)

        log_joint = np.zeros((n_samples, self.n_classes_))
        for c in range(self.n_classes_):
            log_joint[:, c] = np.log(self.class_prior_[c]) + np.sum(
                [
                    np.log(likelihood(X[:, i]))
                    for i, likelihood in enumerate(self.likelihood_params_[c])
                ],
                axis=0,
            )

        log_proba = log_joint - np.transpose(
            np.repeat(
                logsumexp(log_joint, axis=1).reshape(1, -1), self.n_classes_, axis=0
            )
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
