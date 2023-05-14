from abc import ABCMeta
from typing import Union, Optional, Sequence, Type
import warnings

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.utils import check_array, as_float_array
from sklearn.utils.validation import check_is_fitted

from ._base import ContinuousDistMixin, DiscreteDistMixin
from ._enums import Distribution
from .dist import AllDistributions


class GeneralNB(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
    """
    A general Naive Bayes classifier that allows you to specify the likelihood distribution for each feature.
    """

    def __init__(self, *,
                 priors: Optional[Union[Sequence[float], np.ndarray]] = None,
                 distributions: Optional[Sequence[Union[
                     str, Distribution, Type[ContinuousDistMixin], Type[DiscreteDistMixin]]]
                 ] = None) -> None:
        """Initializes an object of the class.

        Args:
            priors (Optional[Union[list, np.ndarray]]): Prior probabilities. Defaults to None.
            distributions: Names of the distributions to be used for features' likelihoods. A sequence with same length
                           of the number of features. If not specified, all likelihood will be considered Gaussian.
                           Defaults to None.

        """
        self.priors = priors
        self.distributions = distributions

    def _more_tags(self):
        return {
            'multilabel': True,
            'requires_y': True
        }

    def __check_inputs(self, X, y):
        # Check if only one class is present in label vector
        if self.n_classes_ == 1:
            raise ValueError("Classifier can't train when only one class is present")

        X = check_array(
            array=X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype='numeric',
            force_all_finite=True,
            ensure_2d=True,
            ensure_min_samples=1,
            ensure_min_features=1,
            estimator=self
        )

        # Check if X contains complex values
        if np.iscomplex(X).any() or np.iscomplex(y).any():
            raise ValueError("Complex data not supported")

        # Check that the number of samples and labels are compatible
        if self.__n_samples != y.shape[0]:
            raise ValueError(
                "X.shape[0]=%d and y.shape[0]=%d are incompatible." % (X.shape[0], y.shape[0])
            )

    def __prepare_X_y(self, X=None, y=None):
        if X is not None:
            # Convert to NumPy array if X is Pandas DataFrame
            if isinstance(X, pd.DataFrame):
                X = X.values
            X = as_float_array(X)

        if y is not None:
            # Convert to a NumPy array
            if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                y = y.values
            else:
                y = np.array(y)

            # Warning in case of y being 2d
            if y.ndim > 1:
                warnings.warn("A column-vector y was passed when a 1d array was expected.", DataConversionWarning)

            y = y.flatten()

        output = tuple(item for item in [X, y] if item is not None)
        output = output[0] if len(output) == 1 else output
        return output

    def __prepare_parameters(self, X, y):
        # Set priors if not specified
        if self.priors is None:
            _, class_count_ = np.unique(y, return_counts=True)
            self.class_prior_ = class_count_ / class_count_.sum()  # Calculate empirical prior probabilities

        else:
            # Check that the provided priors match the number of classes
            if len(self.priors) != self.n_classes_:
                raise ValueError('Number of priors must match the number of classes.')
            # Check that the sum of priors is 1
            if not np.isclose(self.priors.sum(), 1.0):
                raise ValueError('The sum of the priors should be 1.')
            # Check that the priors are non-negative
            if (self.priors < 0).any():
                raise ValueError('Priors must be non-negative.')

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
                    "Number of specified distributions must match the number of features "
                    f"({len(self.distributions)} != {self.n_features_in_})"
                )

            # Check that all specified distributions are supported
            for dist in self.distributions:
                if not (
                        isinstance(dist, Distribution) or
                        dist in Distribution.__members__.values() or
                        (hasattr(dist, 'from_data') and hasattr(dist, '__call__'))
                ):
                    raise ValueError(f"Distribution '{dist}' is not supported")

            self.distributions_ = self.distributions

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame, pd.Series]):
        """Fits general Naive Bayes classifier according to X and y.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Array-like of shape (n_samples, n_features).
                                                 Training vectors, where `n_samples` is the number of samples
                                                 and `n_features` is the number of features.
            y (Union[np.ndarray, pd.DataFrame, pd.Series]): Array-like of shape (n_samples,). Target values.

        Returns:
            self: The instance itself.
        """
        X, y = self.__prepare_X_y(X, y)

        self.classes_, y_ = np.unique(y, return_inverse=True)  # Unique class labels and their indices
        self.n_classes_ = len(self.classes_)  # Number of classes
        self.__n_samples, self.n_features_in_ = X.shape  # Number of samples and features

        self.__check_inputs(X, y)

        y = y_
        self.__prepare_parameters(X, y)

        self.likelihood_params_ = {
            c: [
                AllDistributions[self.distributions_[i]].from_data(X[y == c, i])
                if isinstance(self.distributions_[i], Distribution) or
                   self.distributions_[i] in Distribution.__members__.values() else
                self.distributions_[i].from_data(X[y == c, i])
                for i in range(self.n_features_in_)
            ]
            for c in range(self.n_classes_)
        }

        return self

    def __predict(self, X):
        p_hat = self.predict_log_proba(X)
        return np.argmax(p_hat, axis=1)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Performs classification on an array of test vectors X.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Array-like of shape (n_samples, n_features). The input samples.

        Returns:
            np.ndarray: ndarray of shape (n_samples,). Predicted target values for X.
        """
        p_hat = self.predict_log_proba(X)
        y_hat = self.classes_[np.argmax(p_hat, axis=1)]
        return y_hat

    def predict_log_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Returns log-probability estimates for the test vector X.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Array-like of shape (n_samples, n_features). The input samples.

        Returns:
            np.ndarray: Array-like of shape (n_samples, n_classes).
                        The log-probability of the samples for each class in the model.
                        The columns correspond to the classes in sorted order, as they appear in the attribute `classes_`.
        """
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(array=X, accept_large_sparse=False, force_all_finite=True, estimator=self)

        # Check if the number of input features matches the data seen during fit
        if not X.shape[1] == self.n_features_in_:
            raise ValueError(
                "Expected input with %d features, got %d instead." % (self.n_features_in_, X.shape[1])
            )

        n_samples = X.shape[0]

        X = self.__prepare_X_y(X=X)

        log_joint = np.zeros((n_samples, self.n_classes_))
        for c in range(self.n_classes_):
            log_joint[:, c] = np.log(self.class_prior_[c]) + np.sum(
                (
                    np.log(likelihood(X[:, i]))
                    for i, likelihood in enumerate(self.likelihood_params_[c])
                ),
                axis=0
            )

        log_proba = log_joint - np.transpose(
            np.repeat(logsumexp(log_joint, axis=1).reshape(1, -1), self.n_classes_, axis=0)
        )
        return log_proba

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Returns probability estimates for the test vector X.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Array-like of shape (n_samples, n_features). The input samples.

        Returns:
            np.ndarray: Array-like of shape (n_samples, n_classes).
                        The probability of the samples for each class in the model.
                        The columns correspond to the classes in sorted order, as they appear in the attribute `classes_`.
        """
        return np.exp(self.predict_log_proba(X))
