from __future__ import annotations

import numbers
import sys
import warnings
from abc import ABCMeta
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.utils import as_float_array
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import _ensure_no_complex_data, check_is_fitted

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from ._utils import (
    SKLEARN_V1_6_OR_LATER,
    _check_feature_names,
    _check_n_features,
    validate_data,
)
from .typing import ArrayLike, Float, Int, MatrixLike

__all__ = ["GaussianWNB"]


class GaussianWNB(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
    """Binary Gaussian Minimum Log-likelihood Difference Weighted Naive Bayes (MLD-WNB) classifier

    Parameters
    ----------
    priors : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.

    error_weights : array-like of shape (n_classes, n_classes), default=None
        Matrix of error weights. If not specified, equal weight is assigned to the
        errors of both classes.

    max_iter : int, default=25
        Maximum number of gradient descent iterations.

    step_size : float, default=1e-4
        Step size of weight update (i.e., learning rate).

    penalty : str, default="l2"
        Regularization term, either 'l1' or 'l2'.

    C : float, default=1.0
        Regularization strength. Must be strictly positive.

    learning_hist : bool, default=False
        Whether to record the learning history, i.e., the value of cost function
        in each learning iteration.

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

    error_weights_ : ndarray of shape (n_classes, n_classes)
        Specified matrix of error weights.

    theta_ : ndarray of shape (n_features, n_classes)
        Mean of each feature per class.

    std_ : ndarray of shape (n_features, n_classes)
        Standard deviation of each feature per class.

    var_ : ndarray of shape (n_features, n_classes)
        Variance of each feature per class.

    coef_ : ndarray of shape (n_features,)
        Weights assigned to the features.

    cost_hist_ : ndarray of shape (max_iter,)
        Cost value in each iteration of the optimization.

    n_iter_ : int
        Number of iterations run by the optimization routine to fit the model.
    """

    def __init__(
        self,
        *,
        priors: Optional[ArrayLike] = None,
        error_weights: Optional[np.ndarray] = None,
        max_iter: Int = 25,
        step_size: Float = 1e-4,
        penalty: str = "l2",
        C: Float = 1.0,
        learning_hist: bool = False,
    ) -> None:
        self.priors = priors
        self.error_weights = error_weights
        self.max_iter = max_iter
        self.step_size = step_size
        self.penalty = penalty
        self.C = C
        self.learning_hist = learning_hist

    if SKLEARN_V1_6_OR_LATER:

        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.target_tags.required = True
            tags.classifier_tags.multi_class = False
            return tags

    def _more_tags(self) -> dict[str, bool]:
        return {"binary_only": True, "requires_y": True}

    def _check_inputs(self, X, y) -> None:
        # Check if the targets are suitable for classification
        check_classification_targets(y)

        # Check that the dataset has only two unique labels
        if (y_type := type_of_target(y)) != "binary":
            if SKLEARN_V1_6_OR_LATER:
                msg = f"Only binary classification is supported. The type of the target is {y_type}."
            else:
                msg = "Unknown label type: non-binary"
            raise ValueError(msg)

        # Check if only one class is present in label vector
        if self.n_classes_ == 1:
            raise ValueError("Classifier can't train when only one class is present.")

        X = validate_data(
            self,
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype="numeric",
            force_all_finite=True,
            ensure_2d=True,
            ensure_min_samples=1,
            ensure_min_features=1,
        )

        # Check that the number of samples and labels are compatible
        if self.__n_samples != y.shape[0]:
            raise ValueError("X.shape[0]=%d and y.shape[0]=%d are incompatible." % (X.shape[0], y.shape[0]))

        if self.priors is not None:
            # Check that the provided priors match the number of classes
            if len(self.priors) != self.n_classes_:
                raise ValueError("Number of priors must match the number of classes.")
            # Check that the sum of priors is 1
            if not np.isclose(self.priors.sum(), 1.0):
                raise ValueError("The sum of the priors should be 1.")
            # Check that the priors are non-negative
            if (self.priors < 0).any():
                raise ValueError("Priors must be non-negative.")

        if self.error_weights is not None:
            # Check that the size of error weights matrix matches number of classes
            if self.error_weights.shape != (self.n_classes_, self.n_classes_):
                raise ValueError(
                    "The shape of error weights matrix does not match the number of classes, "
                    "must be (n_classes, n_classes)."
                )

        # Check that the regularization type is either 'l1' or 'l2'
        if self.penalty not in ["l1", "l2"]:
            raise ValueError("Regularization type must be either 'l1' or 'l2'.")

        # Check that the regularization parameter is a positive integer
        if not isinstance(self.C, numbers.Number) or self.C < 0:
            raise ValueError("Regularization parameter must be positive; got (C=%r)" % self.C)

        # Check that the maximum number of iterations is a positive integer
        if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
            raise ValueError(
                "Maximum number of iteration must be a positive integer; got (max_iter=%r)." % self.max_iter
            )

    def _prepare_X_y(self, X=None, y=None, from_fit: bool = False):
        if from_fit and y is None:
            raise ValueError("requires y to be passed, but the target y is None.")

        if X is not None:
            # Convert to NumPy array if X is Pandas DataFrame
            if isinstance(X, pd.DataFrame):
                X = X.values
            _ensure_no_complex_data(X)
            X = as_float_array(X)

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

    def _prepare_parameters(self, X, y) -> None:
        # Calculate mean and standard deviation of features for each class
        for c in range(self.n_classes_):
            self.theta_[:, c] = np.mean(X[y == c, :], axis=0)  # Calculate mean of features for class c
            self.std_[:, c] = np.std(X[y == c, :], axis=0)  # Calculate std of features for class c
        self.var_ = np.square(self.std_)  # Calculate variance of features using std

        self.class_prior_: np.ndarray
        # Update if no priors is provided
        if self.priors is None:
            self.class_prior_ = (
                self.class_count_ / self.class_count_.sum()
            )  # Calculate empirical prior probabilities
        else:
            self.class_prior_ = self.priors

        # Convert to NumPy array if input priors is in a list/tuple/set
        if isinstance(self.class_prior_, (list, tuple, set)):
            self.class_prior_ = np.array(list(self.class_prior_))

        # Update if no error weights is provided
        if self.error_weights is None:
            self.error_weights_ = np.array([[0, 1], [-1, 0]])
        else:
            self.error_weights_ = self.error_weights

    def fit(self, X: MatrixLike, y: ArrayLike) -> Self:
        """Fits Gaussian Binary MLD-WNB classifier according to X, y.

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

        self.__n_samples = X.shape[0]  # Number of samples (for internal use)

        self._check_inputs(X, y)
        y = y_

        self.theta_: np.ndarray = np.zeros(
            (self.n_features_in_, self.n_classes_)
        )  # Mean of each feature per class (n_features x n_classes)
        self.std_: np.ndarray = np.zeros(
            (self.n_features_in_, self.n_classes_)
        )  # Standard deviation of each feature per class (n_features x n_classes)
        self.var_: np.ndarray = np.zeros(
            (self.n_features_in_, self.n_classes_)
        )  # Variance of each feature per class (n_features x n_classes)
        self.coef_: np.ndarray = np.ones((self.n_features_in_,))  # WNB coefficients (n_features x 1)
        self.cost_hist_: np.ndarray = np.array(
            [np.nan for _ in range(self.max_iter)]
        )  # Cost value in each iteration

        self._prepare_parameters(X, y)

        # Learn the weights using gradient descent
        self.n_iter_: int = 0
        for self.n_iter_ in range(self.max_iter):
            # Predict on X
            y_hat = self._predict(X)

            # Calculate cost
            self.cost_hist_[self.n_iter_], _lambda = self._calculate_cost(X, y, y_hat, self.learning_hist)

            # Calculate gradients (most time-consuming)
            _grad = self._calculate_grad(X, _lambda)

            # Add regularization
            if self.penalty == "l1":
                _grad += self.C * np.sign(self.coef_)
            elif self.penalty == "l2":
                _grad += 2 * self.C * self.coef_

            # Update weights
            self.coef_ = self.coef_ - self.step_size * _grad

        self.n_iter_ += 1
        self.cost_hist_ = None if not self.learning_hist else self.cost_hist_

        return self

    def _calculate_cost(self, X, y, y_hat, learning_hist: bool) -> tuple[Float, list[Float]]:
        _lambda = [self.error_weights_[y[i], y_hat[i]] for i in range(self.__n_samples)]

        if learning_hist:
            # Calculate cost
            _cost = 0.0
            for i in range(self.__n_samples):
                _sum = np.log(self.class_prior_[1] / self.class_prior_[0])
                x = X[i, :]
                for j in range(self.n_features_in_):
                    _sum += self.coef_[j] * (
                        np.log(1e-20 + norm.pdf(x[j], self.theta_[j, 1], self.std_[j, 1]))
                        - np.log(1e-20 + norm.pdf(x[j], self.theta_[j, 0], self.std_[j, 0]))
                    )
                _cost += _lambda[i] * _sum
        else:
            _cost = np.nan

        return _cost, _lambda

    def _calculate_grad(self, X, _lambda: list[Float]) -> np.ndarray:
        _grad = np.repeat(
            np.log(self.std_[:, 0] / self.std_[:, 1]).reshape(1, -1),
            self.__n_samples,
            axis=0,
        )
        _grad += (
            0.5
            * (
                (X - np.repeat(self.theta_[:, 0].reshape(1, -1), self.__n_samples, axis=0))
                / (np.repeat(self.std_[:, 0].reshape(1, -1), self.__n_samples, axis=0))
            )
            ** 2
        )
        _grad -= (
            0.5
            * (
                (X - np.repeat(self.theta_[:, 1].reshape(1, -1), self.__n_samples, axis=0))
                / (np.repeat(self.std_[:, 1].reshape(1, -1), self.__n_samples, axis=0))
            )
            ** 2
        )
        _grad *= np.transpose(np.repeat(np.array(_lambda).reshape(1, -1), self.n_features_in_, axis=0))
        _grad = np.sum(_grad, axis=0)

        return _grad

    def _predict(self, X: MatrixLike) -> np.ndarray:
        p_hat = self.predict_log_proba(X)
        return np.argmax(p_hat, axis=1)

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
        X = validate_data(self, X, accept_large_sparse=False, force_all_finite=True, reset=False)

        # Check if the number of input features matches the data seen during fit
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Expected input with %d features, got %d instead." % (self.n_features_in_, X.shape[1])
            )

        n_samples = X.shape[0]

        X = self._prepare_X_y(X=X)

        log_priors = np.tile(np.log(self.class_prior_), (n_samples, 1))
        w_reshaped = np.tile(self.coef_.reshape(-1, 1), (1, self.n_classes_))
        term1 = np.sum(np.multiply(w_reshaped, -np.log(np.sqrt(2 * np.pi) * self.std_)))
        var_inv = np.multiply(w_reshaped, 1.0 / np.multiply(self.std_, self.std_))
        mu_by_var = np.multiply(self.theta_, var_inv)
        term2 = -0.5 * (
            np.matmul(np.multiply(X, X), var_inv)
            - 2.0 * np.matmul(X, mu_by_var)
            + np.sum(self.theta_.conj() * mu_by_var, axis=0)
        )
        log_proba = log_priors + term1 + term2

        log_proba -= np.transpose(
            np.repeat(logsumexp(log_proba, axis=1).reshape(1, -1), self.n_classes_, axis=0)
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
