from abc import ABCMeta
import numbers
from typing import Union, Optional, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import logsumexp

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.utils import check_array, as_float_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import type_of_target


class WeightedNB(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
    """
    Binary Gaussian Minimum Log-likelihood Difference Weighted Naive Bayes (MLD-WNB) Classifier
    """

    def __init__(self, priors: Optional[Sequence, np.ndarray] = None, error_weights: Optional[np.ndarray] = None,
                 max_iter: int = 25, step_size: float = 1e-4, penalty: str = 'l2', C: float = 1.0) -> None:
        """Initializes an object of the class.

        Args:
            priors (Optional[Sequence, np.ndarray]): Prior probabilities. Defaults to None.
            error_weights (Optional[np.ndarray]): Matrix of error weights (n_classes * n_classes). Defaults to None.
            max_iter (int): Maximum number of gradient descent iterations. Defaults to 25.
            step_size (float): Step size of weight update (i.e., learning rate). Defaults to 1e-4.
            penalty (str): Regularization term; must be either 'l1' or 'l2'. Defaults to 'l2'.
            C (float): Regularization strength; must be a positive float. Defaults to 1.0.

        Returns:
            self: The instance itself.
        """
        self.priors = priors  # Prior probabilities of classes (n_classes x 1)
        self.error_weights = error_weights  # Matrix of error weights (n_features x n_features)
        self.max_iter = max_iter  # Maximum number of iterations of the learning algorithm
        self.step_size = step_size  # Learning rate
        self.penalty = penalty  # Regularization type ('l1' or 'l2')
        self.C = C  # Regularization parameter

    def _more_tags(self):
        return {
            'binary_only': True,
            'requires_y': True
        }

    def __check_inputs(self, X, y):
        # Check that the dataset has only two unique labels
        if type_of_target(y) != 'binary':
            warnings.warn('This version of MLD-WNB only supports binary classification.')
            raise ValueError('Unknown label type: non-binary')

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

        if self.priors is not None:
            # Check that the provided priors match the number of classes
            if len(self.priors) != self.n_classes_:
                raise ValueError('Number of priors must match the number of classes.')
            # Check that the sum of priors is 1
            if not np.isclose(self.priors.sum(), 1.0):
                raise ValueError('The sum of the priors should be 1.')
            # Check that the priors are non-negative
            if (self.priors < 0).any():
                raise ValueError('Priors must be non-negative.')

        if self.error_weights is not None:
            # Check that the size of error weights matrix matches number of classes
            if self.error_weights.shape != (self.n_classes_, self.n_classes_):
                raise ValueError(
                    'The shape of error weights matrix does not match the number of classes, '
                    'must be (n_classes, n_classes).'
                )

        # Check that the regularization type is either 'l1' or 'l2'
        if self.penalty not in ['l1', 'l2']:
            raise ValueError("Regularization type must be either 'l1' or 'l2'.")

        # Check that the regularization parameter is a positive integer
        if not isinstance(self.C, numbers.Number) or self.C < 0:
            raise ValueError(
                "Regularization parameter must be positive; got (C=%r)"
                % self.C
            )

        # Check that the maximum number of iterations is a positive integer
        if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
            raise ValueError(
                "Maximum number of iteration must be a positive integer; got (max_iter=%r)"
                % self.max_iter
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
        # Calculate mean and standard deviation of features for each class
        for c in range(self.n_classes_):
            self.mu_[:, c] = np.mean(X[y == c, :], axis=0)  # Calculate mean of features for class c
            self.std_[:, c] = np.std(X[y == c, :], axis=0)  # Calculate std of features for class c

        # Update if no priors is provided
        if self.priors is None:
            _, class_count_ = np.unique(y, return_counts=True)
            self.class_prior_ = class_count_ / class_count_.sum()  # Calculate empirical prior probabilities
        else:
            self.class_prior_ = self.priors

        # Convert to NumPy array in input priors is in a list
        if type(self.class_prior_) is list:
            self.class_prior_ = np.array(self.class_prior_)

        # Update if no error weights is provided
        if self.error_weights is None:
            self.error_weights_ = np.array([[0, 1], [-1, 0]])
        else:
            self.error_weights_ = self.error_weights

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame, pd.Series],
            learning_hist: bool = False):
        """Fits Gaussian Binary MLD-WNB according to X and y.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Array-like of shape (n_samples, n_features).
                                                 Training vectors, where `n_samples` is the number of samples
                                                 and `n_features` is the number of features.
            y (Union[np.ndarray, pd.DataFrame, pd.Series]): Array-like of shape (n_samples,). Target values.
            learning_hist (bool): Whether to keep the learning history (i.e., the value of cost function in each
                                  learning iteration)

        Returns:
            self: The instance itself.
        """
        X, y = self.__prepare_X_y(X, y)

        self.classes_, y_ = np.unique(y, return_inverse=True)  # Unique class labels and their indices
        self.n_classes_ = len(self.classes_)  # Number of classes
        self.__n_samples, self.n_features_in_ = X.shape  # Number of samples and features

        self.__check_inputs(X, y)
        y = y_

        self.mu_ = np.zeros((self.n_features_in_, self.n_classes_))  # Mean of features (n_features x 1)
        self.std_ = np.zeros((self.n_features_in_, self.n_classes_))  # Standard deviation of features (n_features x 1)
        self.coef_ = np.ones((self.n_features_in_,))  # WNB coefficients (n_features x 1)
        self.cost_hist_ = np.array([np.nan for _ in range(self.max_iter)])  # To store cost value in each iteration

        self.__prepare_parameters(X, y)

        # Learn the weights using gradient descent
        self.n_iter_ = 0
        for self.n_iter_ in range(self.max_iter):
            # Predict on X
            y_hat = self.__predict(X)

            # Calculate cost
            self.cost_hist_[self.n_iter_], _lambda = self.__calculate_cost(X, y, y_hat, learning_hist)

            # Calculate gradients (most time-consuming)
            _grad = self.__calculate_grad(X, _lambda)

            # Add regularization
            if self.penalty == 'l1':
                _grad += self.C * np.sign(self.coef_)
            elif self.penalty == 'l2':
                _grad += 2 * self.C * self.coef_

            # Update weights
            self.coef_ = self.coef_ - self.step_size * _grad

        return self

    def __calculate_cost(self, X, y, y_hat, learning_hist):
        _lambda = [self.error_weights_[y[i], y_hat[i]] for i in range(self.__n_samples)]

        if learning_hist:
            # Calculate cost
            _cost = 0
            for i in range(self.__n_samples):
                _sum = np.log(self.class_prior_[1] / self.class_prior_[0])
                x = X[i, :]
                for j in range(self.n_features_in_):
                    _sum += self.coef_[j] * (np.log(1e-20 + norm.pdf(x[j], self.mu_[j, 1], self.std_[j, 1]))
                                                - np.log(1e-20 + norm.pdf(x[j], self.mu_[j, 0], self.std_[j, 0])))
                _cost += _lambda[i] * _sum
        else:
            _cost = None

        return _cost, _lambda

    def __calculate_grad(self, X, _lambda):
        _grad = np.repeat(np.log(self.std_[:, 0] / self.std_[:, 1]).reshape(1, -1), self.__n_samples, axis=0)
        _grad += 0.5 * ((X - np.repeat(self.mu_[:, 0].reshape(1, -1), self.__n_samples, axis=0)) /
                        (np.repeat(self.std_[:, 0].reshape(1, -1), self.__n_samples, axis=0))) ** 2
        _grad -= 0.5 * ((X - np.repeat(self.mu_[:, 1].reshape(1, -1), self.__n_samples, axis=0)) /
                        (np.repeat(self.std_[:, 1].reshape(1, -1), self.__n_samples, axis=0))) ** 2
        _grad *= np.transpose(np.repeat(np.array(_lambda).reshape(1, -1), self.n_features_in_, axis=0))
        _grad = np.sum(_grad, axis=0)

        return _grad

    def __calculate_grad_slow(self, X, _lambda):
        _grad = np.zeros((self.n_features_in_,))
        for i in range(self.__n_samples):
            x = X[i, :]
            _log_p = np.array(
                [
                    np.log(self.std_[j, 0] / self.std_[j, 1]) +
                    0.5*((x[j] - self.mu_[j, 0]) / self.std_[j, 0])**2 -
                    0.5*((x[j] - self.mu_[j, 1]) / self.std_[j, 1])**2
                    for j in range(self.n_features_in_)
                ]
            )
            _grad += _lambda[i] * _log_p
        return _grad

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

        log_priors = np.tile(np.log(self.class_prior_), (n_samples, 1))
        w_reshaped = np.tile(self.coef_.reshape(-1, 1), (1, self.n_classes_))
        term1 = np.sum(np.multiply(w_reshaped, -np.log(np.sqrt(2 * np.pi) * self.std_)))
        var_inv = np.multiply(w_reshaped, 1/np.multiply(self.std_, self.std_))
        mu_by_var = np.multiply(self.mu_, var_inv)
        term2 = -0.5*(np.matmul(np.multiply(X, X), var_inv) - 2*np.matmul(X, mu_by_var)
                      + np.sum(self.mu_.conj()*mu_by_var, axis=0))
        log_proba = log_priors + term1 + term2

        log_proba -= np.transpose(np.repeat(logsumexp(log_proba, axis=1).reshape(1, -1), self.n_classes_, axis=0))
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
