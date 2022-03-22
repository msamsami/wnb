import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Union


class WeightedNB:

    def __init__(self, priors: Union[list, np.ndarray, None] = None, error_weights: Union[np.ndarray, None] = None,
                 max_iter: int = 25, step_size: float = 1e-4, penalty: str = 'l2', C: float = 1.0):
        self.priors = priors  # Prior probabilities of classes (n_classes x 1)
        self.__priors = None
        self.error_weights = error_weights  # Matrix of error weights (n_features x n_features)
        self.__error_weights = None
        self.max_iter = max_iter  # Maximum number of iterations of the learning algorithm
        self.step_size = step_size  # Learning rate
        self.penalty = penalty  # Regularization type ('l1' or 'l2')
        self.C = C  # Regularization parameter

        self.n_samples = None  # Number of samples
        self.n_classes = None  # Number of classes
        self.classes_ = None  # Class labels
        self.class_count_ = None  # Number of samples in each class
        self.n_features = None  # Number of features
        self.mu = None  # Mean of features (n_features x 1)
        self.std = None  # Standard deviation of features (n_features x 1)
        self.weights_ = None  # WNB parameters (n_features x 1)

        self._fit_status = False  # True is correctly fitted; False otherwise
        self.cost_hist_ = None  # Cost value in each iteration

    def __check_inputs(self, X, y):
        # Check that the dataset has only two unique labels
        if self.n_classes != 2:
            raise ValueError('This version of MLD-WNB only supports binary classification.')

        if self.priors is not None:
            # Check that the provided priors match the number of classes
            if len(self.priors) != self.n_classes:
                raise ValueError('Number of priors must match the number of classes.')
            # Check that the sum of priors is 1
            if not np.isclose(self.priors.sum(), 1.0):
                raise ValueError('The sum of the priors should be 1.')
            # Check that the priors are non-negative
            if (self.priors < 0).any():
                raise ValueError('Priors must be non-negative.')

        if self.error_weights is not None:
            # Check that the size of error weights matrix matches number of classes
            if self.error_weights.shape != (self.n_classes, self.n_classes):
                raise ValueError(
                    'The shape of error weights matrix does not match the number of classes, '
                    'must be (n_classes, n_classes).'
                )

        # Check that the maximum number of iterations is a positive integer
        if type(self.max_iter) is not int or self.max_iter < 1:
            raise ValueError('Maximum number of iterations must be a positive integer.')

        # Check that the regularization type is either 'l1' or 'l2'
        if self.penalty not in ['l1', 'l2']:
            raise ValueError("Regularization type must be either 'l1' or 'l2'.")

    def __prepare_parameters(self, X, y):

        # Calculate mean and standard deviation of features for each class
        self.mu = np.zeros((self.n_features, self.n_classes))
        self.std = np.zeros((self.n_features, self.n_classes))
        for i, c in enumerate(self.classes_):
            self.mu[:, i] = np.mean(X[y == c, :], axis=0)  # Calculate mean of features for class c
            self.std[:, i] = np.std(X[y == c, :], axis=0)  # Calculate std of features for class c

        # Update if no priors is provided
        if self.priors is None:
            self.__priors = self.class_count_ / self.n_samples  # Calculate empirical prior probabilities
        else:
            self.__priors = self.priors

        # Convert to NumPy array in input priors is in a list
        if type(self.__priors) is list:
            self.__priors = np.array(self.__priors)

        # Update if no error weights is provided
        if self.error_weights is None:
            self.__error_weights = np.array([[0, 1], [-1, 0]])
        else:
            self.__error_weights = self.error_weights

    def __prepare_X_y(self, X=None, y=None):
        if X is not None:
            # Convert to NumPy array if X is Pandas DataFrame
            if isinstance(X, pd.DataFrame):
                X = X.values

        if y is not None:
            # Convert to a flat NumPy array
            if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                y = y.values
            y = y.flatten()

        output = tuple(item for item in [X, y] if item is not None)
        output = output[0] if len(output) == 1 else output
        return output

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame, pd.Series],
            learning_hist: bool = False):

        X, y = self.__prepare_X_y(X, y)

        self.classes_, self.class_count_ = np.unique(y, return_counts=True)  # Unique class labels and their counts
        self.n_classes = len(self.classes_)  # Find the number of classes
        self.n_samples, self.n_features = X.shape  # Find the number of samples and features

        # Check that the number of samples and labels are compatible
        if self.n_samples != y.shape[0]:
            raise ValueError(
                "X.shape[0]=%d and y.shape[0]=%d are incompatible." % (X.shape[0], y.shape[0])
            )

        self.__check_inputs(X, y)

        self.__prepare_parameters(X, y)

        self.weights_ = np.ones((self.n_features,))  # Initialize the weights
        self.cost_hist_ = np.zeros((self.max_iter,))  # To store history of cost changes

        # Learn the weights using gradient descent
        for _iter in range(self.max_iter):
            # Predict on X
            y_hat = self.predict(X)

            # Calculate cost
            self.cost_hist_[_iter], _lambda = self.__calculate_cost(X, y, y_hat, learning_hist)

            # Calculate gradients (most time-consuming)
            _grad = self.__calculate_grad(X, _lambda)

            # Add regularization
            _grad += (self.C * np.sign(self.weights_)) if self.penalty == 'l1' else (2 * self.C * self.weights_)
            # if self.penalty == 'l1':
            #     _grad += self.C * np.sign(self.weights_)
            # elif self.penalty == 'l2':
            #     _grad += 2 * self.C * self.weights_

            # Update weights
            self.weights_ = self.weights_ - self.step_size * _grad

        self._fit_status = True

    def __calculate_cost(self, X, y, y_hat, learning_hist):
        _lambda = [self.__error_weights[y[i], y_hat[i]] for i in range(self.n_samples)]

        if learning_hist:
            # Calculate cost
            _cost = 0
            for i in range(self.n_samples):
                _sum = np.log(self.__priors[1] / self.__priors[0])
                x = X[i, :]
                for j in range(self.n_features):
                    _sum += self.weights_[j] * (np.log(1e-20 + norm.pdf(x[j], self.mu[j, 1], self.std[j, 1]))
                                                - np.log(1e-20 + norm.pdf(x[j], self.mu[j, 0], self.std[j, 0])))
                _cost += _lambda[i] * _sum
        else:
            _cost = None

        return _cost, _lambda

    def __calculate_grad(self, X, _lambda):
        # _grad = np.zeros((self.n_features,))
        # for i in range(self.n_samples):
        #     x = X[i, :]
        #     _log_p = np.array(
        #         [
        #             np.log(self.std[j, 0] / self.std[j, 1]) +
        #             0.5*((x[j] - self.mu[j, 0]) / self.std[j, 0])**2 -
        #             0.5*((x[j] - self.mu[j, 1]) / self.std[j, 1])**2
        #             for j in range(self.n_features)
        #         ]
        #     )
        #     _grad += _lambda[i] * _log_p

        _grad = np.repeat(np.log(self.std[:, 0] / self.std[:, 1]).reshape(1, -1), self.n_samples, axis=0)
        _grad += 0.5 * ((X - np.repeat(self.mu[:, 0].reshape(1, -1), self.n_samples, axis=0)) /
                        (np.repeat(self.std[:, 0].reshape(1, -1), self.n_samples, axis=0))) ** 2
        _grad -= 0.5 * ((X - np.repeat(self.mu[:, 1].reshape(1, -1), self.n_samples, axis=0)) /
                        (np.repeat(self.std[:, 1].reshape(1, -1), self.n_samples, axis=0))) ** 2
        _grad *= np.transpose(np.repeat(np.array(_lambda).reshape(1, -1), self.n_features, axis=0))
        _grad = np.sum(_grad, axis=0)

        return _grad

    def predict(self, X: Union[np.ndarray, pd.DataFrame]):
        p_hat = self.predict_log_proba(X)
        y_hat = self.classes_[np.argmax(p_hat, axis=1)]
        return y_hat

    def predict_log_proba(self, X: Union[np.ndarray, pd.DataFrame]):
        # if not self._fit_status:
        #     raise Exception('Model is not fitted.')

        if not X.shape[1] == self.n_features:
            raise ValueError(
                "Expected input with %d features, got %d instead." % (self.n_features, X.shape[1])
            )

        X = self.__prepare_X_y(X=X)

        log_priors = np.tile(np.log(self.__priors), (self.n_samples, 1))
        w_reshaped = np.tile(self.weights_.reshape(-1, 1), (1, self.n_classes))
        term1 = np.sum(np.multiply(w_reshaped, -np.log(np.sqrt(2 * np.pi) * self.std)))
        var_inv = np.multiply(w_reshaped, 1/np.multiply(self.std, self.std))
        mu_by_var = np.multiply(self.mu, var_inv)
        term2 = -0.5*(np.matmul(np.multiply(X, X), var_inv) - 2*np.matmul(X, mu_by_var)
                      + np.sum(self.mu.conj()*mu_by_var, axis=0))
        log_proba = log_priors + term1 + term2

        return log_proba

    def predict_proba(self, X):
        log_proba = self.predict_log_proba(X)
        proba = np.array([np.exp(row_log_proba) / (np.exp(row_log_proba)).sum() for row_log_proba in log_proba])

        # proba = np.zeros(log_proba.shape)
        # for i in range(1):
        #     s_class = np.arange(1, self.n_classes + 1)
        #     s_class = [c for c in s_class if c != i]
        #     proba[:, i] = 1 / (1 + np.sum(np.exp(log_proba[:, s_class] - log_proba[:, i]), axis=1))
        # proba[:, -1] = 1 - np.sum(proba, axis=1)

        return proba

    def score(self, X, y):
        y_hat = self.predict(X)
        y = self.__prepare_X_y(y=y)
        _score = (y == y_hat).sum() / len(y)
        return _score
