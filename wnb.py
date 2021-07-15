import numpy as np

class WeightedNB:

    def __init__(self, priors=None, L=None, max_iter=100, penalty='l2', C=1.0):
        self.priors = priors
        self.L = L
        self.max_iter = max_iter
        self.penalty = penalty
        self.C = C

    def fit(self, X, y):
        # Code

    def predict(self, X):
        # Code

    def predict_proba(self, X):
        # Code

    def score(self, X, y):
        # Code

    def _gradient_descent(self, f, f_derv, max_iter=100, step_size=1e-4):
        # Code