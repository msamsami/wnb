import numpy as np

import pytest
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import is_classifier
from sklearn.utils._testing import assert_array_equal, assert_array_almost_equal

from wnb import GaussianWNB

# Data is just 6 separable points in the plane
X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
y = np.array([1, 1, 1, 2, 2, 2])


@pytest.fixture
def global_random_seed():
    return np.random.randint(0, 1000)


def get_random_normal_x_binary_y(global_random_seed):
    # A bit more random tests
    rng = np.random.RandomState(global_random_seed)
    X1 = rng.normal(size=(10, 3))
    y1 = (rng.normal(size=10) > 0).astype(int)
    return X1, y1


def test_gnb():
    """Binary Gaussian MLD-WNB classification

    Checks that GaussianWNB implements fit and predict and returns correct values for a simple toy dataset.
    """
    clf = GaussianWNB()
    y_pred = clf.fit(X, y).predict(X)
    assert_array_equal(y_pred, y)

    y_pred_proba = clf.predict_proba(X)
    y_pred_log_proba = clf.predict_log_proba(X)
    assert_array_almost_equal(np.log(y_pred_proba), y_pred_log_proba, 8)


def test_gnb_estimator():
    """
    Test whether GaussianWNB estimator adheres to scikit-learn conventions.
    """
    check_estimator(GaussianWNB())
    assert is_classifier(GaussianWNB)
