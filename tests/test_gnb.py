import numpy as np

import pytest
from sklearn.utils._testing import assert_almost_equal, assert_array_equal, assert_array_almost_equal, assert_allclose

from wnb import GeneralNB, Distribution as D


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
    """General Naive Bayes classification

    Checks that GeneralNB implements fit and predict and returns correct values for a simple toy dataset.
    """
    clf = GeneralNB()
    y_pred = clf.fit(X, y).predict(X)
    assert_array_equal(y_pred, y)

    y_pred_proba = clf.predict_proba(X)
    y_pred_log_proba = clf.predict_log_proba(X)
    assert_array_almost_equal(np.log(y_pred_proba), y_pred_log_proba, 8)


def test_gnb_prior(global_random_seed):
    """
    Test whether class priors are properly set.
    """
    clf = GeneralNB().fit(X, y)
    assert_array_almost_equal(np.array([3, 3]) / 6.0, clf.class_prior_, 8)

    X1, y1 = get_random_normal_x_binary_y(global_random_seed)
    clf = GeneralNB().fit(X1, y1)

    # Check that the class priors sum to 1
    assert_array_almost_equal(clf.class_prior_.sum(), 1)


def test_gnb_neg_priors():
    """
    Test whether an error is raised in case of negative priors.
    """
    clf = GeneralNB(priors=np.array([-1.0, 2.0]))

    msg = "Priors must be non-negative"
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)


def test_gnb_priors():
    """
    Test whether the class prior override is properly used.
    """
    clf = GeneralNB(priors=np.array([0.3, 0.7])).fit(X, y)
    assert_array_almost_equal(
        clf.predict_proba([[-0.1, -0.1]]),
        np.array([[0.825303662161683, 0.174696337838317]]),
        6,
    )
    assert_array_almost_equal(clf.class_prior_, np.array([0.3, 0.7]))


def test_gnb_priors_sum_isclose():
    """
    Test whether the class prior sum is properly tested.
    """
    X = np.array(
        [
            [-1, -1],
            [-2, -1],
            [-3, -2],
            [-4, -5],
            [-5, -4],
            [1, 1],
            [2, 1],
            [3, 2],
            [4, 4],
            [5, 5],
        ]
    )
    priors = np.array([0.08, 0.14, 0.03, 0.16, 0.11, 0.16, 0.07, 0.14, 0.11, 0.0])
    Y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    clf = GeneralNB(priors=priors)
    clf.fit(X, Y)


def test_gnb_wrong_nb_priors():
    """
    Test whether an error is raised if the number of prior is different from the number of class.
    """
    clf = GeneralNB(priors=np.array([0.25, 0.25, 0.25, 0.25]))

    msg = "Number of priors must match the number of classes"
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)


def test_gnb_prior_greater_one():
    """
    Test if an error is raised if the sum of prior greater than one.
    """
    clf = GeneralNB(priors=np.array([2.0, 1.0]))

    msg = "The sum of the priors should be 1"
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)


def test_gnb_prior_large_bias():
    """
    Test if good prediction when class prior favor largely one class.
    """
    clf = GeneralNB(priors=np.array([0.01, 0.99]))
    clf.fit(X, y)
    assert clf.predict(np.array([[-0.1, -0.1]])) == np.array([2])
