import numpy as np
import pytest
from sklearn.base import is_classifier
from sklearn.utils._testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.estimator_checks import check_estimator

from wnb import GaussianWNB

# Data is just 6 separable points in the plane
X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
y = np.array([1, 1, 1, 2, 2, 2])


def get_random_normal_x_binary_y(global_random_seed):
    # A bit more random tests
    rng = np.random.RandomState(global_random_seed)
    X1 = rng.normal(size=(10, 3))
    y1 = (rng.normal(size=10) > 0).astype(int)
    return X1, y1


def test_gwnb():
    """Binary Gaussian MLD-WNB classification

    Checks that GaussianWNB implements fit and predict and returns correct values for a simple toy dataset.
    """
    clf = GaussianWNB()
    y_pred = clf.fit(X, y).predict(X)
    assert_array_equal(y_pred, y)

    y_pred_proba = clf.predict_proba(X)
    y_pred_log_proba = clf.predict_log_proba(X)
    assert_array_almost_equal(np.log(y_pred_proba), y_pred_log_proba, 8)


def test_gwnb_estimator():
    """
    Test whether GaussianWNB estimator adheres to scikit-learn conventions.
    """
    check_estimator(GaussianWNB())
    assert is_classifier(GaussianWNB)


def test_gwnb_prior(global_random_seed):
    """
    Test whether class priors are properly set.
    """
    clf = GaussianWNB().fit(X, y)
    assert_array_almost_equal(np.array([3, 3]) / 6.0, clf.class_prior_, 8)

    X1, y1 = get_random_normal_x_binary_y(global_random_seed)
    clf = GaussianWNB().fit(X1, y1)

    # Check that the class priors sum to 1
    assert_array_almost_equal(clf.class_prior_.sum(), 1)


def test_gwnb_neg_priors():
    """
    Test whether an error is raised in case of negative priors.
    """
    clf = GaussianWNB(priors=np.array([-1.0, 2.0]))

    msg = "Priors must be non-negative"
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)


def test_gwnb_priors():
    """
    Test whether the class priors override is properly used.
    """
    clf = GaussianWNB(priors=np.array([0.3, 0.7])).fit(X, y)
    assert_array_almost_equal(
        clf.predict_proba([[-0.1, -0.1]]),
        np.array([[0.82357095, 0.17642905]]),
        8,
    )
    assert_array_almost_equal(clf.class_prior_, np.array([0.3, 0.7]))


def test_gwnb_wrong_nb_priors():
    """
    Test whether an error is raised if the number of priors is different from the number of classes.
    """
    clf = GaussianWNB(priors=np.array([0.25, 0.25, 0.25, 0.25]))

    msg = "Number of priors must match the number of classes"
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)


def test_gwnb_prior_greater_one():
    """
    Test if an error is raised if the sum of priors greater than one.
    """
    clf = GaussianWNB(priors=np.array([2.0, 1.0]))

    msg = "The sum of the priors should be 1"
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)


def test_gwnb_prior_large_bias():
    """
    Test if good prediction when class priors favor largely one class.
    """
    clf = GaussianWNB(priors=np.array([0.01, 0.99]))
    clf.fit(X, y)
    assert clf.predict(np.array([[-0.1, -0.1]])) == np.array([2])


def test_gwnb_non_binary():
    """
    Test if an error is raised when given non-binary targets.
    """
    X_ = np.array(
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
    y_ = np.array([1, 2, 3, 4, 4, 3, 2, 1, 1, 2])
    clf = GaussianWNB()

    msg = "Unknown label type: non-binary"
    with pytest.raises(ValueError, match=msg):
        clf.fit(X_, y_)


def test_gwnb_wrong_error_weights():
    """
    Test whether an error is raised in case error weight is not a square array of size (n_classes, n_classes).
    """
    clf = GaussianWNB(error_weights=np.array([[1, 2, 0], [0, -2, -3]]))

    msg = "The shape of error weights matrix does not match the number of classes"
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)

    clf = GaussianWNB(error_weights=np.array([[1, 2, 0], [0, -2, -3], [-2, 1, 1.5]]))
    msg = "The shape of error weights matrix does not match the number of classes"
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)


def test_gwnb_wrong_penalty():
    """
    Test whether an error is raised in case regularization penalty is not supported.
    """
    clf = GaussianWNB(penalty="dropout")

    msg = "Regularization type must be either 'l1' or 'l2'"
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)


def test_gwnb_neg_C():
    """
    Test whether an error is raised in case of negative regularization parameter, C.
    """
    clf = GaussianWNB(C=-0.6)

    msg = "Regularization parameter must be positive"
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)


def test_gwnb_neg_max_iter():
    """
    Test whether an error is raised in case number of iteration is negative.
    """
    clf = GaussianWNB(max_iter=-10)

    msg = "Maximum number of iteration must be a positive integer"
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)


def test_gwnb_no_cost_hist():
    """
    Test whether cost_hist_ is None if learning_hist is not enabled.
    """
    clf = GaussianWNB(max_iter=10)
    clf.fit(X, y)
    assert clf.cost_hist_ is None
