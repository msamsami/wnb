from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from sklearn.base import is_classifier
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB
from sklearn.utils._testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.estimator_checks import check_estimator

from wnb import Distribution as D
from wnb import GeneralNB

# Data is just 6 separable points in the plane
X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
y = np.array([1, 1, 1, 2, 2, 2])


def get_random_normal_x_binary_y(global_random_seed: int) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
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


def test_gnb_vs_sklearn_gaussian():
    """General Naive Bayes classification vs sklearn Gaussian Naive Bayes classification.

    Test GeneralNB with gaussian likelihoods returns the same outputs as the sklearn MultinomialNB.
    """
    clf1 = GeneralNB()
    clf1.fit(X, y)

    clf2 = GaussianNB()
    clf2.fit(X, y)

    y_pred1 = clf1.predict(X)
    y_pred2 = clf2.predict(X)
    assert_array_equal(y_pred1, y_pred2)

    y_pred_proba1 = clf1.predict_proba(X)
    y_pred_proba2 = clf2.predict_proba(X)
    assert_array_almost_equal(y_pred_proba1, y_pred_proba2, 6)

    y_pred_log_proba1 = clf1.predict_log_proba(X)
    y_pred_log_proba2 = clf2.predict_log_proba(X)
    assert_array_almost_equal(y_pred_log_proba1, y_pred_log_proba2, 5)


def test_gnb_vs_sklearn_bernoulli():
    """General Naive Bayes classification vs sklearn Bernoulli Naive Bayes classification.

    Test GeneralNB with bernoulli likelihoods returns the same outputs as the sklearn BernoulliNB.
    """
    rng = np.random.RandomState(1)
    X_ = rng.randint(2, size=(150, 100))
    y_ = rng.randint(1, 5, size=(150,))

    clf1 = GeneralNB(distributions=[D.BERNOULLI] * 100)
    clf1.fit(X_, y_)

    clf2 = BernoulliNB(alpha=1e-10, force_alpha=True)
    clf2.fit(X_, y_)

    y_pred1 = clf1.predict(X_[2:3])
    y_pred2 = clf2.predict(X_[2:3])
    assert_array_equal(y_pred1, y_pred2)

    y_pred_proba1 = clf1.predict_proba(X_[2:3])
    y_pred_proba2 = clf2.predict_proba(X_[2:3])
    assert_array_almost_equal(y_pred_proba1, y_pred_proba2, 6)

    y_pred_log_proba1 = clf1.predict_log_proba(X_[2:3])
    y_pred_log_proba2 = clf2.predict_log_proba(X_[2:3])
    assert_array_almost_equal(y_pred_log_proba1, y_pred_log_proba2, 5)


def test_gnb_vs_sklearn_categorical():
    """General Naive Bayes classification vs sklearn Categorical Naive Bayes classification.

    Test GeneralNB with categorical likelihoods returns the same outputs as the sklearn CategoricalNB.
    """
    categorical_values = [
        ["cat", "dog"],
        ["morning", "noon", "afternoon", "evening"],
        ["apple", "orange", "watermelon"],
        ["black", "white"],
    ]
    rng = np.random.RandomState(24)
    X_str_ = np.empty((150, 4)).astype("str")
    X_ = np.zeros((150, 4))
    for i, options in enumerate(categorical_values):
        rnd_values = rng.randint(len(options), size=(150,))
        X_str_[:, i] = np.array(options)[rnd_values]
        X_[:, i] = rnd_values
    y_ = rng.randint(1, 4, size=(150,))

    clf1 = GeneralNB(distributions=[D.CATEGORICAL for _ in range(len(categorical_values))])
    clf1.fit(X_str_, y_)

    clf2 = CategoricalNB(alpha=1e-10, force_alpha=True)
    clf2.fit(X_, y_)

    y_pred1 = clf1.predict(X_str_[2:3])
    y_pred2 = clf2.predict(X_[2:3])
    assert_array_equal(y_pred1, y_pred2)

    y_pred_proba1 = clf1.predict_proba(X_str_[2:3])
    y_pred_proba2 = clf2.predict_proba(X_[2:3])
    assert_array_almost_equal(y_pred_proba1, y_pred_proba2, 6)

    y_pred_log_proba1 = clf1.predict_log_proba(X_str_[2:3])
    y_pred_log_proba2 = clf2.predict_log_proba(X_[2:3])
    assert_array_almost_equal(y_pred_log_proba1, y_pred_log_proba2, 5)


def test_gnb_estimator():
    """
    Test whether GeneralNB estimator adheres to scikit-learn conventions.
    """
    check_estimator(GeneralNB())
    assert is_classifier(GeneralNB)


def test_gnb_dist_none():
    """
    Test whether the default distribution is set to Gaussian (normal) when no distributions are specified.
    """
    clf = GeneralNB().fit(X, y)
    assert clf.distributions_ == [D.NORMAL] * X.shape[1]


def test_gnb_dist_default_normal():
    """
    Test whether the default distribution is set to Gaussian (normal) when no distributions are specified.
    """
    clf = GeneralNB(distributions=[(D.RAYLEIGH, [0])]).fit(X, y)
    assert clf.distributions_ == [D.RAYLEIGH, D.NORMAL]


def test_gnb_wrong_dist_length():
    """
    Test whether an error is raised if the number of distributions is different from the number of features.
    """
    clf = GeneralNB(distributions=[D.NORMAL] * (X.shape[1] + 1))
    msg = "Number of specified distributions must match the number of features"
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)


@pytest.mark.parametrize(
    "distributions",
    [
        "Normal",
        {"Normal": 0},
        [(D.NORMAL, [0]), D.BERNOULLI],
        ["Normal", (D.BERNOULLI, [1])],
    ],
)
def test_gnb_wrong_dist_value(distributions):
    """
    Test whether an error is raised if invalid value is provided for the distributions parameter.
    """
    clf = GeneralNB(distributions=distributions)
    msg_1 = "distributions parameter must be a sequence of distributions or a sequence of tuples"
    msg_2 = "The 'distributions' parameter of GeneralNB must be an array-like or None"
    with pytest.raises(ValueError, match=rf"{msg_1}|{msg_2}"):
        clf.fit(X, y)


@pytest.mark.parametrize("distributions", [[("Normal", "x1")], [(D.RAYLEIGH, [1, "f1"])]])
def test_gnb_feature_name_dist_with_array(distributions):
    clf = GeneralNB(distributions=distributions)
    with pytest.raises(
        ValueError,
        match="Feature names are only supported when input data X is a DataFrame with named columns",
    ):
        clf.fit(X, y)


class InvalidDistA:
    def __call__(self, *args, **kwargs):
        return 0.0


class InvalidDistB(InvalidDistA):
    @classmethod
    def from_data(cls, *args, **kwargs):
        return cls()


@pytest.mark.parametrize(
    "distributions",
    [
        ["Normal", "Borel"],
        [(D.NORMAL, 0), ("Weibull", [1])],
        [D.NORMAL, InvalidDistA],
        [(InvalidDistA, 0), (InvalidDistB, 1)],
    ],
)
def test_gnb_unsupported_dist(distributions):
    """
    Test whether an error is raised if an unsupported distribution is provided.
    """
    clf = GeneralNB(distributions=distributions)
    msg = r"Distribution .* is not supported"
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)


@pytest.mark.parametrize("dist", [D.BERNOULLI, D.POISSON, D.CATEGORICAL])
def test_gnb_dist_identical(dist: D):
    """
    Test whether GeneralNB returns the same outputs when identical distributions are specified in different formats.
    """

    rng = np.random.RandomState(10)
    X_ = rng.randint(2, size=(150, 100))
    y_ = rng.randint(1, 5, size=(150,))

    clf1 = GeneralNB(distributions=[dist] * 100)
    clf1.fit(X_, y_)

    clf2 = GeneralNB(distributions=[(dist, [i for i in range(100)])])
    clf2.fit(X_, y_)

    df_X = pd.DataFrame(X_, columns=[f"x{i}" for i in range(100)])
    clf3 = GeneralNB(distributions=[(dist, [f"x{i}" for i in range(100)])])
    clf3.fit(df_X, y_)

    y_pred1 = clf1.predict(X_[10:15])
    y_pred2 = clf2.predict(X_[10:15])
    y_pred3 = clf3.predict(df_X.iloc[10:15])
    assert np.array_equal(y_pred1, y_pred2)
    assert np.array_equal(y_pred2, y_pred3)

    y_pred_proba1 = clf1.predict_proba(X_[10:15])
    y_pred_proba2 = clf2.predict_proba(X_[10:15])
    y_pred_proba3 = clf3.predict_proba(df_X.iloc[10:15])
    assert np.array_equal(y_pred_proba1, y_pred_proba2)
    assert np.array_equal(y_pred_proba2, y_pred_proba3)

    y_pred_log_proba1 = clf1.predict_log_proba(X_[10:15])
    y_pred_log_proba2 = clf2.predict_log_proba(X_[10:15])
    y_pred_log_proba3 = clf3.predict_log_proba(df_X.iloc[10:15])
    assert np.array_equal(y_pred_log_proba1, y_pred_log_proba2)
    assert np.array_equal(y_pred_log_proba2, y_pred_log_proba3)


def test_gnb_prior(global_random_seed: int):
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
    Test whether the class priors override is properly used.
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
    Test whether the class priors sum is properly tested.
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
    priors = np.array([0.08, 0.14, 0.03, 0.16, 0.11, 0.16, 0.07, 0.14, 0.11, 0.0])
    y_ = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    clf = GeneralNB(priors=priors)
    clf.fit(X_, y_)


def test_gnb_wrong_nb_priors():
    """
    Test whether an error is raised if the number of priors is different from the number of classes.
    """
    clf = GeneralNB(priors=np.array([0.25, 0.25, 0.25, 0.25]))

    msg = "Number of priors must match the number of classes"
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)


def test_gnb_prior_greater_one():
    """
    Test if an error is raised if the sum of priors greater than one.
    """
    clf = GeneralNB(priors=np.array([2.0, 1.0]))

    msg = "The sum of the priors should be 1"
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)


def test_gnb_prior_large_bias():
    """
    Test if good prediction when class priors favor largely one class.
    """
    clf = GeneralNB(priors=np.array([0.01, 0.99]))
    clf.fit(X, y)
    assert clf.predict(np.array([[-0.1, -0.1]])) == np.array([2])


def test_gnb_wrong_nb_dist():
    """
    Test whether an error is raised if the number of distributions is different from the number of features.
    """
    clf = GeneralNB(distributions=[D.NORMAL, D.GAMMA, D.PARETO])

    msg = "Number of specified distributions must match the number of features"
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)


def test_gnb_var_smoothing():
    """
    Test whether var_smoothing parameter properly affects the variances of normal distributions.
    """
    X = np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]])  # First feature has variance 2.0
    y = np.array([1, 1, 2, 2, 2])

    clf1 = GeneralNB(var_smoothing=0.0)
    clf1.fit(X, y)

    clf2 = GeneralNB(var_smoothing=1.0)
    clf2.fit(X, y)

    test_point = np.array([[2.5, 0]])
    prob1 = clf1.predict_proba(test_point)
    prob2 = clf2.predict_proba(test_point)

    assert not np.allclose(prob1, prob2)
    assert clf1.epsilon_ == 0.0
    assert clf2.epsilon_ > clf1.epsilon_


def test_gnb_var_smoothing_non_numeric():
    """
    Test that var_smoothing is ignored for non-numeric features.
    """
    X = np.array([["a", 1], ["b", 2], ["a", 2], ["b", 1]])
    y = np.array([1, 1, 2, 2])

    clf = GeneralNB(distributions=[D.CATEGORICAL, D.CATEGORICAL], var_smoothing=1e-6)
    clf.fit(X, y)
    assert clf.epsilon_ == 0


def test_gnb_neg_var_smoothing():
    """
    Test whether an error is raised in case of negative var_smoothing.
    """
    clf = GeneralNB(var_smoothing=-1.0)

    msg_1 = "Variance smoothing parameter must be a non-negative real number"
    msg_2 = "'var_smoothing' parameter of GeneralNB must be a float in the range \[0.0, inf\)"
    with pytest.raises(ValueError, match=rf"{msg_1}|{msg_2}"):
        clf.fit(X, y)


def test_gnb_neg_alpha():
    """
    Test whether an error is raised in case of negative alpha.
    """
    clf = GeneralNB(alpha=-1.0)

    msg_1 = "Alpha must be a non-negative real number"
    msg_2 = "'alpha' parameter of GeneralNB must be a float in the range \[0.0, inf\)"
    with pytest.raises(ValueError, match=rf"{msg_1}|{msg_2}"):
        clf.fit(X, y)


def test_gnb_attrs():
    """
    Test whether the attributes are properly set.
    """
    clf = GeneralNB().fit(X, y)
    assert np.array_equal(clf.class_count_, np.array([3, 3]))
    assert np.array_equal(clf.class_prior_, np.array([0.5, 0.5]))
    assert np.array_equal(clf.classes_, np.array([1, 2]))
    assert clf.n_classes_ == 2
    assert clf.epsilon_ > 0
    assert clf.n_features_in_ == 2
    assert not hasattr(clf, "feature_names_in_")
    assert clf.distributions_ == [D.NORMAL, D.NORMAL]
    assert len(clf.likelihood_params_) == 2

    feature_names = [f"x{i}" for i in range(X.shape[1])]
    clf = GeneralNB().fit(pd.DataFrame(X, columns=feature_names), y)
    assert np.array_equal(clf.feature_names_in_, np.array(feature_names))


def test_gnb_check_X_wrong_features():
    """
    Test whether an error is raised in case of providing wrong number of features to the predict method.
    """
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    clf = GeneralNB()
    clf.fit(X, y)

    # Mock validate_data to return X but skip sklearn's feature check, so we can test our custom feature count check
    X_wrong = np.array([[1, 2, 3]])  # 3 features instead of 2
    msg = "Expected input with 2 features, got 3 instead"
    with patch("wnb.gnb.validate_data", return_value=X_wrong):
        with pytest.raises(ValueError, match=msg):
            clf.predict(X_wrong)
