import numpy as np
import pytest
from scipy import stats
from sklearn.utils._testing import assert_array_almost_equal

from wnb.stats import BernoulliDist, CategoricalDist, GeometricDist, PoissonDist

out_of_support_warn_msg = "Value doesn't lie within the support of the distribution"


def test_bernoulli_pdf():
    """
    Test whether pmf method of `BernoulliDist` returns the same result as pmf method of `scipy.stats.bernoulli`.
    """
    bernoulli_wnb = BernoulliDist(p=0.4)
    bernoulli_scipy = stats.bernoulli(p=0.4)
    X = np.random.randint(0, 2, size=10000)
    assert_array_almost_equal(bernoulli_wnb(X), bernoulli_scipy.pmf(X), decimal=10)


def test_bernoulli_out_of_support_data():
    """
    Test whether a warning is issued when calling `BernoulliDist` with out-of-support data.
    """
    X = np.random.randint(0, 2, size=1000)
    bernoulli_wnb = BernoulliDist.from_data(X)

    with pytest.warns(RuntimeWarning, match=out_of_support_warn_msg):
        bernoulli_wnb(2)


def test_categorical_pdf():
    """
    Test whether pmf method of `CategoricalDist` returns the correct expected values.
    """
    cat_wnb = CategoricalDist(prob={-1: 0.1, 0: 0.3, 1: 0.6})
    assert cat_wnb(-1) == 0.1
    assert cat_wnb(0) == 0.3
    assert cat_wnb(1) == 0.6
    assert cat_wnb(2) == 0.0
    assert cat_wnb(3) == 0.0
    assert cat_wnb(-2) == 0.0


def test_categorical_from_data():
    """
    Test whether `CategoricalDist.from_data` method correctly estimates the probabilities.
    """
    X = [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    cat_wnb = CategoricalDist.from_data(X)
    assert set(cat_wnb.prob.keys()) == {1, 0}
    assert_array_almost_equal(sorted(np.array(list(cat_wnb.prob.values()))), np.array([0.4, 0.6]), decimal=10)
    assert cat_wnb.support in ([1, 0], [0, 1])


def test_categorical_out_of_support_data():
    """
    Test whether a warning is issued when calling `CategoricalDist` with out-of-support data.
    """
    X = [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    cat_wnb = CategoricalDist.from_data(X)
    assert cat_wnb.support in ([1, 0], [0, 1])
    with pytest.warns(RuntimeWarning, match=out_of_support_warn_msg):
        cat_wnb(2)


def test_categorical_non_numeric_values():
    X = [1, 1, "0", 1, "0", 1, "0", 1, "0", 1]
    cat_wnb = CategoricalDist.from_data(X)
    assert set(cat_wnb.prob.keys()) == {1, "0"}
    assert_array_almost_equal(sorted(np.array(list(cat_wnb.prob.values()))), np.array([0.4, 0.6]), decimal=10)
    assert cat_wnb.support in ([1, "0"], ["0", 1])
    assert_array_almost_equal(cat_wnb("0"), 0.4, decimal=10)
    assert cat_wnb(np.nan) == 0
    assert cat_wnb(None) == 0
    assert cat_wnb(np.inf) == 0
    assert cat_wnb("2") == 0


def test_geometric_pdf():
    """
    Test whether pmf method of `GeometricDist` returns the same result as pmf method of `scipy.stats.geom`.
    """
    geom_wnb = GeometricDist(p=0.3)
    geom_scipy = stats.geom(p=0.3)
    X = np.random.randint(1, 1000, size=10000)
    assert_array_almost_equal(geom_wnb(X), geom_scipy.pmf(X), decimal=10)


def test_geometric_out_of_support_data():
    """
    Test whether a warning is issued when calling `GeometricDist` with out-of-support data.
    """
    X = np.random.geometric(p=0.3, size=1000)
    geom_wnb = GeometricDist.from_data(X)

    with pytest.warns(RuntimeWarning, match=out_of_support_warn_msg):
        geom_wnb(0)


def test_poisson_pdf():
    """
    Test whether pmf method of `PoissonDist` returns the same result as pmf method of `scipy.stats.poisson`.
    """
    poisson_wnb = PoissonDist(rate=1.5)
    poisson_scipy = stats.poisson(mu=1.5)
    X = np.random.randint(0, 100, size=10000)
    assert_array_almost_equal(poisson_wnb(X), poisson_scipy.pmf(X), decimal=10)


def test_poisson_out_of_support_data():
    """
    Test whether a warning is issued when calling `PoissonDist` with out-of-support data.
    """
    X = np.random.poisson(lam=1.5, size=1000)
    poisson_wnb = PoissonDist.from_data(X)

    with pytest.warns(RuntimeWarning, match=out_of_support_warn_msg):
        poisson_wnb(-1)
