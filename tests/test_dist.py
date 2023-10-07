import numpy as np

import pytest
from scipy.stats import norm, lognorm, expon, uniform, pareto
from sklearn.utils._testing import assert_array_almost_equal

from wnb import Distribution as D
from wnb.dist import (
    AllDistributions,
    NormalDist,
    LognormalDist,
    ExponentialDist,
    UniformDist,
    ParetoDist,
)

out_of_support_warn_msg = "Value doesn't lie within the support of the distribution"


def test_distributions_correct_name_attr():
    """
    Test if all defined distributions have correct `name` attributes.
    """
    for dist_name in AllDistributions.keys():
        assert isinstance(dist_name, (str, D))


def test_distributions_correct_support_attr():
    """
    Test if all defined distributions have correct `_support` attributes.
    """
    for dist in AllDistributions.values():
        if dist.name in [D.UNIFORM, D.PARETO, D.CATEGORICAL]:
            assert dist._support is None
            continue

        assert isinstance(dist._support, (list, tuple))

        if isinstance(dist._support, list):
            for x in dist._support:
                assert isinstance(x, (float, int))
        else:
            assert len(dist._support) == 2
            for x in dist._support:
                assert isinstance(x, (float, int))


def test_normal_pdf():
    """
    Test whether pdf method of NormalDist returns the same result as pdf method of scipy.stats.norm.
    """
    norm_wnb = NormalDist(mu=1, sigma=3)
    norm_scipy = norm(loc=1, scale=3)
    X = np.random.uniform(-50, 50, size=10000)
    assert_array_almost_equal(norm_wnb(X), norm_scipy.pdf(X), decimal=10)


def test_lognormal_pdf():
    """
    Test whether pdf method of LognormalDist returns the same result as pdf method of scipy.stats.lognorm.
    """
    lognorm_wnb = LognormalDist(mu=1, sigma=3)
    lognorm_scipy = lognorm(scale=np.exp(1), s=3)
    X = np.random.uniform(0, 50, size=10000)
    assert_array_almost_equal(lognorm_wnb(X), lognorm_scipy.pdf(X), decimal=10)


def test_lognormal_out_of_support_data():
    """
    Test whether a warning is issued when calling LognormalDist with out-of-support data.
    """
    X = np.random.lognormal(mean=1.5, sigma=3.75, size=1000)
    lognorm_wnb = LognormalDist.from_data(X)

    with pytest.warns(RuntimeWarning, match=out_of_support_warn_msg):
        lognorm_wnb(-1)


def test_exponential_pdf():
    """
    Test whether pdf method of ExponentialDist returns the same result as pdf method of scipy.stats.expon.
    """
    expon_wnb = ExponentialDist(rate=4)
    expon_scipy = expon(scale=1 / 4)
    X = np.random.uniform(0, 100, size=10000)
    assert_array_almost_equal(expon_wnb(X), expon_scipy.pdf(X), decimal=10)


def test_exponential_out_of_support_data():
    """
    Test whether a warning is issued when calling ExponentialDist with out-of-support data.
    """
    X = np.random.exponential(scale=2.5, size=1000)
    expon_wnb = ExponentialDist.from_data(X)

    with pytest.warns(RuntimeWarning, match=out_of_support_warn_msg):
        expon_wnb(-1)


def test_uniform_pdf():
    """
    Test whether pdf method of UniformDist returns the same result as pdf method of scipy.stats.uniform.
    """
    uniform_wnb = UniformDist(a=-5, b=10)
    uniform_scipy = uniform(loc=-5, scale=15)
    X = np.random.uniform(0, 100, size=10000)
    assert_array_almost_equal(uniform_wnb(X), uniform_scipy.pdf(X), decimal=10)


def test_uniform_out_of_support_data():
    """
    Test whether a warning is issued when calling UniformDist with out-of-support data.
    """
    X = np.random.uniform(low=5, high=10, size=1000)
    uniform_wnb = UniformDist.from_data(X)

    with pytest.warns(RuntimeWarning, match=out_of_support_warn_msg):
        uniform_wnb(3)


def test_pareto_pdf():
    """
    Test whether pdf method of ParetoDist returns the same result as pdf method of scipy.stats.pareto.
    """
    pareto_wnb = ParetoDist(x_m=5, alpha=0.5)
    pareto_scipy = pareto(b=0.5, scale=5)
    X = np.random.uniform(0, 100, size=10000)
    assert_array_almost_equal(pareto_wnb(X), pareto_scipy.pdf(X), decimal=10)


def test_pareto_out_of_support_data():
    """
    Test whether a warning is issued when calling ParetoDist with out-of-support data.
    """
    X = np.random.pareto(a=0.5, size=1000)
    pareto_wnb = ParetoDist.from_data(X)

    with pytest.warns(RuntimeWarning, match=out_of_support_warn_msg):
        pareto_wnb(-5)
