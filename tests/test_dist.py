import numpy as np

import pytest
from scipy.stats import norm, lognorm
from sklearn.utils._testing import assert_array_equal, assert_array_almost_equal

from wnb import Distribution as D
from wnb.dist import (
    AllDistributions,
    NormalDist,
    LognormalDist,
    ExponentialDist,
    UniformDist,
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


def test_normal_dist_pdf():
    """
    Test whether pdf method of NormalDist returns the same result as pdf method of scipy.stats.norm.
    """
    norm_wnb = NormalDist(mu=1, sigma=3)
    norm_scipy = norm(loc=1, scale=3)
    X = np.random.uniform(-50, 50, size=10000)
    assert_array_almost_equal(norm_wnb(X), norm_scipy.pdf(X), decimal=15)


def test_lognormal_dist_pdf():
    """
    Test whether pdf method of LognormalDist returns the same result as pdf method of scipy.stats.lognorm.
    """
    lognorm_wnb = LognormalDist(mu=1, sigma=3)
    lognorm_scipy = lognorm(scale=np.exp(1), s=3)
    X = np.random.uniform(0, 50, size=10000)
    assert_array_almost_equal(lognorm_wnb(X), lognorm_scipy.pdf(X), decimal=15)


def test_lognormal_dist_out_of_support_data():
    """
    Test whether a warning is issued when calling LognormalDist with out-of-support data.
    """
    X = np.random.lognormal(mean=1.5, sigma=3.75, size=1000)
    lognormal = LognormalDist.from_data(X)

    with pytest.warns(RuntimeWarning, match=out_of_support_warn_msg):
        lognormal(-1)


def test_exponential_dist_out_of_support_data():
    """
    Test whether a warning is issued when calling ExponentialDist with out-of-support data.
    """
    X = np.random.exponential(scale=2.5, size=1000)
    exp = ExponentialDist.from_data(X)

    with pytest.warns(RuntimeWarning, match=out_of_support_warn_msg):
        exp(-1)


def test_exponential_dist_out_of_support_data():
    """
    Test whether a warning is issued when calling UniformDist with out-of-support data.
    """
    X = np.random.uniform(low=5, high=10, size=10000)
    uniform = UniformDist.from_data(X)

    with pytest.warns(RuntimeWarning, match=out_of_support_warn_msg):
        uniform(3)
