import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats
from sklearn.utils._testing import assert_array_almost_equal

from wnb.stats import (
    BetaDist,
    ChiSquaredDist,
    ExponentialDist,
    GammaDist,
    LaplaceDist,
    LognormalDist,
    NormalDist,
    ParetoDist,
    RayleighDist,
    TDist,
    UniformDist,
)

out_of_support_warn_msg = "Value doesn't lie within the support of the distribution"


def test_normal_pdf():
    """
    Test whether pdf method of `NormalDist` returns the same result as pdf method of `scipy.stats.norm`.
    """
    norm_wnb = NormalDist(mu=1, sigma=3)
    norm_scipy = stats.norm(loc=1, scale=3)
    X = np.random.uniform(-100, 100, size=10000)
    assert_array_almost_equal(norm_wnb(X), norm_scipy.pdf(X), decimal=10)


@pytest.mark.parametrize("epsilon", [1e-10, 1e-9, 1e-6, 1e-3])
def test_normal_with_epsilon(epsilon: float):
    """
    Test whether epsilon is correctly applied for `NormalDist`.
    """
    norm_1 = NormalDist(mu=1, sigma=0)
    norm_2 = NormalDist(mu=1, sigma=0, epsilon=epsilon)
    norm_3 = NormalDist(mu=1, sigma=np.sqrt(epsilon))
    assert norm_1.sigma == norm_2.sigma == 0
    assert norm_3.sigma == np.sqrt(epsilon)
    X = np.random.uniform(-100, 100, size=10000)
    assert np.isnan(norm_1(X)).all()
    assert_array_almost_equal(norm_2(X), norm_3(X), decimal=10)


def test_lognormal_pdf(random_uniform: NDArray[np.float64]):
    """
    Test whether pdf method of `LognormalDist` returns the same result as pdf method of `scipy.stats.lognorm`.
    """
    lognorm_wnb = LognormalDist(mu=1, sigma=3)
    lognorm_scipy = stats.lognorm(scale=np.exp(1), s=3)
    X = random_uniform
    assert_array_almost_equal(lognorm_wnb(X), lognorm_scipy.pdf(X), decimal=10)


def test_lognormal_out_of_support_data():
    """
    Test whether a warning is issued when calling `LognormalDist` with out-of-support data.
    """
    X = np.random.lognormal(mean=1.5, sigma=3.75, size=1000)
    lognorm_wnb = LognormalDist.from_data(X)

    with pytest.warns(RuntimeWarning, match=out_of_support_warn_msg):
        lognorm_wnb(-1)


def test_exponential_pdf(random_uniform: NDArray[np.float64]):
    """
    Test whether pdf method of `ExponentialDist` returns the same result as pdf method of `scipy.stats.expon`.
    """
    expon_wnb = ExponentialDist(rate=4)
    expon_scipy = stats.expon(scale=1 / 4)
    X = random_uniform
    assert_array_almost_equal(expon_wnb(X), expon_scipy.pdf(X), decimal=10)


def test_exponential_out_of_support_data():
    """
    Test whether a warning is issued when calling `ExponentialDist` with out-of-support data.
    """
    X = np.random.exponential(scale=2.5, size=1000)
    expon_wnb = ExponentialDist.from_data(X)

    with pytest.warns(RuntimeWarning, match=out_of_support_warn_msg):
        expon_wnb(-1)


def test_uniform_pdf(random_uniform: NDArray[np.float64]):
    """
    Test whether pdf method of `UniformDist` returns the same result as pdf method of `scipy.stats.uniform`.
    """
    uniform_wnb = UniformDist(a=-5, b=10)
    uniform_scipy = stats.uniform(loc=-5, scale=15)
    X = random_uniform
    assert_array_almost_equal(uniform_wnb(X), uniform_scipy.pdf(X), decimal=10)


def test_uniform_out_of_support_data():
    """
    Test whether a warning is issued when calling `UniformDist` with out-of-support data.
    """
    X = np.random.uniform(low=5, high=10, size=1000)
    uniform_wnb = UniformDist.from_data(X)

    with pytest.warns(RuntimeWarning, match=out_of_support_warn_msg):
        uniform_wnb(3)


def test_pareto_pdf(random_uniform: NDArray[np.float64]):
    """
    Test whether pdf method of `ParetoDist` returns the same result as pdf method of `scipy.stats.pareto`.
    """
    pareto_wnb = ParetoDist(x_m=5, alpha=0.5)
    pareto_scipy = stats.pareto(b=0.5, scale=5)
    X = random_uniform
    assert_array_almost_equal(pareto_wnb(X), pareto_scipy.pdf(X), decimal=10)


def test_pareto_out_of_support_data():
    """
    Test whether a warning is issued when calling `ParetoDist` with out-of-support data.
    """
    X = np.random.pareto(a=0.5, size=1000)
    pareto_wnb = ParetoDist.from_data(X)

    with pytest.warns(RuntimeWarning, match=out_of_support_warn_msg):
        pareto_wnb(-5)


def test_gamma_pdf(random_uniform: NDArray[np.float64]):
    """
    Test whether pdf method of `GammaDist` returns the same result as pdf method of `scipy.stats.gamma`.
    """
    gamma_wnb = GammaDist(k=1, theta=3)
    gamma_scipy = stats.gamma(a=1, scale=3)
    X = random_uniform
    assert_array_almost_equal(gamma_wnb(X), gamma_scipy.pdf(X), decimal=10)


def test_gamma_out_of_support_data():
    """
    Test whether a warning is issued when calling `GammaDist` with out-of-support data.
    """
    X = np.random.gamma(shape=1, scale=3, size=1000)
    gamma_wnb = GammaDist.from_data(X)

    with pytest.warns(RuntimeWarning, match=out_of_support_warn_msg):
        gamma_wnb(-5)


def test_beta_pdf():
    """
    Test whether pdf method of `BetaDist` returns the same result as pdf method of `scipy.stats.beta`.
    """
    beta_wnb = BetaDist(alpha=1, beta=5)
    beta_scipy = stats.beta(a=1, b=5)
    X = np.random.uniform(0.01, 0.99, size=10000)
    assert_array_almost_equal(beta_wnb(X), beta_scipy.pdf(X), decimal=10)


def test_beta_out_of_support_data():
    """
    Test whether a warning is issued when calling `BetaDist` with out-of-support data.
    """
    X = np.random.beta(a=1, b=5, size=1000)
    beta_wnb = BetaDist.from_data(X)

    with pytest.warns(RuntimeWarning, match=out_of_support_warn_msg):
        beta_wnb(1.01)


def test_chi2_pdf(random_uniform: NDArray[np.float64]):
    """
    Test whether pdf method of `ChiSquaredDist` returns the same result as pdf method of `scipy.stats.chi2`.
    """
    chi2_wnb = ChiSquaredDist(k=4)
    chi2_scipy = stats.chi2(df=4)
    X = random_uniform
    assert_array_almost_equal(chi2_wnb(X), chi2_scipy.pdf(X), decimal=10)


def test_chi2_out_of_support_data():
    """
    Test whether a warning is issued when calling `ChiSquaredDist` with out-of-support data.
    """
    X = np.random.chisquare(df=4, size=1000)
    chi2_wnb = ChiSquaredDist.from_data(X)

    with pytest.warns(RuntimeWarning, match=out_of_support_warn_msg):
        chi2_wnb(-5)


def test_t_pdf(random_uniform: NDArray[np.float64]):
    """
    Test whether pdf method of `TDist` returns the same result as pdf method of `scipy.stats.t`.
    """
    t_wnb = TDist(df=10)
    t_scipy = stats.t(df=10)
    X = random_uniform
    assert_array_almost_equal(t_wnb(X), t_scipy.pdf(X), decimal=10)


def test_t_from_data(random_uniform: NDArray[np.float64]):
    """
    Test whether `TDist.from_data` correctly estimates the degrees of freedom parameter.
    """
    t_wnb = TDist.from_data(random_uniform)
    assert_array_almost_equal(t_wnb.df, stats.t.fit(random_uniform)[0], decimal=8)


def test_rayleigh_pdf(random_uniform: NDArray[np.float64]):
    """
    Test whether pdf method of `RayleighDist` returns the same result as pdf method of `scipy.stats.rayleigh`.
    """
    rayleigh_wnb = RayleighDist(sigma=5)
    rayleigh_scipy = stats.rayleigh(scale=5)
    X = random_uniform
    assert_array_almost_equal(rayleigh_wnb(X), rayleigh_scipy.pdf(X), decimal=10)


def test_rayleigh_out_of_support_data():
    """
    Test whether a warning is issued when calling `RayleighDist` with out-of-support data.
    """
    X = np.random.rayleigh(scale=4, size=1000)
    rayleigh_wnb = RayleighDist.from_data(X)

    with pytest.warns(RuntimeWarning, match=out_of_support_warn_msg):
        rayleigh_wnb(-5)


def test_laplace_pdf(random_uniform: NDArray[np.float64]):
    """
    Test whether pdf method of `LaplaceDist` returns the same result as pdf method of `scipy.stats.laplace`.
    """
    laplace_wnb = LaplaceDist(mu=1, b=3)
    laplace_scipy = stats.laplace(loc=1, scale=3)
    X = random_uniform
    assert_array_almost_equal(laplace_wnb(X), laplace_scipy.pdf(X), decimal=10)


def test_laplace_from_data(random_uniform: NDArray[np.float64]):
    """
    Test whether `LaplaceDist.from_data` correctly estimates the scale parameter.
    """
    laplace_wnb = LaplaceDist.from_data(random_uniform)
    assert_array_almost_equal(laplace_wnb.b, stats.laplace.fit(random_uniform)[1], decimal=8)


def test_laplace_scale_non_positive():
    """
    Test whether a ValueError is raised when calling `LaplaceDist` with a non-positive scale.
    """
    with pytest.raises(ValueError):
        LaplaceDist(mu=1, b=-1)
