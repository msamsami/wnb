import numpy as np
import pytest
from scipy import stats
from sklearn.utils._testing import assert_array_almost_equal

from wnb import Distribution as D
from wnb.stats import (
    AllDistributions,
    BernoulliDist,
    BetaDist,
    ChiSquaredDist,
    ExponentialDist,
    GammaDist,
    GeometricDist,
    LognormalDist,
    NormalDist,
    ParetoDist,
    PoissonDist,
    RayleighDist,
    TDist,
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


def test_lognormal_pdf(random_uniform):
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


def test_exponential_pdf(random_uniform):
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


def test_uniform_pdf(random_uniform):
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


def test_pareto_pdf(random_uniform):
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


def test_gamma_pdf(random_uniform):
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


def test_chi2_pdf(random_uniform):
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


def test_t_pdf(random_uniform):
    """
    Test whether pdf method of `TDist` returns the same result as pdf method of `scipy.stats.t`.
    """
    t_wnb = TDist(df=10)
    t_scipy = stats.t(df=10)
    X = random_uniform
    assert_array_almost_equal(t_wnb(X), t_scipy.pdf(X), decimal=10)


def test_rayleigh_pdf(random_uniform):
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
