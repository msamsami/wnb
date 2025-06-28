from wnb.stats import AllDistributions, BernoulliDist, NormalDist
from wnb.stats._utils import get_dist_class, is_dist_supported
from wnb.stats.enums import Distribution


class MockDistribution:
    """
    Mock class that mimics a distribution but doesn't inherit from DistMixin.
    """

    @classmethod
    def from_data(cls, data, **kwargs):
        return cls()

    @property
    def support(self):
        return (0, 1)

    def __call__(self, x):
        return x


class IncompleteMockDistribution:
    """
    Mock class that doesn't have all required attributes.
    """

    def some_method(self):
        pass


def test_is_dist_supported_with_distmixin_subclass():
    """
    Test whether `is_dist_supported` returns True for valid DistMixin subclasses.
    """
    assert is_dist_supported(NormalDist) is True
    assert is_dist_supported(BernoulliDist) is True


def test_is_dist_supported_with_distribution_enum():
    """
    Test whether `is_dist_supported` returns True for Distribution enum values.
    """
    assert is_dist_supported(Distribution.NORMAL) is True
    assert is_dist_supported(Distribution.BERNOULLI) is True
    assert is_dist_supported(Distribution.CATEGORICAL) is True


def test_is_dist_supported_with_distribution_enum_members():
    """
    Test whether `is_dist_supported` returns True for Distribution enum member values.
    """
    for dist in Distribution.__members__.values():
        assert is_dist_supported(dist) is True


def test_is_dist_supported_with_mock_distribution():
    """
    Test whether `is_dist_supported` returns True for objects with required attributes.
    """
    mock_dist = MockDistribution()
    assert is_dist_supported(mock_dist) is True


def test_is_dist_supported_with_invalid_inputs():
    """
    Test whether `is_dist_supported` returns False for invalid inputs.
    """
    assert is_dist_supported("normal") is False
    assert is_dist_supported(42) is False
    assert is_dist_supported(None) is False
    assert is_dist_supported([]) is False
    assert is_dist_supported({}) is False


def test_is_dist_supported_with_incomplete_mock():
    """
    Test whether `is_dist_supported` returns False for objects missing required attributes.
    """
    incomplete_mock = IncompleteMockDistribution()
    assert is_dist_supported(incomplete_mock) is False


def test_get_dist_class_with_distmixin_subclass():
    """
    Test whether `get_dist_class` returns the same class for DistMixin subclasses.
    """
    assert get_dist_class(NormalDist) is NormalDist
    assert get_dist_class(BernoulliDist) is BernoulliDist


def test_get_dist_class_with_distribution_enum():
    """
    Test whether `get_dist_class` returns correct class for Distribution enum values.
    """
    assert get_dist_class(Distribution.NORMAL) is AllDistributions[Distribution.NORMAL]
    assert get_dist_class(Distribution.BERNOULLI) is AllDistributions[Distribution.BERNOULLI]
    assert get_dist_class(Distribution.CATEGORICAL) is AllDistributions[Distribution.CATEGORICAL]


def test_get_dist_class_with_distribution_enum_members():
    """
    Test whether `get_dist_class` works with all Distribution enum member values.
    """
    for dist_enum in Distribution.__members__.values():
        result = get_dist_class(dist_enum)
        assert result is AllDistributions[dist_enum]


def test_get_dist_class_with_uppercase_string():
    """
    Test whether `get_dist_class` returns correct class for uppercase string names.
    """
    assert get_dist_class("NORMAL") is AllDistributions[Distribution.NORMAL]
    assert get_dist_class("BERNOULLI") is AllDistributions[Distribution.BERNOULLI]
    assert get_dist_class("CHI_SQUARED") is AllDistributions[Distribution.CHI_SQUARED]


def test_get_dist_class_with_title_case_string():
    """
    Test whether `get_dist_class` returns correct class for title case string values.
    """
    assert get_dist_class("Normal") is AllDistributions[Distribution.NORMAL]
    assert get_dist_class("Bernoulli") is AllDistributions[Distribution.BERNOULLI]
    assert get_dist_class("Chi-squared") is AllDistributions[Distribution.CHI_SQUARED]


def test_get_dist_class_with_lowercase_string():
    """
    Test whether `get_dist_class` handles lowercase strings correctly by converting to title case.
    """
    # Should work for lowercase versions that match Distribution enum names when title-cased
    result_normal = get_dist_class("normal")
    result_bernoulli = get_dist_class("bernoulli")

    # The function converts to title case, so these should work
    assert result_normal is AllDistributions[Distribution.NORMAL]
    assert result_bernoulli is AllDistributions[Distribution.BERNOULLI]


def test_get_dist_class_with_invalid_inputs():
    """
    Test whether `get_dist_class` returns None for invalid inputs.
    """
    assert get_dist_class("invalid_dist") is None
    assert get_dist_class(42) is None
    assert get_dist_class(None) is None
    assert get_dist_class([]) is None
    assert get_dist_class({}) is None


def test_get_dist_class_with_non_distmixin_class():
    """
    Test whether `get_dist_class` returns None for non-DistMixin classes.
    """
    assert get_dist_class(str) is None
    assert get_dist_class(int) is None
    assert get_dist_class(MockDistribution) is None


def test_get_dist_class_with_lowercase_special_chars():
    """
    Test whether `get_dist_class` handles lowercase strings with special characters (case-insensitive matching against distribution values).
    """
    # Test lowercase string with hyphen that should match "Chi-squared" via case-insensitive comparison
    result_chi_squared = get_dist_class("chi-squared")
    assert result_chi_squared is AllDistributions[Distribution.CHI_SQUARED]

    # Test other lowercase variations that should work
    result_lognormal = get_dist_class("lognormal")
    assert result_lognormal is AllDistributions[Distribution.LOGNORMAL]
