import warnings

import numpy as np
import pytest
from sklearn.utils._testing import assert_array_almost_equal

from wnb.stats.base import ContinuousDistMixin, DiscreteDistMixin, DistMixin, vectorize

out_of_support_warn_msg = "Value doesn't lie within the support of the distribution"


# Test implementations for abstract classes
class TestDistMixin(DistMixin):
    """
    Test implementation of `DistMixin` for testing purposes.
    """

    def __init__(self, param1: float = 1.0, param2: str = "test", param3: int = 42):
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3
        self.name = "TestDist"
        self._support = (0.0, 10.0)

    @classmethod
    def from_data(cls, data, **kwargs):
        return cls(param1=float(np.mean(data)), param2="from_data")

    def __call__(self, x):
        return x * self.param1


class TestContinuousDistMixin(ContinuousDistMixin):
    """
    Test implementation of `ContinuousDistMixin` for testing purposes.
    """

    def __init__(self, scale: float = 1.0, loc: float = 0.0):
        super().__init__()
        self.scale = scale
        self.loc = loc
        self.name = "TestContinuous"
        self._support = (-np.inf, np.inf)

    @classmethod
    def from_data(cls, data, **kwargs):
        return cls(scale=float(np.std(data)), loc=float(np.mean(data)))

    def pdf(self, x: float) -> float:
        return np.exp(-0.5 * ((x - self.loc) / self.scale) ** 2) / (self.scale * np.sqrt(2 * np.pi))


class TestDiscreteDistMixin(DiscreteDistMixin):
    """
    Test implementation of `DiscreteDistMixin` for testing purposes.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.name = "TestDiscrete"
        self._support = [0, 1]

    @classmethod
    def from_data(cls, data, **kwargs):
        return cls(p=float(np.mean(data)))

    def pmf(self, x: float) -> float:
        if x == 1:
            return self.p
        elif x == 0:
            return 1 - self.p
        else:
            return 0.0


def test_dist_mixin_get_param_names():
    """
    Test whether `_get_param_names` method correctly extracts parameter names from `__init__` signature.
    """
    param_names = TestDistMixin._get_param_names()
    expected_params = ["param1", "param2", "param3"]
    assert sorted(param_names) == sorted(expected_params)


def test_dist_mixin_get_param_names_no_init():
    """
    Test whether `_get_param_names` returns empty list for classes with no custom `__init__`.
    """

    class NoInitClass(DistMixin):
        """Class that uses object.__init__ with no custom parameters."""

        @classmethod
        def from_data(cls, data, **kwargs):
            return cls()

        def __call__(self, x):
            return x

    param_names = NoInitClass._get_param_names()
    assert param_names == []


def test_dist_mixin_get_param_names_with_varargs():
    """
    Test whether `_get_param_names` raises RuntimeError for classes with varargs in `__init__`.
    """

    class VarargsClass(DistMixin):
        """Class with varargs in __init__ which should raise RuntimeError."""

        def __init__(self, *args, param1=1.0):
            self.param1 = param1

        @classmethod
        def from_data(cls, data, **kwargs):
            return cls()

        def __call__(self, x):
            return x

    with pytest.raises(RuntimeError, match="wnb distributions should always specify their parameters"):
        VarargsClass._get_param_names()


def test_dist_mixin_get_params():
    """
    Test whether `get_params` method correctly returns parameter values as a dictionary.
    """
    dist = TestDistMixin(param1=2.5, param2="custom", param3=100)
    params = dist.get_params()
    expected_params = {"param1": 2.5, "param2": "custom", "param3": 100}
    assert params == expected_params


def test_dist_mixin_support_property():
    """
    Test whether `support` property correctly returns the distribution support.
    """
    dist = TestDistMixin()
    assert dist.support == (0.0, 10.0)

    # Test with list support
    dist._support = [1, 2, 3]
    assert dist.support == [1, 2, 3]


def test_dist_mixin_check_support_continuous():
    """
    Test whether `_check_support` method correctly validates values within continuous support.
    """
    dist = TestDistMixin()
    dist._support = (0.0, 10.0)

    # Value within support should not raise warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dist._check_support(5.0)

    # Value outside support should raise warning
    with pytest.warns(RuntimeWarning, match=out_of_support_warn_msg):
        dist._check_support(15.0)

    with pytest.warns(RuntimeWarning, match=out_of_support_warn_msg):
        dist._check_support(-1.0)


def test_dist_mixin_check_support_discrete():
    """
    Test whether `_check_support` method correctly validates values within discrete support.
    """
    dist = TestDistMixin()
    dist._support = [1, 2, 3]

    # Value within support should not raise warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dist._check_support(2)

    # Value outside support should raise warning
    with pytest.warns(RuntimeWarning, match=out_of_support_warn_msg):
        dist._check_support(5)


def test_dist_mixin_repr():
    """
    Test whether `__repr__` method correctly formats the string representation.
    """
    dist = TestDistMixin(param1=1.23456, param2="test", param3=42)
    repr_str = repr(dist)

    assert repr_str.startswith("<TestDistMixin(")
    assert repr_str.endswith(")>")
    assert "param1=1.2346" in repr_str  # Should be rounded to 4 decimal places
    assert "param2=test" in repr_str
    assert "param3=42" in repr_str


def test_dist_mixin_from_data():
    """
    Test whether `from_data` class method works correctly.
    """
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dist = TestDistMixin.from_data(data)

    assert dist.param1 == 3.0  # mean of data
    assert dist.param2 == "from_data"
    assert dist.param3 == 42  # default value


def test_vectorize_decorator():
    """
    Test whether the `vectorize` decorator correctly vectorizes functions.
    """

    # Test simple function without signature
    @vectorize()
    def simple_func(x):
        return x**2

    # Test with scalar
    result_scalar = simple_func(3.0)
    assert result_scalar == 9.0

    # Test with array
    X = np.array([1.0, 2.0, 3.0, 4.0])
    result_array = simple_func(X)
    expected = np.array([1.0, 4.0, 9.0, 16.0])
    assert_array_almost_equal(result_array, expected)

    # Test that the decorator preserves function properties
    assert hasattr(simple_func, "__call__")
    assert callable(simple_func)


def test_continuous_dist_mixin_call():
    """
    Test whether `ContinuousDistMixin.__call__` method correctly calls pdf with vectorization.
    """
    dist = TestContinuousDistMixin(scale=1.0, loc=0.0)

    # Test with scalar
    result_scalar = dist(0.0)
    expected_scalar = dist.pdf(0.0)
    assert result_scalar == expected_scalar

    # Test with array
    X = np.array([0.0, 1.0, -1.0])
    result_array = dist(X)
    expected_array = np.array([dist.pdf(x) for x in X])
    assert_array_almost_equal(result_array, expected_array, decimal=10)


def test_discrete_dist_mixin_call():
    """
    Test whether `DiscreteDistMixin.__call__` method correctly calls pmf with vectorization.
    """
    dist = TestDiscreteDistMixin(p=0.7)

    # Test with scalar
    result_scalar = dist(1)
    expected_scalar = dist.pmf(1)
    assert result_scalar == expected_scalar

    # Test with array
    X = np.array([0, 1, 0, 1])
    result_array = dist(X)
    expected_array = np.array([dist.pmf(x) for x in X])
    assert_array_almost_equal(result_array, expected_array, decimal=10)


def test_continuous_dist_mixin_type():
    """
    Test whether `ContinuousDistMixin` has correct type attribute.
    """
    dist = TestContinuousDistMixin()
    assert dist._type == "continuous"


def test_discrete_dist_mixin_type():
    """
    Test whether `DiscreteDistMixin` has correct type attribute.
    """
    dist = TestDiscreteDistMixin()
    assert dist._type == "discrete"


def test_abstract_methods():
    """
    Test whether abstract methods are properly defined as abstract.
    """
    # DistMixin should be abstract
    assert DistMixin.__abstractmethods__ == {"from_data", "__call__"}

    # ContinuousDistMixin should require pdf implementation
    assert "pdf" in ContinuousDistMixin.__abstractmethods__
    assert "from_data" in ContinuousDistMixin.__abstractmethods__

    # DiscreteDistMixin should require pmf implementation
    assert "pmf" in DiscreteDistMixin.__abstractmethods__
    assert "from_data" in DiscreteDistMixin.__abstractmethods__


def test_continuous_dist_mixin_support_warning():
    """
    Test whether `ContinuousDistMixin` properly warns for out-of-support values.
    """
    dist = TestContinuousDistMixin()
    dist._support = (0.0, 1.0)

    with pytest.warns(RuntimeWarning, match=out_of_support_warn_msg):
        dist(-1.0)


def test_discrete_dist_mixin_support_warning():
    """
    Test whether `DiscreteDistMixin` properly warns for out-of-support values.
    """
    dist = TestDiscreteDistMixin()

    with pytest.warns(RuntimeWarning, match=out_of_support_warn_msg):
        dist(2)  # Support is [0, 1]
