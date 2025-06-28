import pytest

from wnb import Distribution as D
from wnb.stats import AllDistributions
from wnb.stats.typing import DistributionLike


@pytest.mark.parametrize("dist_name", AllDistributions.keys())
def test_distributions_correct_name_attr(dist_name):
    """
    Test if all defined distributions have correct `name` attributes.
    """
    assert isinstance(dist_name, (str, D))


@pytest.mark.parametrize("dist", AllDistributions.values())
def test_distributions_correct_support_attr(dist: DistributionLike):
    """
    Test if all defined distributions have correct `_support` attributes.
    """
    if dist.name in [D.UNIFORM, D.PARETO, D.CATEGORICAL]:
        assert dist._support is None

    else:
        assert isinstance(dist._support, (list, tuple))
        if isinstance(dist._support, list):
            for x in dist._support:
                assert isinstance(x, (float, int))
        else:
            assert len(dist._support) == 2
            for x in dist._support:
                assert isinstance(x, (float, int))
