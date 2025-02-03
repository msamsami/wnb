from .base import DistMixin
from .continuous import (
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
from .discrete import BernoulliDist, CategoricalDist, GeometricDist, PoissonDist
from .enums import Distribution

_all_distribution_classes = {
    name: obj
    for name, obj in locals().items()
    if isinstance(obj, type) and issubclass(obj, DistMixin) and obj != DistMixin
}

AllDistributions = {cls.name: cls for cls in _all_distribution_classes.values()}

NonNumericDistributions = [Distribution.CATEGORICAL]

__all__ = [
    "BetaDist",
    "ChiSquaredDist",
    "ExponentialDist",
    "GammaDist",
    "LaplaceDist",
    "LognormalDist",
    "NormalDist",
    "ParetoDist",
    "RayleighDist",
    "TDist",
    "UniformDist",
    "BernoulliDist",
    "CategoricalDist",
    "GeometricDist",
    "PoissonDist",
    "Distribution",
    "AllDistributions",
    "NonNumericDistributions",
]
