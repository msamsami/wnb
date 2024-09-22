from .base import DistMixin
from .continuous import *
from .discrete import *
from .enums import Distribution

_all_distribution_classes = {
    name: obj
    for name, obj in locals().items()
    if isinstance(obj, type) and issubclass(obj, DistMixin) and obj != DistMixin
}

AllDistributions = {cls.name: cls for cls in _all_distribution_classes.values()}

NonNumericDistributions = [Distribution.CATEGORICAL]

__all__ = list(_all_distribution_classes.keys()) + [
    "Distribution",
    "AllDistributions",
    "NonNumericDistributions",
]
