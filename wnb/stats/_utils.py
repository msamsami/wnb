import contextlib
from typing import Optional

from . import AllDistributions
from .base import DistMixin
from .enums import Distribution
from .typing import DistributionLike


def is_dist_supported(dist: DistributionLike) -> bool:
    with contextlib.suppress(TypeError):
        return issubclass(dist, DistMixin)

    if (
        isinstance(dist, Distribution)
        or dist in Distribution.__members__.values()
        or all(hasattr(dist, attr_name) for attr_name in ["from_data", "support", "__call__"])
    ):
        return True

    return False


def get_dist_class(name_or_type: DistributionLike) -> Optional[DistMixin]:
    with contextlib.suppress(TypeError):
        if issubclass(name_or_type, DistMixin):
            return name_or_type

    d_names = [d.name for d in Distribution]
    d_values = [d.value for d in Distribution]

    if isinstance(name_or_type, Distribution) or name_or_type in d_values:
        return AllDistributions[name_or_type]
    elif isinstance(name_or_type, str) and name_or_type.upper() in d_names:
        return AllDistributions[Distribution.__members__[name_or_type.upper()]]
    elif isinstance(name_or_type, str) and name_or_type.lower() in [value.lower() for value in d_values]:
        idx = next(i for i, value in enumerate(d_values) if value.lower() == name_or_type.lower())
        name = d_names[idx]
        return AllDistributions[Distribution.__members__[name]]
    else:
        return
