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

    if isinstance(name_or_type, Distribution) or name_or_type in Distribution.__members__.values():
        return AllDistributions[name_or_type]
    elif isinstance(name_or_type, str) and name_or_type.upper() in [d.name for d in Distribution]:
        return AllDistributions[Distribution.__members__[name_or_type.upper()]]
    elif isinstance(name_or_type, str) and name_or_type.title() in [d.value for d in Distribution]:
        return AllDistributions[Distribution(name_or_type.title())]
    else:
        return
