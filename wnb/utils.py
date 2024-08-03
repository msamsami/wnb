import contextlib
from typing import Optional

from ._base import DistMixin
from ._typing import DistibutionLike
from .dist import AllDistributions
from .enums import Distribution

__all__ = ["is_dist_supported", "get_dist_class"]


def is_dist_supported(dist: DistibutionLike) -> bool:
    with contextlib.suppress(TypeError):
        issubclass(dist, DistMixin)
        return True

    if (
        isinstance(dist, Distribution)
        or dist in Distribution.__members__.values()
        or all(
            hasattr(dist, attr_name)
            for attr_name in ["from_data", "support", "__call__"]
        )
    ):
        return True

    return False


def get_dist_class(name_or_type: DistibutionLike) -> Optional[DistMixin]:
    with contextlib.suppress(TypeError):
        issubclass(name_or_type, DistMixin)
        return name_or_type

    if (
        isinstance(name_or_type, Distribution)
        or name_or_type in Distribution.__members__.values()
    ):
        return AllDistributions[name_or_type]
    elif isinstance(name_or_type, str) and name_or_type.upper() in [
        d.name for d in Distribution
    ]:
        return AllDistributions[Distribution.__members__[name_or_type.upper()]]
    elif isinstance(name_or_type, str) and name_or_type.title() in [
        d.value for d in Distribution
    ]:
        return AllDistributions[Distribution(name_or_type.title())]
    else:
        return
