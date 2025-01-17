import warnings

from wnb.stats import *  # noqa: F403

warnings.warn(
    "The `wnb.dist` module is deprecated and will be removed in a future release. "
    "Please update your imports to use `wnb` or `wnb.stats` directly. "
    "Using `wnb.dist` will continue to work in this version, but it may be removed in future versions.",
    DeprecationWarning,
    stacklevel=2,
)
