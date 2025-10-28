from . import validating
from . import protocol
from . import base
from . import utilities
from .global_config import CONFIG

__all__ = [
    "validating",
    "protocol",
    "base",
    "utilities",
    "CONFIG",
]

__version__ = "2.1.7"

version_split = __version__.split(".")
__spec_version__ = (
    (100 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)
