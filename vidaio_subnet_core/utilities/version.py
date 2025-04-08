from .. import __version__
from ..protocol import Version
from loguru import logger

def check_version(version: Version):
    """
    Check the version of request is up to date with subnet
    """
    if (version is not None 
        and compare_version(version, get_version()) > 0
    ):
        logger.warning(
            f"Received request with version {version}, is newer than miner running version {get_version()}"
        )

def get_version():
    version_split = __version__.split(".")
    return Version(
        major=int(version_split[0]),
        minor=int(version_split[1]),
        patch=int(version_split[2]),
    )

def compare_version(version: Version, other: Version) -> int:
    if version.major > other.major:
        return 1
    elif version.major < other.major:
        return -1
    else:
        if version.minor > other.minor:
            return 1
        elif version.minor < other.minor:
            return -1
        else:
            if version.patch > other.patch:
                return 1
            elif version.patch < other.patch:
                return -1
            else:
                return 0
