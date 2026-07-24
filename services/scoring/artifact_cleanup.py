import logging
import os
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

logger = logging.getLogger(__name__)


async def cleanup_compression_artifacts(
    distorted_file_paths: Iterable[str | None],
    reference_video_paths: Iterable[str | None],
    uploaded_object_names: Iterable[str | None],
    delete_uploaded_object: Callable[[str], Awaitable[Any]],
) -> None:
    """Delete every local compression artifact and uploaded challenge object."""
    for artifact_type, paths in (
        ("distorted video", distorted_file_paths),
        ("reference video", reference_video_paths),
    ):
        for path in dict.fromkeys(path for path in paths if path):
            if not os.path.exists(path):
                continue
            try:
                os.unlink(path)
            except Exception as exc:
                logger.error(f"Error deleting {artifact_type} file {path}: {exc}")

    for object_name in dict.fromkeys(name for name in uploaded_object_names if name):
        try:
            await delete_uploaded_object(object_name)
        except Exception as exc:
            logger.error(
                f"Error deleting uploaded compression challenge object "
                f"{object_name}: {exc}"
            )
