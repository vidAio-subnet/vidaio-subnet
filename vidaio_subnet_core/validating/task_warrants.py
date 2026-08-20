"""Helpers for resolving failed inference task-warrant requests."""

from typing import Literal


InferenceTaskType = Literal["compression", "upscaling"]


def resolve_failed_task_warrant(
    stored_task_type: str | None,
) -> InferenceTaskType | None:
    """Use a known historical inference type without guessing a default.

    A missing TaskWarrant response can mean that the miner only exposes
    competition handlers, so an absent or invalid stored type must remain
    unresolved rather than being interpreted as upscaling.
    """

    if stored_task_type in ("compression", "upscaling"):
        return stored_task_type
    return None
