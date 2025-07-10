from typing import Tuple


def get_task_target_resolution(task_type: str) -> Tuple[int, int]:
    TASK_RESOLUTIONS = {
        "HD24K": (3840, 2160),
        "SD2HD": (1920, 1080),
        "SD24K": (3840, 2160),
        # "4K28K": (7680, 4320),
        # "HD28K": (7680, 4320),
    }
    return TASK_RESOLUTIONS.get(task_type, (3840, 2160))


def get_max_optimized_bitrate(task_type: str) -> int:
    """
    Get the maximum bitrate of optimized videos (task responses) for a given
    task type.

    Args:
        task_type (str): The type of task, e.g. "SD2HD", "HD24K" etc
    Returns:
        int: Maximum bitrate of optimized videos in kbit/s
    """
    # Bitrate budget: 40 Mbit/s for 4K - necessitates rate control when
    # encoding H264 videos. Effectively caps 20s 4K videos at ~100 MB.
    width, height = get_task_target_resolution(task_type)
    n_pixels = width * height
    return int(40000 * n_pixels / (1920 * 1080))
