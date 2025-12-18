"""
Codec Feasibility Validator

This module validates that compression parameter combinations are achievable.
Ensures that target_bitrate is sufficient for the requested VMAF quality at a given resolution.

Based on empirical codec performance data and industry standards.
"""

from typing import Tuple, Optional
import logging
import subprocess
import json

logger = logging.getLogger(__name__)


# Minimum bitrate requirements (in Mbps) to achieve specific VMAF scores
# Format: [codec][resolution][vmaf_threshold] = minimum_bitrate_mbps
#
# Data based on empirical testing and industry standards:
# - Lower resolutions need less bitrate
# - Higher VMAF thresholds need more bitrate
# - Different codecs have different efficiency
MINIMUM_BITRATE_REQUIREMENTS = {
    "av1": {
        # AV1 - Most efficient codec
        "480p": {85: 1.5, 89: 2.0, 93: 3.0},
        "720p": {85: 2.5, 89: 3.5, 93: 5.0},
        "1080p": {85: 4.0, 89: 6.0, 93: 8.0},
        "1440p": {85: 7.0, 89: 10.0, 93: 14.0},
        "2160p": {85: 12.0, 89: 18.0, 93: 25.0},  # 4K
    },
    "hevc": {
        # HEVC/H.265 - Very efficient
        "480p": {85: 2.0, 89: 2.5, 93: 3.5},
        "720p": {85: 3.5, 89: 4.5, 93: 6.5},
        "1080p": {85: 5.5, 89: 8.0, 93: 11.0},
        "1440p": {85: 9.0, 89: 13.0, 93: 18.0},
        "2160p": {85: 16.0, 89: 23.0, 93: 32.0},  # 4K
    },
    "h264": {
        # H.264/AVC - Moderate efficiency
        "480p": {85: 2.5, 89: 3.5, 93: 5.0},
        "720p": {85: 5.0, 89: 7.0, 93: 10.0},
        "1080p": {85: 8.0, 89: 12.0, 93: 16.0},
        "1440p": {85: 14.0, 89: 20.0, 93: 28.0},
        "2160p": {85: 25.0, 89: 35.0, 93: 50.0},  # 4K
    },
    "vp9": {
        # VP9 - Similar to HEVC
        "480p": {85: 2.0, 89: 2.5, 93: 3.5},
        "720p": {85: 3.5, 89: 5.0, 93: 7.0},
        "1080p": {85: 6.0, 89: 8.5, 93: 12.0},
        "1440p": {85: 10.0, 89: 14.0, 93: 20.0},
        "2160p": {85: 18.0, 89: 25.0, 93: 35.0},  # 4K
    },
}


# CBR mode typically requires 20-30% more bitrate than CRF to achieve same quality
# VBR is between CRF and CBR
MODE_BITRATE_MULTIPLIERS = {
    "CRF": 1.0,   # Baseline (most efficient)
    "VBR": 1.15,  # 15% more bitrate needed
    "CBR": 1.25,  # 25% more bitrate needed (least efficient)
}


def get_resolution_category(width: int, height: int) -> str:
    """
    Categorize video resolution into standard categories.

    Args:
        width: Video width in pixels
        height: Video height in pixels

    Returns:
        Resolution category string (e.g., "1080p", "2160p")
    """
    pixels = width * height

    if pixels <= 640 * 480:
        return "480p"
    elif pixels <= 1280 * 720:
        return "720p"
    elif pixels <= 1920 * 1080:
        return "1080p"
    elif pixels <= 2560 * 1440:
        return "1440p"
    else:
        return "2160p"  # 4K and above


def get_nearest_vmaf_threshold(vmaf_threshold: float) -> int:
    """
    Round VMAF threshold to nearest standard value.

    Args:
        vmaf_threshold: Target VMAF threshold

    Returns:
        Nearest standard threshold (85, 89, or 93)
    """
    standard_thresholds = [85, 89, 93]
    return min(standard_thresholds, key=lambda x: abs(x - vmaf_threshold))


def get_minimum_bitrate(
    codec: str,
    resolution_width: int,
    resolution_height: int,
    vmaf_threshold: float,
    codec_mode: str = "CRF"
) -> float:
    """
    Get minimum bitrate required to achieve target VMAF at given resolution.

    Args:
        codec: Target codec ("av1", "hevc", "h264", "vp9")
        resolution_width: Video width in pixels
        resolution_height: Video height in pixels
        vmaf_threshold: Target VMAF threshold (e.g., 85, 89, 93)
        codec_mode: Encoding mode ("CRF", "VBR", "CBR")

    Returns:
        Minimum bitrate in Mbps required
    """
    # Normalize codec name
    codec = codec.lower().strip()

    # Get resolution category
    resolution_category = get_resolution_category(resolution_width, resolution_height)

    # Get nearest standard VMAF threshold
    vmaf_key = get_nearest_vmaf_threshold(vmaf_threshold)

    # Get base bitrate requirement for CRF mode
    try:
        base_bitrate = MINIMUM_BITRATE_REQUIREMENTS[codec][resolution_category][vmaf_key]
    except KeyError:
        logger.warning(f"No bitrate data for codec={codec}, resolution={resolution_category}, vmaf={vmaf_key}")
        # Fallback: use conservative estimate
        if resolution_category == "2160p":
            base_bitrate = 30.0
        elif resolution_category == "1440p":
            base_bitrate = 15.0
        elif resolution_category == "1080p":
            base_bitrate = 8.0
        else:
            base_bitrate = 4.0

    # Apply mode multiplier
    mode_multiplier = MODE_BITRATE_MULTIPLIERS.get(codec_mode.upper(), 1.0)
    minimum_bitrate = base_bitrate * mode_multiplier

    return minimum_bitrate


def validate_compression_parameters(
    codec: str,
    resolution_width: int,
    resolution_height: int,
    vmaf_threshold: float,
    codec_mode: str,
    target_bitrate: float,
    safety_margin: float = 1.15
) -> Tuple[bool, str, Optional[float]]:
    """
    Validate that compression parameters are feasible.

    Args:
        codec: Target codec ("av1", "hevc", "h264", "vp9")
        resolution_width: Video width in pixels
        resolution_height: Video height in pixels
        vmaf_threshold: Target VMAF threshold
        codec_mode: Encoding mode ("CRF", "VBR", "CBR")
        target_bitrate: Target bitrate in Mbps
        safety_margin: Safety multiplier (1.15 = require 15% extra headroom)

    Returns:
        Tuple of (is_valid, reason, suggested_bitrate)
        - is_valid: True if parameters are feasible
        - reason: Human-readable explanation
        - suggested_bitrate: Recommended minimum bitrate (or None if valid)
    """
    # Get minimum required bitrate
    min_bitrate = get_minimum_bitrate(
        codec, resolution_width, resolution_height, vmaf_threshold, codec_mode
    )

    # Apply safety margin
    recommended_bitrate = min_bitrate * safety_margin

    resolution_category = get_resolution_category(resolution_width, resolution_height)

    # Check if target bitrate is sufficient
    if target_bitrate < min_bitrate:
        reason = (
            f"Target bitrate {target_bitrate:.1f} Mbps is too low for "
            f"{codec.upper()} @ {resolution_category} with VMAF {vmaf_threshold:.0f} in {codec_mode} mode. "
            f"Minimum required: {min_bitrate:.1f} Mbps (recommended: {recommended_bitrate:.1f} Mbps)"
        )
        return False, reason, recommended_bitrate

    elif target_bitrate < recommended_bitrate:
        reason = (
            f"Target bitrate {target_bitrate:.1f} Mbps is marginal for "
            f"{codec.upper()} @ {resolution_category} with VMAF {vmaf_threshold:.0f} in {codec_mode} mode. "
            f"Recommended: {recommended_bitrate:.1f} Mbps for reliable quality"
        )
        logger.warning(reason)
        return True, reason, recommended_bitrate

    else:
        reason = (
            f"Parameters validated: {codec.upper()} @ {resolution_category}, "
            f"VMAF {vmaf_threshold:.0f}, {codec_mode} mode, {target_bitrate:.1f} Mbps"
        )
        return True, reason, None


def adjust_parameters_to_feasible(
    codec: str,
    resolution_width: int,
    resolution_height: int,
    vmaf_threshold: float,
    codec_mode: str,
    target_bitrate: float
) -> Tuple[float, float, str]:
    """
    Adjust parameters to make them feasible while preserving as much of the original intent as possible.

    Priority order:
    1. Keep codec and mode (user-specified)
    2. Lower VMAF threshold if needed (quality vs bitrate tradeoff)
    3. Increase bitrate as last resort (if lowering VMAF isn't enough)

    Args:
        codec: Target codec
        resolution_width: Video width
        resolution_height: Video height
        vmaf_threshold: Target VMAF threshold
        codec_mode: Encoding mode
        target_bitrate: Target bitrate in Mbps

    Returns:
        Tuple of (adjusted_vmaf_threshold, adjusted_bitrate, reason)
    """
    is_valid, reason, _ = validate_compression_parameters(
        codec, resolution_width, resolution_height, vmaf_threshold, codec_mode, target_bitrate
    )

    if is_valid:
        return vmaf_threshold, target_bitrate, "Parameters already feasible"

    # Try lowering VMAF threshold first
    vmaf_options = [93, 89, 85]
    for lower_vmaf in vmaf_options:
        if lower_vmaf >= vmaf_threshold:
            continue

        is_valid, _, _ = validate_compression_parameters(
            codec, resolution_width, resolution_height, lower_vmaf, codec_mode, target_bitrate
        )

        if is_valid:
            reason = f"Lowered VMAF threshold from {vmaf_threshold:.0f} to {lower_vmaf} to match bitrate {target_bitrate:.1f} Mbps"
            logger.info(reason)
            return lower_vmaf, target_bitrate, reason

    # If lowering VMAF isn't enough, increase bitrate
    min_bitrate = get_minimum_bitrate(
        codec, resolution_width, resolution_height, 85, codec_mode  # Use lowest VMAF
    )
    adjusted_bitrate = min_bitrate * 1.2  # 20% safety margin

    reason = f"Increased bitrate from {target_bitrate:.1f} to {adjusted_bitrate:.1f} Mbps and lowered VMAF to 85"
    logger.info(reason)
    return 85, adjusted_bitrate, reason


def is_crf_mode_recommended(
    codec: str,
    resolution_width: int,
    resolution_height: int,
    vmaf_threshold: float,
    target_bitrate: float
) -> bool:
    """
    Check if CRF mode would be better than CBR/VBR for the given parameters.

    CRF is recommended when the target bitrate is well above minimum requirements,
    as it can better optimize for quality.

    Args:
        codec: Target codec
        resolution_width: Video width
        resolution_height: Video height
        vmaf_threshold: Target VMAF threshold
        target_bitrate: Target bitrate in Mbps

    Returns:
        True if CRF mode is recommended
    """
    min_bitrate_crf = get_minimum_bitrate(
        codec, resolution_width, resolution_height, vmaf_threshold, "CRF"
    )

    # If target bitrate is 50% or more above CRF minimum, CRF mode is better
    return target_bitrate >= (min_bitrate_crf * 1.5)


def get_optimal_bitrate_range(
    codec: str,
    resolution_width: int,
    resolution_height: int,
    vmaf_threshold: float,
    codec_mode: str = "CRF"
) -> Tuple[float, float, float]:
    """
    Get optimal bitrate range for given parameters.

    Returns minimum, optimal, and maximum bitrate values that make sense
    for the given codec/resolution/quality combination.

    Args:
        codec: Target codec
        resolution_width: Video width
        resolution_height: Video height
        vmaf_threshold: Target VMAF threshold
        codec_mode: Encoding mode

    Returns:
        Tuple of (minimum_bitrate, optimal_bitrate, maximum_bitrate) in Mbps
    """
    min_bitrate = get_minimum_bitrate(
        codec, resolution_width, resolution_height, vmaf_threshold, codec_mode
    )

    # Optimal is 20% above minimum (sweet spot for quality/size tradeoff)
    optimal_bitrate = min_bitrate * 1.2

    # Maximum is where further increases provide diminishing returns
    # For most codecs, 2x minimum is past the point of diminishing returns
    max_bitrate = min_bitrate * 2.0

    return min_bitrate, optimal_bitrate, max_bitrate


def select_smart_bitrate(
    codec: str,
    resolution_width: int,
    resolution_height: int,
    vmaf_threshold: float,
    codec_mode: str = "CRF",
    preference: str = "balanced"
) -> float:
    """
    Intelligently select an appropriate bitrate based on codec/resolution/quality.

    Args:
        codec: Target codec
        resolution_width: Video width
        resolution_height: Video height
        vmaf_threshold: Target VMAF threshold
        codec_mode: Encoding mode
        preference: Bitrate preference ("low", "balanced", "high", or "variable")

    Returns:
        Recommended bitrate in Mbps
    """
    min_bitrate, optimal_bitrate, max_bitrate = get_optimal_bitrate_range(
        codec, resolution_width, resolution_height, vmaf_threshold, codec_mode
    )

    if preference == "low":
        # Conservative: minimum + 10% safety margin
        return min_bitrate * 1.1
    elif preference == "balanced":
        # Recommended: optimal sweet spot
        return optimal_bitrate
    elif preference == "high":
        # High quality: near maximum for best quality
        return max_bitrate * 0.9
    elif preference == "variable":
        # Random within optimal range for testing
        import random
        return random.uniform(optimal_bitrate * 0.9, optimal_bitrate * 1.3)
    else:
        logger.warning(f"Unknown preference '{preference}', using balanced")
        return optimal_bitrate


def get_bitrate_options_for_testing(
    codec: str,
    resolution_width: int,
    resolution_height: int,
    vmaf_threshold: float,
    codec_mode: str = "CRF",
    num_options: int = 4
) -> list[float]:
    """
    Generate a list of sensible bitrate options for testing miners.

    Creates a spread of bitrates from minimum to maximum that are all
    achievable but test different compression efficiencies.

    Args:
        codec: Target codec
        resolution_width: Video width
        resolution_height: Video height
        vmaf_threshold: Target VMAF threshold
        codec_mode: Encoding mode
        num_options: Number of bitrate options to generate

    Returns:
        List of bitrate values in Mbps, sorted from lowest to highest
    """
    min_bitrate, optimal_bitrate, max_bitrate = get_optimal_bitrate_range(
        codec, resolution_width, resolution_height, vmaf_threshold, codec_mode
    )

    # Generate evenly spaced options between min and max
    # Using logarithmic spacing for better distribution
    import math

    if num_options == 1:
        return [optimal_bitrate]

    # Generate log-spaced values
    log_min = math.log(min_bitrate * 1.1)  # Slight margin above minimum
    log_max = math.log(max_bitrate * 0.9)  # Slight margin below maximum

    options = []
    for i in range(num_options):
        log_value = log_min + (log_max - log_min) * i / (num_options - 1)
        bitrate = math.exp(log_value)
        # Round to 1 decimal place for cleaner values
        options.append(round(bitrate, 1))

    return sorted(options)


def get_video_resolution_from_url(video_url: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract video resolution from a video URL using ffprobe.

    Args:
        video_url: URL to the video file

    Returns:
        Tuple of (width, height) or (None, None) if extraction fails
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            video_url
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            data = json.loads(result.stdout)
            stream = data.get("streams", [{}])[0]
            width = stream.get("width")
            height = stream.get("height")

            if width and height:
                logger.info(f"Extracted resolution from {video_url}: {width}x{height}")
                return width, height

        logger.warning(f"Failed to extract resolution from {video_url}")
        return None, None

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout while extracting resolution from {video_url}")
        return None, None
    except Exception as e:
        logger.error(f"Error extracting resolution from {video_url}: {e}")
        return None, None


def get_video_resolution_from_path(video_path: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract video resolution from a local video file using ffprobe.

    Args:
        video_path: Path to the video file

    Returns:
        Tuple of (width, height) or (None, None) if extraction fails
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            video_path
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            if len(parts) == 2:
                width = int(parts[0])
                height = int(parts[1])
                logger.info(f"Extracted resolution from {video_path}: {width}x{height}")
                return width, height

        logger.warning(f"Failed to extract resolution from {video_path}")
        return None, None

    except Exception as e:
        logger.error(f"Error extracting resolution from {video_path}: {e}")
        return None, None
