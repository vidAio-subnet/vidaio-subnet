"""
Codec-Aware Scoring Module

This module provides fairness adjustments for compression scoring based on:
- Codec efficiency (AV1 > HEVC/VP9 > H.264)
- Encoding mode constraints (CRF > VBR > CBR)
- Bitrate constraints

Ensures miners with harder tasks (H.264 CBR) get fair scores compared to
easier tasks (AV1 CRF) when both perform equally well relative to expectations.
"""

from typing import Tuple
import logging

logger = logging.getLogger(__name__)


# Codec efficiency baselines (relative compression capability)
# AV1 is the baseline (1.0), others are relative to it
CODEC_EFFICIENCY = {
    "av1": 1.0,       # Baseline: most efficient
    "hevc": 0.83,     # ~17% less efficient than AV1
    "vp9": 0.85,      # ~15% less efficient than AV1
    "h264": 0.67,     # ~33% less efficient than AV1
}

# Mode difficulty multipliers (how much harder is this mode?)
# Higher = more difficult to achieve good compression
MODE_DIFFICULTY = {
    "CRF": 1.0,       # Baseline: quality-based, most flexible
    "VBR": 1.12,      # 12% harder (bitrate constraints)
    "CBR": 1.25,      # 25% harder (strict bitrate requirement)
}

# Bitrate constraint difficulty
# Lower bitrate relative to optimal = harder task
def get_bitrate_difficulty(actual_bitrate: float, optimal_bitrate: float) -> float:
    """
    Calculate difficulty multiplier based on how constrained the bitrate is.

    Args:
        actual_bitrate: Target bitrate given to miner (Mbps)
        optimal_bitrate: Optimal bitrate for this codec/resolution/quality (Mbps)

    Returns:
        Difficulty multiplier (1.0 = optimal, higher = more difficult)
    """
    if actual_bitrate >= optimal_bitrate:
        # Above optimal: easier task
        return 1.0

    # Below optimal: harder task
    # Linear difficulty increase as bitrate decreases
    # At 50% of optimal: 1.5x difficulty
    ratio = actual_bitrate / optimal_bitrate
    difficulty = 1.0 + (1.0 - ratio) * 1.0  # Max 2.0x at 0% (theoretical)

    return min(2.0, difficulty)  # Cap at 2x difficulty


def normalize_codec_family_scoring(codec_name: str) -> str:
    """
    Normalize codec name to family for scoring purposes.

    Args:
        codec_name: Raw codec name (e.g., "av1_nvenc", "libx264")

    Returns:
        Normalized codec family ("av1", "hevc", "h264", "vp9")
    """
    codec_lower = codec_name.lower().strip()

    # AV1 variants
    if any(v in codec_lower for v in ["av1", "libaom", "libsvtav1", "svt-av1"]):
        return "av1"

    # HEVC variants
    if any(v in codec_lower for v in ["hevc", "h265", "x265", "libx265"]):
        return "hevc"

    # H.264 variants
    if any(v in codec_lower for v in ["h264", "avc", "x264", "libx264"]):
        return "h264"

    # VP9 variants
    if "vp9" in codec_lower or "libvpx" in codec_lower:
        return "vp9"

    # Default to h264 (most conservative)
    logger.warning(f"Unknown codec '{codec_name}', defaulting to h264 for scoring")
    return "h264"


def get_expected_compression_ratio(
    codec: str,
    mode: str,
    reference_ratio: float = 15.0
) -> float:
    """
    Get expected compression ratio for a codec/mode combination.

    Args:
        codec: Codec family ("av1", "hevc", "h264", "vp9")
        mode: Encoding mode ("CRF", "VBR", "CBR")
        reference_ratio: Reference ratio for AV1 CRF (default: 15.0)

    Returns:
        Expected compression ratio for this combination
    """
    codec_norm = normalize_codec_family_scoring(codec)

    # Get codec efficiency
    codec_eff = CODEC_EFFICIENCY.get(codec_norm, 0.67)

    # Get mode difficulty (inverse for expected ratio)
    mode_diff = MODE_DIFFICULTY.get(mode.upper(), 1.0)

    # Expected ratio = reference × codec_efficiency / mode_difficulty
    # Example: H.264 CBR = 15 × 0.67 / 1.25 = 8.04x expected
    expected_ratio = reference_ratio * codec_eff / mode_diff

    return expected_ratio


def calculate_normalized_compression_score(
    actual_compression_rate: float,
    codec: str,
    mode: str,
    target_bitrate: float = None,
    optimal_bitrate: float = None
) -> Tuple[float, float, str]:
    """
    Calculate normalized compression score accounting for codec/mode difficulty.

    Instead of scoring raw compression ratio, this scores performance
    relative to what's expected for the given codec/mode combination.

    Args:
        actual_compression_rate: Actual compression rate (dist_size/ref_size)
        codec: Target codec
        mode: Encoding mode
        target_bitrate: Target bitrate given to miner (optional)
        optimal_bitrate: Optimal bitrate for this task (optional)

    Returns:
        Tuple of (normalized_score, difficulty_multiplier, explanation)
    """
    codec_norm = normalize_codec_family_scoring(codec)
    mode_upper = mode.upper()

    # Get difficulty factors
    codec_eff = CODEC_EFFICIENCY.get(codec_norm, 0.67)
    mode_diff = MODE_DIFFICULTY.get(mode_upper, 1.0)

    # Calculate bitrate difficulty if provided
    bitrate_diff = 1.0
    if target_bitrate and optimal_bitrate:
        bitrate_diff = get_bitrate_difficulty(target_bitrate, optimal_bitrate)

    # Overall difficulty multiplier
    total_difficulty = mode_diff * bitrate_diff

    # Convert compression rate to ratio
    actual_ratio = 1.0 / actual_compression_rate if actual_compression_rate > 0 else 1.0

    # Get expected ratio for this codec/mode
    expected_ratio = get_expected_compression_ratio(codec_norm, mode_upper)

    # Normalize: how much better/worse than expected?
    # normalized_ratio = actual / expected
    # If actual = 12x and expected = 8x: normalized = 1.5 (50% better!)
    # If actual = 6x and expected = 8x: normalized = 0.75 (25% worse)
    normalized_ratio = actual_ratio / expected_ratio

    # Convert to score using standard compression scoring
    # This gives same score curve as before, but relative to expectation
    if normalized_ratio <= 1.25:  # Below 1.25x expected
        # Poor performance relative to codec capability
        normalized_score = ((normalized_ratio - 0.0) / 1.25) ** 1.5
    else:
        # Good performance - exceeding expectations
        normalized_score = 1.0 + 0.3 * min(1.0, (normalized_ratio - 1.25) / 0.75)

    normalized_score = min(1.3, normalized_score)

    # Generate explanation
    explanation = (
        f"Codec: {codec_norm.upper()} (eff={codec_eff:.2f}), "
        f"Mode: {mode_upper} (diff={mode_diff:.2f}), "
        f"Expected: {expected_ratio:.1f}x, Actual: {actual_ratio:.1f}x, "
        f"Performance: {normalized_ratio:.2f}x expected"
    )

    logger.info(f"Normalized compression: {explanation}")

    return normalized_score, total_difficulty, explanation


def apply_codec_aware_scoring(
    base_compression_component: float,
    base_quality_component: float,
    codec: str,
    mode: str,
    compression_weight: float = 0.70,
    quality_weight: float = 0.30
) -> Tuple[float, str]:
    """
    Apply codec-aware adjustments to scoring components.

    For harder codecs/modes:
    - Increase quality weight (harder to achieve good compression, focus on quality)
    - Decrease compression weight (lower expectations)

    Args:
        base_compression_component: Raw compression score (0-1.3)
        base_quality_component: Raw quality score (0.7-1.0)
        codec: Target codec
        mode: Encoding mode
        compression_weight: Base compression weight
        quality_weight: Base quality weight

    Returns:
        Tuple of (adjusted_final_score, explanation)
    """
    codec_norm = normalize_codec_family_scoring(codec)
    mode_upper = mode.upper()

    # Get difficulty
    codec_eff = CODEC_EFFICIENCY.get(codec_norm, 0.67)
    mode_diff = MODE_DIFFICULTY.get(mode_upper, 1.0)

    # Adjust weights based on task difficulty
    # Harder tasks: shift weight toward quality (more achievable)
    # Easier tasks: keep weight on compression (showcase efficiency)

    adjusted_compression_weight = compression_weight
    adjusted_quality_weight = quality_weight

    # For constrained modes (CBR/VBR), quality matters more
    if mode_upper in ["CBR", "VBR"]:
        adjusted_compression_weight = 0.30
        adjusted_quality_weight = 0.70

    # For less efficient codecs, slightly boost quality importance
    if codec_eff < 0.75:  # H.264
        adjusted_quality_weight = min(0.80, adjusted_quality_weight + 0.10)
        adjusted_compression_weight = 1.0 - adjusted_quality_weight

    # Calculate adjusted final score
    adjusted_score = (
        base_compression_component * adjusted_compression_weight +
        base_quality_component * adjusted_quality_weight
    )

    # Apply difficulty bonus
    # Miners who succeed on harder tasks get slight bonus
    difficulty_factor = mode_diff / codec_eff  # Higher = harder
    if difficulty_factor > 1.2:  # Harder than baseline
        difficulty_bonus = min(0.05, (difficulty_factor - 1.0) * 0.05)  # Max 5% bonus
        adjusted_score = min(1.0, adjusted_score + difficulty_bonus)
        explanation = (
            f"Adjusted scoring for {codec_norm.upper()}/{mode_upper}: "
            f"weights=({adjusted_compression_weight:.2f}/{adjusted_quality_weight:.2f}), "
            f"difficulty_bonus={difficulty_bonus:.3f}"
        )
    else:
        explanation = (
            f"Adjusted scoring for {codec_norm.upper()}/{mode_upper}: "
            f"weights=({adjusted_compression_weight:.2f}/{adjusted_quality_weight:.2f})"
        )

    logger.info(explanation)

    return adjusted_score, explanation


def get_codec_mode_fairness_report(
    av1_crf_score: float,
    test_codec: str,
    test_mode: str,
    test_score: float
) -> str:
    """
    Generate fairness report comparing different codec/mode combinations.

    Args:
        av1_crf_score: Score achieved with AV1 CRF (baseline)
        test_codec: Codec being compared
        test_mode: Mode being compared
        test_score: Score achieved with test codec/mode

    Returns:
        Human-readable fairness report
    """
    codec_norm = normalize_codec_family_scoring(test_codec)
    expected_ratio = get_expected_compression_ratio(codec_norm, test_mode)
    baseline_ratio = get_expected_compression_ratio("av1", "CRF")

    efficiency_ratio = expected_ratio / baseline_ratio

    report = f"""
Codec/Mode Fairness Analysis:
---------------------------------------------------------
Baseline (AV1 CRF):
  - Expected compression: {baseline_ratio:.1f}x
  - Score: {av1_crf_score:.3f}

Test ({codec_norm.upper()} {test_mode}):
  - Expected compression: {expected_ratio:.1f}x ({efficiency_ratio:.1%} of AV1 CRF)
  - Score: {test_score:.3f}
  - Fair comparison: {'[OK] Yes' if abs(test_score - av1_crf_score) < 0.05 else '[?] Review'}

Interpretation:
  - If both miners performed equally well relative to their codec/mode,
    scores should be within +/-0.05 of each other.
  - Current difference: {abs(test_score - av1_crf_score):.3f}
---------------------------------------------------------
"""
    return report
