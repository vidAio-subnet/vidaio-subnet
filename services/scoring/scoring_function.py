import math
from services.scoring.codec_aware_scoring import (
    calculate_normalized_compression_score,
    apply_codec_aware_scoring,
    normalize_codec_family_scoring
)


def calculate_compression_score(
    vmaf_score: float,
    compression_rate: float,
    vmaf_threshold: float,
    compression_weight: float = 0.70,
    quality_weight: float = 0.30,
    soft_threshold_margin: float = 5.0,
    target_codec: str = "av1",
    codec_mode: str = "CRF",
    target_bitrate: float = 10.0) -> tuple[float, float, float, str]:
    """
    Calculate compression score that rewards hitting VMAF threshold with good compression.

    Scoring Philosophy:
    - Compression is primary goal (70% weight by default for CRF mode)
    - VMAF at/above threshold is rewarded with diminishing returns (30% weight)
    - 15x compression ratio = 1.0 compression component for AV1 CRF (baseline)
    - Codec-aware normalization adjusts expectations based on codec efficiency and mode difficulty
    - Exceeding VMAF threshold gives modest bonus (not penalty)

    Three scoring zones:
    1. Below hard cutoff (threshold - 5): Score = 0
    2. Soft zone (threshold - 5 to threshold): Gradual recovery
    3. Above threshold: Full scoring with quality bonus and codec-aware adjustments

    Args:
        vmaf_score: VMAF quality score (0-100)
        compression_rate: Size ratio compressed/original (0-1, lower is better)
        vmaf_threshold: Required VMAF threshold (e.g., 85, 89, 93)
        compression_weight: Weight for compression component (default: 0.70)
        quality_weight: Weight for quality component (default: 0.30)
        soft_threshold_margin: Points below threshold before hard cutoff (default: 5.0)
        target_codec: Target codec (av1, hevc, h264, vp9) for codec-aware scoring
        codec_mode: Encoding mode (CRF, VBR, CBR) for difficulty adjustment
        target_bitrate: Target bitrate in Mbps for bitrate difficulty calculation

    Returns:
        Tuple of (final_score, compression_component, quality_component, reason)
        where final_score is in range [0.0, 1.0]
    """
    
    # Validate that weights sum to 1.0
    if abs(compression_weight + quality_weight - 1.0) > 0.01:
        raise ValueError(f"Weights must sum to 1.0, got {compression_weight + quality_weight}")
    
    # Calculate hard cutoff point
    hard_cutoff = vmaf_threshold - soft_threshold_margin
    
    # ========================================================================
    # CASE 0: No Meaningful Compression - Immediate Failure
    # ========================================================================
    # If compression_rate >= 0.80 (less than 1.25x compression), this is likely
    # the miner returning the same video or minimal compression to exploit high VMAF.
    # Zero the entire score to prevent this exploit.
    if compression_rate >= 0.80:
        compression_ratio = 1 / compression_rate if compression_rate > 0 else 1.0
        return 0.0, 0.0, 0.0, f"No meaningful compression (ratio: {compression_ratio:.2f}x, rate: {compression_rate:.2f}). Minimum 1.25x required."
    
    # ========================================================================
    # CASE 1: Below Hard Cutoff - Immediate Failure
    # ========================================================================
    # If VMAF is more than 5 points below threshold, quality is unacceptable
    if vmaf_score < hard_cutoff:
        return 0.0, 0.0, 0.0, f"VMAF {vmaf_score:.2f} below hard cutoff ({hard_cutoff:.2f})"
    
    # ========================================================================
    # CASE 2: Soft Threshold Zone (threshold-5 to threshold)
    # ========================================================================
    # Gradual recovery zone where both quality and compression matter
    if vmaf_score < vmaf_threshold:
        # Calculate position in soft zone (0.0 to 1.0)
        # Example: VMAF 93 with threshold 95 and margin 5
        # soft_zone_position = (93 - 90) / 5 = 0.6
        soft_zone_position = (vmaf_score - hard_cutoff) / soft_threshold_margin
        
        """
        Quality factor uses scaled function to reach 0.7 at threshold:
        f(x) = 0.7 * x²
        
        - At hard cutoff (x=0): factor = 0.0
        - At midpoint (x=0.5): factor = 0.175
        - At threshold (x=1): factor = 0.7
        
        This ensures continuity with the above-threshold quality component
        which starts at 0.7 when VMAF equals threshold.
        """
        quality_factor = 0.7 * (soft_zone_position ** 2)
        
        # Calculate compression component (compression_rate < 0.80 guaranteed by CASE 0)
        compression_ratio = 1 / compression_rate
        
        if compression_ratio <= 20:
            """
            Compression scoring for 1x to 20x:
            f(r) = ((r - 1) / 19) ^ 1.5
            
            Where r is compression ratio (e.g., 5 for 5x compression)
            - At 1x: component = 0
            - At 2x: component ≈ 0.23
            - At 5x: component ≈ 0.65
            - At 10x: component ≈ 0.89
            - At 15x: component ≈ 0.97
            - At 20x: component = 1.0
            
            The exponent 1.5 provides:
            - Slow growth initially (encouraging minimum viable compression)
            - Steeper growth in practical range (5-15x)
            - Reaches 1.0 exactly at 20x
            """
            compression_component = ((compression_ratio - 1) / 19) ** 1.5
        else:
            """
            Bonus for exceptional compression (>20x):
            f(r) = 1.0 + 0.3 * ln(r / 20)
            
            Logarithmic bonus rewards exceptional performance:
            - At 20x: component = 1.0 (continuous at boundary)
            - At 30x: component ≈ 1.12
            - At 40x: component ≈ 1.21
            - At 60x: component ≈ 1.30 (capped)
            
            Natural log ensures smooth transition from linear region.
            """
            compression_component = 1.0 + 0.3 * math.log(compression_ratio / 20)
        
        compression_component = min(1.3, compression_component)
        
        # In soft zone, both compression AND quality factor matter
        # If you're below threshold, you need good compression to recover
        final_score = compression_component * quality_factor
        
        return min(1.0, final_score), compression_component, quality_factor, f"VMAF {vmaf_score:.2f} in soft zone (quality factor: {quality_factor:.2f})"
    
    # ========================================================================
    # CASE 3: Above Threshold - Full Scoring with Codec-Aware Adjustments
    # ========================================================================
    else:
        # Calculate how much VMAF exceeds threshold
        vmaf_excess = vmaf_score - vmaf_threshold
        max_vmaf_excess = 100 - vmaf_threshold  # Maximum possible excess to VMAF 100

        """
        Quality component rewards exceeding threshold with linear interpolation:
        f(x) = 0.7 + 0.3 * (x / max_excess)

        Where x is VMAF points above threshold:
        - At threshold (x=0): component = 0.7
        - At threshold + 25% of range: component = 0.775
        - At threshold + 50% of range: component = 0.85
        - At threshold + 75% of range: component = 0.925
        - At VMAF 100: component = 1.0

        Example with threshold=85:
        - At 85: 0.7 + 0.3 * (0/15) = 0.7
        - At 90: 0.7 + 0.3 * (5/15) = 0.8
        - At 95: 0.7 + 0.3 * (10/15) = 0.9
        - At 100: 0.7 + 0.3 * (15/15) = 1.0

        This provides steady linear growth from 0.7 to 1.0 as VMAF approaches perfect quality.
        """
        quality_component = 0.7 + 0.3 * min(1.0, vmaf_excess / max_vmaf_excess)

        # Use codec-aware normalized compression scoring
        # This adjusts expectations based on codec efficiency and mode difficulty
        normalized_score, total_difficulty, explanation = calculate_normalized_compression_score(
            actual_compression_rate=compression_rate,
            codec=target_codec,
            mode=codec_mode,
            target_bitrate=target_bitrate,
            optimal_bitrate=None  # Will use default expectations
        )

        # The normalized score is already in the 0-1.3 range
        compression_component = normalized_score

        compression_ratio = 1 / compression_rate if compression_rate > 0 else 1.0
        reason_suffix = "" if compression_ratio < 10 else " (excellent compression)"

        """
        Apply codec-aware weight adjustments and difficulty bonuses:
        - For CBR/VBR modes: shifts weight toward quality (harder to compress)
        - For less efficient codecs (H.264): slightly boosts quality importance
        - Adds difficulty bonus for harder codec/mode combinations

        This ensures miners with harder tasks (H.264 CBR) get fair scores
        compared to easier tasks (AV1 CRF) when both perform equally well.
        """
        final_score, adjustment_explanation = apply_codec_aware_scoring(
            base_compression_component=compression_component,
            base_quality_component=quality_component,
            codec=target_codec,
            mode=codec_mode,
            compression_weight=compression_weight,
            quality_weight=quality_weight
        )

        final_score = min(1.0, final_score)

        codec_norm = normalize_codec_family_scoring(target_codec)
        reason = f"success{reason_suffix} [{codec_norm.upper()}/{codec_mode.upper()}] {adjustment_explanation}"

        return final_score, compression_component, quality_component, reason