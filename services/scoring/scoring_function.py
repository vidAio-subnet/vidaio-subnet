import math


def calculate_compression_score(
    vmaf_score: float,
    compression_rate: float,
    vmaf_threshold: float,
    compression_weight: float = 0.70,
    quality_weight: float = 0.30,
    soft_threshold_margin: float = 5.0) -> tuple[float, float, float, str]:
    """
    Calculate compression score that rewards hitting VMAF threshold with good compression.
    
    Scoring Philosophy:
    - Compression is primary goal (70% weight by default)
    - VMAF at/above threshold is rewarded with diminishing returns (30% weight)
    - 15x compression ratio = 1.0 compression component
    - Exceeding VMAF threshold gives modest bonus (not penalty)
    
    Three scoring zones:
    1. Below hard cutoff (threshold - 5): Score = 0
    2. Soft zone (threshold - 5 to threshold): Gradual recovery
    3. Above threshold: Full scoring with quality bonus
    
    Args:
        vmaf_score: VMAF quality score (0-100)
        compression_rate: Size ratio compressed/original (0-1, lower is better)
        vmaf_threshold: Required VMAF threshold (e.g., 85, 90, 95)
        compression_weight: Weight for compression component (default: 0.70)
        quality_weight: Weight for quality component (default: 0.30)
        soft_threshold_margin: Points below threshold before hard cutoff (default: 5.0)
    
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
    # CASE 1: Below Hard Cutoff - Immediate Failure
    # ========================================================================
    # If VMAF is more than 5 points below threshold, quality is unacceptable
    if vmaf_score < hard_cutoff:
        return 0.0, 0.0, 0.0, f"VMAF {vmaf_score:.1f} below hard cutoff ({hard_cutoff:.1f})"
    
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
        Quality factor uses quadratic function: f(x) = x²
        - At hard cutoff (x=0): factor = 0
        - At threshold (x=1): factor = 1
        - Provides smooth transition from 0 to 1
        """
        quality_factor = soft_zone_position ** 2
        
        # Calculate compression component
        if compression_rate >= 0.95:
            # Less than 1.05x compression - essentially no compression
            compression_component = 0.0
        else:
            compression_ratio = 1 / compression_rate
            
            if compression_ratio <= 15:
                """
                Compression scoring for 1x to 15x:
                f(r) = ((r - 1) / 14) ^ 1.5
                
                Where r is compression ratio (e.g., 5 for 5x compression)
                - At 1x: component = 0
                - At 2x: component ≈ 0.26
                - At 5x: component ≈ 0.69
                - At 10x: component ≈ 0.92
                - At 15x: component = 1.0
                
                The exponent 1.5 provides:
                - Slow growth initially (encouraging minimum viable compression)
                - Steeper growth in practical range (5-10x)
                - Reaches 1.0 exactly at 15x
                """
                compression_component = ((compression_ratio - 1) / 14) ** 1.5
            else:
                """
                Bonus for exceptional compression (>15x):
                f(r) = 1.0 + 0.2 * ln(r / 15)
                
                Logarithmic bonus rewards exceptional performance:
                - At 20x: component ≈ 1.06
                - At 30x: component ≈ 1.14
                - Capped at 1.2
                """
                compression_component = 1.0 + 0.2 * math.log(compression_ratio / 15)
            
            compression_component = min(1.2, compression_component)
        
        # In soft zone, both compression AND quality factor matter
        # If you're below threshold, you need good compression to recover
        final_score = compression_component * quality_factor
        
        return min(1.0, final_score), compression_component, quality_factor, f"VMAF {vmaf_score:.1f} in soft zone (quality factor: {quality_factor:.2f})"
    
    # ========================================================================
    # CASE 3: Above Threshold - Full Scoring with Quality Bonus
    # ========================================================================
    else:
        # Calculate how much VMAF exceeds threshold
        vmaf_excess = vmaf_score - vmaf_threshold
        
        """
        Quality component rewards exceeding threshold with diminishing returns:
        f(x) = 0.9 + 0.1 * (1 - e^(-x/4))
        
        Where x is VMAF points above threshold:
        - At threshold (x=0): component = 0.9
        - At threshold+1 (x=1): component ≈ 0.92
        - At threshold+2 (x=2): component ≈ 0.94
        - At threshold+5 (x=5): component ≈ 0.97
        - At threshold+10 (x=10): component ≈ 0.99
        - Asymptotically approaches 1.0
        
        This rewards quality improvements but with diminishing returns,
        ensuring compression remains the dominant factor.
        """
        quality_component = 0.9 + 0.1 * (1 - math.exp(-vmaf_excess / 4))
        
        # Calculate compression component
        if compression_rate >= 0.95:
            # Less than 1.05x compression - essentially no compression
            compression_component = 0.0
            reason_suffix = " (WARNING: minimal compression)"
            
        elif compression_rate >= 0.80:
            """
            Poor compression zone (1.25x or less):
            f(r) = ((r - 1)² * 0.4
            
            Where r is compression ratio:
            - At 1.0x: component = 0
            - At 1.125x: component ≈ 0.004
            - At 1.25x: component = 0.016
            - Maximum 0.4 at this range
            
            Heavy quadratic penalty discourages minimal compression.
            """
            compression_ratio = 1 / compression_rate
            compression_component = (compression_ratio - 1) ** 2 * 0.4
            reason_suffix = " (poor compression)"
            
        else:
            # Good compression zone
            compression_ratio = 1 / compression_rate
            
            if compression_ratio <= 15:
                """
                Good compression scoring (1.25x to 15x):
                f(r) = ((r - 1.25) / 13.75) ^ 1.2
                
                Starting from 1.25x to give smooth transition from poor zone:
                - At 1.25x: component = 0.025
                - At 2x: component ≈ 0.26
                - At 5x: component ≈ 0.69
                - At 10x: component ≈ 0.92
                - At 15x: component = 1.0
                
                The exponent 1.2 provides balanced reward curve.
                """
                compression_component = ((compression_ratio - 1.25) / 13.75) ** 1.2 + 0.025
            else:
                """
                Exceptional compression bonus (>15x):
                f(r) = 1.0 + 0.3 * ln(r / 15)
                
                Logarithmic bonus up to 1.3 cap:
                - At 20x: component ≈ 1.09
                - At 30x: component ≈ 1.21
                - At 45x: component ≈ 1.30 (capped)
                """
                compression_component = 1.0 + 0.3 * math.log(compression_ratio / 15)
            
            compression_component = min(1.3, compression_component)
            reason_suffix = "" if compression_ratio < 10 else " (excellent compression)"
        
        """
        Final score is weighted combination:
        final_score = w_c * compression_component + w_q * quality_component
        
        Default: 70% compression + 30% quality
        Capped at 1.0 to maintain normalized scoring
        """
        final_score = (compression_weight * compression_component + 
                      quality_weight * quality_component)
        
        final_score = min(1.0, final_score)

        return final_score, compression_component, quality_component, f"success{reason_suffix}"