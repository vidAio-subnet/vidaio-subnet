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
    - 100x compression ratio = 1.0 compression component
    - Exceeding VMAF threshold gives modest bonus (not penalty)
    
    Three scoring zones:
    1. Below hard cutoff (threshold - 5): Score = 0
    2. Soft zone (threshold - 5 to threshold): Gradual recovery
    3. Above threshold: Full scoring with quality bonus
    
    Args:
        vmaf_score: VMAF quality score (0-100)
        compression_rate: Size ratio compressed/original (0-1, lower is better)
        vmaf_threshold: Required VMAF threshold (e.g., 85, 89, 93)
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
            f(r) = 0.7 * sqrt((r - 1) / 19)
            
            Where r is compression ratio (e.g., 5 for 5x compression)
            - At 1x: component = 0
            - At 2x: component ≈ 0.16
            - At 5x: component ≈ 0.32
            - At 10x: component ≈ 0.48
            - At 15x: component ≈ 0.60
            - At 20x: component = 0.70
            
            Square root provides concave growth rewarding early compression.
            Reaches 0.70 at 20x, leaving room for 20-100x incentives.
            """
            compression_component = 0.7 * math.sqrt((compression_ratio - 1) / 19)
        elif compression_ratio <= 100:
            """
            Extended compression scoring for 20x to 100x:
            f(r) = 0.7 + 0.3 * sqrt((r - 20) / 80)
            
            Continuous at 20x boundary (0.70), reaching 1.0 at 100x:
            - At 20x: component = 0.70
            - At 30x: component ≈ 0.81
            - At 50x: component ≈ 0.88
            - At 75x: component ≈ 0.95
            - At 100x: component = 1.00
            
            Square root ensures meaningful incentive throughout 20-100x range.
            """
            compression_component = 0.7 + 0.3 * math.sqrt((compression_ratio - 20) / 80)
        else:
            # Beyond 100x: cap at 1.0 to prevent extreme over-compression
            compression_component = 1.0
        
        # In soft zone, both compression AND quality factor matter
        # If you're below threshold, you need good compression to recover
        final_score = compression_component * quality_factor
        
        return min(1.0, final_score), compression_component, quality_factor, f"VMAF {vmaf_score:.2f} in soft zone (quality factor: {quality_factor:.2f})"
    
    # ========================================================================
    # CASE 3: Above Threshold - Full Scoring with Quality Bonus
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
        
        # Calculate compression component (compression_rate < 0.80 guaranteed by CASE 0)
        compression_ratio = 1 / compression_rate
        
        if compression_ratio <= 20:
            """
            Compression scoring for 1.25x to 20x:
            f(r) = 0.7 * sqrt((r - 1) / 19)
            
            - At 1.25x: component ≈ 0.08
            - At 2x: component ≈ 0.16
            - At 5x: component ≈ 0.32
            - At 10x: component ≈ 0.48
            - At 15x: component ≈ 0.60
            - At 20x: component = 0.70
            
            Square root provides concave growth rewarding early compression.
            Reaches 0.70 at 20x, leaving room for 20-100x incentives.
            """
            compression_component = 0.7 * math.sqrt((compression_ratio - 1) / 19)
        elif compression_ratio <= 100:
            """
            Extended compression scoring for 20x to 100x:
            f(r) = 0.7 + 0.3 * sqrt((r - 20) / 80)
            
            Continuous at 20x boundary (0.70), reaching 1.0 at 100x:
            - At 20x: component = 0.70
            - At 30x: component ≈ 0.81
            - At 50x: component ≈ 0.88
            - At 75x: component ≈ 0.95
            - At 100x: component = 1.00
            
            Square root ensures meaningful incentive throughout 20-100x range.
            """
            compression_component = 0.7 + 0.3 * math.sqrt((compression_ratio - 20) / 80)
        else:
            # Beyond 100x: cap at 1.0 to prevent extreme over-compression
            compression_component = 1.0
        
        reason_suffix = "" if compression_ratio < 50 else " (excellent compression)"
        
        """
        Final score is weighted combination:
        final_score = w_c * compression_component + w_q * quality_component
        
        Default: 70% compression + 30% quality
        Capped at 1.0 — only achievable with ~100x compression AND perfect VMAF
        """
        final_score = (compression_weight * compression_component + 
                      quality_weight * quality_component)
        
        final_score = min(1.0, final_score)

        return final_score, compression_component, quality_component, f"success{reason_suffix}"


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    VMAF_THRESHOLD = 85.0

    # ── Plot 1: final_score vs compression ratio for different VMAF values ──
    ratios = np.linspace(1.3, 150, 500)
    vmaf_values = [80, 82, 85, 90, 95, 100]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax1 = axes[0]
    for vmaf in vmaf_values:
        scores = []
        for r in ratios:
            rate = 1.0 / r
            s, _, _, _ = calculate_compression_score(vmaf, rate, VMAF_THRESHOLD)
            scores.append(s)
        ax1.plot(ratios, scores, label=f"VMAF={vmaf}")

    ax1.set_xlabel("Compression Ratio (x)")
    ax1.set_ylabel("Final Score")
    ax1.set_title("Final Score vs Compression Ratio\n(across VMAF values, threshold=85)")
    ax1.legend()
    ax1.set_xlim(1, 150)
    ax1.set_ylim(-0.02, 1.05)
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
    ax1.axvline(x=20, color="gray", linestyle=":", alpha=0.4, label="20x")
    ax1.axvline(x=100, color="gray", linestyle=":", alpha=0.4, label="100x")
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: final_score vs VMAF for different compression ratios ──
    vmafs = np.linspace(75, 100, 500)
    ratio_values = [2, 5, 10, 20, 50, 100]

    ax2 = axes[1]
    for r in ratio_values:
        scores = []
        rate = 1.0 / r
        for v in vmafs:
            s, _, _, _ = calculate_compression_score(v, rate, VMAF_THRESHOLD)
            scores.append(s)
        ax2.plot(vmafs, scores, label=f"Ratio={r}x")

    ax2.set_xlabel("VMAF Score")
    ax2.set_ylabel("Final Score")
    ax2.set_title("Final Score vs VMAF\n(across compression ratios, threshold=85)")
    ax2.legend()
    ax2.set_xlim(75, 100)
    ax2.set_ylim(-0.02, 1.05)
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
    ax2.axvline(x=VMAF_THRESHOLD, color="red", linestyle=":", alpha=0.4, label="threshold")
    ax2.axvline(x=VMAF_THRESHOLD - 5, color="orange", linestyle=":", alpha=0.4, label="hard cutoff")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("scoring_curve.png", dpi=150)
    print("Saved scoring_curve.png")
    plt.show()