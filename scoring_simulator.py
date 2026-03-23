#!/usr/bin/env python3
"""
SN85 Validator Scoring Simulator

Simulates validator scoring for compression tasks to predict performance
without actual validator submission. Use this to tune CQ values and codec choices.

Usage:
    python scoring_simulator.py --vmaf 89 --ratio 15 --threshold 89
    python scoring_simulator.py --sweep-vmaf --ratio 15 --threshold 89
"""

import math
import argparse
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ScoreResult:
    """Result of a scoring calculation."""
    final_score: float
    compression_component: float
    quality_component: float
    reason: str
    is_valid: bool  # False if below hard cutoff or no compression


def calculate_validator_score(
    vmaf_score: float,
    compression_ratio: float,
    vmaf_threshold: float,
    compression_weight: float = 0.70,
    quality_weight: float = 0.30,
    soft_threshold_margin: float = 5.0
) -> ScoreResult:
    """
    Calculate validator score matching the actual validator scoring function.

    Score formula (from scoring_function.py):
    - 70% compression weight + 30% quality weight
    - Score = 0 if VMAF < (threshold - 5)
    - Score = 0 if compression_rate >= 0.80 (< 1.25x compression)

    Args:
        vmaf_score: VMAF quality score (0-100)
        compression_ratio: Compression ratio (e.g., 15 for 15x smaller)
        vmaf_threshold: Required VMAF threshold (85, 89, or 93)
        compression_weight: Weight for compression (default 0.70)
        quality_weight: Weight for quality (default 0.30)
        soft_threshold_margin: Points below threshold before hard cutoff (default 5)

    Returns:
        ScoreResult with detailed breakdown
    """
    # Validate weights
    assert abs(compression_weight + quality_weight - 1.0) < 0.01, "Weights must sum to 1.0"

    hard_cutoff = vmaf_threshold - soft_threshold_margin
    compression_rate = 1.0 / compression_ratio if compression_ratio > 0 else 1.0

    # CASE 0: No meaningful compression
    if compression_rate >= 0.80:
        return ScoreResult(
            final_score=0.0,
            compression_component=0.0,
            quality_component=0.0,
            reason=f"FAIL: No meaningful compression (ratio: {compression_ratio:.2f}x < 1.25x)",
            is_valid=False
        )

    # CASE 1: Below hard cutoff
    if vmaf_score < hard_cutoff:
        return ScoreResult(
            final_score=0.0,
            compression_component=0.0,
            quality_component=0.0,
            reason=f"FAIL: VMAF {vmaf_score:.2f} below hard cutoff ({hard_cutoff:.2f})",
            is_valid=False
        )

    # CASE 2: Soft zone (threshold-5 to threshold)
    if vmaf_score < vmaf_threshold:
        soft_zone_position = (vmaf_score - hard_cutoff) / soft_threshold_margin
        quality_factor = 0.7 * (soft_zone_position ** 2)

        # Compression component
        if compression_ratio <= 20:
            compression_component = ((compression_ratio - 1) / 19) ** 1.5
        else:
            compression_component = 1.0 + 0.3 * math.log(compression_ratio / 20)

        normalization_factor = 1.12
        final_score = (compression_component * quality_factor) / normalization_factor

        return ScoreResult(
            final_score=min(1.0, final_score),
            compression_component=compression_component,
            quality_component=quality_factor,
            reason=f"SOFT: VMAF {vmaf_score:.2f} in soft zone (quality_factor={quality_factor:.2f})",
            is_valid=True
        )

    # CASE 3: Above threshold - Full scoring
    vmaf_excess = vmaf_score - vmaf_threshold
    max_vmaf_excess = 100 - vmaf_threshold
    quality_component = 0.7 + 0.3 * min(1.0, vmaf_excess / max_vmaf_excess)

    # Compression component
    if compression_ratio <= 20:
        compression_component = ((compression_ratio - 1.25) / 18.75) ** 0.9
    else:
        compression_component = 1.0 + 0.1 * math.log(compression_ratio / 20)

    normalization_factor = 1.12
    final_score = (compression_weight * compression_component +
                  quality_weight * quality_component) / normalization_factor

    reason_suffix = " (excellent)" if compression_ratio >= 10 else ""

    return ScoreResult(
        final_score=min(1.0, final_score),
        compression_component=compression_component,
        quality_component=quality_component,
        reason=f"SUCCESS{reason_suffix}: VMAF {vmaf_score:.2f} >= {vmaf_threshold}",
        is_valid=True
    )


def print_score_table(threshold: int):
    """Print a scoring table for different VMAF/compression combinations."""
    print(f"\n{'='*80}")
    print(f"Validator Scoring Table - VMAF Threshold: {threshold}")
    print(f"{'='*80}")
    print(f"Formula: Score = (0.7 * compression + 0.3 * quality) / 1.12")
    print(f"{'='*80}\n")

    vmaf_values = list(range(threshold - 5, min(100, threshold + 11), 1))
    compression_ratios = [2, 3, 5, 7, 10, 12, 15, 18, 20, 25, 30, 40]

    # Header
    header = "VMAF  | " + " | ".join(f"{r:>4}x" for r in compression_ratios)
    print(header)
    print("-" * len(header))

    # Rows
    for vmaf in vmaf_values:
        row = f"{vmaf:>4}  |"
        for ratio in compression_ratios:
            result = calculate_validator_score(vmaf, ratio, threshold)
            if result.is_valid:
                row += f" {result.final_score:.2f} |"
            else:
                row += "   -  |"
        print(row)

    print(f"\n{'='*80}")
    print("Notes:")
    print("  -  = Score is 0 (below hard cutoff or < 1.25x compression)")
    print("  Sweet spot: 15-20x compression at target VMAF = ~0.79 score")
    print(f"{'='*80}\n")


def analyze_sweet_spot(threshold: int):
    """Analyze the competitive sweet spots for given threshold."""
    print(f"\n{'='*80}")
    print(f"Competitive Analysis - Threshold: {threshold}")
    print(f"{'='*80}\n")

    # Fixed variables
    vmaf_target = threshold

    print(f"At target VMAF = {vmaf_target} (exact threshold):\n")
    ratios = [5, 10, 12, 15, 18, 20, 25, 30]

    print(f"{'Ratio':>8} {'Comp%':>8} {'Qual%':>8} {'Score':>8} {'Rank':>10}")
    print("-" * 50)

    results = []
    for ratio in ratios:
        result = calculate_validator_score(vmaf_target, ratio, threshold)
        rank_score = result.final_score
        results.append((ratio, result))

    # Sort by score
    results.sort(key=lambda x: x[1].final_score, reverse=True)

    for ratio, result in sorted(results, key=lambda x: x[0]):
        rank = "★ BEST" if result.final_score >= 0.75 else "good" if result.final_score >= 0.70 else "fair"
        print(f"{ratio:>8}x {result.compression_component:>8.2f} {result.quality_component:>8.2f} "
              f"{result.final_score:>8.2f} {rank:>10}")

    print(f"\n{'='*80}")
    print("Recommendations:")
    print("  ★ Target 15-20x compression ratio for optimal validator score")
    print("  ★ Aim for VMAF = threshold + 1-3 for safety margin")
    print("  ★ Below 10x compression: difficulty achieving competitive scores")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="SN85 Validator Scoring Simulator"
    )
    parser.add_argument("--vmaf", type=float, help="VMAF score to evaluate")
    parser.add_argument("--ratio", type=float, help="Compression ratio (e.g., 15 for 15x)")
    parser.add_argument("--threshold", type=int, default=89,
                       choices=[85, 89, 93], help="VMAF threshold (default: 89)")
    parser.add_argument("--table", action="store_true",
                       help="Print full scoring table")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze sweet spot for given threshold")

    args = parser.parse_args()

    if args.table:
        print_score_table(args.threshold)
        return

    if args.analyze:
        analyze_sweet_spot(args.threshold)
        return

    if args.vmaf is not None and args.ratio is not None:
        result = calculate_validator_score(args.vmaf, args.ratio, args.threshold)
        print(f"\n{'='*60}")
        print(f"Score Calculation Result")
        print(f"{'='*60}")
        print(f"VMAF Score:          {args.vmaf}")
        print(f"Compression Ratio:   {args.ratio}x")
        print(f"VMAF Threshold:      {args.threshold}")
        print(f"{'='*60}")
        print(f"Final Score:         {result.final_score:.4f}")
        print(f"Compression Comp:    {result.compression_component:.4f}")
        print(f"Quality Component:   {result.quality_component:.4f}")
        print(f"Status:              {'VALID' if result.is_valid else 'INVALID'}")
        print(f"{'='*60}")
        print(f"Reason: {result.reason}")
        print(f"{'='*60}\n")
    else:
        print("Usage examples:")
        print("  python scoring_simulator.py --vmaf 89 --ratio 15 --threshold 89")
        print("  python scoring_simulator.py --table --threshold 89")
        print("  python scoring_simulator.py --analyze --threshold 93")


if __name__ == "__main__":
    main()
