#!/usr/bin/env python3
"""
SN85 Compression Service Benchmark Tool

Benchmarks video compression with different codec/CQ combinations
to optimize for validator scoring (70% compression ratio + 30% VMAF quality).

Usage:
    python benchmark_compression.py test_video.mp4
    python benchmark_compression.py test_video.mp4 --codec av1_nvenc --quality High
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Result of a single compression benchmark."""
    codec: str
    quality_level: str
    cq_value: int
    scene_type: str

    # File metrics
    input_size_mb: float = 0.0
    output_size_mb: float = 0.0
    compression_ratio: float = 0.0
    compression_rate: float = 0.0

    # Quality metrics
    vmaf_score: Optional[float] = None

    # Performance
    encode_time_seconds: float = 0.0
    realtime_factor: float = 0.0  # encode_time / video_duration

    # Scoring
    predicted_score: float = 0.0
    compression_component: float = 0.0
    quality_component: float = 0.0


def get_video_duration(video_path: str) -> Optional[float]:
    """Get video duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting duration: {e}")
    return None


def calculate_vmaf(reference_path: str, distorted_path: str, num_samples: int = 3) -> Optional[float]:
    """
    Calculate VMAF score using ffmpeg-quality-metrics or ffmpeg libvmaf.
    Uses sampling for faster results.
    """
    # Try ffmpeg-quality-metrics first
    try:
        import ffmpeg_quality_metrics
        metrics = ffmpeg_quality_metrics.FFmpegQualityMetrics(
            distorted_path, reference_path
        )
        data = metrics.calculate([
            "vmaf", "psnr", "ssim"
        ])
        if 'vmaf' in data and data['vmaf']:
            # Average VMAF across all frames
            vmaf_values = [float(f.get('vmaf', 0)) for f in data['vmaf'] if f.get('vmaf')]
            return sum(vmaf_values) / len(vmaf_values) if vmaf_values else None
    except ImportError:
        pass
    except Exception as e:
        print(f"ffmpeg-quality-metrics failed: {e}")

    # Fallback to direct ffmpeg libvmaf
    try:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", distorted_path,
            "-i", reference_path,
            "-lavfi", f"libvmaf=log_path=/dev/null:log_fmt=json:n_subsample={max(1, 24 // num_samples)}",
            "-f", "null", "-"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            # Parse VMAF from stderr or stdout (libvmaf outputs differently per version)
            for output in [result.stderr, result.stdout]:
                for line in output.split('\n'):
                    if 'VMAF score:' in line:
                        try:
                            return float(line.split('VMAF score:')[1].strip().split()[0])
                        except (ValueError, IndexError):
                            continue
    except Exception as e:
        print(f"libvmaf calculation failed: {e}")

    return None


def calculate_compression_score(
    vmaf_score: float,
    compression_rate: float,
    vmaf_threshold: float
) -> Tuple[float, float, float, str]:
    """
    Calculate predicted validator score.
    Based on services/scoring/scoring_function.py
    """
    soft_threshold_margin = 5.0
    compression_weight = 0.70
    quality_weight = 0.30

    hard_cutoff = vmaf_threshold - soft_threshold_margin

    # Case 0: No meaningful compression
    if compression_rate >= 0.80:
        return 0.0, 0.0, 0.0, "No meaningful compression (need < 0.80 rate)"

    # Case 1: Below hard cutoff
    if vmaf_score < hard_cutoff:
        return 0.0, 0.0, 0.0, f"VMAF {vmaf_score:.1f} below hard cutoff ({hard_cutoff:.1f})"

    compression_ratio = 1 / compression_rate

    # Compression component calculation
    if compression_ratio <= 20:
        if vmaf_score < vmaf_threshold:
            # Soft zone
            compression_component = ((compression_ratio - 1) / 19) ** 1.5
        else:
            # Above threshold
            compression_component = ((compression_ratio - 1.25) / 18.75) ** 0.9
    else:
        if vmaf_score < vmaf_threshold:
            compression_component = 1.0 + 0.3 * __import__('math').log(compression_ratio / 20)
        else:
            compression_component = 1.0 + 0.1 * __import__('math').log(compression_ratio / 20)

    # Quality component
    if vmaf_score < vmaf_threshold:
        # Soft zone - quadratic recovery
        soft_zone_position = (vmaf_score - hard_cutoff) / soft_threshold_margin
        quality_component = 0.7 * (soft_zone_position ** 2)
    else:
        # Above threshold
        max_vmaf_excess = 100 - vmaf_threshold
        vmaf_excess = vmaf_score - vmaf_threshold
        quality_component = 0.7 + 0.3 * min(1.0, vmaf_excess / max_vmaf_excess)

    # Final score
    if vmaf_score < vmaf_threshold:
        normalization_factor = 1.12
        final_score = (compression_component * quality_component) / normalization_factor
    else:
        normalization_factor = 1.12
        final_score = (compression_weight * compression_component +
                      quality_weight * quality_component) / normalization_factor

    final_score = min(1.0, final_score)

    reason = f"VMAF={vmaf_score:.1f}, Ratio={compression_ratio:.1f}x"
    return final_score, compression_component, quality_component, reason


def run_benchmark(
    video_path: str,
    codec: str,
    quality_level: str,
    cq_value: int,
    scene_type: str = "default",
    use_gpu: bool = True
) -> Optional[BenchmarkResult]:
    """Run a single compression benchmark."""

    input_path = Path(video_path)
    if not input_path.exists():
        print(f"Video not found: {video_path}")
        return None

    input_size = input_path.stat().st_size / (1024 * 1024)
    duration = get_video_duration(video_path) or 0.0

    # Build ffmpeg command
    output_path = input_path.parent / f"{input_path.stem}_bench_{codec}_cq{cq_value}.mp4"

    # Map codec to encoder
    codec_map = {
        'av1': 'av1_nvenc' if use_gpu else 'libsvtav1',
        'hevc': 'hevc_nvenc' if use_gpu else 'libx265',
        'h264': 'h264_nvenc' if use_gpu else 'libx264',
    }
    encoder = codec_map.get(codec, codec)

    # Build command
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-c:v", encoder,
    ]

    # Add codec-specific settings
    if 'nvenc' in encoder:
        if 'av1' in encoder:
            cmd.extend(["-preset", "p2"])
            cmd.extend(["-rc", "constqp"])
            cmd.extend(["-qp", str(cq_value)])
        else:
            cmd.extend(["-preset", "slow"])
            cmd.extend(["-rc", "constqp"])
            cmd.extend(["-qp", str(cq_value)])
    else:
        # CPU codecs
        if encoder == 'libsvtav1':
            cmd.extend(["-preset", "6"])  # SVT-AV1 uses numeric presets 0-12
            cmd.extend(["-crf", str(cq_value)])
        elif encoder in ['libx265', 'libx264']:
            cmd.extend(["-preset", "medium"])
            cmd.extend(["-crf", str(cq_value)])

    # Add output
    cmd.extend(["-c:a", "copy", str(output_path)])

    print(f"  Running: {' '.join(cmd[-6:])}...")

    # Execute encoding
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        encode_time = time.time() - start_time

        if result.returncode != 0:
            print(f"  Encode failed: {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        print("  Encode timeout!")
        return None

    # Calculate results
    if output_path.exists():
        output_size = output_path.stat().st_size / (1024 * 1024)
        compression_ratio = input_size / output_size if output_size > 0 else 0
        compression_rate = output_size / input_size if input_size > 0 else 1.0

        # Calculate VMAF
        print(f"  Calculating VMAF...")
        vmaf = calculate_vmaf(video_path, str(output_path))

        # Calculate predicted score
        vmaf_threshold = {'High': 93, 'Medium': 89, 'Low': 85}.get(quality_level, 89)
        if vmaf:
            final_score, comp_comp, qual_comp, reason = calculate_compression_score(
                vmaf, compression_rate, vmaf_threshold
            )
        else:
            final_score, comp_comp, qual_comp, reason = 0, 0, 0, "VMAF calculation failed"

        result = BenchmarkResult(
            codec=codec,
            quality_level=quality_level,
            cq_value=cq_value,
            scene_type=scene_type,
            input_size_mb=input_size,
            output_size_mb=output_size,
            compression_ratio=compression_ratio,
            compression_rate=compression_rate,
            vmaf_score=vmaf,
            encode_time_seconds=encode_time,
            realtime_factor=encode_time / duration if duration > 0 else 0,
            predicted_score=final_score,
            compression_component=comp_comp,
            quality_component=qual_comp
        )

        # Cleanup
        output_path.unlink(missing_ok=True)

        return result

    return None


def find_optimal_cq(
    video_path: str,
    codec: str,
    quality_level: str,
    target_vmaf: float,
    scene_type: str = "default"
) -> List[BenchmarkResult]:
    """Binary search for optimal CQ value for target VMAF."""
    print(f"\nFinding optimal CQ for {codec} / {quality_level} / target VMAF {target_vmaf}")

    results = []

    # Test range of CQ values
    if quality_level == "High":
        cq_range = [18, 20, 22, 24]
    elif quality_level == "Medium":
        cq_range = [22, 24, 26, 28]
    else:
        cq_range = [26, 28, 30, 32]

    for cq in cq_range:
        result = run_benchmark(video_path, codec, quality_level, cq, scene_type)
        if result:
            results.append(result)
            vmaf_str = f"{result.vmaf_score:.1f}" if result.vmaf_score else "N/A"
            print(f"  CQ={cq}: VMAF={vmaf_str}, Ratio={result.compression_ratio:.1f}x, "
                  f"Score={result.predicted_score:.3f}, Time={result.encode_time_seconds:.1f}s")

    return results


def print_summary(results: List[BenchmarkResult]):
    """Print benchmark summary table."""
    print("\n" + "="*100)
    print("BENCHMARK SUMMARY")
    print("="*100)
    print(f"{'Codec':<12} {'Quality':<8} {'CQ':<4} {'VMAF':>6} {'Ratio':>8} "
          f"{'Rate':>6} {'Score':>6} {'Comp':>6} {'Qual':>6} {'Time':>8}")
    print("-"*100)

    sorted_results = sorted(results, key=lambda x: x.predicted_score, reverse=True)

    for r in sorted_results:
        vmaf_str = f"{r.vmaf_score:.1f}" if r.vmaf_score else "N/A"
        print(f"{r.codec:<12} {r.quality_level:<8} {r.cq_value:<4} {vmaf_str:>6} "
              f"{r.compression_ratio:>8.1f}x {r.compression_rate:>6.2f} "
              f"{r.predicted_score:>6.3f} {r.compression_component:>6.3f} "
              f"{r.quality_component:>6.3f} {r.encode_time_seconds:>7.1f}s")

    print("="*100)

    # Best by score
    if sorted_results:
        best = sorted_results[0]
        vmaf_str = f"{best.vmaf_score:.1f}" if best.vmaf_score else "N/A"
        print(f"\nBest overall: {best.codec} CQ={best.cq_value} "
              f"Score={best.predicted_score:.3f} (VMAF={vmaf_str}, "
              f"{best.compression_ratio:.1f}x compression)")


def main():
    parser = argparse.ArgumentParser(description="SN85 Compression Benchmark Tool")
    parser.add_argument("video", help="Path to test video")
    parser.add_argument("--codec", choices=["av1", "hevc", "h264", "av1_nvenc", "hevc_nvenc"],
                       default="av1", help="Codec to test")
    parser.add_argument("--quality", choices=["High", "Medium", "Low"],
                       default="High", help="Quality level")
    parser.add_argument("--cq", type=int, nargs="+", help="Specific CQ values to test")
    parser.add_argument("--range", action="store_true",
                       help="Test full CQ range to find optimum")
    parser.add_argument("--cpu", action="store_true",
                       help="Use CPU codecs instead of GPU")
    parser.add_argument("--output", "-o", help="Save results to JSON file")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Video not found: {args.video}")
        sys.exit(1)

    # Check for ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg not found in PATH")
        sys.exit(1)

    print(f"SN85 Compression Benchmark")
    print(f"Video: {args.video}")
    print(f"Codec: {args.codec}")
    print(f"Quality Level: {args.quality}")
    print(f"GPU: {'No (CPU mode)' if args.cpu else 'Yes (NVENC if available)'}")

    results = []

    if args.cq:
        # Test specific CQ values
        for cq in args.cq:
            result = run_benchmark(args.video, args.codec, args.quality, cq, use_gpu=not args.cpu)
            if result:
                results.append(result)
    elif args.range:
        # Find optimal
        target_vmaf = {'High': 93, 'Medium': 89, 'Low': 85}.get(args.quality, 89)
        results = find_optimal_cq(args.video, args.codec, args.quality, target_vmaf,
                                  use_gpu=not args.cpu)
    else:
        # Default test set
        cq_values = [20, 22, 24] if args.quality == "High" else [24, 26, 28]
        for cq in cq_values:
            result = run_benchmark(args.video, args.codec, args.quality, cq, use_gpu=not args.cpu)
            if result:
                results.append(result)

    if results:
        print_summary(results)

        if args.output:
            output_data = {
                "video": args.video,
                "codec": args.codec,
                "quality": args.quality,
                "timestamp": time.time(),
                "results": [
                    {
                        "codec": r.codec,
                        "quality_level": r.quality_level,
                        "cq_value": r.cq_value,
                        "vmaf_score": r.vmaf_score,
                        "compression_ratio": r.compression_ratio,
                        "compression_rate": r.compression_rate,
                        "predicted_score": r.predicted_score,
                        "encode_time_seconds": r.encode_time_seconds,
                    }
                    for r in results
                ]
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to {args.output}")
    else:
        print("\nNo successful benchmarks completed.")


if __name__ == "__main__":
    main()
