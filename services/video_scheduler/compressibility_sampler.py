"""
Latin Hypercube Sampling (LHS) module for video compressibility factor space.

Generates diverse FFmpeg parameter combinations that cover the full range of
video compressibility factors: resolution, noise, QP, FPS, blur/sharpen,
speed, denoise, codec, and H.264 profile.

Usage:
    from compressibility_sampler import generate_lhs_samples, build_ffmpeg_command

    samples = generate_lhs_samples(n_samples=50)
    for sample in samples:
        cmd = build_ffmpeg_command(sample, "input.mp4", "output.mp4")
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from scipy.stats.qmc import LatinHypercube

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Factor definitions — each maps a factor name to its discrete levels.
# Order matters: it matches the LHS dimension index.
# ──────────────────────────────────────────────────────────────────────

COMPRESSIBILITY_FACTORS = {
    "resolution": [
        (854, 480),
        (1280, 720),
        (1920, 1080),
        (3840, 2160),
    ],
    "noise": [0, 10, 20, 35, 50],
    "qp": [8, 15, 23, 30, 38],
    "fps": [15, 24, 30, 60],
    "blur_sharpen": [
        "gblur=sigma=2",       # heavy blur — most compressible
        "gblur=sigma=1",       # light blur
        None,                  # no change
        "unsharp=5:5:0.8",     # light sharpen
        "unsharp=5:5:1.5",     # heavy sharpen — least compressible
    ],
    "speed": [0.5, 1.0, 2.0],
    "denoise": [
        None,                  # no denoise
        "hqdn3d=4:4:6:6",     # light denoise
        "hqdn3d=8:8:12:12",   # heavy denoise
    ],
    "codec": [
        "h264_nvenc",
        "hevc_nvenc",
        "libvpx-vp9",
        "libsvtav1",
    ],
    "profile": [
        "baseline",
        "main",
        "high",
    ],
}


# ──────────────────────────────────────────────────────────────────────
# Codec configuration — encoder flags, rate-control, and file extension
# ──────────────────────────────────────────────────────────────────────

CODEC_CONFIG = {
    "h264_nvenc": {
        "encoder": "h264_nvenc",
        "rc_mode": "constqp",
        "qp_flag": "-qp",
        "extension": ".mp4",
        "is_nvenc": True,
        "extra_flags": [],
    },
    "hevc_nvenc": {
        "encoder": "hevc_nvenc",
        "rc_mode": "constqp",
        "qp_flag": "-qp",
        "extension": ".mp4",
        "is_nvenc": True,
        "extra_flags": [],
    },
    "libvpx-vp9": {
        "encoder": "libvpx-vp9",
        "rc_mode": "crf",
        "qp_flag": "-crf",
        "extension": ".webm",
        "is_nvenc": False,
        "extra_flags": ["-b:v", "0", "-row-mt", "1", "-threads", "4"],
    },
    "libsvtav1": {
        "encoder": "libsvtav1",
        "rc_mode": "crf",
        "qp_flag": "-crf",
        "extension": ".mp4",
        "is_nvenc": False,
        "extra_flags": ["-preset", "8"],
    },
}

# CPU fallback encoders when GPU encoders are unavailable
CPU_FALLBACKS = {
    "h264_nvenc": "libx264",
    "hevc_nvenc": "libx265",
}

CPU_FALLBACK_CONFIG = {
    "libx264": {
        "encoder": "libx264",
        "rc_mode": "crf",
        "qp_flag": "-crf",
        "extension": ".mp4",
        "is_nvenc": False,
        "extra_flags": ["-preset", "fast"],
    },
    "libx265": {
        "encoder": "libx265",
        "rc_mode": "crf",
        "qp_flag": "-crf",
        "extension": ".mp4",
        "is_nvenc": False,
        "extra_flags": ["-preset", "fast"],
    },
}


def generate_lhs_samples(
    n_samples: int = None,
    seed: int = None,
    available_encoders: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate a Latin Hypercube Sample across all compressibility factors.

    Each returned sample is a dictionary mapping factor names to concrete values.
    LHS guarantees that every level of every factor appears at least
    ``n_samples // max_levels`` times, giving thorough coverage.

    Args:
        n_samples: Number of parameter combinations to generate.
                   Defaults to ``LHS_SAMPLE_COUNT`` env var or 50.
        seed: RNG seed for reproducibility (None = random each call).
        available_encoders: If provided, restrict codec choices to only
                            these encoder names (e.g. after probing FFmpeg).

    Returns:
        List of sample dicts ready for ``build_ffmpeg_command()``.
    """
    if n_samples is None:
        n_samples = int(os.getenv("LHS_SAMPLE_COUNT", "50"))

    factors = dict(COMPRESSIBILITY_FACTORS)  # shallow copy

    # Filter codecs to only those available
    if available_encoders is not None:
        valid_codecs = []
        for codec in factors["codec"]:
            if codec in available_encoders:
                valid_codecs.append(codec)
            elif codec in CPU_FALLBACKS and CPU_FALLBACKS[codec] in available_encoders:
                # Replace NVENC codec with its CPU fallback
                valid_codecs.append(CPU_FALLBACKS[codec])
            # else: skip this codec entirely
        if valid_codecs:
            factors["codec"] = list(set(valid_codecs))  # deduplicate
        else:
            logger.warning("No available encoders match any codec levels, keeping defaults")

    n_dims = len(factors)
    sampler = LatinHypercube(d=n_dims, seed=seed)
    lhs_raw = sampler.random(n=n_samples)  # shape (n_samples, n_dims)

    factor_names = list(factors.keys())
    samples: List[Dict[str, Any]] = []

    for row in lhs_raw:
        sample: Dict[str, Any] = {}
        for dim_idx, name in enumerate(factor_names):
            levels = factors[name]
            level_idx = int(row[dim_idx] * len(levels))
            level_idx = min(level_idx, len(levels) - 1)
            sample[name] = levels[level_idx]

        # Conditional logic: profile only applies to H.264 variants
        if sample["codec"] not in ("h264_nvenc", "libx264"):
            sample["profile"] = None

        samples.append(sample)

    logger.info(
        "Generated %d LHS samples across %d factors (%s)",
        len(samples),
        n_dims,
        ", ".join(factor_names),
    )
    return samples


def _build_filter_chain(sample: Dict[str, Any]) -> Optional[str]:
    """
    Build a single ``-vf`` filter chain string from a sample dict.

    Chaining all CPU filters into one ``-vf`` avoids multiple
    decode→encode passes and is 2–4× faster (per the GPU doc).

    Returns None if no filters are needed.
    """
    filters: List[str] = []

    # 1. Resolution scaling
    w, h = sample["resolution"]
    filters.append(f"scale={w}:{h}:flags=lanczos")

    # 2. FPS conversion
    fps = sample["fps"]
    if fps != 30:  # 30 is the most common default
        filters.append(f"fps={fps}")

    # 3. Speed change
    speed = sample["speed"]
    if speed != 1.0:
        pts_factor = 1.0 / speed  # 2× speed → setpts=0.5*PTS
        filters.append(f"setpts={pts_factor:.2f}*PTS")

    # 4. Noise injection
    noise = sample["noise"]
    if noise > 0:
        filters.append(f"noise=alls={noise}:allf=t")

    # 5. Blur / sharpen
    blur_sharpen = sample["blur_sharpen"]
    if blur_sharpen is not None:
        filters.append(blur_sharpen)

    # 6. Denoise
    denoise = sample["denoise"]
    if denoise is not None:
        filters.append(denoise)

    return ",".join(filters) if filters else None


def get_output_extension(codec: str) -> str:
    """Return the correct file extension for a given codec."""
    cfg = CODEC_CONFIG.get(codec) or CPU_FALLBACK_CONFIG.get(codec)
    if cfg:
        return cfg["extension"]
    return ".mp4"


def build_ffmpeg_command(
    sample: Dict[str, Any],
    input_path: str,
    output_path: str,
    gpu_id: int = 0,
) -> List[str]:
    """
    Convert a sample dict into a complete FFmpeg command list.

    The command follows the GPU-optimised pipeline pattern:
      CPU decode → CPU filter chain → GPU/CPU encode

    IMPORTANT: Never uses ``-hwaccel cuda`` or ``-hwaccel cuvid``
    since A6000/A100 have no NVDEC.

    Args:
        sample: A parameter dict from ``generate_lhs_samples()``.
        input_path: Path to the input seed video.
        output_path: Path for the encoded output.
        gpu_id: GPU device index for NVENC (ignored for CPU codecs).

    Returns:
        List of command-line arguments suitable for ``subprocess.run()``.
    """
    codec_name = sample["codec"]
    cfg = CODEC_CONFIG.get(codec_name) or CPU_FALLBACK_CONFIG.get(codec_name)
    if cfg is None:
        raise ValueError(f"Unknown codec: {codec_name}")

    cmd = ["ffmpeg", "-y", "-loglevel", "warning"]

    # Input (CPU decode only — no hwaccel)
    cmd.extend(["-i", input_path])

    # Filter chain
    vf = _build_filter_chain(sample)
    if vf:
        cmd.extend(["-vf", vf])

    # Encoder
    cmd.extend(["-c:v", cfg["encoder"]])

    # GPU selection for NVENC
    if cfg["is_nvenc"]:
        cmd.extend(["-gpu", str(gpu_id)])
        # NVENC preset
        cmd.extend(["-preset", "p4"])

    # Rate control
    qp = sample["qp"]
    if cfg["rc_mode"] == "constqp":
        cmd.extend(["-rc", "constqp", cfg["qp_flag"], str(qp)])
    else:
        cmd.extend([cfg["qp_flag"], str(qp)])

    # H.264 profile (only if codec is H.264)
    if sample.get("profile") and codec_name in ("h264_nvenc", "libx264"):
        cmd.extend(["-profile:v", sample["profile"]])

    # Extra codec-specific flags
    cmd.extend(cfg["extra_flags"])

    # Strip audio (test data doesn't need it)
    cmd.append("-an")

    # Handle speed change: drop audio when PTS is altered
    # (already covered by -an above)

    # Output
    cmd.append(output_path)

    return cmd


def get_codec_type(codec: str) -> str:
    """
    Classify a codec as 'nvenc' or 'cpu' for parallelism scheduling.
    """
    cfg = CODEC_CONFIG.get(codec) or CPU_FALLBACK_CONFIG.get(codec)
    if cfg and cfg["is_nvenc"]:
        return "nvenc"
    return "cpu"


def sample_to_metadata(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a sample dict into a JSON-serialisable metadata dict
    suitable for storage alongside the output video.
    """
    meta = dict(sample)
    # Convert resolution tuple to string for JSON
    if "resolution" in meta and isinstance(meta["resolution"], tuple):
        meta["resolution"] = f"{meta['resolution'][0]}x{meta['resolution'][1]}"
    return meta


def validate_samples(samples: List[Dict[str, Any]]) -> bool:
    """
    Validate that an LHS sample set covers every level of every factor
    at least once. Logs warnings for any missing levels.

    Returns True if all levels are covered.
    """
    all_covered = True
    for factor_name, levels in COMPRESSIBILITY_FACTORS.items():
        seen = set()
        for s in samples:
            val = s.get(factor_name)
            # Handle tuple comparison
            if isinstance(val, (list, tuple)):
                val = tuple(val)
            seen.add(val)

        # Profile is conditional — only check when codec is h264
        if factor_name == "profile":
            # Only check that some non-None profiles exist
            non_none = {v for v in seen if v is not None}
            if not non_none:
                logger.warning("Factor '%s': no H.264 samples generated, profile not tested", factor_name)
            continue

        level_set = set()
        for lv in levels:
            if isinstance(lv, (list, tuple)):
                level_set.add(tuple(lv))
            else:
                level_set.add(lv)

        missing = level_set - seen
        if missing:
            logger.warning("Factor '%s': missing levels %s", factor_name, missing)
            all_covered = False

    if all_covered:
        logger.info("LHS validation passed: all factor levels covered")
    return all_covered
