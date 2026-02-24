"""
GPU-optimised video compressibility transform pipeline.

Takes seed videos and applies LHS-sampled compressibility transforms using
the NVENC-only pattern (CPU decode → CPU filter → GPU encode).

Handles parallel execution with separate pools for NVENC and CPU-only codecs
to saturate both GPU and CPU hardware simultaneously.

Usage:
    from compressibility_pipeline import CompressibilityPipeline

    pipeline = CompressibilityPipeline()
    results = pipeline.run_batch(seed_path="seed.mp4", n_samples=10)
"""

import json
import logging
import os
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from compressibility_sampler import (
    CODEC_CONFIG,
    CPU_FALLBACK_CONFIG,
    CPU_FALLBACKS,
    build_ffmpeg_command,
    generate_lhs_samples,
    get_codec_type,
    get_output_extension,
    sample_to_metadata,
    validate_samples,
)

logger = logging.getLogger(__name__)


def probe_available_encoders() -> List[str]:
    """
    Probe FFmpeg for available video encoders.

    Returns a list of encoder names that are compiled into the local FFmpeg.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = result.stdout.splitlines()
        encoders: List[str] = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].startswith("V"):
                encoders.append(parts[1])
        return encoders
    except Exception as e:
        logger.warning("Failed to probe FFmpeg encoders: %s", e)
        return []


def check_nvenc_available() -> bool:
    """Check if any NVENC encoder is available."""
    encoders = probe_available_encoders()
    return any(enc in encoders for enc in ("h264_nvenc", "hevc_nvenc"))


def check_gpu_available() -> bool:
    """Check if nvidia-smi reports a GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and len(result.stdout.strip()) > 0
    except FileNotFoundError:
        return False
    except Exception:
        return False


def get_gpu_count() -> int:
    """Return the number of GPUs visible to nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return 0
        return len([l for l in result.stdout.strip().splitlines() if l.strip()])
    except Exception:
        return 0


def get_cpu_count() -> int:
    """Return ``os.cpu_count()`` with a sane fallback."""
    return os.cpu_count() or 4


def trim_seed_video(
    input_path: str,
    output_path: str,
    start_seconds: float = 0,
    duration_seconds: float = 60,
) -> bool:
    """
    Trim a seed video using stream copy (instant, no re-encode).

    Returns True on success.
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_seconds),
        "-i", input_path,
        "-t", str(duration_seconds),
        "-c", "copy",
        "-avoid_negative_ts", "make_start",
        output_path,
        "-hide_banner", "-loglevel", "error",
    ]
    try:
        subprocess.run(cmd, check=True, timeout=60)
        return os.path.exists(output_path)
    except Exception as e:
        logger.error("Failed to trim seed video: %s", e)
        return False


class CompressibilityPipeline:
    """
    Orchestrates LHS-sampled compressibility transforms on seed videos.

    Architecture:
    - CPU decode → CPU filter chain → GPU or CPU encode
    - NVENC jobs (H.264/H.265) run in a GPU thread pool
    - CPU-only jobs (VP9/AV1) run in a separate CPU thread pool
    - Both pools execute concurrently to saturate all hardware

    IMPORTANT: Never uses ``-hwaccel cuda`` / ``-hwaccel cuvid``.
    A6000/A100 data-centre GPUs have no NVDEC.
    """

    def __init__(self, output_dir: str = "videos"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Probe hardware
        self.gpu_available = check_gpu_available()
        self.num_gpus = get_gpu_count() if self.gpu_available else 0
        self.cpu_count = get_cpu_count()

        # Probe encoders
        self.available_encoders = probe_available_encoders()
        self.nvenc_available = check_nvenc_available()

        # Pool sizing (per plan Step 4)
        # NVENC pool: min(nproc // 2, 12) — each FFmpeg uses ~2 CPU threads for decode
        self.nvenc_pool_size = min(self.cpu_count // 2, 12) if self.nvenc_available else 0
        # CPU-only pool: 4 (VP9/AV1 encoders are multi-threaded internally)
        self.cpu_pool_size = 4

        # Track job counter for round-robin GPU assignment
        self._nvenc_job_counter = 0

        self._log_hardware_info()

    def _log_hardware_info(self):
        """Log detected hardware configuration."""
        logger.info("═══ Compressibility Pipeline Hardware ═══")
        logger.info("  GPU available: %s (%d GPUs)", self.gpu_available, self.num_gpus)
        logger.info("  CPU cores: %d", self.cpu_count)
        logger.info("  NVENC available: %s", self.nvenc_available)
        logger.info("  NVENC pool size: %d", self.nvenc_pool_size)
        logger.info("  CPU pool size: %d", self.cpu_pool_size)

        nvenc_encoders = [e for e in self.available_encoders if "nvenc" in e]
        cpu_encoders = [e for e in ["libx264", "libx265", "libvpx-vp9", "libsvtav1"]
                        if e in self.available_encoders]
        logger.info("  NVENC encoders: %s", nvenc_encoders or "none")
        logger.info("  CPU encoders: %s", cpu_encoders or "none")

        # Warn about missing encoders
        for codec in ["h264_nvenc", "hevc_nvenc"]:
            if codec not in self.available_encoders:
                fallback = CPU_FALLBACKS.get(codec)
                if fallback and fallback in self.available_encoders:
                    logger.warning("  %s not available, will fall back to %s", codec, fallback)
                else:
                    logger.warning("  %s not available and no CPU fallback found", codec)

    def _get_gpu_id(self) -> int:
        """Round-robin GPU assignment for NVENC jobs."""
        if self.num_gpus <= 1:
            return 0
        gpu_id = self._nvenc_job_counter % self.num_gpus
        self._nvenc_job_counter += 1
        return gpu_id

    def _run_single_transform(
        self,
        sample: Dict[str, Any],
        seed_path: str,
        job_index: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a single FFmpeg transform for one LHS sample point.

        Returns a result dict on success, None on failure.
        Each failure is isolated — it does not affect other jobs.
        """
        variant_id = str(uuid.uuid4())
        codec = sample["codec"]
        ext = get_output_extension(codec)
        output_filename = f"compress_{variant_id}{ext}"
        output_path = os.path.join(self.output_dir, output_filename)

        # Determine GPU ID for NVENC jobs
        gpu_id = self._get_gpu_id() if get_codec_type(codec) == "nvenc" else 0

        try:
            cmd = build_ffmpeg_command(sample, seed_path, output_path, gpu_id=gpu_id)

            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout per job
            )
            elapsed = time.time() - start_time

            if result.returncode != 0:
                logger.error(
                    "Job %d failed (codec=%s): %s",
                    job_index, codec, result.stderr[:500],
                )
                # Cleanup failed output
                if os.path.exists(output_path):
                    os.remove(output_path)
                return None

            # Calculate output file size
            output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0

            # Estimate FPS throughput
            try:
                probe = subprocess.run(
                    [
                        "ffprobe", "-v", "error",
                        "-select_streams", "v:0",
                        "-show_entries", "stream=nb_frames,duration",
                        "-of", "csv=p=0",
                        output_path,
                    ],
                    capture_output=True, text=True, timeout=10,
                )
                parts = probe.stdout.strip().split(",")
                if len(parts) >= 1 and parts[0].isdigit():
                    nb_frames = int(parts[0])
                    fps_throughput = nb_frames / elapsed if elapsed > 0 else 0
                else:
                    fps_throughput = 0
            except Exception:
                fps_throughput = 0

            logger.info(
                "✅ Job %d done: codec=%s, %.1fs, %.1f fps, %.1f MB → %s",
                job_index, codec, elapsed, fps_throughput,
                output_size / (1024 * 1024), output_filename,
            )

            return {
                "variant_id": variant_id,
                "output_path": output_path,
                "output_filename": output_filename,
                "codec": codec,
                "extension": ext,
                "elapsed_seconds": elapsed,
                "fps_throughput": fps_throughput,
                "output_size_bytes": output_size,
                "compressibility_params": sample_to_metadata(sample),
            }

        except subprocess.TimeoutExpired:
            logger.error("Job %d timed out (codec=%s)", job_index, codec)
            if os.path.exists(output_path):
                os.remove(output_path)
            return None
        except Exception as e:
            logger.error("Job %d error (codec=%s): %s", job_index, codec, e)
            if os.path.exists(output_path):
                os.remove(output_path)
            return None

    def run_batch(
        self,
        seed_path: str,
        n_samples: int = None,
        seed_rng: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Run a full batch of compressibility transforms on a seed video.

        Steps:
        1. Generate LHS sample set
        2. Partition samples into NVENC and CPU-only jobs
        3. Execute both pools concurrently
        4. Return results

        Args:
            seed_path: Path to the seed (input) video.
            n_samples: Number of LHS samples. Defaults to env LHS_SAMPLE_COUNT.
            seed_rng: RNG seed for LHS reproducibility.

        Returns:
            List of result dicts for successful transforms.
        """
        if not os.path.exists(seed_path):
            logger.error("Seed video not found: %s", seed_path)
            return []

        batch_start = time.time()

        # 1. Generate samples
        samples = generate_lhs_samples(
            n_samples=n_samples,
            seed=seed_rng,
            available_encoders=self.available_encoders,
        )
        validate_samples(samples)

        # 2. Partition by codec type
        nvenc_jobs: List[Tuple[int, Dict[str, Any]]] = []
        cpu_jobs: List[Tuple[int, Dict[str, Any]]] = []

        for i, sample in enumerate(samples):
            codec_type = get_codec_type(sample["codec"])
            if codec_type == "nvenc" and self.nvenc_available:
                nvenc_jobs.append((i, sample))
            else:
                cpu_jobs.append((i, sample))

        logger.info(
            "Batch: %d total samples → %d NVENC jobs, %d CPU jobs",
            len(samples), len(nvenc_jobs), len(cpu_jobs),
        )

        # 3. Execute both pools concurrently
        all_results: List[Dict[str, Any]] = []

        def submit_and_collect(jobs, pool_size, pool_name):
            if not jobs or pool_size <= 0:
                return

            with ThreadPoolExecutor(max_workers=pool_size) as executor:
                future_map = {
                    executor.submit(
                        self._run_single_transform, sample, seed_path, idx
                    ): idx
                    for idx, sample in jobs
                }

                for future in as_completed(future_map):
                    job_idx = future_map[future]
                    try:
                        result = future.result()
                        if result:
                            all_results.append(result)
                    except Exception as e:
                        logger.error(
                            "%s pool job %d raised: %s", pool_name, job_idx, e
                        )

        # Run NVENC and CPU pools concurrently via threads
        from concurrent.futures import ThreadPoolExecutor as TPE
        with TPE(max_workers=2) as meta_pool:
            f_nvenc = meta_pool.submit(
                submit_and_collect, nvenc_jobs, self.nvenc_pool_size, "NVENC"
            )
            f_cpu = meta_pool.submit(
                submit_and_collect, cpu_jobs, self.cpu_pool_size, "CPU"
            )
            f_nvenc.result()
            f_cpu.result()

        batch_elapsed = time.time() - batch_start
        logger.info(
            "🎉 Batch complete: %d/%d succeeded in %.1fs",
            len(all_results), len(samples), batch_elapsed,
        )

        return all_results

    def run_single(
        self,
        seed_path: str,
        sample: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Run a single transform (useful for integration with existing worker loops).

        Args:
            seed_path: Path to the seed video.
            sample: A single LHS sample dict.

        Returns:
            Result dict on success, None on failure.
        """
        return self._run_single_transform(sample, seed_path, job_index=0)
