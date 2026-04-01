from pathlib import Path
import shutil
import subprocess
import re
import time


class Miner:
    """
    Minimal upscaling miner using ffmpeg with NVENC GPU acceleration.
    Uses CUDA hardware decoding, scale_cuda for GPU-based scaling,
    and NVENC for hardware encoding.
    Miners should fork this and improve upon it.

    Required interface:
    - Class must be named `Miner`
    - __init__(self, path_hf_repo: Path) -> None
    - process_video(self, input_path: Path, task_type: str) -> Path
    - File must be named `miner.py` in the root of the HF repo
    """

    def __init__(self, path_hf_repo: Path) -> None:
        self.repo_path = path_hf_repo

    def __repr__(self) -> str:
        return "UpscalingMiner(ffmpeg+nvenc)"

    @staticmethod
    def _run_cmd(cmd: list[str], step_name: str) -> None:
        print(f"[miner] Running {step_name}: {' '.join(cmd)}")
        start = time.time()
        # NOTE: `close_fds` is required for Chutes, else it would trigger `aegis[integ] FATAL: security violation (secure authorization fd not valid)`
        result = subprocess.run(cmd, capture_output=True, text=True, close_fds=False)
        elapsed = time.time() - start
        if result.stdout:
            print(f"[miner] {step_name} stdout:\n{result.stdout}")
        if result.stderr:
            print(f"[miner] {step_name} stderr:\n{result.stderr}")
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )
        print(f"[miner] {step_name} completed in {elapsed:.2f}s (exit code {result.returncode})")

    def process_video(self, input_path: Path, task_type: str) -> Path:
        scale_factor = 4 if task_type == "SD24K" else 2
        output_path = input_path.with_name(f"{input_path.stem}_upscaled.mp4")

        print(f"[miner] process_video called: input={input_path}, task_type={task_type}, scale={scale_factor}")
        print(f"[miner] Input file exists: {input_path.exists()}, size: {input_path.stat().st_size if input_path.exists() else 'N/A'} bytes")

        # Check tool availability
        ffmpeg_path = shutil.which("ffmpeg")
        print(f"[miner] ffmpeg location: {ffmpeg_path or 'NOT FOUND'}")

        # GPU-accelerated upscale: CUDA decode -> scale_cuda -> NVENC encode
        self._run_cmd(
            [
                "ffmpeg",
                "-hwaccel", "cuda",
                "-hwaccel_output_format", "cuda",
                "-i", str(input_path),
                "-vf", f"scale_cuda=iw*{scale_factor}:ih*{scale_factor}",
                "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23",
                "-c:a", "copy",
                str(output_path),
            ],
            step_name="ffmpeg-nvenc-upscale",
        )
        print(f"[miner] Upscaled file exists: {output_path.exists()}, size: {output_path.stat().st_size if output_path.exists() else 'N/A'} bytes")

        return output_path
