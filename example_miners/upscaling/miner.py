from pathlib import Path
import shutil
import subprocess
import re
import time


class Miner:
    """
    Minimal upscaling miner using video2x with realesrgan.
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
        return "UpscalingMiner(video2x+realesrgan)"

    def _get_frame_rate(self, input_path: Path) -> float:
        print(f"[miner] Extracting frame rate from {input_path}")
        # NOTE: `close_fds` is required for Chutes, else it would trigger `aegis[integ] FATAL: security violation (secure authorization fd not valid)`
        result = subprocess.run(
            ["ffmpeg", "-i", str(input_path), "-hide_banner"],
            capture_output=True, text=True, close_fds=False
        )
        match = re.search(r"(\d+(?:\.\d+)?) fps", result.stderr)
        if match:
            fps = float(match.group(1))
            print(f"[miner] Detected frame rate: {fps} fps")
            return fps
        print("[miner] Could not detect frame rate, defaulting to 30.0 fps")
        return 30.0

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
        scale_factor = "4" if task_type == "SD24K" else "2"
        output_path = input_path.with_name(f"{input_path.stem}_upscaled.mp4")

        print(f"[miner] process_video called: input={input_path}, task_type={task_type}, scale={scale_factor}")
        print(f"[miner] Input file exists: {input_path.exists()}, size: {input_path.stat().st_size if input_path.exists() else 'N/A'} bytes")

        # Check tool availability
        for tool in ("ffmpeg", "video2x"):
            tool_path = shutil.which(tool)
            print(f"[miner] {tool} location: {tool_path or 'NOT FOUND'}")

        # Pad last frame to avoid video2x artifacts
        frame_rate = self._get_frame_rate(input_path)
        stop_duration = 2 / frame_rate
        padded_path = input_path.with_name(f"{input_path.stem}_padded.mp4")

        self._run_cmd(
            [
                "ffmpeg", "-i", str(input_path),
                "-vf", f"tpad=stop_mode=clone:stop_duration={stop_duration}",
                "-c:v", "libx264", "-crf", "28", "-preset", "fast",
                str(padded_path),
            ],
            step_name="ffmpeg-pad",
        )
        print(f"[miner] Padded file exists: {padded_path.exists()}, size: {padded_path.stat().st_size if padded_path.exists() else 'N/A'} bytes")

        # Upscale with video2x
        self._run_cmd(
            [
                "video2x", "-i", str(padded_path), "-o", str(output_path),
                "-p", "realesrgan", "-s", scale_factor,
                "-c", "libx264",
                "-e", "preset=fast",
                "-e", "crf=20",
                "-e", "profile=main",
                "-e", "pix_fmt=yuv420p",
                "-e", "sar=1:1",
                "-e", "color_primaries=bt709",
                "-e", "color_trc=bt709",
                "-e", "colorspace=bt709",
                "-e", "movflags=+faststart",
            ],
            step_name="video2x-upscale",
        )
        print(f"[miner] Upscaled file exists: {output_path.exists()}, size: {output_path.stat().st_size if output_path.exists() else 'N/A'} bytes")

        padded_path.unlink(missing_ok=True)
        return output_path
