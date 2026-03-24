from pathlib import Path
import subprocess
import re


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
        result = subprocess.run(
            ["ffmpeg", "-i", str(input_path), "-hide_banner"],
            capture_output=True, text=True,
        )
        match = re.search(r"(\d+(?:\.\d+)?) fps", result.stderr)
        if match:
            return float(match.group(1))
        return 30.0

    def process_video(self, input_path: Path, task_type: str) -> Path:
        scale_factor = "4" if task_type == "SD24K" else "2"
        output_path = input_path.with_name(f"{input_path.stem}_upscaled.mp4")

        # Pad last frame to avoid video2x artifacts
        frame_rate = self._get_frame_rate(input_path)
        stop_duration = 2 / frame_rate
        padded_path = input_path.with_name(f"{input_path.stem}_padded.mp4")

        subprocess.run(
            [
                "ffmpeg", "-i", str(input_path),
                "-vf", f"tpad=stop_mode=clone:stop_duration={stop_duration}",
                "-c:v", "libx264", "-crf", "28", "-preset", "fast",
                str(padded_path),
            ],
            check=True, capture_output=True,
        )

        # Upscale with video2x
        subprocess.run(
            [
                "video2x", "-i", str(padded_path), "-o", str(output_path),
                "-p", "realesrgan", "-s", scale_factor,
                "-c", "libx265",
                "-e", "preset=slow",
                "-e", "crf=20",
                "-e", "profile=main",
                "-e", "pix_fmt=yuv420p",
                "-e", "sar=1:1",
                "-e", "color_primaries=bt709",
                "-e", "color_trc=bt709",
                "-e", "colorspace=bt709",
                "-e", "movflags=+faststart",
            ],
            check=True, capture_output=True,
        )

        padded_path.unlink(missing_ok=True)
        return output_path
