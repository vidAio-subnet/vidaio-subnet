from pathlib import Path
import subprocess


class Miner:
    """
    Minimal compression miner using ffmpeg.
    Miners should fork this and improve upon it.

    Required interface:
    - Class must be named `Miner`
    - __init__(self, path_hf_repo: Path) -> None
    - process_video(self, input_path: Path, vmaf_threshold: float,
                    target_codec: str, codec_mode: str, target_bitrate: float) -> Path
    - File must be named `miner.py` in the root of the HF repo
    """

    CODEC_MAP = {
        "av1": "libsvtav1",
        "hevc": "libx265",
        "h264": "libx264",
        "vp9": "libvpx-vp9",
    }

    def __init__(self, path_hf_repo: Path) -> None:
        self.repo_path = path_hf_repo

    def __repr__(self) -> str:
        return "CompressionMiner(ffmpeg)"

    def _vmaf_to_crf(self, vmaf: float, encoder: str) -> int:
        if "x264" in encoder or "x265" in encoder:
            if vmaf >= 93:
                return 18
            elif vmaf >= 89:
                return 23
            return 28
        else:
            if vmaf >= 93:
                return 25
            elif vmaf >= 89:
                return 30
            return 35

    def process_video(
        self,
        input_path: Path,
        vmaf_threshold: float,
        target_codec: str,
        codec_mode: str,
        target_bitrate: float,
    ) -> Path:
        output_path = input_path.with_name(f"{input_path.stem}_compressed.mp4")
        encoder = self.CODEC_MAP.get(target_codec, "libsvtav1")

        cmd = ["ffmpeg", "-i", str(input_path)]

        if codec_mode == "CRF":
            crf = self._vmaf_to_crf(vmaf_threshold, encoder)
            cmd += ["-c:v", encoder, "-crf", str(crf)]
        elif codec_mode == "CBR":
            cmd += ["-c:v", encoder, "-b:v", f"{target_bitrate}M"]
        else:  # VBR
            cmd += [
                "-c:v", encoder,
                "-b:v", f"{target_bitrate}M",
                "-maxrate", f"{target_bitrate * 1.5}M",
                "-bufsize", f"{target_bitrate * 2}M",
            ]

        cmd += ["-c:a", "copy", str(output_path)]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
