from pathlib import Path
import shutil
import subprocess
import time


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
        "av1": "av1_nvenc",      # hardware, needs Ada Lovelace+
        "hevc": "hevc_nvenc",    # hardware, or "libx265" as software fallback
        "h264": "libx264",       # or "h264_nvenc" for hardware
        "vp9": "libvpx-vp9",    # software only, no NVENC support for VP9
    }

    def __init__(self, path_hf_repo: Path) -> None:
        self.repo_path = path_hf_repo

    def __repr__(self) -> str:
        return "CompressionMiner(ffmpeg)"

    def _vmaf_to_crf(self, vmaf: float, encoder: str) -> int:
        if "x264" in encoder or "x265" in encoder:
            if vmaf >= 93:
                crf = 18
            elif vmaf >= 89:
                crf = 23
            else:
                crf = 28
        else:
            if vmaf >= 93:
                crf = 25
            elif vmaf >= 89:
                crf = 30
            else:
                crf = 35
        print(f"[miner] VMAF threshold {vmaf} with encoder {encoder} -> CRF {crf}")
        return crf

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

        print(f"[miner] process_video called: input={input_path}, vmaf_threshold={vmaf_threshold}, "
              f"target_codec={target_codec}, codec_mode={codec_mode}, target_bitrate={target_bitrate}")
        print(f"[miner] Input file exists: {input_path.exists()}, size: {input_path.stat().st_size if input_path.exists() else 'N/A'} bytes")
        print(f"[miner] Resolved encoder: {encoder}")

        # Check tool availability
        tool_path = shutil.which("ffmpeg")
        print(f"[miner] ffmpeg location: {tool_path or 'NOT FOUND'}")

        cmd = ["ffmpeg", "-i", str(input_path)]

        if codec_mode == "CRF":
            crf = self._vmaf_to_crf(vmaf_threshold, encoder)
            cmd += ["-c:v", encoder, "-crf", str(crf)]
            print(f"[miner] Using CRF mode: crf={crf}")
        elif codec_mode == "CBR":
            cmd += ["-c:v", encoder, "-b:v", f"{target_bitrate}M"]
            print(f"[miner] Using CBR mode: bitrate={target_bitrate}M")
        else:  # VBR
            cmd += [
                "-c:v", encoder,
                "-b:v", f"{target_bitrate}M",
                "-maxrate", f"{target_bitrate * 1.5}M",
                "-bufsize", f"{target_bitrate * 2}M",
            ]
            print(f"[miner] Using VBR mode: bitrate={target_bitrate}M, "
                  f"maxrate={target_bitrate * 1.5}M, bufsize={target_bitrate * 2}M")

        cmd += ["-c:a", "copy", str(output_path)]

        self._run_cmd(cmd, step_name="ffmpeg-compress")
        print(f"[miner] Compressed file exists: {output_path.exists()}, size: {output_path.stat().st_size if output_path.exists() else 'N/A'} bytes")

        return output_path
