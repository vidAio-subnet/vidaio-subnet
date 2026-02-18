import subprocess
import xml.etree.ElementTree as ET
import json
import os
from moviepy.editor import VideoFileClip
from loguru import logger
import tempfile
import shutil


_vmaf_ffmpeg_available = None  # cached result

def is_vmaf_ffmpeg_available(docker_image="vmaf_ffmpeg"):
    """
    Check whether the vmaf_ffmpeg Docker image is available locally and
    its FFmpeg build supports the libvmaf_cuda filter.
    The result is cached so the check only runs once per process.
    """
    global _vmaf_ffmpeg_available
    if _vmaf_ffmpeg_available is not None:
        return _vmaf_ffmpeg_available

    try:
        # 1. Check if Docker image exists
        result = subprocess.run(
            ["docker", "images", "-q", docker_image],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            logger.info(f"vmaf_ffmpeg Docker image '{docker_image}' not found locally.")
            _vmaf_ffmpeg_available = False
            return False

        # 2. Check if FFmpeg in the image supports libvmaf_cuda
        result = subprocess.run(
            ["docker", "run", "--rm", docker_image, "-filters"],
            capture_output=True, text=True, timeout=30,
        )
        if "libvmaf_cuda" not in (result.stdout + result.stderr):
            logger.info(f"vmaf_ffmpeg Docker image does not support libvmaf_cuda filter.")
            _vmaf_ffmpeg_available = False
            return False

        logger.info("vmaf_ffmpeg Docker image with libvmaf_cuda support detected ✅")
        _vmaf_ffmpeg_available = True
        return True

    except Exception as e:
        logger.warning(f"Error checking vmaf_ffmpeg availability: {e}")
        _vmaf_ffmpeg_available = False
        return False


def trim_video(input_path, start_time, trim_duration=1, target_crf=18, reencode=False):
    """
    Trims a video segment. By default uses stream copy (no re-encoding) for
    lossless, fast trimming. Set reencode=True to re-encode to H.264, which
    is needed when the output must be in a standard codec (e.g. for upscale_video).
    
    Args:
        input_path (str): Path to the source (AV1, HEVC, etc.)
        start_time (float): Start point in seconds
        trim_duration (int): Duration in seconds
        target_crf (int): Quality level (17-18 for visually transparent), only used when reencode=True
        reencode (bool): If True, re-encode to H.264. If False, use stream copy (default).
    """
    filename, ext = os.path.splitext(input_path)
    output_path = f"{filename}_trimmed_{start_time:.2f}.mp4"
    
    if reencode:
        cmd = [
            "ffmpeg",
            "-y",                  # Overwrite if exists
            "-ss", str(start_time),
            "-t", str(trim_duration),
            "-i", input_path,
            "-c:v", "libx264",     # Ensure H.264 output
            "-preset", "fast",     # Better compression efficiency
            "-crf", str(target_crf),# High quality setting
            "-c:a", "aac",         # Standard audio codec for MP4
            "-b:a", "192k",        # Good audio bitrate
            output_path
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",                  # Overwrite if exists
            "-ss", str(start_time),
            "-t", str(trim_duration),
            "-i", input_path,
            "-c:v", "copy",        # Stream copy — no re-encoding
            "-c:a", "copy",        # Stream copy audio
            output_path
        ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr.decode()}")
        return None

def get_video_fps(video_path):
    """
    Get the frame rate of a video using ffprobe.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        output = subprocess.check_output(cmd).decode().strip()
        num, den = map(int, output.split('/'))
        return num / den
    except Exception as e:
        print(f"Error getting FPS for {video_path}: {e}")
        # Fallback to 30 fps if detection fails, though this might cause drift
        return 30.0

def convert_mp4_to_y4m(input_path, random_frames, upscale_factor=1):
    """
    Converts an MP4 video file to Y4M format using FFmpeg and upscales selected frames.
    
    Args:
        input_path (str): Path to the input MP4 file.
        random_frames (list): List of frame indices to select.
        upscale_factor (int): Factor by which to upscale the frames (2 or 4).
    
    Returns:
        str: Path to the converted Y4M file.
    """
    if not input_path.lower().endswith(".mp4"):
        raise ValueError("Input file must be an MP4 file.")

    # Change extension to .y4m and keep it in the same directory
    output_path = os.path.splitext(input_path)[0] + ".y4m"

    try:
        select_expr = "+".join([f"eq(n\\,{f})" for f in random_frames])
        
        if upscale_factor >= 2:

            scale_width = f"iw*{upscale_factor}"
            scale_height = f"ih*{upscale_factor}"

            subprocess.run([
                "ffmpeg",
                "-i", input_path,
                "-vf", f"select='{select_expr}',scale={scale_width}:{scale_height}",
                "-pix_fmt", "yuv420p",
                "-vsync", "vfr",
                output_path,
                "-y"
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        else:
            subprocess.run([
                "ffmpeg",
                "-i", input_path,
                "-vf", f"select='{select_expr}'",
                "-pix_fmt", "yuv420p",
                "-vsync", "vfr",
                output_path,
                "-y"
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        return output_path

    except Exception as e:
        print(f"Error in vmaf_metric_batch: {e}")
        raise
    
def vmaf_metric(ref_path, dist_path, output_file="vmaf_output.xml", neg_model=False):
    """
    Calculate VMAF score using the VMAF tool and parse the harmonic mean value from the output.
    
    Args:
        ref_path (str): Path to the reference Y4M video.
        dist_path (str): Path to the distorted Y4M video.
        output_file (str): Path to the output XML file.
    
    Returns:
        float: The VMAF harmonic mean score.
    """
    
    if neg_model:
        logger.info("Using VMAF NEG model for scoring.")
        model_version = "version=vmaf_v0.6.1neg"
    else:
        logger.info("Using standard VMAF model for scoring.")
        model_version = "version=vmaf_v0.6.1"
    command = [
        "vmaf",  
        "-r", ref_path,
        "-d", dist_path,
        "--model", model_version,
        "-out-fmt", "xml",
        "-o", output_file  
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Error calculating VMAF: {result.stderr.strip()}")
        
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Expected output file '{output_file}' not found.")
        
        tree = ET.parse(output_file)
        root = tree.getroot()
        
        vmaf_metric = root.find(".//metric[@name='vmaf']")
        if vmaf_metric is None:
            raise ValueError("VMAF metric not found in the output.")
        
        vmaf_harmonic_mean = float(vmaf_metric.attrib['harmonic_mean'])
        return vmaf_harmonic_mean
    
    except Exception as e:
        print(f"Error in calculate_vmaf: {e}")
        raise


def vmaf_metric_ffmpeg(
    dist_path,
    ref_path,
    skip_frames=0,
    n_subsample=1,
    docker_image="vmaf_ffmpeg",
):
    """
    Calculate VMAF score using the vmaf_ffmpeg Docker container with GPU-accelerated
    libvmaf_cuda, operating directly on MP4 inputs.

    The libvmaf_cuda filter expects: distorted first, reference second.
    This matches FFmpeg's convention where the first -i is the distorted video
    and the second -i is the reference video.

    Args:
        dist_path (str): Absolute path to the distorted MP4 video.
        ref_path (str): Absolute path to the reference MP4 video.
        skip_frames (int): Number of leading frames to skip (e.g. 51 to
            drop the first 51 frames from both streams).
        n_subsample (int): Subsample every N-th frame for faster scoring.
        docker_image (str): Name of the Docker image (default "vmaf_ffmpeg").

    Returns:
        float: The VMAF harmonic mean score.
    """
    dist_path = os.path.abspath(dist_path)
    ref_path = os.path.abspath(ref_path)

    # Collect unique directories that need to be mounted
    dist_dir = os.path.dirname(dist_path)
    ref_dir = os.path.dirname(ref_path)

    # Use a temporary directory for the JSON output
    tmp_dir = tempfile.mkdtemp(prefix="vmaf_ffmpeg_")
    output_json = os.path.join(tmp_dir, "vmaf_output.json")

    try:
        # Build volume mounts – map each host dir to the same path inside the
        # container so that absolute paths work unchanged.
        volumes = set()
        volumes.add(dist_dir)
        volumes.add(ref_dir)
        volumes.add(tmp_dir)

        vol_args = []
        for vol in volumes:
            vol_args.extend(["-v", f"{vol}:{vol}"])

        # Build the filter_complex string
        # skip_frames: select frames with index > skip_frames (i.e. drop first skip_frames+1 frames)
        if skip_frames > 0:
            select_filter = f"select=gt(n\\,{skip_frames}),"
        else:
            select_filter = ""

        filter_complex = (
            f"[0:v]{select_filter}format=yuv420p,hwupload_cuda[dis];"
            f"[1:v]{select_filter}format=yuv420p,hwupload_cuda[ref];"
            f"[dis][ref]libvmaf_cuda=n_subsample={n_subsample}"
            f":log_fmt=json:log_path={output_json}"
        )

        # Assemble the full docker command
        # Entrypoint of the image is "ffmpeg", so we only pass ffmpeg args
        cmd = [
            "docker", "run", "--rm",
            "--gpus", "all",
            *vol_args,
            docker_image,
            "-i", dist_path,
            "-i", ref_path,
            "-filter_complex", filter_complex,
            "-f", "null", "-",
        ]

        logger.info(f"Running VMAF via Docker: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(
                f"Docker VMAF command failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

        # Parse the JSON log produced by libvmaf_cuda
        if not os.path.exists(output_json):
            raise FileNotFoundError(
                f"Expected VMAF JSON output '{output_json}' not found."
            )

        with open(output_json, "r") as f:
            vmaf_data = json.load(f)

        # Extract per-frame VMAF scores
        frames = vmaf_data.get("frames", [])
        if not frames:
            raise ValueError("No frames found in VMAF JSON output.")

        scores = []
        for frame in frames:
            metrics = frame.get("metrics", {})
            vmaf_score = metrics.get("vmaf")
            if vmaf_score is not None and vmaf_score > 0:
                scores.append(vmaf_score)

        if not scores:
            raise ValueError("No valid (> 0) VMAF scores found in output.")

        # Compute harmonic mean: n / sum(1/x_i)
        harmonic_mean = len(scores) / sum(1.0 / s for s in scores)
        logger.info(f"VMAF harmonic mean (ffmpeg/docker): {harmonic_mean:.4f}")
        return harmonic_mean

    except Exception as e:
        logger.error(f"Error in vmaf_metric_ffmpeg: {e}")
        raise

    finally:
        # Clean up temporary directory
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

def calculate_vmaf(ref_y4m_path, dist_mp4_path, random_frames, neg_model=False, return_y4m_path=False):
    """
    Calculate VMAF score between reference and distorted videos.
    
    Args:
        ref_y4m_path: Path to reference Y4M file
        dist_mp4_path: Path to distorted MP4 file
        random_frames: List of frame indices to sample
        neg_model: Whether to use negative VMAF model
        return_y4m_path: If True, returns (score, dist_y4m_path) instead of just score
        
    Returns:
        If return_y4m_path=False: vmaf_score (float or None)
        If return_y4m_path=True: (vmaf_score, dist_y4m_path) tuple
    """
    dist_y4m_path = None
    try:
        print("Converting distorted MP4 to Y4M...")
        dist_y4m_path = convert_mp4_to_y4m(dist_mp4_path, random_frames)
        
        print("Calculating VMAF score...")
        vmaf_harmonic_mean = vmaf_metric(ref_y4m_path, dist_y4m_path, neg_model=neg_model)
        print(f"VMAF harmonic_mean Value as Float: {vmaf_harmonic_mean}")
        
        if return_y4m_path:
            # Return Y4M path for reuse (caller is responsible for cleanup)
            return vmaf_harmonic_mean, dist_y4m_path
        else:
            # Original behavior: cleanup and return score only
            return vmaf_harmonic_mean
        
    except Exception as e:
        print(f"Failed to calculate VMAF: {e}")
        if return_y4m_path:
            return None, dist_y4m_path
        return None

    finally:
        # Only cleanup if NOT returning the Y4M path
        if not return_y4m_path and dist_y4m_path and os.path.exists(dist_y4m_path):
            try:
                os.remove(dist_y4m_path)
                print("Intermediate Y4M files deleted.")
            except Exception as e:
                print(f"Warning: Could not delete {dist_y4m_path}: {e}")