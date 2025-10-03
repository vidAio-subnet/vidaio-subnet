import subprocess
import xml.etree.ElementTree as ET
import os
import json
import hashlib
import tempfile
from loguru import logger
from moviepy.editor import VideoFileClip
from typing import List, Tuple, Optional

# Configuration
VMAF_SEGMENTS = 2  # Number of segments to check
VMAF_SEGMENT_DURATION = 1.0  # Duration of each segment in seconds


def get_video_properties(video_path: str) -> dict:
    """Get video properties using ffprobe"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,duration',
        '-of', 'json',
        video_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    
    if not info.get('streams'):
        raise ValueError(f"Could not probe video: {video_path}")
    
    stream = info['streams'][0]
    fps_str = stream['r_frame_rate']
    num, den = map(int, fps_str.split('/'))
    
    return {
        'width': stream['width'],
        'height': stream['height'],
        'fps': num / den,
        'duration': float(stream.get('duration', 0))
    }

def generate_deterministic_segments(
    video_path: str,
    reference_path: str,
    num_segments: int = VMAF_SEGMENTS,
    segment_duration: float = VMAF_SEGMENT_DURATION) -> List[Tuple[float, float]]:
    """
    Generate deterministic segment timestamps distributed across the video.
    
    Uses hash-based selection to ensure:
    - Same video pair always gets same segments (can't retry for better scores)
    - Segments are unpredictable before submission (can't optimize specific times)
    - Distributed coverage across entire video (can't just optimize beginning/end)
    """
    
    ref_props = get_video_properties(reference_path)
    duration = ref_props['duration']
    
    # Ensure enough video for requested segments
    min_spacing = 2.0
    total_segment_time = num_segments * segment_duration
    total_spacing_time = (num_segments - 1) * min_spacing
    min_required = total_segment_time + total_spacing_time
    
    if duration < min_required:
        num_segments = max(1, int(duration / (segment_duration + min_spacing)))
        print(f"Video too short, reducing to {num_segments} segments")
    
    # Create deterministic seed from video hashes
    with open(video_path, 'rb') as f:
        video_hash = hashlib.md5(f.read(1024 * 1024)).hexdigest()
    
    with open(reference_path, 'rb') as f:
        ref_hash = hashlib.md5(f.read(1024 * 1024)).hexdigest()
    
    combined_hash = hashlib.md5(f"{video_hash}{ref_hash}".encode()).hexdigest()
    
    # Divide video into regions and pick one point from each
    segments = []
    usable_duration = duration - segment_duration - 1.0
    region_size = usable_duration / num_segments
    
    for i in range(num_segments):
        region_start = i * region_size
        region_end = (i + 1) * region_size - segment_duration
        
        # Deterministic "random" point within region
        offset_hash = hashlib.md5(f"{combined_hash}{i}".encode()).hexdigest()
        offset_ratio = int(offset_hash[:8], 16) / 0xFFFFFFFF
        
        start_time = region_start + (offset_ratio * (region_end - region_start))
        end_time = start_time + segment_duration
        
        segments.append((start_time, end_time))
    
    return segments

def calculate_vmaf_neg_segments(
    ref_path: str,
    dist_path: str,
    num_segments: int = VMAF_SEGMENTS,
    segment_duration: float = VMAF_SEGMENT_DURATION,
    vmaf_model: str = "version=vmaf_v0.6.1neg") -> Optional[float]:
    """
    Calculate VMAF NEG score using FFmpeg's libvmaf filter with segment sampling.
    
    This uses FFmpeg directly - no Y4M conversion needed!
    
    Args:
        ref_path: Path to reference video (MP4)
        dist_path: Path to distorted video (MP4)
        num_segments: Number of segments to check
        segment_duration: Duration of each segment
        vmaf_model: VMAF model to use (default: NEG model)
    
    Returns:
        VMAF NEG harmonic mean score (0-100) or None on error
    """
    
    output_json = None
    
    try:
        # Get video properties
        ref_props = get_video_properties(ref_path)
        dist_props = get_video_properties(dist_path)
        
        # Generate deterministic segments
        segments = generate_deterministic_segments(
            dist_path,
            ref_path,
            num_segments,
            segment_duration
        )
        
        logger.info(f"Checking {len(segments)} segments for VMAF NEG:")
        for i, (start, end) in enumerate(segments):
            logger.info(f"  Segment {i+1}: {start:.2f}s - {end:.2f}s")
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            output_json = f.name
        
        # Build filter complex for segment extraction and concatenation
        # Distorted video segments
        dist_trims = []
        for i, (start, end) in enumerate(segments):
            dist_trims.append(
                f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS,"
                f"scale={ref_props['width']}:{ref_props['height']}:flags=bicubic,"
                f"fps={ref_props['fps']}[dist{i}]"
            )
        
        # Reference video segments
        ref_trims = []
        for i, (start, end) in enumerate(segments):
            ref_trims.append(
                f"[1:v]trim=start={start}:end={end},setpts=PTS-STARTPTS,"
                f"fps={ref_props['fps']}[ref{i}]"
            )
        
        # Concatenate segments
        dist_concat = "".join([f"[dist{i}]" for i in range(len(segments))])
        ref_concat = "".join([f"[ref{i}]" for i in range(len(segments))])
        
        # Complete filter with VMAF calculation
        filter_complex = (
            ";".join(dist_trims) + ";" +
            ";".join(ref_trims) + ";" +
            f"{dist_concat}concat=n={len(segments)}:v=1:a=0[distall];" +
            f"{ref_concat}concat=n={len(segments)}:v=1:a=0[refall];" +
            f"[distall][refall]libvmaf="
            f"model='{vmaf_model}':"
            f"log_fmt=json:log_path={output_json}"
        )
        
        # FFmpeg command with explicit software decoding
        cmd = [
            'ffmpeg',
            '-hwaccel', 'none',
            '-hwaccel_device', 'none',
            '-threads', '4', 
            '-i', dist_path,
            '-i', ref_path,
            '-filter_complex', filter_complex,
            '-f', 'null', '-'
        ]
        
        logger.info("Calculating VMAF NEG score using FFmpeg libvmaf...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180
        )
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return None
        
        # Parse JSON output
        if not os.path.exists(output_json):
            print(f"VMAF output file not found: {output_json}")
            return None
        
        with open(output_json, 'r') as f:
            vmaf_data = json.load(f)
        
        vmaf_score = vmaf_data['pooled_metrics']['vmaf'].get(
            'harmonic_mean',
            vmaf_data['pooled_metrics']['vmaf']['mean']
        )
        
        logger.info(f"VMAF NEG score: {vmaf_score:.2f}")
        
        return vmaf_score
        
    except subprocess.TimeoutExpired:
        logger.error("VMAF calculation timed out")
        return None
    except Exception as e:
        logger.error(f"Error calculating VMAF NEG: {e}")
        return None
    
    finally:
        # Cleanup
        if output_json and os.path.exists(output_json):
            try:
                os.unlink(output_json)
            except Exception as e:
                logger.warning(f"Could not delete {output_json}: {e}")

def trim_video(video_path, start_time, trim_duration=1):
    """
    Trims a video at a specific start point for a given duration, or uses the whole video if it's shorter than the trim duration.

    Args:
        video_path (str): Path to the video to be trimmed.
        start_time (float): The start time for trimming.
        trim_duration (int): Duration of the clip to be extracted in seconds.

    Returns:
        str: Path to the trimmed video.
    """
    # Load the video file
    video_clip = VideoFileClip(video_path)
    video_duration = video_clip.duration

    # If the video is shorter than the trim duration, use the whole video
    if video_duration < trim_duration:
        # Use the whole video
        trimmed_clip = video_clip
    else:
        # Create the subclip from the specified start time
        trimmed_clip = video_clip.subclip(start_time, start_time + trim_duration)
    
    # Define the output path for the trimmed video
    output_path = video_path.replace(".mp4", f"_trimmed_{start_time:.2f}.mp4")
    
    # Write the trimmed video
    trimmed_clip.write_videofile(output_path, codec="libx264", verbose=False, logger=None)

    # Return the path to the trimmed video
    return output_path

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


def vmaf_metric(ref_path, dist_path, output_file="vmaf_output.xml"):
    """
    Calculate VMAF score using the VMAF tool and parse the harmonic mean value from the output.
    
    Args:
        ref_path (str): Path to the reference Y4M video.
        dist_path (str): Path to the distorted Y4M video.
        output_file (str): Path to the output XML file.
    
    Returns:
        float: The VMAF harmonic mean score.
    """
    command = [
        "vmaf",  
        "-r", ref_path,
        "-d", dist_path,
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

def calculate_vmaf(ref_y4m_path, dist_mp4_path, random_frames):
    dist_y4m_path = None
    try:
        print("Converting distorted MP4 to Y4M...")
        dist_y4m_path = convert_mp4_to_y4m(dist_mp4_path, random_frames)
        
        print("Calculating VMAF score...")
        vmaf_harmonic_mean = vmaf_metric(ref_y4m_path, dist_y4m_path)
        print(f"VMAF harmonic_mean Value as Float: {vmaf_harmonic_mean}")
        
        return vmaf_harmonic_mean
        
    except Exception as e:
        print(f"Failed to calculate VMAF: {e}")
        return None

    finally:
        if dist_y4m_path and os.path.exists(dist_y4m_path):
            try:
                os.remove(dist_y4m_path)
                print("Intermediate Y4M files deleted.")
            except Exception as e:
                print(f"Warning: Could not delete {dist_y4m_path}: {e}")