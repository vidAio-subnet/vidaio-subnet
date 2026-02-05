import subprocess
import xml.etree.ElementTree as ET
import os
from moviepy.editor import VideoFileClip
from loguru import logger
import tempfile
import shutil


def trim_video(input_path, start_time, trim_duration=1, target_crf=18):
    """
    Trims a video and re-encodes it to H.264 with minimal quality loss.
    
    Args:
        input_path (str): Path to the source (AV1, HEVC, etc.)
        start_time (float): Start point in seconds
        trim_duration (int): Duration in seconds
        target_crf (int): Quality level (17-18 for visually transparent)
    """
    filename, ext = os.path.splitext(input_path)
    output_path = f"{filename}_trimmed_{start_time:.2f}.mp4"
    
    cmd = [
        "ffmpeg",
        "-y",                  # Overwrite if exists
        "-ss", str(start_time),
        "-t", str(trim_duration),
        "-i", input_path,
        "-c:v", "libx264",     # Ensure H.264 output
        "-preset", "slow",     # Better compression efficiency
        "-crf", str(target_crf),# High quality setting
        "-c:a", "aac",         # Standard audio codec for MP4
        "-b:a", "192k",        # Good audio bitrate
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
        "-show_entries", "stream=r_frame_rate",
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

    output_path = os.path.splitext(input_path)[0] + ".y4m"
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 1. Extract specific frames using fast seek (-ss)
        
        # Get FPS to calculate timestamps
        fps = get_video_fps(input_path)
        print(f"DEBUG: Video FPS: {fps}")
        
        extracted_frames_count = 0
        
        for i, frame_idx in enumerate(random_frames):
            timestamp = frame_idx / fps
            frame_output = os.path.join(temp_dir, f"frame_{i:05d}.png")
            
            # Construct fast seek command
            # -ss before -i is fast seek (keyframe spacing)
            # -vsync 0 prevents frame duplication/drop logic messing with single frame extraction
            extract_cmd = [
                "ffmpeg", 
                "-ss", f"{timestamp:.6f}",
                "-i", input_path,
                "-vf", f"scale=iw*{upscale_factor}:ih*{upscale_factor}",
                "-frames:v", "1",
                "-vsync", "0",
                "-pix_fmt", "rgb24",
                frame_output,
                "-y"
            ]
            
            # print(f"DEBUG: Extracting frame {frame_idx} at {timestamp:.6f}s")
            try:
                subprocess.run(
                    extract_cmd, 
                    check=True, 
                    capture_output=True
                )
                if os.path.exists(frame_output):
                    extracted_frames_count += 1
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to extract frame {frame_idx} at {timestamp}s: {e}")
                # We continue to try other frames even if one fails
        
        # Check if frames were actually generated
        generated_files = sorted(os.listdir(temp_dir))
        if not generated_files:
            print("ERROR: No frames were extracted!")
            raise RuntimeError("FFmpeg extraction produced no frames.")
        
        print(f"DEBUG: Extracted {len(generated_files)} frames using seek.")

        # 2. Re-assemble PNGs into a Y4M
        # We use '-f image2' to read the sequential images as a video stream
        assemble_cmd = [
            "ffmpeg",
            "-framerate", "30", # We set a 'sane' rate for the Y4M header
            "-i", os.path.join(temp_dir, "frame_%05d.png"),
            "-pix_fmt", "yuv420p", # Convert back to target pix_fmt
            output_path,
            "-y"
        ]
        subprocess.run(assemble_cmd, check=True, capture_output=True)

        return output_path

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg command failed with exit code {e.returncode}")
        print(f"Command: {e.cmd}")
        if e.stdout:
            print(f"Stdout: {e.stdout.decode(errors='ignore')}")
        if e.stderr:
            print(f"Stderr: {e.stderr.decode(errors='ignore')}")
        
        # List contents of temp directory to check if frames were generated
        if os.path.exists(temp_dir):
            print(f"Contents of {temp_dir}: {os.listdir(temp_dir)}")
        else:
            print(f"Temp dir {temp_dir} does not exist.")
            
        raise
    except Exception as e:
        print(f"Nuclear extraction failed: {e}")
        raise
    finally:
        # Cleanup the image bridge
        shutil.rmtree(temp_dir)


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