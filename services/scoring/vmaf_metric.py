import subprocess
import xml.etree.ElementTree as ET
import os
from moviepy.editor import VideoFileClip
from loguru import logger

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