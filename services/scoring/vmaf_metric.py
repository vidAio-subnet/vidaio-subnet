import subprocess
import xml.etree.ElementTree as ET
import os
from moviepy.editor import VideoFileClip

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

def convert_mp4_to_y4m(input_path):
    """
    Converts an MP4 video file to Y4M format using FFmpeg.
    
    Args:
        input_path (str): Path to the input MP4 file.
    
    Returns:
        str: Path to the converted Y4M file.
    """
    if not input_path.lower().endswith(".mp4"):
        raise ValueError("Input file must be an MP4 file.")

    # Change extension to .y4m and keep it in the same directory
    output_path = os.path.splitext(input_path)[0] + ".y4m"

    command = [
        "ffmpeg",
        "-i", input_path,
        "-pix_fmt", "yuv420p",  # Ensure pixel format is compatible
        "-f", "yuv4mpegpipe",   # Set output format to Y4M
        output_path
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Error converting MP4 to Y4M: {result.stderr.strip()}")
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Expected output file '{output_path}' not found.")
        return output_path
    except Exception as e:
        print(f"Error in convert_mp4_to_y4m: {e}")
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

def calculate_vmaf(ref_y4m_path, dist_mp4_path, start_point):
    try:

        print("Trimming distorted MP4")
        dist_trim_path = trim_video(dist_mp4_path, start_point)

        print("Converting distorted MP4 to Y4M...")
        dist_y4m_path = convert_mp4_to_y4m(dist_trim_path)
        
        # Step 2: Calculate VMAF
        print("Calculating VMAF score...")
        vmaf_harmonic_mean = vmaf_metric(ref_y4m_path, dist_y4m_path)
        print(f"VMAF harmonic_mean Value as Float: {vmaf_harmonic_mean}")
        
        os.remove(dist_y4m_path)
        os.remove(dist_trim_path)
        print("Intermediate Y4M files deleted.")

        return vmaf_harmonic_mean
        
    except Exception as e:
        print(f"Failed to calculate VMAF: {e}")
