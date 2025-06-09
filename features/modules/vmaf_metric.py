import subprocess
import xml.etree.ElementTree as ET
import os
import time

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
    if os.path.exists(output_path):
        print(f"Output file already exists: {output_path}")
        os.remove(output_path)
        print(f"Output file removed: {output_path}")

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

        os.remove(output_file)
        return vmaf_harmonic_mean
    
    except Exception as e:
        print(f"Error in calculate_vmaf: {e}")
        raise

def evaluate_vmaf(ref_path, dist_path):
    try:
        start_time = time.time()
        ref_y4m_path = convert_mp4_to_y4m(ref_path)
        dist_y4m_path = convert_mp4_to_y4m(dist_path)
        vmaf_file = ref_y4m_path + ".xml"

        vmaf_harmonic_mean = vmaf_metric(ref_y4m_path, dist_y4m_path, vmaf_file)
        os.remove(dist_y4m_path)
        os.remove(ref_y4m_path)
        
        elapsed_time = time.time() - start_time
        print(f"VMAF evaluation completed in {elapsed_time:.2f} seconds")
        return vmaf_harmonic_mean
    except Exception as e:
        print(f"Failed to calculate VMAF: {e}")
        return None

