import subprocess
import xml.etree.ElementTree as ET
import os

def convert_mp4_to_y4m(input_path, output_path):
    """
    Converts an MP4 video file to Y4M format using FFmpeg.
    
    Args:
        input_path (str): Path to the input MP4 file.
        output_path (str): Path to the output Y4M file.
    
    Returns:
        str: Path to the converted Y4M file.
    """
    command = [
        "ffmpeg",
        "-i", input_path,
        "-pix_fmt", "yuv420p",  # Ensure the pixel format is compatible
        "-f", "yuv4mpegpipe",  # Output format is Y4M
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

def calculate_vmaf(ref_mp4_path, dist_mp4_path):
    try:
        print(ref_mp4_path, dist_mp4_path)

        # Define paths to the Y4M files
        ref_y4m_path = "ref.y4m"
        dist_y4m_path = "dist.y4m"
        
        # Check if ref_y4m_path exists and delete it
        if os.path.exists(ref_y4m_path):
            print(f"Deleting existing reference Y4M file: {ref_y4m_path}")
            os.remove(ref_y4m_path)
        
        # Check if dist_y4m_path exists and delete it
        if os.path.exists(dist_y4m_path):
            print(f"Deleting existing distorted Y4M file: {dist_y4m_path}")
            os.remove(dist_y4m_path)
            
        # Step 1: Convert MP4 to Y4M
        print("Converting reference MP4 to Y4M...")
        convert_mp4_to_y4m(ref_mp4_path, ref_y4m_path)
        print("Converting distorted MP4 to Y4M...")
        convert_mp4_to_y4m(dist_mp4_path, dist_y4m_path)
        
        # Step 2: Calculate VMAF
        print("Calculating VMAF score...")
        vmaf_harmonic_mean = vmaf_metric(ref_y4m_path, dist_y4m_path)
        print(f"VMAF harmonic_mean Value as Float: {vmaf_harmonic_mean}")
        
        # Optional: Clean up intermediate Y4M files
        os.remove(ref_y4m_path)
        os.remove(dist_y4m_path)
        print("Intermediate Y4M files deleted.")
        return vmaf_harmonic_mean
        
    except Exception as e:
        print(f"Failed to calculate VMAF: {e}")
