import subprocess
import xml.etree.ElementTree as ET
import os

def calculate_vmaf(ref_path, dist_path, output_file="vmaf_output.xml"):
    """Calculate VMAF score using the VMAF tool and parse the min value from the output."""
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

if __name__ == "__main__":
    try:
        vmaf_min_value = calculate_vmaf(
            "/workspace/vidaio-subnet/output.y4m", 
            "/workspace/vidaio-subnet/output1.y4m"
        )
        print(f"VMAF harmonic_mean Value as Float: {vmaf_min_value}")
    except Exception as e:
        print(f"Failed to calculate VMAF: {e}")
