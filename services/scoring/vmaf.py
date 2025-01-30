import subprocess
import xml.etree.ElementTree as ET
import os

def calculate_vmaf(ref_path, dist_path, output_file="vmaf_output.xml"):
    """Calculate VMAF score using the VMAF tool and parse the min value from the output."""
    command = [
        "vmaf",  # Ensure this is the correct command for your VMAF tool
        "-r", ref_path,
        "-d", dist_path,
        "-out-fmt", "xml",
        "-o", output_file  # Specify the output file
    ]
    
    try:
        # Run the VMAF command
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Check if the command was successful
        if result.returncode != 0:
            raise Exception(f"Error calculating VMAF: {result.stderr.strip()}")
        
        # Check if the output file exists
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Expected output file '{output_file}' not found.")
        
        # Parse the XML file
        tree = ET.parse(output_file)
        root = tree.getroot()
        
        # Find the <metric> tag with name="vmaf"
        vmaf_metric = root.find(".//metric[@name='vmaf']")
        if vmaf_metric is None:
            raise ValueError("VMAF metric not found in the output.")
        
        # Extract the 'min' attribute and convert it to a float
        vmaf_harmonic_mean = float(vmaf_metric.attrib['harmonic_mean'])
        return vmaf_harmonic_mean  # Convert to integer if needed
    
    except Exception as e:
        # Print the error and re-raise it for debugging purposes
        print(f"Error in calculate_vmaf: {e}")
        raise

# Test the function
try:
    vmaf_min_value = calculate_vmaf(
        "/workspace/vidaio-subnet/output.y4m", 
        "/workspace/vidaio-subnet/output1.y4m"
    )
    print(f"VMAF harmonic_mean Value as Float: {vmaf_min_value}")
except Exception as e:
    print(f"Failed to calculate VMAF: {e}")
