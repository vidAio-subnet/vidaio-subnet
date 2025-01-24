import json
import subprocess
import lpips

def calculate_vmaf(ref_path, dist_path):
    """Calculate VMAF score using the VMAF tool."""
    command = [
        "vmaf",
        "--reference", ref_path,
        "--distorted", dist_path,
        "--out-fmt", "json"
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception("Error calculating VMAF: " + result.stderr)

    vmaf_data = json.loads(result.stdout)
    return vmaf_data['pooled_metrics']['vmaf']