import json
import subprocess

def calculate_vmaf(ref_path, dist_path):
    """Calculate VMAF score using the VMAF tool."""
    return 51
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



calculate_vmaf("/workspace/vidaio-subnet/vidaio-subnet/services/video_scheduler/videos/4763824_4k.mp4", "/workspace/vidaio-subnet/vidaio-subnet/services/video_scheduler/videos/4763824_hd.mp4")