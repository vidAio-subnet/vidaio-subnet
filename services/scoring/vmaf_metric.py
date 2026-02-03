import subprocess
import xml.etree.ElementTree as ET
import os
from moviepy.editor import VideoFileClip
from loguru import logger

def trim_video(video_path, start_time, trim_duration=1, reencode=True):
    """
    Trims a video. Can either re-encode (MoviePy) or stream copy (FFmpeg).

    Args:
        video_path (str): Path to input video.
        start_time (float): Start time in seconds.
        trim_duration (int): Duration in seconds.
        reencode (bool): If True, re-encodes using libx264 (MoviePy). 
                         If False, uses FFmpeg stream copy (no quality loss).

    Returns:
        str: Path to the trimmed video.
    """
    mode_suffix = "reenc" if reencode else "copy"
    output_path = video_path.replace(".mp4", f"_trimmed_{start_time:.2f}_{mode_suffix}.mp4")

    # 1. Check Duration using MoviePy
    with VideoFileClip(video_path) as video_clip:
        video_duration = video_clip.duration
        
        if video_duration < trim_duration:
            actual_duration = video_duration
            actual_end = video_duration
        else:
            actual_duration = trim_duration
            actual_end = start_time + trim_duration

        if reencode:
            # --- METHOD A: Re-encode (MoviePy/libx264) ---
            print(f"   [Trim] Mode: Re-encode (libx264) on {os.path.basename(video_path)}")
            if video_duration < trim_duration:
                trimmed_clip = video_clip
            else:
                trimmed_clip = video_clip.subclip(start_time, actual_end)
            
            trimmed_clip.write_videofile(
                output_path, 
                codec="libx264", 
                verbose=False, 
                logger=None
            )
        else:
            # --- METHOD B: Stream Copy (FFmpeg) ---
            print(f"   [Trim] Mode: Stream Copy (Original Codec) on {os.path.basename(video_path)}")
            
            # Build command dynamically
            cmd = ["ffmpeg", "-y"]
            
            # FIX: Only add -ss if we are actually seeking. 
            # -ss 0.0 with -c copy can sometimes corrupt the bitstream start.
            if start_time > 0:
                cmd.extend(["-ss", str(start_time)])
            
            cmd.extend([
                "-i", video_path,
                "-t", str(actual_duration),
                "-map", "0:v",          # Fix: Only copy video stream (prevents audio errors)
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                output_path
            ])
            
            subprocess.run(
                cmd, 
                check=True, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.STDOUT
            )

    return output_path

def vmaf_metric(ref_path, dist_path, output_file="vmaf_output.xml", neg_model=False):
    """
    Calculate VMAF score using FFmpeg's libvmaf filter (No Y4M conversion needed).
    """
    # Configure model string for libvmaf
    if neg_model:
        model_cfg = "model=version=vmaf_v0.6.1neg"
    else:
        model_cfg = "model=version=vmaf_v0.6.1"
    
    # Construct FFmpeg command with libvmaf
    # We map both inputs and pass them into the libvmaf filter
    command = [
        "ffmpeg",
        "-i", dist_path, # Input 0: Distorted
        "-i", ref_path,  # Input 1: Reference
        "-filter_complex", 
        f"[0:v][1:v]libvmaf={model_cfg}:log_path={output_file}:log_fmt=xml",
        "-f", "null", 
        "-"
    ]
    
    try:
        # Run FFmpeg
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Note: FFmpeg writes log to stderr, but VMAF XML is written to log_path
        if not os.path.exists(output_file):
             # Fallback: Sometimes libvmaf fails if inputs are different resolutions/framerates.
             # Check stderr for hints if needed.
            raise FileNotFoundError(f"VMAF output file '{output_file}' not generated. FFmpeg stderr:\n{result.stderr[-500:]}")
        
        tree = ET.parse(output_file)
        root = tree.getroot()
        
        # libvmaf XML output usually puts the aggregate score in <aggregate metrics="..."/>
        # We look for any 'metric' tag with name='vmaf'
        vmaf_node = root.find(".//metric[@name='vmaf']")
        
        if vmaf_node is None:
            # Fallback for different XML structures
            vmaf_node = root.find(".//aggregate/metric[@name='vmaf']")
            
        if vmaf_node is None:
            raise ValueError("VMAF metric not found in the output XML.")
            
        # libvmaf often uses 'mean' or 'harmonic_mean' depending on version
        score = vmaf_node.attrib.get('harmonic_mean') or vmaf_node.attrib.get('mean')
        return float(score)
    
    except Exception as e:
        print(f"Error in vmaf_metric: {e}")
        raise

def calculate_vmaf(ref_path, dist_path, neg_model=False):
    """
    Wrapper for VMAF calculation. Accepts MP4 paths directly.
    """
    try:
        # Pass MP4s directly to FFmpeg/libvmaf
        score = vmaf_metric(ref_path, dist_path, neg_model=neg_model)
        return score
        
    except Exception as e:
        print(f"Failed to calculate VMAF: {e}")
        return None

if __name__ == "__main__":
    # --- TEST CONFIGURATION ---
    ref_video = "reference.mp4"    
    dist_video = "distorted.mp4"   
    
    TRIM_START = 0.0 
    TRIM_DURATION = 5

    if os.path.exists(ref_video) and os.path.exists(dist_video):
        print(f"--- VMAF Re-encode Impact Test (Direct MP4 Mode) ---")
        print(f"Reference File: {ref_video}")
        print(f"Distorted File: {dist_video}")
        
        temp_files = []

        try:
            # --- SCENARIO A: WITH Re-encoding on BOTH ---
            print(f"\n[SCENARIO A] Trimming BOTH with Re-encoding (libx264)...")
            
            ref_A = trim_video(ref_video, TRIM_START, TRIM_DURATION, reencode=True)
            dist_A = trim_video(dist_video, TRIM_START, TRIM_DURATION, reencode=True)
            temp_files.extend([ref_A, dist_A])

            # Calculate Score (Direct MP4 comparison)
            score_a = calculate_vmaf(ref_A, dist_A, neg_model=True)
            print(f"   >>> VMAF Score A (Both Re-encoded): {score_a}")


            # --- SCENARIO B: WITHOUT Re-encoding on BOTH ---
            print(f"\n[SCENARIO B] Trimming BOTH with Stream Copy (No Re-encode)...")
            
            ref_B = trim_video(ref_video, TRIM_START, TRIM_DURATION, reencode=False)
            dist_B = trim_video(dist_video, TRIM_START, TRIM_DURATION, reencode=False)
            temp_files.extend([ref_B, dist_B])

            # Calculate Score (Direct MP4 comparison)
            score_b = calculate_vmaf(ref_B, dist_B, neg_model=True)
            print(f"   >>> VMAF Score B (Both Copied): {score_b}")


            # --- SUMMARY ---
            print(f"\n--- FINAL RESULTS ---")
            print(f"Score A (Re-encoded): {score_a}")
            print(f"Score B (Stream Copy): {score_b}")
            if score_b and score_a:
                diff = score_b - score_a
                print(f"Difference: {diff:.4f} (Positive means Stream Copy was higher quality)")

        except Exception as e:
            logger.exception(f"Test Execution Failed: {e}")
            
        finally:
            print("\nCleaning up...")
            for f in temp_files:
                if f and os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass
            print("Cleanup complete.")
    else:
        print("Error: Ensure both 'reference.mp4' and 'distorted.mp4' exist.")