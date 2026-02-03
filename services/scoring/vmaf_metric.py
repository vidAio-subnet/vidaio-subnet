import subprocess
import xml.etree.ElementTree as ET
import os
import time
from moviepy.editor import VideoFileClip
from loguru import logger

def trim_video(video_path, start_time, trim_duration=1, reencode=False):
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
            
            if start_time > 0:
                cmd.extend(["-ss", str(start_time)])
            
            cmd.extend([
                "-i", video_path,
                "-t", str(actual_duration),
                "-map", "0:v",          # Fix: Only copy video stream
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
    if neg_model:
        model_cfg = "model=version=vmaf_v0.6.1neg"
    else:
        model_cfg = "model=version=vmaf_v0.6.1"
    
    command = [
        "ffmpeg",
        "-i", dist_path, 
        "-i", ref_path,  
        "-filter_complex", 
        f"[0:v][1:v]libvmaf={model_cfg}:log_path={output_file}:log_fmt=xml",
        "-f", "null", 
        "-"
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"VMAF output file '{output_file}' not generated. FFmpeg stderr:\n{result.stderr[-500:]}")
        
        tree = ET.parse(output_file)
        root = tree.getroot()
        
        vmaf_node = root.find(".//metric[@name='vmaf']")
        if vmaf_node is None:
            vmaf_node = root.find(".//aggregate/metric[@name='vmaf']")
            
        if vmaf_node is None:
            raise ValueError("VMAF metric not found in the output XML.")
            
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
            # --- SCENARIO A: WITH Re-encoding ---
            print(f"\n[SCENARIO A] Trimming with Re-encoding (libx264)...")
            t0 = time.perf_counter()
            
            # Trim
            ref_A = trim_video(ref_video, TRIM_START, TRIM_DURATION, reencode=True)
            dist_A = trim_video(dist_video, TRIM_START, TRIM_DURATION, reencode=True)
            temp_files.extend([ref_A, dist_A])
            
            # Score
            score_a = calculate_vmaf(ref_A, dist_A, neg_model=True)
            
            t1 = time.perf_counter()
            duration_a = t1 - t0
            print(f"   >>> Score A: {score_a}")
            print(f"   >>> Time A:  {duration_a:.4f} seconds")


            # --- SCENARIO B: WITHOUT Re-encoding ---
            print(f"\n[SCENARIO B] Trimming with Stream Copy (No Re-encode)...")
            t2 = time.perf_counter()
            
            # Trim
            ref_B = trim_video(ref_video, TRIM_START, TRIM_DURATION, reencode=False)
            dist_B = trim_video(dist_video, TRIM_START, TRIM_DURATION, reencode=False)
            temp_files.extend([ref_B, dist_B])
            
            # Score
            score_b = calculate_vmaf(ref_B, dist_B, neg_model=True)
            
            t3 = time.perf_counter()
            duration_b = t3 - t2
            print(f"   >>> Score B: {score_b}")
            print(f"   >>> Time B:  {duration_b:.4f} seconds")


            # --- SUMMARY ---
            print(f"\n--- FINAL RESULTS ---")
            print(f"{'Metric':<15} | {'Scenario A (Re-encode)':<25} | {'Scenario B (Stream Copy)':<25}")
            print("-" * 70)
            print(f"{'VMAF Score':<15} | {score_a:<25} | {score_b:<25}")
            print(f"{'Total Time':<15} | {duration_a:.4f}s{'':<19} | {duration_b:.4f}s")
            
            if score_b and score_a:
                print("-" * 70)
                time_diff = duration_a - duration_b
                print(f"Stream Copy was {time_diff:.4f}s faster.")

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