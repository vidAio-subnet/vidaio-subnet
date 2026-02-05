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

import json

def vmaf_metric(ref_path, dist_path, output_file="vmaf_output.json", neg_model=False):
    """
    Calculate VMAF score using Dockerized FFmpeg with CUDA acceleration (libvmaf_cuda).
    """
    if neg_model:
        model_version = "vmaf_v0.6.1neg"
    else:
        model_version = "vmaf_v0.6.1"
    
    # Ensure absolute paths for mounting
    ref_path = os.path.abspath(ref_path)
    dist_path = os.path.abspath(dist_path)
    output_file = os.path.abspath(output_file.replace(".xml", ".json")) # Ensure JSON extension
    
    # Construct paths inside the container (mounting host root to /host_root)
    # We strip the leading slash from the absolute path to append to /host_root
    # e.g. /tmp/video.mp4 -> /host_root/tmp/video.mp4
    def container_path(p):
        return f"/host_root{p}"

    # Verify input files exist
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference file not found: {ref_path}")
    if not os.path.exists(dist_path):
        raise FileNotFoundError(f"Distorted file not found: {dist_path}")

    # Remove existing output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Docker command
    # Using --rm to clean up container after run
    # Mounting / to /host_root to handle any absolute path
    # Using specific filter_complex for CUDA acceleration (hwupload -> scale_cuda -> libvmaf_cuda)
    
    command = [
        "docker", "run", "--rm", "--gpus", "all",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,video,utility",
        "-v", "/:/host_root",
        "vmaf_ffmpeg:latest",
        "-init_hw_device", "cuda=gpu", "-filter_hw_device", "gpu",
        "-i", container_path(dist_path),
        "-i", container_path(ref_path),
        "-filter_complex",
        f"[0:v]format=yuv420p,hwupload,scale_cuda=format=yuv420p[dis];[1:v]format=yuv420p,hwupload,scale_cuda=format=yuv420p[ref];[dis][ref]libvmaf_cuda=model=version={model_version}:pool=harmonic_mean:log_fmt=json:log_path={container_path(output_file)}",
        "-f", "null",
        "-"
    ]
    
    try:
        logger.info(f"Running VMAF (CUDA/Docker) on {os.path.basename(dist_path)}")
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Docker VMAF execution failed: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, command, result.output, result.stderr)
        
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"VMAF output file '{output_file}' not generated. Docker stderr:\n{result.stderr[-500:]}")
        
        # Parse JSON output
        with open(output_file, 'r') as f:
            data = json.load(f)
            
        # Extract Harmonic Mean VMAF score
        # Structure: { "version": "...", "frames": [...], "pooled_metrics": { "vmaf": { "min": ..., "max": ..., "mean": ..., "harmonic_mean": ... } } }
        try:
            score = data['pooled_metrics']['vmaf']['harmonic_mean']
            return float(score)
        except KeyError:
            # Fallback to mean if harmonic_mean is missing (unlikely with this pool config)
            score = data['pooled_metrics']['vmaf']['mean']
            return float(score)

    except Exception as e:
        logger.exception(f"Error in vmaf_metric: {e}")
        raise
    finally:
        # Cleanup output file
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except:
                pass

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