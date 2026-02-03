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
    # suffix helps distinguish files
    mode_suffix = "reenc" if reencode else "copy"
    output_path = video_path.replace(".mp4", f"_trimmed_{start_time:.2f}_{mode_suffix}.mp4")

    # 1. Check Duration using MoviePy (lightweight check)
    with VideoFileClip(video_path) as video_clip:
        video_duration = video_clip.duration
        
        # Determine actual end time
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
            
            # Note: -ss before -i is faster, but -ss after -i is more accurate.
            # However, with -c copy, it must snap to keyframes regardless.
            # We put -ss before -i for standard trimming behavior.
            cmd = [
                "ffmpeg",
                "-y",                   # Overwrite output
                "-ss", str(start_time), # Seek start
                "-i", video_path,       # Input
                "-t", str(actual_duration), # Duration
                "-c", "copy",           # Stream copy (NO RE-ENCODING)
                "-avoid_negative_ts", "make_zero", # Reset timestamps
                output_path
            ]
            
            # Suppress FFmpeg output unless error
            subprocess.run(
                cmd, 
                check=True, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.STDOUT
            )

    return output_path

def convert_mp4_to_y4m(input_path, random_frames, upscale_factor=1):
    """
    Converts an MP4 video file to Y4M format using FFmpeg and upscales selected frames.
    """
    if not input_path.lower().endswith(".mp4"):
        raise ValueError("Input file must be an MP4 file.")

    output_path = os.path.splitext(input_path)[0] + ".y4m"

    try:
        select_expr = "+".join([f"eq(n\\,{f})" for f in random_frames])
        
        if upscale_factor >= 2:
            scale_width = f"iw*{upscale_factor}"
            scale_height = f"ih*{upscale_factor}"
            vf_filter = f"select='{select_expr}',scale={scale_width}:{scale_height}"
        else:
            vf_filter = f"select='{select_expr}'"

        subprocess.run([
            "ffmpeg",
            "-i", input_path,
            "-vf", vf_filter,
            "-pix_fmt", "yuv420p",
            "-vsync", "vfr",
            output_path,
            "-y"
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        return output_path

    except Exception as e:
        print(f"Error in convert_mp4_to_y4m: {e}")
        raise

def vmaf_metric(ref_path, dist_path, output_file="vmaf_output.xml", neg_model=False):
    """
    Calculate VMAF score using the VMAF tool.
    """
    if neg_model:
        model_version = "version=vmaf_v0.6.1neg"
    else:
        model_version = "version=vmaf_v0.6.1"
        
    command = [
        "vmaf",  
        "-r", ref_path,
        "-d", dist_path,
        "--model", model_version,
        "-out-fmt", "xml",
        "-o", output_file  
    ]
    
    try:
        # Run VMAF
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Error calculating VMAF: {result.stderr.strip()}")
        
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Expected output file '{output_file}' not found.")
        
        tree = ET.parse(output_file)
        root = tree.getroot()
        
        vmaf_node = root.find(".//metric[@name='vmaf']")
        if vmaf_node is None:
            raise ValueError("VMAF metric not found in the output XML.")
        
        return float(vmaf_node.attrib['harmonic_mean'])
    
    except Exception as e:
        print(f"Error in vmaf_metric: {e}")
        raise

def calculate_vmaf(ref_y4m_path, dist_mp4_path, random_frames, neg_model=False):
    """
    Calculates VMAF given a Reference Y4M and a Distorted MP4.
    """
    dist_y4m_path = None
    try:
        # print(f"   Converting dist to Y4M: {dist_mp4_path}...")
        dist_y4m_path = convert_mp4_to_y4m(dist_mp4_path, random_frames)
        
        # print("   Running VMAF binary...")
        score = vmaf_metric(ref_y4m_path, dist_y4m_path, neg_model=neg_model)
        return score
        
    except Exception as e:
        print(f"Failed to calculate VMAF: {e}")
        return None

    finally:
        if dist_y4m_path and os.path.exists(dist_y4m_path):
            try:
                os.remove(dist_y4m_path)
            except:
                pass

if __name__ == "__main__":
    # --- TEST CONFIGURATION ---
    ref_video = "reference.mp4"    # Your high quality source
    dist_video = "distorted.mp4"   # Your compressed/processed version
    
    frames_to_test = [0, 10, 20, 30] 
    TRIM_START = 0.0 
    TRIM_DURATION = 5

    if os.path.exists(ref_video) and os.path.exists(dist_video):
        print(f"--- VMAF Re-encode Impact Test ---")
        print(f"Reference File: {ref_video}")
        print(f"Distorted File: {dist_video}")
        
        # Track files for cleanup
        temp_files = []

        try:
            # --- SCENARIO A: WITH Re-encoding on BOTH ---
            print(f"\n[SCENARIO A] Trimming BOTH with Re-encoding (libx264)...")
            
            # 1. Trim both inputs with re-encoding
            ref_A = trim_video(ref_video, TRIM_START, TRIM_DURATION, reencode=True)
            dist_A = trim_video(dist_video, TRIM_START, TRIM_DURATION, reencode=True)
            temp_files.extend([ref_A, dist_A])

            # 2. Convert Reference A to Y4M (Required for VMAF func)
            ref_A_y4m = convert_mp4_to_y4m(ref_A, frames_to_test)
            temp_files.append(ref_A_y4m)

            # 3. Calculate Score
            score_a = calculate_vmaf(ref_A_y4m, dist_A, frames_to_test, neg_model=True)
            print(f"   >>> VMAF Score A (Both Re-encoded): {score_a}")


            # --- SCENARIO B: WITHOUT Re-encoding on BOTH ---
            print(f"\n[SCENARIO B] Trimming BOTH with Stream Copy (No Re-encode)...")
            
            # 1. Trim both inputs using stream copy
            ref_B = trim_video(ref_video, TRIM_START, TRIM_DURATION, reencode=False)
            dist_B = trim_video(dist_video, TRIM_START, TRIM_DURATION, reencode=False)
            temp_files.extend([ref_B, dist_B])

            # 2. Convert Reference B to Y4M
            ref_B_y4m = convert_mp4_to_y4m(ref_B, frames_to_test)
            temp_files.append(ref_B_y4m)

            # 3. Calculate Score
            score_b = calculate_vmaf(ref_B_y4m, dist_B, frames_to_test, neg_model=True)
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
            # Cleanup all intermediate files
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