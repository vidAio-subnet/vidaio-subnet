from scenedetect import AdaptiveDetector, SceneManager, open_video
import time
import os
import subprocess
import traceback 
import gc 

def has_audio(video_path):
    """
    Returns True if the given video file has an audio stream.
    """
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'a',
        '-show_entries', 'stream=codec_type',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    is_windows = os.name == 'nt'
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, shell=is_windows)
        return bool(result.stdout.strip())
    except FileNotFoundError:
        print("Error: ffprobe command not found. Make sure FFmpeg is installed and in your PATH.")
        return False
    except subprocess.CalledProcessError as e:
        if "Stream specifier 'a' matches no streams" not in e.stderr:
             print(f"Error running ffprobe: {e.stderr}")
        return False

def create_temp_downscaled_video(input_path, downscale_factor, temp_suffix="_downscaled.mkv"):
    """Creates a temporary downscaled video using FFmpeg."""
    if downscale_factor <= 1:
        return None # No downscaling needed

    temp_output_path = os.path.splitext(input_path)[0] + temp_suffix
    scale_filter = f"scale=-1:ih/{downscale_factor}"

    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-vf', scale_filter,
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '23',
        '-an',
        temp_output_path
    ]
    print(f"Creating temporary downscaled video (factor {downscale_factor}): {temp_output_path}")
    print(f"FFmpeg command: {' '.join(cmd)}")
    is_windows = os.name == 'nt'
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, shell=is_windows)
        print("Temporary downscaled video created successfully.")
        return temp_output_path
    except Exception as e:
        print(f"Error creating temporary downscaled video: {e}")
        if hasattr(e, 'stderr'):
            print(f"FFmpeg stderr: {e.stderr}")
        return None

def split_video_into_scenes(video_path, temp_dir='./videos/temp_scenes', detection_downscale=1):
    """
    Splits the original video into scenes using FFmpeg's segment muxer (no re-encoding).
    ... (Args and Returns documentation remains the same) ...
    """
    print(f"\n--- Running split_video_into_scenes ---")
    print(f"Video: {video_path}")
    print(f"Temp Dir: {temp_dir}")
    print(f"Detection Downscale: {detection_downscale}")
    print("-" * 35)

    temp_downscaled_video = None
    video_to_detect = video_path

    # --- Create temporary downscaled video if needed ---
    if detection_downscale is not None and detection_downscale > 1:
        actual_downscale = int(detection_downscale)
        print(f"Attempting to create temporary video downscaled by factor {actual_downscale}.")
        temp_downscaled_video = create_temp_downscaled_video(video_path, actual_downscale)
        if temp_downscaled_video:
            video_to_detect = temp_downscaled_video
        else:
            print("Failed to create downscaled video, proceeding with original resolution.")
            actual_downscale = 0
    else:
        actual_downscale = 0
        print("Using original resolution for detection.")

    print(f"Detecting scenes on: {video_to_detect}")
    # --- Timing for Scene Detection ---
    start_detection_time = time.time()
    detection_duration = 0.0
    scene_list_seconds = []
    # --- End Timing ---

    video_manager = None # Initialize outside try
    try:
        # --- Open video WITHOUT 'with' statement ---
        video_manager = open_video(video_to_detect)
        scene_manager = SceneManager()
        scene_manager.add_detector(AdaptiveDetector())
        scene_manager.detect_scenes(video=video_manager, show_progress=True)
        scene_list_frames = scene_manager.get_scene_list()

        end_detection_time = time.time()
        detection_duration = end_detection_time - start_detection_time
        print(f"Scene detection completed in {detection_duration:.2f} seconds.")

        scene_list_seconds = [(start.get_seconds(), end.get_seconds()) for start, end in scene_list_frames]

        if not scene_list_seconds:
            print("No scenes detected.")
            return [], detection_duration

        print(f"Detected {len(scene_list_seconds)} scenes.")

    except Exception as e:
        # --- Timing for Scene Detection (in case of error) ---
        end_detection_time = time.time()
        detection_duration = end_detection_time - start_detection_time
        print(f"Scene detection failed after {detection_duration:.2f} seconds.")
        # --- End Timing ---
        print(f"Error during scene detection: {e}")
        traceback.print_exc()
        # We rely on the finally block for cleanup
        return None, detection_duration
    finally:
        # --- Force release/cleanup BEFORE deleting temp file ---
        if video_manager is not None:
            print("Attempting to release video manager resources...")
            try:
                # Explicitly delete the reference
                del video_manager
                # Force garbage collection to hopefully close handles
                gc.collect()
                print("Video manager reference deleted and GC run.")
            except Exception as e_del:
                print(f"Warning: Error during manual video manager cleanup: {e_del}")

        # --- Clean up the temporary downscaled video ---
        if temp_downscaled_video and os.path.exists(temp_downscaled_video):
            print(f"Removing temporary downscaled video: {temp_downscaled_video}")
            try:
                # Add a small delay just in case the OS needs a moment after GC
                time.sleep(0.2) # Slightly longer delay
                os.remove(temp_downscaled_video)
            except Exception as e_rem:
                print(f"Warning: Failed to remove temporary file {temp_downscaled_video}: {e_rem}")

    # --- Proceed with FFmpeg splitting ---
    output_pattern = os.path.join(temp_dir, 'scene_%03d.mkv')
    split_times = [start for start, end in scene_list_seconds[1:]]

    if not split_times:
        print("Only one scene detected. Copying the whole file as scene_000.mkv")
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-c', 'copy',  # Copy both video and audio without re-encoding
            '-map', '0',   # Map all streams from input
            '-reset_timestamps', '1',
            os.path.join(temp_dir, 'scene_000.mkv')
        ]
    else:
        split_times_str = ",".join(map(str, split_times))
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-c', 'copy',  # Copy both video and audio without re-encoding
            '-f', 'segment',
            '-segment_times', split_times_str,
            '-reset_timestamps', '1',
            '-map', '0',   # Map all streams from input
            output_pattern
        ]

    print("\nRunning FFmpeg segment command...")
    print(f"Command: {' '.join(cmd)}")

    is_windows = os.name == 'nt'
    try:
        # Run FFmpeg command
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, shell=is_windows)
        print("FFmpeg splitting completed successfully.")

        # Find the generated files
        scene_files = sorted([
            os.path.join(temp_dir, f) for f in os.listdir(temp_dir)
            if f.startswith('scene_') and f.endswith('.mkv')
        ])

        # Verification
        if len(scene_files) != len(scene_list_seconds):
            print(f"Warning: Number of detected scenes ({len(scene_list_seconds)}) "
                  f"does not match number of generated files ({len(scene_files)}).")

        print(f"Generated {len(scene_files)} scene files.")
        return scene_files, detection_duration

    except FileNotFoundError:
        print("Error: ffmpeg command not found. Make sure FFmpeg is installed and in your PATH.")
        return None, detection_duration
    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg for splitting:")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return Code: {e.returncode}")
        print(f"Stderr: {e.stderr}")
        return None, detection_duration
    except Exception as e:
        print(f"Unexpected error during FFmpeg splitting: {e}")
        traceback.print_exc()
        return None, detection_duration
