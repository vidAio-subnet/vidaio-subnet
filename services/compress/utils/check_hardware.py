import tempfile
import os
import subprocess
import time
import sys
import shutil # For cleaning up directories
import ffmpeg # For dummy video creation in __main__

# Adjust sys.path to allow imports from the 'utils' directory
# This assumes check_hardware.py is in src/utilities/
# and other modules like encode_video.py are in utils/
UTILS_DIR = os.path.abspath(os.path.dirname(__file__))
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

try:
    from encode_video import encode_video, ENCODER_SETTINGS
    from calculate_vmaf_adv import calculate_vmaf_advanced
except ImportError as e:
    print(f"Error importing modules for comparison in check_hardware.py: {e}")
    print("Ensure encode_video.py and calculate_vmaf_adv.py are in the src directory.")
    # Define dummy functions if imports fail, so the rest of check_hardware can be tested
    def encode_video(*args, **kwargs):
        print("Warning: encode_video function not imported. Comparison will not run.")
        return None, 0
    def calculate_vmaf_advanced(*args, **kwargs):
        print("Warning: calculate_vmaf_advanced function not imported. VMAF comparison will not run.")
        return 0.0
    ENCODER_SETTINGS = {}


# --- Define these at the module level for import by Streamlit app ---
# Hardware encoders in preference order
HW_ENCODERS = [
    'h264_videotoolbox', 'hevc_videotoolbox', # Apple VideoToolbox (macOS)
    'av1_nvenc', 'h264_nvenc', 'hevc_nvenc',  # NVIDIA NVENC (Windows/Linux)
    'h264_qsv', 'hevc_qsv', 'av1_qsv',       # Intel Quick Sync Video (Windows/Linux)
    'h264_amf', 'hevc_amf',                  # AMD AMF (Windows)
    # Add others if needed, e.g., 'h264_omx' for Raspberry Pi
]
# Software encoders in preference order
SW_ENCODERS = ['libx264', 'libx265', 'libsvtav1', 'libaom-av1']
# ---

def test_encoder_works(encoder_name, test_duration=1):
    """
    Actually test if an encoder works by encoding a small test clip.
    Returns True if encoding succeeds, False otherwise.
    """
    temp_input_file_h = None
    temp_input_filename = None
    temp_output_file_h = None
    temp_output_filename = None
    
    try:
        # Create a temporary file for the input video
        temp_input_file_h = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_input_filename = temp_input_file_h.name
        temp_input_file_h.close() # Close it so ffmpeg can write to it

        # Generate a test input (solid color video) and save as MP4
        gen_input_cmd = [
            'ffmpeg', '-y', '-f', 'lavfi', '-i', 
            f'color=c=blue:s=320x240:d={test_duration}', # Small res for faster test
            '-pix_fmt', 'yuv420p', temp_input_filename
        ]
        gen_result = subprocess.run(gen_input_cmd, check=False, capture_output=True, text=True, errors='ignore')
        
        if gen_result.returncode != 0:
            # print(f"Failed to generate test input for {encoder_name}. FFmpeg stderr:\n{gen_result.stderr}")
            return False
        if not os.path.exists(temp_input_filename) or os.path.getsize(temp_input_filename) == 0:
            # print(f"Test input file {temp_input_filename} not created or empty for {encoder_name}.")
            return False

        # Create a temporary file for the output video
        temp_output_file_h = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_output_filename = temp_output_file_h.name
        temp_output_file_h.close() # Close it so ffmpeg can write to it

        # Try encoding with the specified encoder
        encode_cmd = ['ffmpeg', '-y', '-i', temp_input_filename, '-c:v', encoder_name]

        # Add appropriate preset for speed and compatibility during testing
        if encoder_name in ['libx264', 'libx265']:
            encode_cmd.extend(['-preset', 'ultrafast', '-b:v', '500k'])
        elif encoder_name in ['libsvtav1', 'libaom-av1']:
             encode_cmd.extend(['-preset', '10', '-b:v', '500k']) # SVT-AV1 preset 10 is fast
        elif 'nvenc' in encoder_name:
            encode_cmd.extend(['-preset', 'p1', '-b:v', '500k']) # NVENC fastest preset
        elif 'qsv' in encoder_name:
            encode_cmd.extend(['-preset', 'ultrafast', '-b:v', '500k']) # QSV preset
        elif 'videotoolbox' in encoder_name or 'amf' in encoder_name:
            # These might not use 'preset' or have different quality controls.
            # Often, just testing basic functionality is enough. Add bitrate for consistency.
            encode_cmd.extend(['-b:v', '500k'])
        
        encode_cmd.extend(['-t', str(test_duration), '-f', 'mp4', temp_output_filename])
        
        encode_result = subprocess.run(encode_cmd, capture_output=True, text=True, errors='ignore')
        
        if encode_result.returncode != 0:
            # print(f"Encoding failed for {encoder_name}. FFmpeg stderr:\n{encode_result.stderr}")
            return False
            
        success = os.path.exists(temp_output_filename) and os.path.getsize(temp_output_filename) > 0
        # if not success:
            # print(f"Output file {temp_output_filename} not created or empty for {encoder_name}.")
        return success
    
    except Exception as e:
        # print(f"Exception during test for {encoder_name}: {e}")
        return False
    finally:
        # Clean up temporary files
        if temp_input_filename and os.path.exists(temp_input_filename):
            try: os.remove(temp_input_filename)
            except OSError: pass
        if temp_output_filename and os.path.exists(temp_output_filename):
            try: os.remove(temp_output_filename)
            except OSError: pass
                
def get_best_working_codec():
    """Select the best actually working codec from predefined lists."""
    # Uses HW_ENCODERS and SW_ENCODERS defined at module level
    print("--- Testing Hardware Encoders (this may take a moment) ---") # For console, not Streamlit UI
    for encoder in HW_ENCODERS:
        print(f"Testing {encoder}...", end=" ", flush=True) # Console print
        if test_encoder_works(encoder):
            print("✅ Working!") # Console print
            return encoder
        print("❌ Failed.") # Console print
    
    print("\n--- Testing Software Encoders (this may take a moment) ---") # For console
    for encoder in SW_ENCODERS:
        print(f"Testing {encoder}...", end=" ", flush=True) # Console print
        if test_encoder_works(encoder):
            print("✅ Working!") # Console print
            return encoder
        print("❌ Failed.") # Console print
    
    print("\n--- Fallback: No preferred encoders working, defaulting to libx264 ---") # Console print
    # As a last resort, test libx264 again if it wasn't already the last one in SW_ENCODERS
    # or if you want to be absolutely sure.
    if test_encoder_works('libx264'):
        print("libx264 is working as a fallback.")
        return 'libx264'
    else:
        print("CRITICAL: libx264 (ultimate fallback) also failed to work. Check FFmpeg installation and codecs.")
        return None # Indicate no working codec found

# Usage in your main code
if __name__ == "__main__":
    print("--- Starting Codec Hardware Check and Comparison ---")
    
    # --- Configuration for Comparison ---
    # Option 1: Specify a path to your own video file to test.
    # If None, a dummy video will be created.
    # user_input_file_path = "./videos/input_videos/base_segments/elephants_dream_1080p24_scene_029.mp4"
    user_input_file_path = None # Set to a path or None to create dummy

    main_temp_dir = tempfile.mkdtemp(prefix="codec_compare_")
    print(f"Temporary files for comparison will be stored in: {os.path.abspath(main_temp_dir)}")

    input_to_use = None
    dummy_input_filename_for_creation = os.path.join(main_temp_dir, "dummy_comparison_input.mp4")
    
    # --- Default codec for comparison if best_selected_codec is the same ---
    # This is the key for ENCODER_SETTINGS for a standard H.264 encode
    # Ensure 'h264' key exists in your ENCODER_SETTINGS in encode_video.py
    DEFAULT_COMPARISON_CODEC_KEY = "h264" 
    test_rate = 30 # Example rate for model-based CQ prediction (if ENCODER_SETTINGS uses it)
    # --- End Configuration ---

    # 1. Prepare input video (user-provided or dummy)
    if user_input_file_path:
        potential_user_file = os.path.abspath(user_input_file_path)
        if os.path.exists(potential_user_file):
            input_to_use = potential_user_file
            print(f"\nUsing user-provided input video: {input_to_use}")
        else:
            print(f"\nUser-provided input video not found: {potential_user_file}. Will attempt to create a dummy video.")

    if not input_to_use:
        print(f"\nCreating dummy input video: {dummy_input_filename_for_creation}...")
        try:
            (
                ffmpeg
                .input('testsrc=duration=5:size=640x360:rate=24', format='lavfi') # Using ffmpeg-python
                .output(dummy_input_filename_for_creation, pix_fmt='yuv420p', acodec='aac', vcodec='libx264', preset='ultrafast') # Ensure dummy is encoded
                .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            )
            print("Dummy input video created successfully.")
            input_to_use = dummy_input_filename_for_creation
        except ffmpeg.Error as e:
            print(f"Failed to create dummy input video with ffmpeg-python. FFmpeg stderr:\n{e.stderr.decode('utf8', errors='ignore')}")
        except Exception as e:
            print(f"Unexpected error creating dummy input video: {e}")

    if not input_to_use or not os.path.exists(input_to_use):
        print("Cannot proceed with comparison without a valid input video. Exiting.")
        if os.path.exists(main_temp_dir):
            shutil.rmtree(main_temp_dir)
        sys.exit(1)

    input_file_size_bytes = os.path.getsize(input_to_use)
    print(f"Input video for comparison: {os.path.basename(input_to_use)}")
    print(f"Input size: {input_file_size_bytes / (1024*1024):.2f} MB")

    # 2. Test and select the best working codec
    print("\n--- Determining Best Available Codec ---")
    best_selected_codec = get_best_working_codec()
    
    if not best_selected_codec:
        print("No working codec could be determined. Cannot proceed with comparison. Exiting.")
        if os.path.exists(main_temp_dir):
            shutil.rmtree(main_temp_dir)
        sys.exit(1)
        
    print(f"\nSelected best working codec by hardware check: {best_selected_codec}")

    # --- Perform Encodings and Comparisons ---
    results = {}
    
    # Map display names to the actual codec names to use with encode_video
    # The keys for this map are the actual codec names (e.g., 'libx264', 'av1_nvenc')
    # that match keys in your ENCODER_SETTINGS in encode_video.py
    codecs_to_compare = {}

    # Add the default comparison codec (e.g., libx264 via 'h264' key)
    if DEFAULT_COMPARISON_CODEC_KEY in ENCODER_SETTINGS:
        codecs_to_compare[f"Default ({DEFAULT_COMPARISON_CODEC_KEY} -> {ENCODER_SETTINGS[DEFAULT_COMPARISON_CODEC_KEY]['codec']})"] = DEFAULT_COMPARISON_CODEC_KEY
    else:
        print(f"Warning: Default comparison codec key '{DEFAULT_COMPARISON_CODEC_KEY}' not in ENCODER_SETTINGS. Skipping its test.")

    # Add the best selected codec if it's different and valid
    if best_selected_codec != ENCODER_SETTINGS.get(DEFAULT_COMPARISON_CODEC_KEY, {}).get('codec'):
        if best_selected_codec in ENCODER_SETTINGS: # If best_selected_codec is a direct key in ENCODER_SETTINGS
             codecs_to_compare[f"Best Auto ({best_selected_codec} -> {ENCODER_SETTINGS[best_selected_codec]['codec']})"] = best_selected_codec
        else: # Try to find if best_selected_codec matches a 'codec' value within ENCODER_SETTINGS
            found_key_for_best = None
            for key, settings in ENCODER_SETTINGS.items():
                if settings.get('codec') == best_selected_codec:
                    found_key_for_best = key
                    break
            if found_key_for_best:
                codecs_to_compare[f"Best Auto ({found_key_for_best} -> {best_selected_codec})"] = found_key_for_best
            else:
                print(f"Warning: Best selected codec '{best_selected_codec}' not found as a key or 'codec' value in ENCODER_SETTINGS. Skipping its comparison encode.")
    else:
        print(f"Best auto-selected codec ({best_selected_codec}) is the same as the default comparison codec. Only one test will run for it.")


    for display_name, codec_key_for_settings in codecs_to_compare.items():
        print(f"\n--- Processing: {display_name} ---")
        
        # The actual codec name to pass to ffmpeg comes from ENCODER_SETTINGS
        actual_ffmpeg_codec = ENCODER_SETTINGS[codec_key_for_settings]['codec']
        
        encoded_output_filename = os.path.join(main_temp_dir, f"output_{os.path.basename(input_to_use).split('.')[0]}_{codec_key_for_settings.replace('/', '_')}.mp4")
        
        print(f"Encoding with settings for '{codec_key_for_settings}' (FFmpeg codec: {actual_ffmpeg_codec}, Rate: {test_rate})...")
        _log, time_taken = encode_video(
            input_path=input_to_use,
            output_path=encoded_output_filename,
            codec=codec_key_for_settings, # Use the key for ENCODER_SETTINGS
            rate=test_rate,
            logging_enabled=False # Keep logs minimal for comparison summary
        )

        if time_taken is not None and os.path.exists(encoded_output_filename) and os.path.getsize(encoded_output_filename) > 0:
            output_file_size_bytes = os.path.getsize(encoded_output_filename)
            compression_ratio = input_file_size_bytes / output_file_size_bytes if output_file_size_bytes > 0 else float('inf')
            
            print("Calculating VMAF...")
            vmaf_score = calculate_vmaf_advanced(input_to_use, encoded_output_filename, use_sampling=False)
            
            results[display_name] = {
                "Encoding Time (s)": f"{time_taken:.2f}",
                "Output Size (MB)": f"{output_file_size_bytes / (1024*1024):.2f}",
                "Compression Ratio": f"{compression_ratio:.2f}:1",
                "VMAF Score": f"{vmaf_score:.2f}" if vmaf_score is not None else "Error"
            }
        else:
            print(f"Encoding failed or output file invalid for {display_name}.")
            results[display_name] = {
                "Encoding Time (s)": "Failed",
                "Output Size (MB)": "N/A",
                "Compression Ratio": "N/A",
                "VMAF Score": "N/A"
            }

    # --- Print Summary ---
    print("\n\n--- Comparison Summary ---")
    print(f"Input Video: {os.path.basename(input_to_use)} (Size: {input_file_size_bytes / (1024*1024):.2f} MB)")
    print(f"Target Rate (for model scale if applicable): {test_rate}")
    print("-" * 70)
    header = "| Encoder Config                         | Enc. Time (s) | Output (MB) | Comp. Ratio | VMAF   |"
    print(header)
    print("-" * 70)
    for name, data in results.items():
        print(f"| {name:<38} | {data['Encoding Time (s)']:<13} | {data['Output Size (MB)']:<11} | {data['Compression Ratio']:<11} | {data['VMAF Score']:<6} |")
    print("-" * 70)

    # --- Cleanup ---
    try:
        print(f"\nCleaning up temporary directory: {main_temp_dir}")
        shutil.rmtree(main_temp_dir)
        print("Temporary directory cleaned up successfully.")
    except Exception as e:
        print(f"Error cleaning up temporary directory {main_temp_dir}: {e}")
    
    print("\n--- Test Completed ---")