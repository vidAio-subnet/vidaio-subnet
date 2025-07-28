import cv2
import numpy as np
import subprocess
import os
import re


def calculate_perceptual_contrast(frame_input):
    """
    Calculate contrast from frame(s) - handles both single frame arrays and file paths.
    
    Args:
        frame_input: Can be:
                    - numpy array (single frame)
                    - string (single frame file path)
                    - list of strings (frame file paths - uses middle frame)
    
    Returns:
        float: Normalized contrast value (0.0 to 1.0)
    """
    try:
        import cv2
        import numpy as np
        import os
        
        # Handle different input types
        if isinstance(frame_input, np.ndarray):
            # Direct frame array (legacy usage)
            frame = frame_input
        elif isinstance(frame_input, str):
            # Single frame file path
            if not os.path.exists(frame_input):
                print(f"⚠️ Frame file not found: {frame_input}")
                return 0.5
            frame = cv2.imread(frame_input)
        elif isinstance(frame_input, list):
            # List of frame paths - use middle frame
            if not frame_input:
                print("⚠️ Empty frame path list provided")
                return 0.5
            
            mid_index = len(frame_input) // 2
            frame_path = frame_input[mid_index]
            
            if not os.path.exists(frame_path):
                print(f"⚠️ Middle frame file not found: {frame_path}")
                return 0.5
            
            frame = cv2.imread(frame_path)
            print(f"   🖼️ Using middle frame ({mid_index+1}/{len(frame_input)}) for contrast")
        else:
            print(f"⚠️ Unsupported frame input type: {type(frame_input)}")
            return 0.5
        
        # Validate frame
        if frame is None:
            print("⚠️ Could not load frame for contrast calculation")
            return 0.5
        
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            print("⚠️ Invalid frame data for contrast calculation")
            return 0.5
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Calculate RMS contrast
        gray_float = gray.astype(np.float64)
        mean_intensity = np.mean(gray_float)
        
        if mean_intensity > 0:
            contrast = np.sqrt(np.mean((gray_float - mean_intensity) ** 2)) / mean_intensity
        else:
            contrast = np.std(gray_float) / 128.0
        
        # Normalize to 0-1 range
        normalized_contrast = min(contrast / 1.5, 1.0)
        
        return float(normalized_contrast)
        
    except Exception as e:
        print(f"⚠️ Error in contrast calculation: {str(e)[:50]}...")
        return 0.5


def get_video_duration(video_path):
    """Get the duration of a video in seconds"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file to get duration: {video_path}")
        return 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps is None or fps == 0 or frame_count is None:
        cap.release()
        print(f"Warning: Could not get valid FPS ({fps}) or frame count ({frame_count}) for {video_path}")
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', video_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as e:
            print(f"Error getting duration with ffprobe fallback for {video_path}: {e}")
            return 0.0
    else:
        duration = frame_count / fps
        cap.release()
        return duration

def get_video_codec(video_path):
    """Get the codec of a video stream using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return result.stdout.strip()
    except FileNotFoundError:
        print("Error: ffprobe not found. Please ensure FFmpeg is installed and in your PATH.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error getting codec for {video_path}: {e.stderr}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while getting video codec: {e}")
        return None


def calculate_contrast_adjusted_cq(scene_type, contrast_value, base_cq, scene_cq_offsets):
    """Calculate contrast-adjusted CQ value based on scene type and contrast."""
    base_offset = scene_cq_offsets.get(scene_type, 0)
    contrast_adjustment = 0

    if scene_type == 'Screen Content / Text':
        if contrast_value > 0.7:
            contrast_adjustment = -1
        elif contrast_value < 0.3:
            contrast_adjustment = +2
    elif scene_type == 'Faces / People':
        if contrast_value > 0.6:
            contrast_adjustment = -2
        elif contrast_value < 0.3:
            contrast_adjustment = +1
    elif scene_type == 'Animation / Cartoon / Rendered Graphics':
        if contrast_value > 0.6:
            contrast_adjustment = -1
        elif contrast_value < 0.3:
            contrast_adjustment = +1
    elif scene_type == 'Gaming Content':
        if contrast_value > 0.7:
            contrast_adjustment = -1
        elif contrast_value < 0.4:
            contrast_adjustment = +2
    else:
        if contrast_value > 0.7:
            contrast_adjustment = -1
        elif contrast_value < 0.3:
            contrast_adjustment = +1

    final_cq = max(10, min(63, base_cq + base_offset + contrast_adjustment))
    return final_cq




def sort_scene_files_by_number(scene_files):
    """
    Sort scene files by their scene number extracted from filename.
    
    Args:
        scene_files (list): List of scene file paths
        
    Returns:
        list: Sorted list of scene file paths
    """
    def extract_scene_number(filename):
        """Extract scene number from filename like 'scene_001.mp4'"""
        match = re.search(r'scene_(\d+)', os.path.basename(filename))
        return int(match.group(1)) if match else 0
    
    return sorted(scene_files, key=extract_scene_number)

def calculate_contrast_adjusted_cq(content_type, contrast_value, optimal_cq, scene_cq_offsets):
    """
    Adjust CQ based on content type and contrast value.
    
    Args:
        content_type (str): Classified content type
        contrast_value (float): Calculated contrast value (0.0-1.0)
        optimal_cq (int): Base optimal CQ value
        scene_cq_offsets (dict): CQ adjustments by content type
        
    Returns:
        int: Adjusted CQ value
    """
    # Get base offset for content type
    base_offset = scene_cq_offsets.get(content_type, 0)
    
    # Apply contrast-based adjustment
    # High contrast content can handle slightly higher CQ
    contrast_adjustment = 0
    if contrast_value > 0.7:  # High contrast
        contrast_adjustment = 1
    elif contrast_value < 0.3:  # Low contrast
        contrast_adjustment = -1
    
    adjusted_cq = optimal_cq + base_offset + contrast_adjustment
    
    # Clamp to valid range
    adjusted_cq = max(10, min(51, adjusted_cq))
    
    return adjusted_cq
