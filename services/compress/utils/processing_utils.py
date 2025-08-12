import os
import time
import json
import shutil
import subprocess
from .encode_video import encode_video
from .classify_scene import classify_scene_with_model, extract_frames_from_scene
from .analyze_video_fast import analyze_video_fast

def get_video_bitrate(video_path):
    """Get the bitrate of a video file in kbps."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_format', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        bitrate = data.get('format', {}).get('bit_rate')
        if bitrate:
            return float(bitrate) / 1000  # Convert to kbps
        return None
    except Exception as e:
        print(f"Error getting bitrate for {video_path}: {e}")
        return None

def analyze_input_compression(video_path):
    """Analyze if input video is already well compressed."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_streams', '-show_format', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        video_stream = next((s for s in data.get('streams', []) if s.get('codec_type') == 'video'), None)
        if not video_stream:
            return None
            
        codec = video_stream.get('codec_name', '')
        bitrate = data.get('format', {}).get('bit_rate')
        duration = float(data.get('format', {}).get('duration', 0))
        
        # Calculate pixels per second
        width = int(video_stream.get('width', 0))
        height = int(video_stream.get('height', 0))
        fps_str = video_stream.get('r_frame_rate', '0/1')
        fps = eval(fps_str) if '/' in fps_str else float(fps_str)
        
        pixels_per_second = width * height * fps
        
        return {
            'codec': codec,
            'bitrate_kbps': float(bitrate) / 1000 if bitrate else None,
            'duration': duration,
            'resolution': (width, height),
            'fps': fps,
            'pixels_per_second': pixels_per_second,
            'bits_per_pixel': (float(bitrate) / pixels_per_second) if bitrate and pixels_per_second > 0 else None
        }
    except Exception as e:
        print(f"Error analyzing input compression: {e}")
        return None

def should_skip_encoding(input_path, estimated_cq, target_vmaf, logging_enabled=True):
    """Determine if encoding should be skipped to prevent size increase."""
    
    input_analysis = analyze_input_compression(input_path)
    if not input_analysis:
        return False, estimated_cq + 1
    
    input_bitrate = input_analysis.get('bitrate_kbps', 0)
    bits_per_pixel = input_analysis.get('bits_per_pixel', 0)
    codec = input_analysis.get('codec', '').lower()
    
    if bits_per_pixel and bits_per_pixel < 0.02 and target_vmaf < 90:  
        if logging_enabled:
            print(f"⚠️ Input very well compressed (bits/pixel: {bits_per_pixel:.4f}). Encoding anyway with conservative CQ.")
        return False, min(estimated_cq + 8, 51)
    
    if target_vmaf >= 90:
        if logging_enabled:
            print(f"🎯 High target VMAF ({target_vmaf}) - encoding anyway with adjusted CQ")
        return False, estimated_cq
    
    adjusted_cq = calculate_bitrate_aware_cq(estimated_cq, input_analysis, target_vmaf, "unknown")
    
    if input_bitrate > 10000:  
        adjusted_cq = min(adjusted_cq + 4, 51)
        if logging_enabled:
            print(f"Very high bitrate input ({input_bitrate:.0f} kbps) - adding safety margin. CQ: {adjusted_cq}")
    
    return False, adjusted_cq 

def encode_scene_with_size_check(scene_path, output_path, codec, adjusted_cq, content_type, contrast_value, max_retries=2, logging_enabled=True):
    
    original_size = os.path.getsize(scene_path)
    input_analysis = analyze_input_compression(scene_path)
    
    # Pre-check if encoding should be skipped
    skip_encoding, final_cq = should_skip_encoding(scene_path, adjusted_cq, 93, logging_enabled)
    
    if skip_encoding:
        if logging_enabled:
            print("Skipping encoding - input already optimally compressed")
        return None, 0.0
    
    if final_cq != adjusted_cq:
        if logging_enabled:
            print(f"Adjusted CQ from {adjusted_cq} to {final_cq} based on input analysis")
        adjusted_cq = final_cq
    
    input_bitrate = input_analysis.get('bitrate_kbps', 0) if input_analysis else 0
    if input_bitrate > 4000:
        max_size_ratio = 1.15  # Allow 15% increase for high-bitrate content
    else:
        max_size_ratio = 1.05  # Only allow 5% for normal content
    
    for attempt in range(max_retries + 1):
        if logging_enabled and attempt > 0:
            print(f"Retry attempt {attempt} - increasing CQ to prevent size increase")
        
        # FIXED: Actually increase CQ on retries
        retry_cq = adjusted_cq + (attempt * 4)  # Start with base CQ, then +4, +8
        retry_cq = min(retry_cq, 51)  # Cap at 51
        
        if logging_enabled and attempt > 0:
            print(f"Using CQ {retry_cq} for attempt {attempt}")
        
        start_time = time.time()
        _, encoding_time = encode_video(
            input_path=scene_path,
            output_path=output_path,
            codec=codec,
            rate=retry_cq,  # FIXED: Use retry_cq instead of adjusted_cq
            scene_type=content_type if attempt == 0 else None,
            contrast_value=contrast_value if attempt == 0 else None,
            logging_enabled=logging_enabled
        )
        
        if not os.path.exists(output_path):
            continue
            
        output_size = os.path.getsize(output_path)
        size_ratio = output_size / original_size
        
        if logging_enabled:
            print(f"Size ratio: {size_ratio:.2f} (output/input) with CQ {retry_cq}")
        
        # Accept the result if within threshold OR it's the final attempt
        if size_ratio <= max_size_ratio or attempt == max_retries:
            if size_ratio > max_size_ratio and logging_enabled:
                print(f"⚠️  Accepting oversized output ({size_ratio:.2f}x) - final attempt")
            return output_path, encoding_time
        else:
            if logging_enabled:
                print(f"Output too large ({size_ratio:.2f}x), retrying with higher CQ")
            os.remove(output_path)  # Remove oversized file
    
    return None, 0.0

def classify_scene_from_path(scene_path, temp_dir, scene_classifier_model, available_metrics, 
                           device='cpu', class_mapping=None, logging_enabled=True, 
                           num_frames=3, metrics_scaler=None):
    """
    Classify a scene directly from video path - wrapper that uses enhanced functions
    
    This function is a simplified wrapper that leverages the robust implementations
    
    Args:
        scene_path: Path to the video file
        temp_dir: Temporary directory for frame extraction
        scene_classifier_model: Trained scene classifier model
        available_metrics: List of available metrics for the model
        device: Device to run inference on ('cpu' or 'cuda')
        class_mapping: Class mapping dictionary (optional)
        logging_enabled: Whether to enable logging
        num_frames: Number of frames to extract for classification
        metrics_scaler: Metrics scaler (optional)
        
    Returns:
        tuple: (classification_label, detailed_results, video_features)
        classification_label: Scene classification result
        detailed_results: Classification confidence and probabilities
        video_features: Video analysis metrics from analyze_video_fast
    """
    try:
        
        
        # Step 1: Get video duration using ffprobe
        if logging_enabled:
            print(f"Extracting {num_frames} frames from {scene_path}")
        
        try:
            import subprocess
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', scene_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
        except Exception as e:
            if logging_enabled:
                print(f"Warning: Could not get video duration: {e}, using default 30s")
            duration = 30.0  # Fallback duration
        
        # Step 2: Create temp directory for frame extraction
        import tempfile
        frame_temp_dir = os.path.join(temp_dir, f"frames_{int(time.time())}")
        os.makedirs(frame_temp_dir, exist_ok=True)
        
        # Step 3: Extract frames using the robust enhanced function
        frame_paths = extract_frames_from_scene(
            video_path=scene_path,
            start_time=0.0,
            end_time=duration,
            num_frames=num_frames,
            output_dir=frame_temp_dir
        )
        
        if not frame_paths:
            if logging_enabled:
                print("Warning: No frames could be extracted from video")
            return "unclear", {
                'confidence_score': 0.0,
                'prob_screen_content': 0.0,
                'prob_animation': 0.0,
                'prob_faces': 0.0,
                'prob_gaming': 0.0,
                'prob_other': 0.0,
                'prob_unclear': 1.0,
                'frame_predictions': [],
                'error': 'Frame extraction failed'
            }, {}
        
        # Step 4: Extract video features
        if logging_enabled:
            print(f"Extracting video features from {scene_path}")
        
        try:
            
            video_features = analyze_video_fast(scene_path, logging_enabled=False)
            
            if not video_features or 'error' in video_features:
                # Create default video features if extraction fails
                if logging_enabled:
                    print("Warning: Video feature extraction failed, using defaults")
                video_features = {}
                for metric in available_metrics:
                    video_features[metric] = 0.5  # Default neutral values
        except ImportError:
            if logging_enabled:
                print("Warning: analyze_video_fast not available, using default features")
            # Create default video features
            video_features = {}
            for metric in available_metrics:
                video_features[metric] = 0.5  # Default neutral values
        
        # Step 5: Classify using the enhanced classification function
        classification_label, detailed_results = classify_scene_with_model(
            frame_paths=frame_paths,
            video_features=video_features,
            scene_classifier=scene_classifier_model,
            metrics_scaler=metrics_scaler,
            available_metrics=available_metrics,
            device=device,
            logging_enabled=logging_enabled
        )
        
        # Cleanup temporary frames
        try:
            shutil.rmtree(frame_temp_dir)
        except Exception as e:
            if logging_enabled:
                print(f"Warning: Could not cleanup temp frames: {e}")
        
        return classification_label, detailed_results, video_features
        
    except Exception as e:
        if logging_enabled:
            print(f"Error in classify_scene_from_path: {e}")
        
        # Cleanup on error
        try:
            if 'frame_temp_dir' in locals():
                shutil.rmtree(frame_temp_dir)
        except:
            pass
        
        return "unclear", {
            'confidence_score': 0.0,
            'prob_screen_content': 0.0,
            'prob_animation': 0.0,
            'prob_faces': 0.0,
            'prob_gaming': 0.0,
            'prob_other': 0.0,
            'prob_unclear': 1.0,
            'frame_predictions': [],
            'error': str(e)
        }, {}