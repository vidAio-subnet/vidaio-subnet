import os
import time
import cv2
import pandas as pd
import gc
import subprocess
import json
import numpy as np
import sys
# Adjust sys.path to allow imports from the 'utils' directory
# This assumes check_hardware.py is in src/utilities/
# and other modules like encode_video.py are in utils/
UTILS_DIR = os.path.abspath(os.path.dirname(__file__))
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)



from analyze_video_fast import analyze_video_fast
from find_optimal_cq import find_optimal_cq
from encode_video import encode_video
from classify_scene import classify_scene_with_model, extract_frames_from_scene
from calculate_vmaf_adv import calculate_parallel_vmaf  # Preferred for batching
from video_utils import get_video_duration, calculate_contrast_adjusted_cq
from logging_utils import VideoProcessingLogger


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


def calculate_bitrate_aware_cq(base_cq, input_analysis, target_vmaf, scene_type):
    """Adjust CQ based on input video characteristics to prevent size increases."""
    
    if not input_analysis:
        return base_cq
    
    input_bitrate = input_analysis.get('bitrate_kbps', 0)
    bits_per_pixel = input_analysis.get('bits_per_pixel', 0)
    codec = input_analysis.get('codec', '').lower()
    
    # If input is already very compressed (low bits per pixel), increase CQ
    if bits_per_pixel and bits_per_pixel < 0.1:
        print(f"Input appears heavily compressed (bits/pixel: {bits_per_pixel:.4f}). Increasing CQ by 3.")
        base_cq = min(base_cq + 3, 51)
    
    # If input uses modern codec and we're using older codec, increase CQ
    modern_codecs = ['hevc', 'h265', 'av1', 'vp9']
    older_codecs = ['h264', 'avc', 'vp8']
    
    if any(mc in codec for mc in modern_codecs):
        print(f"Input uses modern codec ({codec}). Adjusting CQ for re-encoding.")
        base_cq = min(base_cq + 2, 51)
    
    # For very high target VMAF with already good input, be more conservative
    if target_vmaf > 95 and bits_per_pixel and bits_per_pixel > 0.05:
        print(f"High target VMAF ({target_vmaf}) with decent input quality. Conservative CQ adjustment.")
        base_cq = min(base_cq + 1, 51)
    
    # Scene-specific adjustments for already compressed content
    if bits_per_pixel and bits_per_pixel < 0.08:
        if scene_type in ['Screen Content / Text', 'Animation / Cartoon / Rendered Graphics']:
            base_cq = min(base_cq + 2, 51)  # These compress well, don't over-encode
    
    return base_cq


def process_scene_analysis_and_cq(scene_path, scene_number, temp_directory,
                                 scene_classifier_model, pipeline_obj, feature_scaler_step,
                                 vmaf_scaler, cq_min, cq_max, available_metrics,
                                 target_vmaf, device, feature_names, 
                                 vmaf_prediction_model_path: str = None,
                                 logger=None, config=None,codec_name = None, logging_enabled=True):
    """
    Process scene analysis with comprehensive feature extraction, classification,
    and CQ optimization, similar to the analysis part of process_short_video_as_single_scene.
    """
    # Step 1: Initialization and Setup
    analysis_start_time = time.time()
    timings = {}
    frames_dir_path = None
    # Determine codec to use (from parameter or config fallback)
    if codec_name:
        codec_for_this_run = codec_name
        if logging_enabled:
            print(f"      🎥 Using provided codec: {codec_for_this_run}")
    else:
        # Extract from config with fallbacks
        codec_for_this_run = (
            config.get('resolved_codec') or 
            config.get('codec') or 
            'libx264'
        ) if config else 'libx264'
        if logging_enabled:
            print(f"      🎥 Using fallback codec: {codec_for_this_run}")

    # set maximum frames used for video analysis algorithm
    max_frames_for_analysis = 100 
    if config and 'max_frames_for_metrics_analysis' in config:
        max_frames_for_analysis = int(config.get('max_frames_for_metrics_analysis', 100))
    # Initialize scene data structure with default values
    scene_data = {
        'scene_number': scene_number,
        'original_path': scene_path, 
        'duration_seconds': 0.0,
        'file_size_input_mb': 0.0,
        'scene_type': 'unknown',
        'confidence_score': 0.0,
        'prob_screen_content': 0.0,
        'prob_animation': 0.0,
        'prob_faces': 0.0,
        'prob_gaming': 0.0,
        'prob_other': 0.0,
        'prob_unclear': 0.0,
        'optimal_cq': None,
        'predicted_vmaf_at_optimal_cq': None,
        'contrast_value': 0.5,
        'adjusted_cq': None,
        'target_vmaf': target_vmaf,
        'notes': '',
        'metrics_resolution_width': 0, 'metrics_resolution_height': 0, 'metrics_frame_rate': 0.0,
        'metrics_bit_depth': 8, 'input_bitrate_kbps': 0.0, 'input_codec': 'unknown', 'bits_per_pixel': 0.0,
        'metrics_avg_motion': 0.0, 'metrics_avg_edge_density': 0.0, 'metrics_avg_texture': 0.0,
        'metrics_avg_color_complexity': 0.0, 'metrics_avg_motion_variance': 0.0,
        'metrics_avg_grain_noise': 0.0, 'metrics_avg_spatial_information': 0.0,
        'metrics_avg_temporal_information': 0.0,
    }

    try:
        # Step 2: Video Feature Analysis
        if logging_enabled: print(f"      📊 Analyzing scene features for: {os.path.basename(scene_path)}...")
        step_start = time.time()
        # Extract comprehensive video metrics using fast analysis
        features_original = analyze_video_fast(scene_path, max_frames=max_frames_for_analysis, logging_enabled=logging_enabled)
        timings['feature_analysis'] = time.time() - step_start

        if features_original is None:
            scene_data['notes'] = 'Feature analysis failed'
            scene_data['processing_time_seconds'] = time.time() - analysis_start_time
            if logger: logger.log_scene_processing(scene_path, scene_number, scene_data, "analysis_failed")
            return scene_data, None, timings 
        # Update scene_data with extracted metrics:
        # - Resolution (width/height)
        # - Frame rate, bit depth, codec
        # - Motion, edge density, texture
        # - Color complexity, grain noise
        # - Spatial/temporal information
        # - Input bitrate and compression ratios
        scene_data.update({k: features_original.get(k, scene_data.get(k)) for k in scene_data if k.startswith('metrics_') or k in ['input_bitrate_kbps', 'input_codec', 'bits_per_pixel']})
        scene_data['duration_seconds'] = get_video_duration(scene_path) or 0.0
        scene_data['file_size_input_mb'] = os.path.getsize(scene_path) / (1024 * 1024) if os.path.exists(scene_path) else 0.0
        # Check for missing critical features
        critical_features_to_check = [ 'metrics_frame_rate', 'input_bitrate_kbps','metrics_resolution']
        if any(scene_data.get(f, 0) == 0 for f in critical_features_to_check):
            if logging_enabled: print(f"      ⚠️ Missing or zero critical features, attempting recovery...")
            # Attempt recovery using alternative methods (ffprobe)
            recovered = recover_missing_features(scene_path, scene_data, logging_enabled) 
            scene_data.update(recovered)
        # Log feature extracted data
        if logger: logger.log_scene_processing(scene_path, scene_number, scene_data, "feature_extraction_complete")
         #Step 4: Frame Extraction for AI Analysis
        if logging_enabled: print(f"      🖼️ Extracting frames for AI analysis...")
        step_start = time.time()
        # Create temporary directory for frame analysis
        frames_dir_path = os.path.join(temp_directory, f"scene_{scene_number}_frames_analysis")
        os.makedirs(frames_dir_path, exist_ok=True)
        # Extract representative frames for content classification
        extracted_frame_paths = extract_frames_from_scene(scene_path, 0, scene_data['duration_seconds'],
         num_frames=3, # Extract 3 representative frames
         output_dir=frames_dir_path)
        timings['frame_extraction'] = time.time() - step_start
        
        if not extracted_frame_paths:
            scene_data['notes'] = 'Frame extraction failed'
            scene_data['processing_time_seconds'] = time.time() - analysis_start_time
            if logger: logger.log_scene_processing(scene_path, scene_number, scene_data, "analysis_failed")
            return scene_data, frames_dir_path, timings 
        # Step 5: AI-Based Content Classification
        if logging_enabled: print(f"      🎯 Classifying content type using AI...")
        step_start = time.time()
        # Classify scene content using neural network
        classification_result = classify_scene_with_model(
            frame_paths=extracted_frame_paths, video_features=features_original,
            scene_classifier=scene_classifier_model, metrics_scaler=feature_scaler_step,
            available_metrics=available_metrics, device=device, logging_enabled=logging_enabled
        )
        # Update scene_data with classification results:
        # - scene_type (e.g., "Screen Content", "Animation", "Faces", etc.)
        # - confidence_score
        # - Individual class probabilities (prob_screen_content, prob_animation, etc.)
        timings['scene_classification'] = time.time() - step_start

        content_type = "unknown" 
        if isinstance(classification_result, tuple) and len(classification_result) == 2:
            content_type, classification_details = classification_result
            scene_data.update({
                'scene_type': content_type,
                'confidence_score': classification_details.get('confidence_score', 0.0),
                'prob_screen_content': classification_details.get('prob_screen_content', 0.0),
                'prob_animation': classification_details.get('prob_animation', 0.0),
                'prob_faces': classification_details.get('prob_faces', 0.0),
                'prob_gaming': classification_details.get('prob_gaming', 0.0),
                'prob_other': classification_details.get('prob_other', 0.0),
                'prob_unclear': classification_details.get('prob_unclear', 0.0)
            })
        elif isinstance(classification_result, str): 
            content_type = classification_result
            scene_data['scene_type'] = content_type
            scene_data['confidence_score'] = 0.5 
        else:
            scene_data['notes'] = "Classification returned unexpected result"
        
        if logger: logger.log_scene_processing(scene_path, scene_number, scene_data, "classification_complete")
        # Step 6: Quality Parameter (CQ) Optimization
        if logging_enabled: print(f"      🎚️ Optimizing quality parameters (CQ)...")
        step_start = time.time()
        # Find optimal CQ value using machine learning model
        result = find_optimal_cq(
            features_original, target_vmaf_original=target_vmaf,
            pipeline_obj=pipeline_obj, vmaf_scaler=vmaf_scaler,
            cq_min=cq_min, cq_max=cq_max,
            required_feature_names=feature_names, 
            vmaf_prediction_model_path=vmaf_prediction_model_path,
            codec_name=codec_for_this_run, # ✅ Codec-specific optimization
            conservative_cq_limits=True, # Apply codec-specific CQ limits
            logging_enabled=logging_enabled,
            logger=logger 
        )
        if isinstance(result, tuple):
            optimal_cq_result, predicted_vmaf_val = result
        else:
            optimal_cq_result = result
            predicted_vmaf_val = 0.0
        timings['cq_optimization'] = time.time() - step_start
        
        optimal_cq_val = -1
        if isinstance(optimal_cq_result, (int, float)) and optimal_cq_result != -1:
             optimal_cq_val = optimal_cq_result

        # Handle predicted VMAF properly
        if isinstance(predicted_vmaf_val, (int, float)) and predicted_vmaf_val > 0:
            predicted_vmaf_original = predicted_vmaf_val
        elif logging_enabled:
            print(f"      ⚠️ VMAF prediction returned invalid value: {predicted_vmaf_val}")

        # Store optimization results
        scene_data['optimal_cq'] = optimal_cq_val
        scene_data['predicted_vmaf_at_optimal_cq'] = predicted_vmaf_val

        if optimal_cq_val == -1:
            scene_data['notes'] = 'CQ optimization failed'
            scene_data['processing_time_seconds'] = time.time() - analysis_start_time
            if logger: logger.log_scene_processing(scene_path, scene_number, scene_data, "analysis_failed")
            return scene_data, frames_dir_path, timings 
        #Step 7: Perceptual Quality Adjustments
        if logging_enabled: print(f"      🌈 Applying perceptual quality adjustments...")
        step_start = time.time()
        # Ensure calculate_contrast_from_frames is correctly defined or imported
        # from .video_utils import calculate_contrast_from_frames # if it's there
        # Calculate visual contrast from extracted frames
        contrast_val = calculate_contrast_from_frames(extracted_frame_paths) 
        # Apply scene-type specific CQ adjustments
        current_scene_cq_offsets = config.get('scene_cq_offsets', {
            'Screen Content / Text': 2, 'Animation / Cartoon / Rendered Graphics': 0,
            'Faces / People': -1, 'Gaming Content': 1, 'other': 0, 'unclear': 0
        }) if config else {}
        # Calculate final adjusted CQ considering contrast and scene type
        adjusted_cq_val = calculate_contrast_adjusted_cq(
            content_type, contrast_val, optimal_cq_val, current_scene_cq_offsets
        )
        timings['contrast_adjustment'] = time.time() - step_start

        scene_data['contrast_value'] = contrast_val
        # Step 8: Input Compression Analysis
        if logging_enabled:
            print(f"      🔍 Analyzing input compression characteristics...")
        
        # Get compression-aware CQ adjustment
        skip_encoding, final_adjusted_cq = should_skip_encoding(
            scene_path, adjusted_cq_val, target_vmaf, logging_enabled
        )
        
        # Since should_skip_encoding now never returns True, we always encode
        # but use the adjusted CQ value
        if final_adjusted_cq != adjusted_cq_val:
            if logging_enabled:
                print(f"      🔧 Adjusted CQ based on input analysis: {adjusted_cq_val} → {final_adjusted_cq}")
            adjusted_cq_val = final_adjusted_cq
        
        scene_data['adjusted_cq'] = adjusted_cq_val
        scene_data['should_force_encoding'] = True  # Always encode


        scene_data['processing_time_seconds'] = time.time() - analysis_start_time
        if logger: logger.log_scene_processing(scene_path, scene_number, scene_data, "analysis_optimization_complete")

        if logging_enabled:
            print(f"      ✅ Scene analysis & CQ opt. completed for {os.path.basename(scene_path)}:")
            print(f"         Content type: {scene_data.get('scene_type', 'unknown')}, Optimal CQ: {scene_data.get('optimal_cq', 'N/A')} -> Adjusted CQ: {scene_data.get('adjusted_cq', 'N/A')}")
            print(f"         Predicted VMAF: {predicted_vmaf_original:.2f}")  # Add this debug line

        return scene_data, frames_dir_path, timings

    except Exception as e:
        if logging_enabled:
            print(f"      ❌ Error during scene analysis for {os.path.basename(scene_path)}: {e}")
            import traceback
            traceback.print_exc()
        scene_data['notes'] = f'Error: {str(e)}'
        scene_data['processing_time_seconds'] = time.time() - analysis_start_time
        if logger: logger.log_scene_processing(scene_path, scene_number, scene_data, "error")
        
        return scene_data, frames_dir_path, timings

def process_scene_analysis_and_cq_modular(scene_path, scene_number, temp_directory,
                                 scene_classifier_model, pipeline_obj, feature_scaler_step,
                                 vmaf_scaler, cq_min, cq_max, available_metrics,
                                 target_vmaf, device, feature_names, 
                                 vmaf_prediction_model_path: str = None,
                                 logger=None, config=None,codec_name = None, logging_enabled=True):
    """
    Process scene analysis with comprehensive feature extraction, classification,
    and CQ optimization, similar to the analysis part of process_short_video_as_single_scene.
    
    Returns:
        tuple: (scene_data, frames_dir_path, detailed_analysis_data)
    """
    # Step 1: Initialization and Setup
    analysis_start_time = time.time()
    timings = {}
    frames_dir_path = None

    #  Initialize detailed analysis data for model training
    detailed_analysis_data = {
        'raw_video_features': None,
        'processed_video_features': None,
        'vmaf_model_features': None,
        'scene_classifier_features': None,
        'scene_classifier_raw_output': None,
        'scene_classifier_probabilities': None,
        'cq_optimization_steps': [],
        'feature_recovery_attempts': [],
        'contrast_calculation_details': None,
        'input_compression_analysis': None,
        'processing_timings': {},
        'debug_info': {}
    }


    # Determine codec to use (from parameter or config fallback)
    if codec_name:
        codec_for_this_run = codec_name
        if logging_enabled:
            print(f"      🎥 Using provided codec: {codec_for_this_run}")
    else:
        # Extract from config with fallbacks
        codec_for_this_run = (
            config.get('resolved_codec') or 
            config.get('codec') or 
            'libx264'
        ) if config else 'libx264'
        if logging_enabled:
            print(f"      🎥 Using fallback codec: {codec_for_this_run}")


     # Store codec info for debugging
    detailed_analysis_data['debug_info']['codec_selection'] = {
        'provided_codec': codec_name,
        'config_codec': config.get('codec') if config else None,
        'resolved_codec': config.get('resolved_codec') if config else None,
        'final_codec': codec_for_this_run
    }


    # set maximum frames used for video analysis algorithm
    max_frames_for_analysis = 100 
    if config and 'max_frames_for_metrics_analysis' in config:
        max_frames_for_analysis = int(config.get('max_frames_for_metrics_analysis', 100))

    # Initialize scene data structure with default values
    scene_data = {
        'scene_number': scene_number,
        'original_path': scene_path, 
        'duration_seconds': 0.0,
        'file_size_input_mb': 0.0,
        'scene_type': 'unknown',
        'confidence_score': 0.0,
        'prob_screen_content': 0.0,
        'prob_animation': 0.0,
        'prob_faces': 0.0,
        'prob_gaming': 0.0,
        'prob_other': 0.0,
        'prob_unclear': 0.0,
        'optimal_cq': None,
        'predicted_vmaf_at_optimal_cq': None,
        'contrast_value': 0.5,
        'adjusted_cq': None,
        'target_vmaf': target_vmaf,
        'notes': '',
        'metrics_resolution':0,
        'metrics_resolution_width': 0, 'metrics_resolution_height': 0, 'metrics_frame_rate': 0.0,
        'metrics_bit_depth': 8, 'input_bitrate_kbps': 0.0, 'input_codec': 'unknown', 'bits_per_pixel': 0.0,
        'metrics_avg_motion': 0.0, 'metrics_avg_edge_density': 0.0, 'metrics_avg_texture': 0.0,
        'metrics_avg_color_complexity': 0.0, 'metrics_avg_motion_variance': 0.0,
        'metrics_avg_grain_noise': 0.0, 'metrics_avg_spatial_information': 0.0,
        'metrics_avg_temporal_information': 0.0,
    }

    try:
        # Step 2: Video Feature Analysis
        if logging_enabled: print(f"      📊 Analyzing scene features for: {os.path.basename(scene_path)}...")
        step_start = time.time()
        #  Store input compression analysis for debugging
        detailed_analysis_data['input_compression_analysis'] = analyze_input_compression(scene_path)
        # Extract comprehensive video metrics using fast analysis
        features_original = analyze_video_fast(scene_path, max_frames=max_frames_for_analysis, logging_enabled=logging_enabled)
        timings['feature_analysis'] = time.time() - step_start
        detailed_analysis_data['processing_timings']['feature_analysis'] = timings['feature_analysis']

        if features_original is None:
            scene_data['notes'] = 'Feature analysis failed'
            scene_data['processing_time_seconds'] = time.time() - analysis_start_time
            detailed_analysis_data['debug_info']['failure_point'] = 'feature_analysis'
            if logger: logger.log_scene_processing(scene_path, scene_number, scene_data, "analysis_failed")
            return scene_data, None, detailed_analysis_data
        
         # Store raw video features for model training
        detailed_analysis_data['raw_video_features'] = features_original.copy()
        
        # Update scene_data with extracted metrics
        scene_data.update({k: features_original.get(k, scene_data.get(k)) for k in scene_data if k.startswith('metrics_') or k in ['input_bitrate_kbps', 'input_codec', 'bits_per_pixel','metrics_resolution']})
        scene_data['duration_seconds'] = get_video_duration(scene_path) or 0.0
        scene_data['file_size_input_mb'] = os.path.getsize(scene_path) / (1024 * 1024) if os.path.exists(scene_path) else 0.0
        
        # Check for missing critical features
        critical_features_to_check = ['metrics_frame_rate', 'input_bitrate_kbps', 'metrics_resolution']
        missing_features = [f for f in critical_features_to_check if scene_data.get(f, 0) == 0]
        
        if missing_features:
            if logging_enabled: print(f"      ⚠️ Missing critical features: {missing_features}, attempting recovery...")
            detailed_analysis_data['feature_recovery_attempts'].append({
                'missing_features': missing_features,
                'before_recovery': {k: scene_data.get(k) for k in critical_features_to_check}
            })
            
            # Attempt recovery using alternative methods
            recovered = recover_missing_features(scene_path, scene_data, logging_enabled) 
            scene_data.update(recovered)
            
            detailed_analysis_data['feature_recovery_attempts'][-1]['after_recovery'] = {
                'recovered_features': recovered,
                'final_values': {k: scene_data.get(k) for k in critical_features_to_check}
            }
        
        # Log feature extracted data
        if logger: logger.log_scene_processing(scene_path, scene_number, scene_data, "feature_extraction_complete")
        
        # Step 4: Frame Extraction for AI Analysis
        if logging_enabled: print(f"      🖼️ Extracting frames for AI analysis...")
        step_start = time.time()
        
        # Create temporary directory for frame analysis
        frames_dir_path = os.path.join(temp_directory, f"scene_{scene_number}_frames_analysis")
        os.makedirs(frames_dir_path, exist_ok=True)
        
        # Extract representative frames for content classification
        extracted_frame_paths = extract_frames_from_scene(
            scene_path, 0, scene_data['duration_seconds'],
            num_frames=3,  # Extract 3 representative frames
            output_dir=frames_dir_path
        )
        timings['frame_extraction'] = time.time() - step_start
        detailed_analysis_data['processing_timings']['frame_extraction'] = timings['frame_extraction']
        
        if not extracted_frame_paths:
            scene_data['notes'] = 'Frame extraction failed'
            scene_data['processing_time_seconds'] = time.time() - analysis_start_time
            detailed_analysis_data['debug_info']['failure_point'] = 'frame_extraction'
            if logger: logger.log_scene_processing(scene_path, scene_number, scene_data, "analysis_failed")
            return scene_data, frames_dir_path, detailed_analysis_data
        
        # Store frame extraction details
        detailed_analysis_data['debug_info']['frame_extraction'] = {
            'extracted_frames': extracted_frame_paths,
            'frame_count': len(extracted_frame_paths),
            'frames_directory': frames_dir_path
        }
        
        # Step 5: AI-Based Content Classification
        if logging_enabled: print(f"      🎯 Classifying content type using AI...")
        step_start = time.time()
        
        # Store features used for scene classification
        scene_classifier_features = {k: features_original.get(k, 0) for k in available_metrics if k in features_original}
        detailed_analysis_data['scene_classifier_features'] = scene_classifier_features
        
        # Classify scene content using neural network
        classification_result = classify_scene_with_model(
            frame_paths=extracted_frame_paths, 
            video_features=features_original,
            scene_classifier=scene_classifier_model, 
            metrics_scaler=feature_scaler_step,
            available_metrics=available_metrics, 
            device=device, 
            logging_enabled=logging_enabled
        )
        
        timings['scene_classification'] = time.time() - step_start
        detailed_analysis_data['processing_timings']['scene_classification'] = timings['scene_classification']

        content_type = "unknown" 
        if isinstance(classification_result, tuple) and len(classification_result) == 2:
            content_type, classification_details = classification_result
            
            # ✅ NEW: Store detailed classification results
            detailed_analysis_data['scene_classifier_raw_output'] = classification_details
            detailed_analysis_data['scene_classifier_probabilities'] = {
                'prob_screen_content': classification_details.get('prob_screen_content', 0.0),
                'prob_animation': classification_details.get('prob_animation', 0.0),
                'prob_faces': classification_details.get('prob_faces', 0.0),
                'prob_gaming': classification_details.get('prob_gaming', 0.0),
                'prob_other': classification_details.get('prob_other', 0.0),
                'prob_unclear': classification_details.get('prob_unclear', 0.0)
            }
            
            scene_data.update({
                'scene_type': content_type,
                'confidence_score': classification_details.get('confidence_score', 0.0),
                'prob_screen_content': classification_details.get('prob_screen_content', 0.0),
                'prob_animation': classification_details.get('prob_animation', 0.0),
                'prob_faces': classification_details.get('prob_faces', 0.0),
                'prob_gaming': classification_details.get('prob_gaming', 0.0),
                'prob_other': classification_details.get('prob_other', 0.0),
                'prob_unclear': classification_details.get('prob_unclear', 0.0)
            })
        elif isinstance(classification_result, str): 
            content_type = classification_result
            scene_data['scene_type'] = content_type
            scene_data['confidence_score'] = 0.5 
        else:
            scene_data['notes'] = "Classification returned unexpected result"
            detailed_analysis_data['debug_info']['classification_error'] = str(classification_result)
        
        if logger: logger.log_scene_processing(scene_path, scene_number, scene_data, "classification_complete")
        
        # Step 6: Quality Parameter (CQ) Optimization
        if logging_enabled: print(f"      🎚️ Optimizing quality parameters (CQ)...")
        step_start = time.time()
        
        # ✅ NEW: Prepare features for VMAF prediction model
        vmaf_features = {}
        for feature_name in feature_names:
            if feature_name in features_original:
                vmaf_features[feature_name] = features_original[feature_name]
            else:
                # Handle missing features with defaults
                vmaf_features[feature_name] = 0.0
                if logging_enabled:
                    print(f"      ⚠️ Missing VMAF feature: {feature_name}, using default: 0.0")
        
        detailed_analysis_data['vmaf_model_features'] = vmaf_features
        detailed_analysis_data['processed_video_features'] = features_original.copy()
        
        # Find optimal CQ value using machine learning model
        result = find_optimal_cq(
            features_original, 
            target_vmaf_original=target_vmaf,
            pipeline_obj=pipeline_obj, 
            vmaf_scaler=vmaf_scaler,
            cq_min=cq_min, cq_max=cq_max,
            required_feature_names=feature_names, 
            vmaf_prediction_model_path=vmaf_prediction_model_path,
            codec_name=codec_for_this_run,  # Codec-specific optimization
            conservative_cq_limits=True,    # Apply codec-specific CQ limits
            logging_enabled=logging_enabled,
            logger=logger 
        )
        if isinstance(result, tuple):
            optimal_cq_result, predicted_vmaf_val = result
        else:
            optimal_cq_result = result
            predicted_vmaf_val = 0.0
        
        timings['cq_optimization'] = time.time() - step_start
        detailed_analysis_data['processing_timings']['cq_optimization'] = timings['cq_optimization']
        
        # Store CQ optimization details
        detailed_analysis_data['cq_optimization_steps'].append({
            'target_vmaf': target_vmaf,
            'cq_range': {'min': cq_min, 'max': cq_max},
            'codec_used': codec_for_this_run,
            'optimal_cq_result': optimal_cq_result,
            'predicted_vmaf': predicted_vmaf_val,
            'features_used': list(vmaf_features.keys()),
            'feature_values': vmaf_features
        })
        
        optimal_cq_val = -1
        if isinstance(optimal_cq_result, (int, float)) and optimal_cq_result != -1:
             optimal_cq_val = optimal_cq_result

        # Handle predicted VMAF properly
        predicted_vmaf_original = 0.0
        if isinstance(predicted_vmaf_val, (int, float)) and predicted_vmaf_val > 0:
            predicted_vmaf_original = predicted_vmaf_val
        elif logging_enabled:
            print(f"      ⚠️ VMAF prediction returned invalid value: {predicted_vmaf_val}")

        # Store optimization results
        scene_data['optimal_cq'] = optimal_cq_val
        scene_data['predicted_vmaf_at_optimal_cq'] = predicted_vmaf_val

        if optimal_cq_val == -1:
            scene_data['notes'] = 'CQ optimization failed'
            scene_data['processing_time_seconds'] = time.time() - analysis_start_time
            detailed_analysis_data['debug_info']['failure_point'] = 'cq_optimization'
            if logger: logger.log_scene_processing(scene_path, scene_number, scene_data, "analysis_failed")
            return scene_data, frames_dir_path, detailed_analysis_data
        
        # Step 7: Perceptual Quality Adjustments
        if logging_enabled: print(f"      🌈 Applying perceptual quality adjustments...")
        step_start = time.time()
        
        # Calculate visual contrast from extracted frames
        contrast_val = calculate_contrast_from_frames(extracted_frame_paths) 
        
        # Store contrast calculation details
        detailed_analysis_data['contrast_calculation_details'] = {
            'frame_paths_used': extracted_frame_paths,
            'calculated_contrast': contrast_val,
            'method': 'standard_deviation_based'
        }
        
        # Apply scene-type specific CQ adjustments
        current_scene_cq_offsets = config.get('scene_cq_offsets', {
            'Screen Content / Text': 2, 'Animation / Cartoon / Rendered Graphics': 0,
            'Faces / People': -1, 'Gaming Content': 1, 'other': 0, 'unclear': 0
        }) if config else {}
        
        # Calculate final adjusted CQ considering contrast and scene type
        adjusted_cq_val = calculate_contrast_adjusted_cq(
            content_type, contrast_val, optimal_cq_val, current_scene_cq_offsets
        )
        
        timings['contrast_adjustment'] = time.time() - step_start
        detailed_analysis_data['processing_timings']['contrast_adjustment'] = timings['contrast_adjustment']
        
        #  Store contrast adjustment details
        detailed_analysis_data['debug_info']['contrast_adjustment'] = {
            'original_cq': optimal_cq_val,
            'content_type': content_type,
            'contrast_value': contrast_val,
            'scene_cq_offsets': current_scene_cq_offsets,
            'applied_offset': current_scene_cq_offsets.get(content_type, 0),
            'adjusted_cq': adjusted_cq_val
        }

        scene_data['contrast_value'] = contrast_val
        
        # Step 8: Input Compression Analysis
        if logging_enabled:
            print(f"      🔍 Analyzing input compression characteristics...")
        
        # Get compression-aware CQ adjustment
        skip_encoding, final_adjusted_cq = should_skip_encoding(
            scene_path, adjusted_cq_val, target_vmaf, logging_enabled
        )
        
        # Store compression analysis results
        detailed_analysis_data['debug_info']['compression_analysis'] = {
            'skip_encoding_recommended': skip_encoding,
            'pre_compression_cq': adjusted_cq_val,
            'post_compression_cq': final_adjusted_cq,
            'cq_adjustment': final_adjusted_cq - adjusted_cq_val,
            'input_analysis': detailed_analysis_data['input_compression_analysis']
        }
        
        # Since should_skip_encoding now never returns True, we always encode
        # but use the adjusted CQ value
        if final_adjusted_cq != adjusted_cq_val:
            if logging_enabled:
                print(f"      🔧 Adjusted CQ based on input analysis: {adjusted_cq_val} → {final_adjusted_cq}")
            adjusted_cq_val = final_adjusted_cq
        
        scene_data['adjusted_cq'] = adjusted_cq_val
        scene_data['should_force_encoding'] = True  # Always encode

        # Step 9: Finalization
        scene_data['processing_time_seconds'] = time.time() - analysis_start_time
        detailed_analysis_data['processing_timings']['total_time'] = scene_data['processing_time_seconds']
        
        if logger: logger.log_scene_processing(scene_path, scene_number, scene_data, "analysis_optimization_complete")

        if logging_enabled:
            print(f"      ✅ Scene analysis & CQ opt. completed for {os.path.basename(scene_path)}:")
            print(f"         Content type: {scene_data.get('scene_type', 'unknown')}, Optimal CQ: {scene_data.get('optimal_cq', 'N/A')} -> Adjusted CQ: {scene_data.get('adjusted_cq', 'N/A')}")
            print(f"         Predicted VMAF: {predicted_vmaf_original:.2f}")

        #  add summary to detailed analysis data
        detailed_analysis_data['debug_info']['processing_summary'] = {
            'success': True,
            'total_processing_time': scene_data['processing_time_seconds'],
            'feature_count': len(features_original) if features_original else 0,
            'vmaf_features_count': len(vmaf_features),
            'frames_extracted': len(extracted_frame_paths),
            'classification_confidence': scene_data.get('confidence_score', 0.0),
            'cq_progression': f"{optimal_cq_val} → {adjusted_cq_val}",
            'predicted_vs_target_vmaf': f"{predicted_vmaf_original:.1f} vs {target_vmaf:.1f}"
        }

        return scene_data, frames_dir_path, detailed_analysis_data

    except Exception as e:
        if logging_enabled:
            print(f"      ❌ Error during scene analysis for {os.path.basename(scene_path)}: {e}")
            import traceback
            traceback.print_exc()
        
        scene_data['notes'] = f'Error: {str(e)}'
        scene_data['processing_time_seconds'] = time.time() - analysis_start_time
        
        # ✅ NEW: Store error details for debugging
        detailed_analysis_data['debug_info']['error_details'] = {
            'error_message': str(e),
            'error_type': type(e).__name__,
            'processing_time_at_error': time.time() - analysis_start_time,
            'traceback': traceback.format_exc() if logging_enabled else None
        }
        
        if logger: logger.log_scene_processing(scene_path, scene_number, scene_data, "error")
        
        return scene_data, frames_dir_path, detailed_analysis_data

def recover_missing_features(video_path, original_features, logging_enabled=True):
    """
    Attempt to recover missing video features using alternative methods.
    """
    recovered = {}
    
    try:
        import subprocess
        import json
        import os
        
        # Try ffprobe with different parameters
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', '-select_streams', 'v:0', video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get('streams', [])
            
            if streams:
                stream = streams[0]
                
                # Recover resolution
                if 'width' in stream and 'height' in stream:
                    recovered['metrics_resolution_width'] = int(stream['width'])
                    recovered['metrics_resolution_height'] = int(stream['height'])
                
                # Recover frame rate
                if 'r_frame_rate' in stream:
                    frame_rate_str = stream['r_frame_rate']
                    if '/' in frame_rate_str:
                        num, den = frame_rate_str.split('/')
                        recovered['metrics_frame_rate'] = float(num) / float(den) if float(den) != 0 else 0
                    else:
                        recovered['metrics_frame_rate'] = float(frame_rate_str)
                
                # Recover bitrate
                if 'bit_rate' in stream:
                    recovered['input_bitrate_kbps'] = int(stream['bit_rate']) / 1000
                elif 'tags' in stream and 'BPS' in stream['tags']:
                    recovered['input_bitrate_kbps'] = int(stream['tags']['BPS']) / 1000
        
        # Estimate bitrate from file size if still missing
        if recovered.get('input_bitrate_kbps', 0) == 0:
            file_size_bytes = os.path.getsize(video_path) if os.path.exists(video_path) else 0
            duration = original_features.get('duration_seconds', 0)
            
            if file_size_bytes > 0 and duration > 0:
                bitrate_bps = (file_size_bytes * 8) / duration
                recovered['input_bitrate_kbps'] = bitrate_bps / 1000
        
        # Recalculate bits per pixel if we have the necessary components
        if (recovered.get('metrics_resolution_width', 0) > 0 and 
            recovered.get('metrics_resolution_height', 0) > 0 and 
            recovered.get('metrics_frame_rate', 0) > 0 and 
            recovered.get('input_bitrate_kbps', 0) > 0):
            
            pixels_per_second = (recovered['metrics_resolution_width'] * 
                               recovered['metrics_resolution_height'] * 
                               recovered['metrics_frame_rate'])
            recovered['bits_per_pixel'] = (recovered['input_bitrate_kbps'] * 1000) / pixels_per_second
        
        if logging_enabled and recovered:
            print(f"      🔧 Recovered features: {list(recovered.keys())}")
        
        return recovered
        
    except Exception as e:
        if logging_enabled:
            print(f"      ⚠️ Feature recovery failed: {e}")
        return {}


def calculate_contrast_from_frames(frame_paths):
    """Calculate contrast from frames without storing frame data."""
    if not frame_paths:
        return 0.5
    
    try:
        contrast_values = []
        for frame_path in frame_paths:
            if os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
                if frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    contrast = np.std(gray) / 255.0
                    contrast_values.append(contrast)
        
        return np.mean(contrast_values) if contrast_values else 0.5
            
    except Exception:
        return 0.5


def should_skip_encoding(input_path, estimated_cq, target_vmaf, logging_enabled=True):
    """Determine if encoding should be skipped to prevent size increase."""
    
    input_analysis = analyze_input_compression(input_path)
    if not input_analysis:
        return False, estimated_cq + 1
    
    input_bitrate = input_analysis.get('bitrate_kbps', 0)
    bits_per_pixel = input_analysis.get('bits_per_pixel', 0)
    codec = input_analysis.get('codec', '').lower()
    
    # FIXED: More conservative thresholds - only skip for extremely compressed content
    if bits_per_pixel and bits_per_pixel < 0.02 and target_vmaf < 90:  # CHANGED: More conservative threshold
        if logging_enabled:
            print(f"⚠️ Input very well compressed (bits/pixel: {bits_per_pixel:.4f}). Encoding anyway with conservative CQ.")
        # CHANGED: Don't skip, just add more conservative CQ adjustment
        return False, min(estimated_cq + 8, 51)  # Add significant CQ increase but don't skip
    
    # For very high target VMAF, never skip encoding
    if target_vmaf >= 90:
        if logging_enabled:
            print(f"🎯 High target VMAF ({target_vmaf}) - encoding anyway with adjusted CQ")
        return False, estimated_cq
    
    adjusted_cq = calculate_bitrate_aware_cq(estimated_cq, input_analysis, target_vmaf, "unknown")
    
    # Add safety margin for high bitrate content
    if input_bitrate > 10000:  # Very high bitrate content
        adjusted_cq = min(adjusted_cq + 4, 51)
        if logging_enabled:
            print(f"Very high bitrate input ({input_bitrate:.0f} kbps) - adding safety margin. CQ: {adjusted_cq}")
    
    return False, adjusted_cq  # CHANGED: Never skip encoding, always process



def encode_scene_with_size_check(scene_path, output_path, codec, adjusted_cq, content_type, contrast_value, max_retries=2, logging_enabled=True):
    """Enhanced encoding with size checking and CQ adjustment."""
    
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
    
    # ENHANCED: More reasonable size ratio threshold for high-bitrate content
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


def calculate_vmaf_for_scenes_df(scenes_df,
                            use_sampling=True,
                            num_clips=3,
                            clip_duration=2,
                            use_downscaling=True,
                            scale_factor=0.5,
                            use_vmafneg=False,
                            default_vmaf_model_path_config=None,
                            vmafneg_model_path_config=None,
                            use_frame_rate_scaling=False,       
                            target_fps=15.0,                     
                            frame_rate_scaling_method='uniform', 
                            logger=None,
                            logging_enabled=True):
    """
    Calculate VMAF scores for all scenes
    """
    
    def log_message(message, level="info"):
        if logger:
            if level == "error": logger.error(message)
            elif level == "warning": logger.warning(message)
            elif level == "debug" and hasattr(logger, 'debug'): logger.debug(message)
            else: logger.info(message)
        elif logging_enabled:
            safe_log_message(message, level)

  
   
    log_message(f"\nCalculating VMAF scores using standard clip sampling ({num_clips} clips per scene)...")

    if scenes_df is None or scenes_df.empty:
        log_message("VMAF calculation: scenes_df is empty or None.", "warning")
        return scenes_df

    required_cols = ['original_path', 'encoded_path', 'encoding_success']
    for col in required_cols:
        if col not in scenes_df.columns:
            log_message(f"VMAF calculation: Missing required column '{col}' in scenes_df.", "error")
            if 'vmaf_score' not in scenes_df.columns:
                 scenes_df['vmaf_score'] = pd.NA
            return scenes_df

    successful_encodes_df = scenes_df[
        scenes_df['encoding_success'].eq(True) &
        scenes_df['original_path'].notna() & scenes_df['original_path'].ne('') &
        scenes_df['encoded_path'].notna() & scenes_df['encoded_path'].ne('')
    ].copy()

    if successful_encodes_df.empty:
        log_message("VMAF calculation: No successfully encoded scenes with valid paths to process.")
        if 'vmaf_score' not in scenes_df.columns:
             scenes_df['vmaf_score'] = pd.NA
        return scenes_df

    log_message(f"Found {len(successful_encodes_df)} successfully encoded scenes for VMAF calculation.")

    reference_paths = successful_encodes_df['original_path'].tolist()
    encoded_paths = successful_encodes_df['encoded_path'].tolist()
    
    
    log_message(f"Using standard parallel VMAF calculation method")
    
    vmaf_scores_list = calculate_parallel_vmaf(
        reference_scenes=reference_paths,
        encoded_files=encoded_paths,
        use_downscaling=use_downscaling,
        scale_factor=scale_factor,
        use_vmafneg=use_vmafneg,
        default_vmaf_model_path_config=default_vmaf_model_path_config,
        vmafneg_model_path_config=vmafneg_model_path_config,
        use_frame_rate_scaling=use_frame_rate_scaling,     
        target_fps=target_fps,                                  
        frame_rate_scaling_method=frame_rate_scaling_method,    
        logger=logger,
        logging_enabled=logging_enabled
    )

    # =================================================================
    # PROCESS RESULTS (SAME FOR BOTH METHODS)
    # =================================================================
    
    # Initialize actual_vmaf column in the original DataFrame if it doesn't exist
    if 'actual_vmaf' not in scenes_df.columns:
        scenes_df['actual_vmaf'] = pd.NA

    # Map scores back to the successful_encodes_df first, then update the original scenes_df
    if vmaf_scores_list is not None and len(vmaf_scores_list) == len(successful_encodes_df):
        # Ensure scores are numeric, convert None from VMAF calc to pd.NA
        processed_scores = [score if score is not None else pd.NA for score in vmaf_scores_list]
        successful_encodes_df['actual_vmaf'] = processed_scores
        
        # Update the original DataFrame using the index from successful_encodes_df
        scenes_df.loc[successful_encodes_df.index, 'actual_vmaf'] = successful_encodes_df['actual_vmaf']
        
        log_message(f"VMAF scores updated for {len(processed_scores)} scenes.")
        
        # Log individual scene results with prediction comparison
        for idx, (scene_idx, row) in enumerate(successful_encodes_df.iterrows()):
            predicted = row.get('predicted_vmaf_at_optimal_cq', 0)
            actual = processed_scores[idx] if pd.notna(processed_scores[idx]) else 0
            error = abs(actual - predicted) if predicted > 0 and actual > 0 else 0
            log_message(f"Scene {row.get('scene_number', idx+1)}: VMAF {actual:.2f} (predicted: {predicted:.2f}, error: ±{error:.2f})")
    elif vmaf_scores_list is None:
        log_message("VMAF calculation returned None. Scores not updated.", "warning")
    else:
        log_message(f"VMAF scores list length mismatch (expected {len(successful_encodes_df)}, got {len(vmaf_scores_list)}) or empty. VMAF scores not updated.", "warning")
        
    # Force garbage collection after potentially heavy processing
    gc.collect()
    
    return scenes_df

def is_streamlit_context():
    """Check if we're running in a Streamlit context safely."""
    try:
        import streamlit as st
        _ = st.session_state
        return True
    except (ImportError, AttributeError, RuntimeError):
        return False

def safe_log_message(message, level="info"):
    """Safe logging that works with or without Streamlit context."""
    if is_streamlit_context():
        try:
            import streamlit as st
            if level == "error":
                st.error(message)
            elif level == "warning":
                st.warning(message)
            else:
                st.info(message)
        except:
            print(message)
    else:
        print(message)
