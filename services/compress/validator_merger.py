"""
Validation and Merging Module

This module handles the final stage of video processing including:
- Scene validation and quality checks
- Video merging from encoded scenes
- Final VMAF calculation
- Comprehensive report generation
"""

import os
import json
import time
import signal
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from utils.calculate_vmaf_adv import calculate_vmaf_advanced
from utils.merge_videos import merge_videos


# ============================================================================
# Constants
# ============================================================================

# Timeout settings
MERGE_TIMEOUT_SECONDS = 300  # 5 minutes
SIGNAL_TIMEOUT_SECONDS = 10


# ============================================================================
# Main Function
# ============================================================================

def validation_and_merging(
    original_video_path: str, 
    encoded_scenes_data: List[Dict[str, Any]], 
    config: Dict[str, Any], 
    logging_enabled: bool = True
) -> Tuple[Optional[str], Optional[float], Optional[Dict[str, Any]]]:
    """
    Part 4: Validation and merging of encoded scenes with existing VMAF data.
    
    Args:
        original_video_path: Path to the original video file
        encoded_scenes_data: List of scene data dictionaries from Part 3
        config: Configuration dictionary
        logging_enabled: Whether to enable detailed logging
        
    Returns:
        tuple: (final_video_path, final_vmaf, comprehensive_report)
    """
    if logging_enabled:
        print(f"\n🔗 --- Part 4: Validation and Merging ---")
        print(f"   🎬 Processing {len(encoded_scenes_data)} scenes with VMAF data")
    
    # Get configuration settings
    vmaf_config = config.get('vmaf_calculation', {})
    output_dir = config.get('directories', {}).get('output_dir', './output')
    temp_dir = config.get('directories', {}).get('temp_dir', './videos/temp_scenes')
    save_individual_reports = config.get('output_settings', {}).get('save_individual_scene_reports', True)
    save_comprehensive_report = config.get('output_settings', {}).get('save_comprehensive_report', True)
    cleanup_temp_files = config.get('output_settings', {}).get('cleanup_temp_files', True)
    
    # Extract base video name for reports
    original_video_name = os.path.splitext(os.path.basename(original_video_path))[0]
    
    # Check if full video VMAF calculation is enabled
    calculate_full_video_vmaf = config.get('vmaf_calculation', {}).get('calculate_full_video_vmaf', True)
    
    if logging_enabled:
        print(f"calculate_full_video_vmaf flag: {calculate_full_video_vmaf}")
        print(f"   📁 Output directory: {output_dir}")
        print(f"   📁 Temp directory: {temp_dir}")
    
    # Create output directory with error handling
    if not _create_output_directory(output_dir, logging_enabled):
        return None, None, None
    
    # Initialize processing start time
    part5_start_time = time.time()
    
    # Validate encoded scenes
    successful_scenes, failed_scenes, temp_files_to_cleanup = _validate_encoded_scenes(
        encoded_scenes_data, logging_enabled
    )
    
    if not successful_scenes:
        if logging_enabled:
            print("❌ No valid scenes found for merging")
        return None, None, None
    
    # Merge video scenes
    final_output_path, merge_time, merged_size = _merge_video_scenes(
        successful_scenes, output_dir, original_video_name, logging_enabled
    )
    
    if not final_output_path:
        return None, None, None
    
    # Clean up temporary files if enabled
    if cleanup_temp_files:
        _cleanup_temp_files(temp_files_to_cleanup, successful_scenes, logging_enabled)
    
    # Calculate final VMAF
    final_vmaf, final_vmaf_time = _calculate_final_vmaf(
        original_video_path, final_output_path, successful_scenes, 
        calculate_full_video_vmaf, vmaf_config, config, logging_enabled
    )
    
    # Generate comprehensive report
    comprehensive_report = _generate_comprehensive_report(
        original_video_path, final_output_path, successful_scenes, failed_scenes,
        part5_start_time, merge_time, final_vmaf_time, config, logging_enabled
    )
    
    # Save reports if enabled
    if save_comprehensive_report:
        _save_reports(comprehensive_report, output_dir, original_video_name, logging_enabled)
    
    if logging_enabled:
        print(f"✅ Part 4 completed successfully!")
        print(f"   📁 Final video: {os.path.basename(final_output_path)}")
        print(f"   🎯 Final VMAF: {final_vmaf:.2f}" if final_vmaf else "   🎯 Final VMAF: Not calculated")
    
    return final_output_path, final_vmaf, comprehensive_report


# ============================================================================
# Core Processing Functions
# ============================================================================

def _create_output_directory(output_dir: str, logging_enabled: bool) -> bool:
    """Create output directory with error handling."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        if logging_enabled:
            print(f"   ✅ Output directory ready: {output_dir}")
        return True
    except Exception as e:
        if logging_enabled:
            print(f"   ❌ Failed to create output directory: {e}")
        return False


def _validate_encoded_scenes(
    encoded_scenes_data: List[Dict[str, Any]], 
    logging_enabled: bool
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """Validate encoded scenes and categorize them."""
    if logging_enabled:
        print(f"   🔍 Validating encoded scenes...")
    
    successful_scenes = []
    failed_scenes = []
    temp_files_to_cleanup = []
    
    for i, scene_data in enumerate(encoded_scenes_data):
        scene_num = scene_data.get('scene_number', i+1)
        
        if logging_enabled:
            print(f"      Validating scene {scene_num}...")
        
        if scene_data.get('encoding_success', False) and scene_data.get('encoded_path'):
            encoded_path = scene_data.get('encoded_path')
            
            # Enhanced file validation with timeout protection
            if _validate_scene_file(encoded_path, scene_num, logging_enabled):
                successful_scenes.append(scene_data)
                
                # Only add original scene file (temp split files) to cleanup
                original_scene_path = scene_data.get('path')
                if _should_cleanup_scene_file(original_scene_path, encoded_path):
                    temp_files_to_cleanup.append(original_scene_path)
            else:
                failed_scenes.append(scene_data)
        else:
            if logging_enabled:
                print(f"         ❌ Scene {scene_num} encoding failed")
            failed_scenes.append(scene_data)
    
    if logging_enabled:
        print(f"   📊 Scene validation complete: {len(successful_scenes)} successful, {len(failed_scenes)} failed")
    
    return successful_scenes, failed_scenes, temp_files_to_cleanup


def _validate_scene_file(encoded_path: str, scene_num: int, logging_enabled: bool) -> bool:
    """Validate a single scene file."""
    try:
        if os.path.exists(encoded_path):
            file_size = os.path.getsize(encoded_path)
            if file_size > 0:
                if logging_enabled:
                    print(f"         ✅ Scene {scene_num} valid ({file_size/1024/1024:.1f} MB)")
                return True
            else:
                if logging_enabled:
                    print(f"         ❌ Scene {scene_num} file is empty")
                return False
        else:
            if logging_enabled:
                print(f"         ❌ Scene {scene_num} file not found: {encoded_path}")
            return False
    except Exception as e:
        if logging_enabled:
            print(f"         ❌ Scene {scene_num} validation error: {e}")
        return False


def _should_cleanup_scene_file(original_scene_path: Optional[str], encoded_path: str) -> bool:
    """Determine if a scene file should be cleaned up."""
    return (
        original_scene_path and 
        os.path.exists(original_scene_path) and 
        original_scene_path != encoded_path and  # Don't delete if same file
        'temp_scenes' in original_scene_path  # Only delete temp scene splits
    )


def _merge_video_scenes(
    successful_scenes: List[Dict[str, Any]], 
    output_dir: str, 
    original_video_name: str, 
    logging_enabled: bool
) -> Tuple[Optional[str], float, float]:
    """Merge video scenes into final output."""
    if logging_enabled:
        print(f"   🔗 Merging {len(successful_scenes)} scenes into final video...")
    
    # Generate output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_output_filename = f"{original_video_name}_compressed_{timestamp}.mp4"
    final_output_path = os.path.join(output_dir, final_output_filename)
    
    if logging_enabled:
        print(f"   📁 Output: {final_output_filename}")
    
    # Set up timeout handler for merging
    def timeout_handler(signum, frame):
        raise TimeoutError("Video merging timed out")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(MERGE_TIMEOUT_SECONDS)
    
    try:
        merge_start_time = time.time()
        
        # Perform video merging
        merge_success = merge_videos(
            scene_videos=[scene.get('encoded_path') for scene in successful_scenes],
            output_video=final_output_path
        )
        
        merge_time = time.time() - merge_start_time
        signal.alarm(0)  # Cancel timeout
        
        if not merge_success or not os.path.exists(final_output_path):
            if logging_enabled:
                print(f"   ❌ Video merging failed")
            return None, merge_time, 0.0
        
        # Get merged file size
        merged_size = os.path.getsize(final_output_path) / (1024 * 1024)  # MB
        
        if logging_enabled:
            print(f"   ✅ Video merging completed in {merge_time:.1f}s")
            print(f"   📁 Final video: {final_output_filename} ({merged_size:.1f} MB)")
        
        return final_output_path, merge_time, merged_size
        
    except TimeoutError:
        signal.alarm(0)
        if logging_enabled:
            print(f"   ❌ Video merging timed out after {MERGE_TIMEOUT_SECONDS} minutes!")
        return None, 0.0, 0.0
    except Exception as e:
        signal.alarm(0)
        if logging_enabled:
            print(f"   ❌ Video merging failed with error: {e}")
            import traceback
            traceback.print_exc()
        return None, 0.0, 0.0


def _cleanup_temp_files(
    temp_files_to_cleanup: List[str], 
    successful_scenes: List[Dict[str, Any]], 
    logging_enabled: bool
) -> None:
    """Clean up temporary files."""
    if logging_enabled:
        print(f"   🧹 Cleaning up temporary files...")
    
    # Clean up original scene files
    temp_cleanup_success = 0
    temp_cleanup_failed = 0
    
    for temp_file in temp_files_to_cleanup:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                temp_cleanup_success += 1
                if logging_enabled:
                    print(f"      🗑️ Deleted: {os.path.basename(temp_file)}")
            except Exception as e:
                temp_cleanup_failed += 1
                if logging_enabled:
                    print(f"      ❌ Failed to delete {os.path.basename(temp_file)}: {e}")
    
    # Clean up encoded scene files
    if logging_enabled:
        print(f"   🧹 Cleaning up encoded scene files...")
    
    encoded_cleanup_success = 0
    encoded_cleanup_failed = 0
    
    for scene_data in successful_scenes:
        encoded_path = scene_data.get('encoded_path')
        if encoded_path and os.path.exists(encoded_path):
            try:
                os.remove(encoded_path)
                encoded_cleanup_success += 1
                if logging_enabled:
                    print(f"      🗑️ Deleted: {os.path.basename(encoded_path)}")
            except Exception as e:
                encoded_cleanup_failed += 1
                if logging_enabled:
                    print(f"      ❌ Failed to delete {os.path.basename(encoded_path)}: {e}")
    
    if logging_enabled:
        print(f"   ✅ Cleanup complete: {temp_cleanup_success + encoded_cleanup_success} deleted, {temp_cleanup_failed + encoded_cleanup_failed} failed")


def _calculate_final_vmaf(
    original_video_path: str, 
    final_output_path: str, 
    successful_scenes: List[Dict[str, Any]], 
    calculate_full_video_vmaf: bool, 
    vmaf_config: Dict[str, Any], 
    config: Dict[str, Any], 
    logging_enabled: bool
) -> Tuple[Optional[float], float]:
    """Calculate final VMAF score."""
    final_vmaf = None
    final_vmaf_time = 0
    
    calculate_full_video_vmaf = False
    
    if calculate_full_video_vmaf:
        if logging_enabled:
            print(f"   📊 Starting final VMAF calculation...")
        
        try:
            final_vmaf_start_time = time.time()
            
            final_vmaf = calculate_vmaf_advanced(
                input_file=original_video_path,
                encoded_file=final_output_path,
                use_downscaling=vmaf_config.get('vmaf_use_downscaling', True),
                scale_factor=vmaf_config.get('vmaf_downscaling_scale_factor', 0.5),
                use_vmafneg=config.get('vmaf_models', {}).get('use_neg_by_default', False),
                default_vmaf_model_path_config=config.get('vmaf_models', {}).get('default_model_path'),
                vmafneg_model_path_config=config.get('vmaf_models', {}).get('neg_model_path'),
                use_frame_rate_scaling=vmaf_config.get('vmaf_use_frame_rate_scaling', False),
                target_fps=vmaf_config.get('vmaf_target_fps', 15.0),
                frame_rate_scaling_method=vmaf_config.get('vmaf_frame_rate_scaling_method', 'uniform'),
                logger=None,
                logging_enabled=logging_enabled
            )
            final_vmaf_time = time.time() - final_vmaf_start_time
            
            if logging_enabled:
                print(f"   ✅ Final VMAF calculation completed in {final_vmaf_time:.1f}s")
                print(f"   🎯 Final VMAF score: {final_vmaf:.2f}")
        
        except Exception as e:
            if logging_enabled:
                print(f"   ❌ Final VMAF calculation failed: {e}")
            final_vmaf = None
    else:
        if logging_enabled:
            print(f"   ⏭️ Skipping full video VMAF calculation (disabled in config)")
    
    # Estimate final VMAF from scene averages if available
    if not final_vmaf:
        scene_vmafs = [scene.get('actual_vmaf', 0) for scene in successful_scenes if scene.get('actual_vmaf', 0) > 0]
        if scene_vmafs:
            final_vmaf = sum(scene_vmafs) / len(scene_vmafs)
            if logging_enabled:
                print(f"   📊 Estimated VMAF from scene average: {final_vmaf:.2f}")
        else:
            if logging_enabled:
                print(f"   ⏭️ No scene VMAF data available for estimation")
    
    return final_vmaf, final_vmaf_time


def _generate_comprehensive_report(
    original_video_path: str, 
    final_output_path: str, 
    successful_scenes: List[Dict[str, Any]], 
    failed_scenes: List[Dict[str, Any]], 
    part5_start_time: float, 
    merge_time: float, 
    final_vmaf_time: float, 
    config: Dict[str, Any], 
    logging_enabled: bool
) -> Dict[str, Any]:
    """Generate comprehensive processing report."""
    if logging_enabled:
        print(f"   📋 Generating comprehensive report...")

    total_processing_time = time.time() - part5_start_time

    # Calculate summary statistics
    input_size_total = sum(scene.get('input_size_mb', 0) for scene in successful_scenes)
    output_size_total = sum(scene.get('encoded_file_size_mb', 0) for scene in successful_scenes)
    overall_compression_ratio = ((input_size_total - output_size_total) / input_size_total * 100) if input_size_total > 0 else 0

    # Calculate VMAF statistics
    scenes_with_valid_vmaf = [s for s in successful_scenes if s.get('actual_vmaf') is not None]
    average_scene_vmaf = sum(s['actual_vmaf'] for s in scenes_with_valid_vmaf) / len(scenes_with_valid_vmaf) if scenes_with_valid_vmaf else 0
    target_vmaf = config.get('video_processing', {}).get('target_vmaf', 93.0)
    scenes_meeting_target = sum(1 for s in scenes_with_valid_vmaf if s['actual_vmaf'] >= target_vmaf)

    # Extract training data from scenes
    training_data = _extract_training_data_from_scenes(successful_scenes)

    # Create comprehensive report
    comprehensive_report = {
        'processing_summary': {
            'original_video_path': original_video_path,
            'final_video_path': final_output_path,
            'total_processing_time_seconds': total_processing_time,
            'merge_time_seconds': merge_time,
            'final_vmaf_calculation_time_seconds': final_vmaf_time,
            'timestamp': datetime.now().isoformat()
        },
        'compression_metrics': {
            'input_size_total_mb': input_size_total,
            'output_size_total_mb': output_size_total,
            'overall_compression_ratio_percent': overall_compression_ratio,
            'scenes_processed': len(successful_scenes),
            'scenes_failed': len(failed_scenes)
        },
        'quality_metrics': {
            'final_vmaf': None,  # Will be updated later
            'average_scene_vmaf': average_scene_vmaf,
            'scenes_meeting_target': scenes_meeting_target,
            'total_scenes_with_vmaf': len(scenes_with_valid_vmaf),
            'target_vmaf': target_vmaf
        },
        'scene_analysis': {
            'successful_scenes': successful_scenes,
            'failed_scenes': failed_scenes,
            'training_data': training_data
        }
    }

    return comprehensive_report


def _extract_training_data_from_scenes(scenes_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract comprehensive training data directly from scene data."""
    training_data = {
        'pipeline_stages_data': {
            'part1_preprocessing': [],
            'part2_scene_detection': [],
            'part3_ai_encoding': [],
            'part4_vmaf_calculation': [],
            'part5_merging': []
        },
        'model_training_features': [],
        'prediction_accuracy_analysis': [],
        'scene_classification_results': [],
        'encoding_optimization_data': [],
        'vmaf_validation_data': [],
        'feature_extraction_performance': []
    }
    
    for scene in scenes_data:
        scene_num = scene.get('scene_number', 'unknown')
        
        # Extract Part 1 data (preprocessing info)
        if 'original_video_metadata' in scene:
            part1_data = scene['original_video_metadata'].copy()
            part1_data['scene_number'] = scene_num
            training_data['pipeline_stages_data']['part1_preprocessing'].append(part1_data)
        
        # Extract Part 2 data (scene detection)
        part2_data = {
            'scene_number': scene_num,
            'start_time': scene.get('start_time'),
            'end_time': scene.get('end_time'),
            'duration': scene.get('duration'),
            'scene_codec': scene.get('scene_codec'),
            'scene_container': scene.get('scene_container'),
            'file_size_mb': scene.get('file_size_mb'),
            'scene_detection_method': 'content_based' if len(scenes_data) > 1 else 'time_based'
        }
        training_data['pipeline_stages_data']['part2_scene_detection'].append(part2_data)
        
        # Extract Part 3 data (AI encoding) - this is the richest data source
        if scene.get('encoding_success', False):
            part3_data = {
                'scene_number': scene_num,
                'scene_type': scene.get('scene_type'),
                'codec_used': scene.get('codec_used'),
                'final_adjusted_cq': scene.get('final_adjusted_cq'),
                'predicted_vmaf_at_optimal_cq': scene.get('predicted_vmaf_at_optimal_cq'),
                'processing_time_seconds': scene.get('processing_time_seconds'),
                'compression_ratio': scene.get('compression_ratio'),
                'size_ratio': scene.get('size_ratio'),
                'encoding_success': True,
                'input_size_mb': scene.get('input_size_mb'),
                'encoded_file_size_mb': scene.get('encoded_file_size_mb')
            }
            training_data['pipeline_stages_data']['part3_ai_encoding'].append(part3_data)
            
            # Extract model training data if available
            model_data = scene.get('model_training_data', {})
            if model_data:
                _extract_model_training_data(training_data, scene_num, model_data, scene)
    
    return training_data


def _extract_model_training_data(
    training_data: Dict[str, Any], 
    scene_num: Any, 
    model_data: Dict[str, Any], 
    scene: Dict[str, Any]
) -> None:
    """Extract model training data from scene."""
    # Extract model features
    model_features = {
        'scene_number': scene_num,
        'raw_video_features': model_data.get('raw_video_features', {}),
        'processed_video_features': model_data.get('processed_video_features', {}),
        'vmaf_model_features': model_data.get('vmaf_model_features', {}),
        'scene_classifier_features': model_data.get('scene_classifier_features', {}),
        'scene_classifier_probabilities': model_data.get('scene_classifier_probabilities', {}),
        'processing_timings': model_data.get('processing_timings', {})
    }
    training_data['model_training_features'].append(model_features)
    
    # Extract prediction accuracy
    prediction_acc = model_data.get('prediction_accuracy', {})
    if prediction_acc:
        training_data['prediction_accuracy_analysis'].append({
            'scene_number': scene_num,
            'predicted_vmaf': prediction_acc.get('predicted_vmaf'),
            'prediction_error': prediction_acc.get('prediction_error'),
            'target_vmaf': scene.get('target_vmaf'),
            'actual_vmaf': scene.get('actual_vmaf')
        })
    
    # Extract feature extraction performance
    if model_data.get('feature_recovery_attempts'):
        training_data['feature_extraction_performance'].append({
            'scene_number': scene_num,
            'recovery_attempts': model_data.get('feature_recovery_attempts', []),
            'feature_extraction_successful': bool(model_data.get('raw_video_features'))
        })


def _save_reports(
    comprehensive_report: Dict[str, Any], 
    output_dir: str, 
    original_video_name: str, 
    logging_enabled: bool
) -> None:
    """Save comprehensive and individual reports."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save comprehensive report
    comprehensive_report_path = os.path.join(
        output_dir, 
        f"comprehensive_processing_report_{original_video_name}_{timestamp}.json"
    )
    
    try:
        with open(comprehensive_report_path, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        if logging_enabled:
            print(f"   📊 Comprehensive report saved: {os.path.basename(comprehensive_report_path)}")
    except Exception as e:
        if logging_enabled:
            print(f"   ❌ Failed to save comprehensive report: {e}")
    
    # Save individual scene reports if enabled
    if comprehensive_report.get('scene_analysis', {}).get('successful_scenes'):
        _save_individual_scene_reports(
            comprehensive_report['scene_analysis']['successful_scenes'],
            output_dir, original_video_name, timestamp, logging_enabled
        )


def _save_individual_scene_reports(
    successful_scenes: List[Dict[str, Any]], 
    output_dir: str, 
    original_video_name: str, 
    timestamp: str, 
    logging_enabled: bool
) -> None:
    """Save individual scene reports."""
    scene_reports_dir = os.path.join(output_dir, 'scene_reports')
    os.makedirs(scene_reports_dir, exist_ok=True)
    
    for scene in successful_scenes:
        scene_num = scene.get('scene_number', 'unknown')
        scene_report_path = os.path.join(
            scene_reports_dir, 
            f"scene_{scene_num}_report_{original_video_name}_{timestamp}.json"
        )
        
        try:
            with open(scene_report_path, 'w') as f:
                json.dump(scene, f, indent=2, default=str)
        except Exception as e:
            if logging_enabled:
                print(f"   ❌ Failed to save scene {scene_num} report: {e}")
    
    if logging_enabled:
        print(f"   📄 Individual scene reports saved in: scene_reports/")



