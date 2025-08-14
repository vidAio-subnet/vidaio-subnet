import os
import json
import time
from datetime import datetime
from utils.calculate_vmaf_adv import calculate_vmaf_advanced
from utils.merge_videos import merge_videos
import signal

def validation_and_merging(original_video_path, encoded_scenes_data, config, logging_enabled=True):
    """
    Part 4: Validation and merging of encoded scenes with existing VMAF data.
    """
    logging_enabled=True  # Ensure logging is enabled for this part
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
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        if logging_enabled:
            print(f"   ✅ Output directory ready: {output_dir}")
    except Exception as e:
        if logging_enabled:
            print(f"   ❌ Failed to create output directory: {e}")
        return None, None, None
    
    # Initialize processing start time
    part5_start_time = time.time()
    
    # ===== SCENE VALIDATION =====
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
            
            try:
                if os.path.exists(encoded_path):
                    file_size = os.path.getsize(encoded_path)
                    if file_size > 0:
                        successful_scenes.append(scene_data)
                        # ❌ REMOVED: Don't add encoded files to general cleanup
                        # temp_files_to_cleanup.append(encoded_path)  # REMOVED THIS LINE
                        
                        # Only add original scene file (temp split files) to cleanup
                        original_scene_path = scene_data.get('path')
                        if (original_scene_path and 
                            os.path.exists(original_scene_path) and 
                            original_scene_path != encoded_path and  # Don't delete if same file
                            'temp_scenes' in original_scene_path):  # Only delete temp scene splits
                            temp_files_to_cleanup.append(original_scene_path)
                        
                        if logging_enabled:
                            print(f"         ✅ Scene {scene_num} valid ({file_size/1024/1024:.1f} MB)")
                    else:
                        if logging_enabled:
                            print(f"         ❌ Scene {scene_num} file is empty")
                        failed_scenes.append(scene_data)
                else:
                    if logging_enabled:
                        print(f"         ❌ Scene {scene_num} file not found: {encoded_path}")
                    failed_scenes.append(scene_data)
            except Exception as e:
                if logging_enabled:
                    print(f"         ❌ Scene {scene_num} validation error: {e}")
                failed_scenes.append(scene_data)
        else:
            if logging_enabled:
                print(f"         ❌ Scene {scene_num} encoding failed")
            failed_scenes.append(scene_data)
    
    # Count scenes with VMAF data (for reporting only)
    scenes_with_vmaf = sum(1 for scene in successful_scenes if scene.get('actual_vmaf') is not None)
    
    if logging_enabled:
        print(f"   📊 Scene validation results:")
        print(f"      ✅ Successfully encoded scenes: {len(successful_scenes)}")
        print(f"      📊 Scenes with VMAF data: {scenes_with_vmaf}")
        print(f"      🔍 Scenes without VMAF: {len(successful_scenes) - scenes_with_vmaf}")
        if failed_scenes:
            print(f"      ❌ Failed scenes: {len(failed_scenes)}")
    
    if not successful_scenes:
        if logging_enabled:
            print(f"   ❌ No successfully encoded scenes to process!")
        return None, None, None
    
    # ===== REMOVED: CALCULATE MISSING VMAF SECTION =====
    # We expect Part 4 to handle all VMAF calculations or skip them entirely
    
    if logging_enabled:
        print(f"   ✅ Using VMAF data from Part 4 (or skipped if disabled)")
    
    # ===== VIDEO MERGING =====
    if logging_enabled:
        print(f"   🔗 Starting video merging process...")
        print(f"      Merging {len(successful_scenes)} encoded scenes...")
    
    merge_start_time = time.time()
    
    # Generate timestamp for all uses (not just fallback)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Use custom filename from config if available
    custom_filename = config.get('output_settings', {}).get('custom_output_filename')

    if custom_filename:
        # Use the custom filename from Streamlit config
        final_output_filename = custom_filename
        if logging_enabled:
            print(f"      📁 Using custom filename: {final_output_filename}")
    else:
        # Fallback to original pattern if no custom filename
        codec_used = successful_scenes[0].get('codec_used', 'unknown') if successful_scenes else 'unknown'
        final_output_filename = f"{original_video_name}_final_{codec_used}_{timestamp}.mp4"
        if logging_enabled:
            print(f"      📁 Using default pattern: {final_output_filename}")

    final_output_path = os.path.join(output_dir, final_output_filename)
    
    if logging_enabled:
        print(f"      📁 Output file: {final_output_filename}")
    
    # Extract encoded scene paths in correct order
    encoded_scene_paths = []
    for scene_data in sorted(successful_scenes, key=lambda x: x.get('scene_number', 0)):
        scene_path = scene_data['encoded_path']
        encoded_scene_paths.append(scene_path)
        if logging_enabled:
            scene_size = os.path.getsize(scene_path) / 1024 / 1024 if os.path.exists(scene_path) else 0
            print(f"         Scene {scene_data.get('scene_number')}: {os.path.basename(scene_path)} ({scene_size:.1f} MB)")
    
    if logging_enabled:
        print(f"      🎬 Executing video merge...")
    
    try:
        def timeout_handler(signum, frame):
            raise TimeoutError("Video merging timeout")
        
        # Only set alarm on Unix systems
        if hasattr(signal, 'alarm'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(300)  # 5 minute timeout
        
        try:
            merge_success = merge_videos(encoded_scene_paths, final_output_path, logging_enabled=logging_enabled)
        finally:
            # Cancel the alarm
            if hasattr(signal, 'alarm'):
                signal.alarm(0)
        
        merge_time = time.time() - merge_start_time
        
        if not merge_success:
            if logging_enabled:
                print(f"   ❌ Video merging failed!")
            return None, None, None
        
        # ✅ D: Verify merged file
        if not os.path.exists(final_output_path):
            if logging_enabled:
                print(f"   ❌ Merged file was not created!")
            return None, None, None
        
        merged_size = os.path.getsize(final_output_path) / 1024 / 1024
        if merged_size == 0:
            if logging_enabled:
                print(f"   ❌ Merged file is empty!")
            return None, None, None
        
        if logging_enabled:
            print(f"   ✅ Video merging completed in {merge_time:.1f}s")
            print(f"   📁 Final video: {final_output_filename} ({merged_size:.1f} MB)")
        if cleanup_temp_files:
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
                print(f"   ✅ Encoded scenes cleanup: {encoded_cleanup_success} deleted, {encoded_cleanup_failed} failed")
    
    except TimeoutError:
        if logging_enabled:
            print(f"   ❌ Video merging timed out after 5 minutes!")
        return None, None, None
    except Exception as e:
        if logging_enabled:
            print(f"   ❌ Video merging failed with error: {e}")
            import traceback
            traceback.print_exc()
        return None, None, None
    
    # ===== FINAL VMAF CALCULATION =====
    final_vmaf = None
    final_vmaf_time = 0
    
    if logging_enabled:
        print(f"   ⏭️ Skipping full video VMAF calculation (disabled in config)")
    
    # Estimate final VMAF from scene averages if available
    scene_vmafs = [scene.get('actual_vmaf', 0) for scene in successful_scenes if scene.get('actual_vmaf', 0) > 0]
    if scene_vmafs:
        final_vmaf = sum(scene_vmafs) / len(scene_vmafs)
        if logging_enabled:
            print(f"   📊 Estimated VMAF from scene average: {final_vmaf:.2f}")
    else:
        if logging_enabled:
            print(f"   ⏭️ No scene VMAF data available for estimation")
    
    # ===== COMPREHENSIVE REPORT GENERATION =====
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

    # ✅ SIMPLE: Extract training data directly from scenes
    def extract_training_data_from_scenes(scenes_data):
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
                
                # Scene classification results
                if scene.get('scene_type'):
                    training_data['scene_classification_results'].append({
                        'scene_number': scene_num,
                        'classified_type': scene.get('scene_type'),
                        'classification_confidence': scene.get('scene_classification_confidence'),
                        'duration': scene.get('duration'),
                        'file_size_mb': scene.get('file_size_mb')
                    })
                
                # Encoding optimization data
                training_data['encoding_optimization_data'].append({
                    'scene_number': scene_num,
                    'scene_type': scene.get('scene_type'),
                    'optimal_cq': scene.get('optimal_cq'),
                    'final_adjusted_cq': scene.get('final_adjusted_cq'),
                    'conservative_adjustment': scene.get('conservative_cq_adjustment', 0),
                    'size_protection_used': scene.get('size_protection_used', False),
                    'compression_achieved': scene.get('compression_ratio', 0),
                    'target_vmaf': scene.get('target_vmaf'),
                    'predicted_vmaf': scene.get('predicted_vmaf_at_optimal_cq')
                })
            
            # Extract Part 4 data (VMAF calculation)
            part4_data = {
                'scene_number': scene_num,
                'actual_vmaf': scene.get('actual_vmaf'),
                'vmaf_calculation_status': scene.get('vmaf_calculation_status'),
                'vmaf_calculation_time': scene.get('vmaf_calculation_time', 0),
                'vmaf_calculation_notes': scene.get('vmaf_calculation_notes'),
                'target_achieved': scene.get('actual_vmaf', 0) >= scene.get('target_vmaf', 0) if scene.get('actual_vmaf') else None
            }
            training_data['pipeline_stages_data']['part4_vmaf_calculation'].append(part4_data)
            
            # VMAF validation data (detailed)
            if scene.get('actual_vmaf') is not None:
                vmaf_data = {
                    'scene_number': scene_num,
                    'actual_vmaf': scene.get('actual_vmaf'),
                    'predicted_vmaf': scene.get('predicted_vmaf_at_optimal_cq'),
                    'target_vmaf': scene.get('target_vmaf'),
                    'vmaf_config_used': scene.get('vmaf_config_used', {}),
                    'calculation_method': scene.get('vmaf_calculation_notes', '').split(' ')[0] if scene.get('vmaf_calculation_notes') else 'unknown'
                }
                
                # Add prediction analysis if available
                if scene.get('vmaf_prediction_analysis'):
                    vmaf_data.update(scene['vmaf_prediction_analysis'])
                
                training_data['vmaf_validation_data'].append(vmaf_data)
        
        # Add Part 4 data (merging performance)
        training_data['pipeline_stages_data']['part5_merging'] = {
            'total_scenes_processed': len(scenes_data),
            'successful_scenes': len([s for s in scenes_data if s.get('encoding_success')]),
            'merging_time': merge_time,
            'final_vmaf_calculation_time': final_vmaf_time,
            'total_part5_time': total_processing_time,
            'overall_compression_ratio': overall_compression_ratio
        }
        
        return training_data

    # Extract training data
    comprehensive_training_data = extract_training_data_from_scenes(successful_scenes)

    comprehensive_report = {
        'processing_summary': {
            'timestamp': datetime.now().isoformat(),
            'original_video_path': original_video_path,
            'final_video_path': final_output_path,
            'total_scenes_processed': len(successful_scenes),
            'failed_scenes_count': len(failed_scenes),
            'processing_time_seconds': total_processing_time,
            'vmaf_calculation_enabled': config.get('vmaf_calculation', {}).get('calculate_scene_vmaf', True)
        },
        
        'quality_metrics': {
            'final_vmaf_score': final_vmaf,
            'final_vmaf_calculated': calculate_full_video_vmaf,
            'final_vmaf_source': 'calculated' if calculate_full_video_vmaf else 'estimated_from_scenes' if scenes_with_valid_vmaf else 'unavailable',
            'average_scene_vmaf': average_scene_vmaf,
            'target_vmaf': target_vmaf,
            'target_achieved': (final_vmaf >= target_vmaf) if final_vmaf else False,
            'scenes_meeting_target': scenes_meeting_target,
            'scenes_with_vmaf_data': len(scenes_with_valid_vmaf),
            'vmaf_calculation_time': final_vmaf_time,
            'scene_vmaf_source': 'part4'
        },
        
        'compression_metrics': {
            'total_input_size_mb': input_size_total,
            'total_output_size_mb': output_size_total,
            'size_ratio': output_size_total / input_size_total if input_size_total > 0 else 0,
            'overall_compression_ratio_percent': overall_compression_ratio,
            'final_file_size_mb': os.path.getsize(final_output_path) / (1024 * 1024) if os.path.exists(final_output_path) else 0
        },
        
        'scene_analysis': {
            'scene_types_distribution': {},
            'codec_usage': {},
            'prediction_accuracy_stats': {
                'scenes_meeting_target': scenes_meeting_target,
                'scenes_with_vmaf': len(scenes_with_valid_vmaf),
                'average_scene_vmaf': average_scene_vmaf
            }
        },
        
        'processing_performance': {
            'part5_merging_time': merge_time,
            'part5_final_vmaf_time': final_vmaf_time,
            'part5_total_time': total_processing_time
        },
        
        'individual_scenes_summary': [],
        
        'comprehensive_training_data': comprehensive_training_data
    }
        
    
    # ===== SAVE REPORTS =====
    if save_comprehensive_report:
        if logging_enabled:
            print(f"   💾 Saving comprehensive processing report...")

        try:
            # ✅ SIMPLE: Extract original filename from the original_video_path parameter
            original_name_from_path = os.path.splitext(os.path.basename(original_video_path))[0]
            
            # Clean up temp prefixes if present
            if original_name_from_path.startswith('tmp'):
                # Try to get from config if available
                config_original_name = config.get('output_settings', {}).get('original_video_name')
                if config_original_name:
                    clean_name = config_original_name
                else:
                    # Try to extract from custom filename in config
                    custom_filename = config.get('output_settings', {}).get('custom_output_filename')
                    if custom_filename and '_encoded_' in custom_filename:
                        clean_name = custom_filename.split('_encoded_')[0]
                    else:
                        clean_name = "video"  # Generic fallback
            else:
                clean_name = original_name_from_path
            
            report_filename = f"{clean_name}_encoding_report.json"
            report_path = os.path.join(output_dir, report_filename)
            
            if logging_enabled:
                print(f"      📁 Report filename: {report_filename}")
                print(f"      📊 Extracted from: {os.path.basename(original_video_path)}")
            
            with open(report_path, 'w') as f:
                json.dump(comprehensive_report, f, indent=2, default=str)
            
            if logging_enabled:
                print(f"      📄 Saved comprehensive report: {report_filename}")
        except Exception as e:
            if logging_enabled:
                print(f"      ❌ Failed to save comprehensive report: {e}")
    # ===== CLEANUP TEMPORARY FILES =====
    if cleanup_temp_files and temp_files_to_cleanup:
        if logging_enabled:
            print(f"   🧹 Cleaning up temporary files...")
        
        cleanup_success = 0
        cleanup_failed = 0
        
        for temp_file in set(temp_files_to_cleanup):
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    cleanup_success += 1
                    if logging_enabled:
                        print(f"      🗑️ Deleted: {os.path.basename(temp_file)}")
            except Exception as e:
                cleanup_failed += 1
                if logging_enabled:
                    print(f"      ❌ Failed to delete {os.path.basename(temp_file)}: {e}")
        
        if logging_enabled:
            print(f"   ✅ Cleanup completed: {cleanup_success} files deleted, {cleanup_failed} failed")
    
    # ===== FINAL SUMMARY =====
    if logging_enabled:
        print(f"\n🎉 --- Part 4 Completed Successfully ---")
        print(f"   📁 Final video: {final_output_filename}")
        if final_vmaf:
            print(f"   🎯 Final VMAF: {final_vmaf:.2f} (target: {target_vmaf:.1f}) - {'✅ ACHIEVED' if final_vmaf >= target_vmaf else '❌ MISSED'}")
        print(f"   🗜️ Overall compression: {overall_compression_ratio:+.1f}%")
        print(f"   ⏱️ Total processing time: {total_processing_time:.1f}s")
        print(f"   📊 Scenes meeting target: {scenes_meeting_target}/{len(scenes_with_valid_vmaf)}")
        print(f"   🔍 VMAF data from Part 4: {len(scenes_with_valid_vmaf)}/{len(successful_scenes)} scenes")
    
    return final_output_path, final_vmaf, comprehensive_report



