import os
import time
import json
import sys

from utils.calculate_vmaf_adv import calculate_vmaf_advanced

def calculate_scene_vmaf(scene_metadata, config, logging_enabled=True):
    """
    Calculate VMAF for a single encoded scene according to config settings.
    
    Args:
        scene_metadata (dict): Scene metadata from Part 3 containing:
                              - 'path': original scene file path
                              - 'encoded_path': encoded scene file path
                              - 'scene_number': scene number
                              - 'encoding_success': boolean
                              - 'original_video_metadata': metadata from Part 1
                              - Other scene processing data
        config (dict): Configuration dictionary with vmaf_calculation settings
        logging_enabled (bool): Whether to enable detailed logging
        
    Returns:
        dict: Updated scene metadata with VMAF results added:
              - 'actual_vmaf': calculated VMAF score
              - 'vmaf_calculation_status': 'success', 'failed', or 'skipped'
              - 'vmaf_calculation_notes': detailed status information
              - 'vmaf_calculation_time': time taken for calculation
              - 'vmaf_config_used': VMAF settings used for calculation
    """
    if logging_enabled:
        print(f"🔍 Calculating VMAF for Scene {scene_metadata.get('scene_number', 'unknown')}")
    
    # Create a copy of scene metadata to avoid modifying original
    updated_metadata = scene_metadata.copy()
    calculation_start_time = time.time()
    
    # Initialize VMAF result fields
    vmaf_result = {
        'actual_vmaf': None,
        'vmaf_calculation_status': 'skipped',
        'vmaf_calculation_notes': '',
        'vmaf_calculation_time': 0.0,
        'vmaf_config_used': {}
    }
    
    try:
        # ===== VALIDATION CHECKS =====
        
        # Check if encoding was successful
        if not scene_metadata.get('encoding_success', False):
            vmaf_result.update({
                'vmaf_calculation_status': 'skipped',
                'vmaf_calculation_notes': 'Scene encoding failed - VMAF calculation skipped'
            })
            if logging_enabled:
                print(f"   ⏭️ Skipping VMAF calculation - encoding failed")
            updated_metadata.update(vmaf_result)
            return updated_metadata
        
        # Check if encoded file exists
        encoded_path = scene_metadata.get('encoded_path')
        if not encoded_path or not os.path.exists(encoded_path):
            vmaf_result.update({
                'vmaf_calculation_status': 'failed',
                'vmaf_calculation_notes': f'Encoded file not found: {encoded_path}'
            })
            if logging_enabled:
                print(f"   ❌ Encoded file not found: {encoded_path}")
            updated_metadata.update(vmaf_result)
            return updated_metadata
        print("scenes path is :", scene_metadata.get('path'), "encoded path is :", encoded_path)
        # Check if original scene file exists
        original_path = scene_metadata.get('path')
        if not original_path or not os.path.exists(original_path):
            vmaf_result.update({
                'vmaf_calculation_status': 'failed',
                'vmaf_calculation_notes': f'Original scene file not found: {original_path}'
            })
            if logging_enabled:
                print(f"   ❌ Original scene file not found: {original_path}")
            updated_metadata.update(vmaf_result)
            return updated_metadata
        
        # ===== EXTRACT VMAF CONFIGURATION =====
        
        vmaf_config = config.get('vmaf_calculation', {})
        vmaf_models_config = config.get('vmaf_models', {})
        
        # Extract VMAF calculation settings
        use_sampling = vmaf_config.get('vmaf_use_sampling', True)
        num_clips = vmaf_config.get('vmaf_num_clips', 3)
        clip_duration = vmaf_config.get('vmaf_clip_duration', 2)
        use_downscaling = vmaf_config.get('vmaf_use_downscaling', True)
        scale_factor = vmaf_config.get('vmaf_downscaling_scale_factor', 0.5)
        use_frame_rate_scaling = vmaf_config.get('vmaf_use_frame_rate_scaling', False)
        target_fps = vmaf_config.get('vmaf_target_fps', 15.0)
        frame_rate_scaling_method = vmaf_config.get('vmaf_frame_rate_scaling_method', 'uniform')
        
        # Extract VMAF model settings
        use_vmafneg = vmaf_models_config.get('use_neg_by_default', False)
        default_model_path = vmaf_models_config.get('default_model_path')
        vmafneg_model_path = vmaf_models_config.get('neg_model_path')
        
        # Store configuration used for this calculation
        vmaf_result['vmaf_config_used'] = {
            'use_sampling': use_sampling,
            'num_clips': num_clips,
            'clip_duration': clip_duration,
            'use_downscaling': use_downscaling,
            'scale_factor': scale_factor,
            'use_frame_rate_scaling': use_frame_rate_scaling,
            'target_fps': target_fps,
            'frame_rate_scaling_method': frame_rate_scaling_method,
            'use_vmafneg': use_vmafneg,
            'default_model_path': default_model_path,
            'vmafneg_model_path': vmafneg_model_path
        }
        
        if logging_enabled:
            print(f"   📊 VMAF Settings:")
            print(f"      Sampling: {use_sampling} ({'clips: ' + str(num_clips) + ', duration: ' + str(clip_duration) + 's' if use_sampling else 'full video'})")
            print(f"      Downscaling: {use_downscaling} ({'scale: ' + str(scale_factor) if use_downscaling else ''})")
            print(f"      Frame rate scaling: {use_frame_rate_scaling} ({'target: ' + str(target_fps) + ' FPS' if use_frame_rate_scaling else ''})")
            print(f"      Model: {'VMAFNEG' if use_vmafneg else 'Default VMAF'}")
        
        # ===== CALCULATE VMAF =====
        
        if logging_enabled:
            print(f"   🔍 Calculating VMAF...")
            print(f"      Reference: {os.path.basename(original_path)}")
            print(f"      Encoded: {os.path.basename(encoded_path)}")
        
        # Perform VMAF calculation using advanced function
        vmaf_score = calculate_vmaf_advanced(
            input_file=original_path,
            encoded_file=encoded_path,
            use_sampling=use_sampling,
            num_clips=num_clips,
            clip_duration=clip_duration,
            use_downscaling=use_downscaling,
            scale_factor=scale_factor,
            use_vmafneg=use_vmafneg,
            default_vmaf_model_path_config=default_model_path,
            vmafneg_model_path_config=vmafneg_model_path,
            use_frame_rate_scaling=use_frame_rate_scaling,
            target_fps=target_fps,
            frame_rate_scaling_method=frame_rate_scaling_method,
            logger=None,
            logging_enabled=logging_enabled
        )
        
        calculation_time = time.time() - calculation_start_time
        
        # ===== PROCESS RESULTS =====
        
        if vmaf_score is not None and vmaf_score > 0:
            # Successful VMAF calculation
            vmaf_result.update({
                'actual_vmaf': vmaf_score,
                'vmaf_calculation_status': 'success',
                'vmaf_calculation_notes': f'VMAF calculated successfully using {"sampling" if use_sampling else "full video"} method',
                'vmaf_calculation_time': calculation_time
            })
            
            # ✅ ENHANCED: Add prediction accuracy analysis if available
            predicted_vmaf = scene_metadata.get('predicted_vmaf_at_optimal_cq')
            target_vmaf = scene_metadata.get('target_vmaf')
            
            if predicted_vmaf is not None:
                prediction_error = abs(predicted_vmaf - vmaf_score)
                prediction_relative_error = (prediction_error / vmaf_score * 100) if vmaf_score > 0 else None
                
                vmaf_result['vmaf_prediction_analysis'] = {
                    'predicted_vmaf': predicted_vmaf,
                    'actual_vmaf': vmaf_score,
                    'prediction_error': prediction_error,
                    'prediction_relative_error': prediction_relative_error,
                    'prediction_accuracy_good': prediction_error <= 2.0  # Within 2 VMAF points
                }
                
                if logging_enabled:
                    print(f"   🎯 Prediction Analysis:")
                    print(f"      Predicted: {predicted_vmaf:.2f}")
                    print(f"      Actual: {vmaf_score:.2f}")
                    print(f"      Error: {prediction_error:.2f} VMAF points")
            
            if target_vmaf is not None:
                target_achieved = vmaf_score >= target_vmaf
                vmaf_result['vmaf_target_analysis'] = {
                    'target_vmaf': target_vmaf,
                    'actual_vmaf': vmaf_score,
                    'target_achieved': target_achieved,
                    'vmaf_margin': vmaf_score - target_vmaf
                }
                
                if logging_enabled:
                    status = "✅ ACHIEVED" if target_achieved else "❌ MISSED"
                    print(f"   🎯 Target Analysis: {vmaf_score:.2f} vs {target_vmaf:.2f} - {status}")
            
            if logging_enabled:
                print(f"   ✅ VMAF calculation successful: {vmaf_score:.2f}")
                print(f"   ⏱️ Calculation time: {calculation_time:.1f}s")
        
        else:
            # VMAF calculation failed
            vmaf_result.update({
                'actual_vmaf': None,
                'vmaf_calculation_status': 'failed',
                'vmaf_calculation_notes': 'VMAF calculation returned invalid result (None or <= 0)',
                'vmaf_calculation_time': calculation_time
            })
            
            if logging_enabled:
                print(f"   ❌ VMAF calculation failed - invalid result: {vmaf_score}")
    
    except Exception as e:
        # Handle any calculation errors
        calculation_time = time.time() - calculation_start_time
        
        vmaf_result.update({
            'actual_vmaf': None,
            'vmaf_calculation_status': 'failed',
            'vmaf_calculation_notes': f'VMAF calculation failed with error: {str(e)}',
            'vmaf_calculation_time': calculation_time
        })
        
        if logging_enabled:
            print(f"   ❌ VMAF calculation failed with error: {e}")
    
    # ===== UPDATE METADATA =====
    
    # Add VMAF results to scene metadata
    updated_metadata.update(vmaf_result)
    
    # ✅ ENHANCED: Update model training data if available
    if 'model_training_data' in updated_metadata:
        if 'vmaf_validation' not in updated_metadata['model_training_data']:
            updated_metadata['model_training_data']['vmaf_validation'] = {}
        
        updated_metadata['model_training_data']['vmaf_validation'].update({
            'actual_vmaf_measured': vmaf_result.get('actual_vmaf'),
            'vmaf_calculation_method': 'sampling' if use_sampling else 'full_video',
            'vmaf_calculation_successful': vmaf_result.get('vmaf_calculation_status') == 'success',
            'vmaf_calculation_time': vmaf_result.get('vmaf_calculation_time', 0),
            'vmaf_config_used': vmaf_result.get('vmaf_config_used', {}),
            'prediction_analysis': vmaf_result.get('vmaf_prediction_analysis'),
            'target_analysis': vmaf_result.get('vmaf_target_analysis')
        })
    
    return updated_metadata


def calculate_multiple_scenes_vmaf(scenes_metadata_list, config, logging_enabled=True):
    """
    Calculate VMAF for multiple scenes sequentially.
    
    Args:
        scenes_metadata_list (list): List of scene metadata dictionaries
        config (dict): Configuration dictionary
        logging_enabled (bool): Whether to enable detailed logging
        
    Returns:
        list: Updated list of scene metadata with VMAF results
    """
    if logging_enabled:
        print(f"\n📊 === Calculating VMAF for {len(scenes_metadata_list)} scenes ===")
    
    updated_scenes = []
    successful_calculations = 0
    failed_calculations = 0
    skipped_calculations = 0
    total_vmaf_time = 0.0
    
    for i, scene_metadata in enumerate(scenes_metadata_list):
        scene_number = scene_metadata.get('scene_number', i + 1)
        
        if logging_enabled:
            print(f"\n🔍 Scene {scene_number}/{len(scenes_metadata_list)}")
        
        # Calculate VMAF for this scene
        updated_scene = calculate_scene_vmaf(scene_metadata, config, logging_enabled)
        updated_scenes.append(updated_scene)
        
        # Track statistics
        status = updated_scene.get('vmaf_calculation_status', 'unknown')
        calc_time = updated_scene.get('vmaf_calculation_time', 0)
        total_vmaf_time += calc_time
        
        if status == 'success':
            successful_calculations += 1
        elif status == 'failed':
            failed_calculations += 1
        elif status == 'skipped':
            skipped_calculations += 1
    
    # ===== SUMMARY =====
    
    if logging_enabled:
        print(f"\n📊 === VMAF Calculation Summary ===")
        print(f"   ✅ Successful: {successful_calculations}")
        print(f"   ❌ Failed: {failed_calculations}")
        print(f"   ⏭️ Skipped: {skipped_calculations}")
        print(f"   ⏱️ Total VMAF time: {total_vmaf_time:.1f}s")
        print(f"   📈 Success rate: {successful_calculations/len(scenes_metadata_list)*100:.1f}%")
        
        # Calculate average VMAF if any successful calculations
        successful_scenes = [s for s in updated_scenes if s.get('actual_vmaf') is not None]
        if successful_scenes:
            avg_vmaf = sum(s['actual_vmaf'] for s in successful_scenes) / len(successful_scenes)
            min_vmaf = min(s['actual_vmaf'] for s in successful_scenes)
            max_vmaf = max(s['actual_vmaf'] for s in successful_scenes)
            
            print(f"   📊 VMAF Statistics:")
            print(f"      Average: {avg_vmaf:.2f}")
            print(f"      Range: {min_vmaf:.2f} - {max_vmaf:.2f}")
            
            # Check target achievement
            target_vmaf = config.get('video_processing', {}).get('target_vmaf', 93.0)
            scenes_meeting_target = sum(1 for s in successful_scenes if s['actual_vmaf'] >= target_vmaf)
            print(f"      Target ({target_vmaf:.1f}) achieved: {scenes_meeting_target}/{len(successful_scenes)} scenes")
    
    return updated_scenes


def scene_vmaf_calculation(encoded_scenes_data, config, logging_enabled=True):
    """
    Part 4: Calculate VMAF for individual encoded scenes.
    
    This new Part 4 focuses solely on VMAF calculation for scenes that were
    successfully encoded in Part 3. It processes scenes individually and
    adds VMAF results to their metadata for use in Part 5.
    
    Args:
        encoded_scenes_data (list): List of scene data dictionaries from Part 3
        config (dict): Configuration dictionary with VMAF settings
        logging_enabled (bool): Whether to enable detailed logging
        
    Returns:
        list: Updated scene data with VMAF results added
    """
    if logging_enabled:
        print(f"\n📊 === Part 4: Scene VMAF Calculation ===")
        print(f"   🎬 Processing {len(encoded_scenes_data)} scenes for VMAF calculation")
    
    part4_start_time = time.time()
    
    # Filter scenes that need VMAF calculation
    scenes_needing_vmaf = []
    scenes_with_vmaf = []
    scenes_failed_encoding = []
    
    for scene_data in encoded_scenes_data:
        # Check if scene encoding was successful
        if not scene_data.get('encoding_success', False):
            scenes_failed_encoding.append(scene_data)
            continue
        
        # Check if VMAF already exists
        existing_vmaf = scene_data.get('actual_vmaf')
        if existing_vmaf is not None and existing_vmaf > 0:
            scenes_with_vmaf.append(scene_data)
            if logging_enabled:
                print(f"   ✅ Scene {scene_data.get('scene_number')} already has VMAF: {existing_vmaf:.2f}")
        else:
            scenes_needing_vmaf.append(scene_data)
    
    if logging_enabled:
        print(f"   📊 VMAF Status Summary:")
        print(f"      ✅ Scenes with existing VMAF: {len(scenes_with_vmaf)}")
        print(f"      🔍 Scenes needing VMAF calculation: {len(scenes_needing_vmaf)}")
        print(f"      ❌ Scenes with failed encoding: {len(scenes_failed_encoding)}")
    
    # Process scenes that need VMAF calculation
    if scenes_needing_vmaf:
        if logging_enabled:
            print(f"\n   🔍 Calculating VMAF for {len(scenes_needing_vmaf)} scenes...")
        
        # Use local function (no circular import)
        updated_scenes_needing_vmaf = calculate_multiple_scenes_vmaf(
            scenes_needing_vmaf, 
            config, 
            logging_enabled=logging_enabled
        )
        
        # Combine all scenes back together
        all_updated_scenes = []
        
        # Add scenes with existing VMAF
        all_updated_scenes.extend(scenes_with_vmaf)
        
        # Add scenes with newly calculated VMAF
        all_updated_scenes.extend(updated_scenes_needing_vmaf)
        
        # Add scenes with failed encoding (no VMAF possible)
        for failed_scene in scenes_failed_encoding:
            failed_scene.update({
                'actual_vmaf': None,
                'vmaf_calculation_status': 'skipped',
                'vmaf_calculation_notes': 'Scene encoding failed - VMAF calculation skipped',
                'vmaf_calculation_time': 0.0
            })
            all_updated_scenes.append(failed_scene)
        
        # Sort by scene number to maintain order
        all_updated_scenes.sort(key=lambda x: x.get('scene_number', 0))
        
    else:
        if logging_enabled:
            print(f"   ✅ All scenes already have VMAF scores - no calculation needed")
        
        # Just combine existing scenes (with and without VMAF)
        all_updated_scenes = scenes_with_vmaf + scenes_failed_encoding
        all_updated_scenes.sort(key=lambda x: x.get('scene_number', 0))
    
    # Calculate summary statistics
    part4_time = time.time() - part4_start_time
    successful_vmaf_calculations = sum(1 for scene in all_updated_scenes 
                                     if scene.get('vmaf_calculation_status') == 'success')
    
    if logging_enabled:
        print(f"\n✅ Part 4 completed in {part4_time:.1f}s:")
        print(f"   📊 Total scenes processed: {len(all_updated_scenes)}")
        print(f"   ✅ Successful VMAF calculations: {successful_vmaf_calculations}")
        print(f"   📈 VMAF calculation success rate: {successful_vmaf_calculations/len(scenes_needing_vmaf)*100:.1f}%" if scenes_needing_vmaf else "   📈 No new calculations needed")
        
        # Show VMAF statistics for scenes with valid scores
        scenes_with_valid_vmaf = [s for s in all_updated_scenes if s.get('actual_vmaf') is not None]
        if scenes_with_valid_vmaf:
            avg_vmaf = sum(s['actual_vmaf'] for s in scenes_with_valid_vmaf) / len(scenes_with_valid_vmaf)
            min_vmaf = min(s['actual_vmaf'] for s in scenes_with_valid_vmaf)
            max_vmaf = max(s['actual_vmaf'] for s in scenes_with_valid_vmaf)
            target_vmaf = config.get('video_processing', {}).get('target_vmaf', 93.0)
            scenes_meeting_target = sum(1 for s in scenes_with_valid_vmaf if s['actual_vmaf'] >= target_vmaf)
            
            print(f"   📊 VMAF Statistics:")
            print(f"      Average: {avg_vmaf:.2f}")
            print(f"      Range: {min_vmaf:.2f} - {max_vmaf:.2f}")
            print(f"      Target achieved: {scenes_meeting_target}/{len(scenes_with_valid_vmaf)} scenes")
    
    return all_updated_scenes


if __name__ == '__main__':
    # Test the new Part 4 functionality
    print("🧪 --- Part 4 Scene VMAF Calculation Testing ---")
    
    # Load configuration
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        print("✅ Configuration loaded successfully")
    except FileNotFoundError:
        print("⚠️ Config file not found, using default configuration")
        config = {
            'vmaf_calculation': {
                'vmaf_use_sampling': True,
                'vmaf_num_clips': 3,
                'vmaf_clip_duration': 2,
                'vmaf_use_downscaling': True,
                'vmaf_downscaling_scale_factor': 0.5
            },
            'vmaf_models': {
                'use_neg_by_default': False,
                'default_model_path': 'src/models/vmaf_v0.6.1.json',
                'neg_model_path': 'src/models/vmaf_v0.6.1neg.json'
            },
            'video_processing': {
                'target_vmaf': 93.0
            }
        }
    
    # Create test scene data (simulating Part 3 output)
    test_scenes_data = [
        {
            'scene_number': 1,
            'path': './videos/ducks_take_off_1080p50_full.mp4',
            'encoded_path': './videos/ducks_take_off_1080p50_full.mp4',  # Using same file for test
            'encoding_success': True,
            'scene_type': 'other',
            'predicted_vmaf_at_optimal_cq': 94.5,
            'target_vmaf': 93.0,
            # No actual_vmaf - should be calculated
        },
        {
            'scene_number': 2,
            'path': './videos/ducks_take_off_1080p50_full.mp4',
            'encoded_path': './videos/ducks_take_off_1080p50_full.mp4',
            'encoding_success': True,
            'scene_type': 'other',
            'predicted_vmaf_at_optimal_cq': 93.2,
            'target_vmaf': 93.0,
            'actual_vmaf': 92.8,  # Already has VMAF - should be preserved
            'vmaf_calculation_status': 'success',
            'vmaf_calculation_notes': 'Pre-calculated'
        },
        {
            'scene_number': 3,
            'encoding_success': False,
            'error_reason': 'Encoding failed',
            # No VMAF possible - should be skipped
        }
    ]
    
    # Test Part 4
    if os.path.exists('./videos/ducks_take_off_1080p50_full.mp4'):
        print(f"\n🧪 Testing Part 4 with {len(test_scenes_data)} scenes...")
        
        updated_scenes = part4_scene_vmaf_calculation(test_scenes_data, config, logging_enabled=True)
        
        print(f"\n📊 Part 4 Test Results:")
        for scene in updated_scenes:
            scene_num = scene.get('scene_number', 'unknown')
            vmaf_score = scene.get('actual_vmaf')
            vmaf_status = scene.get('vmaf_calculation_status', 'unknown')
            
            print(f"   Scene {scene_num}: VMAF = {vmaf_score}, Status = {vmaf_status}")
        
        print(f"\n✅ Part 4 testing completed!")
    else:
        print(f"❌ Test video file not found")
    
    print(f"\n🎉 Part 4 (Scene VMAF Calculation) testing completed!")