import os
import time

from utils.calculate_vmaf_adv import calculate_vmaf_advanced

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
    adds VMAF results to their metadata for use in Part 4.

    scene_number, encoded_path, path, encoding_success, 

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