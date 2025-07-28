import os
import time
import shutil
import pandas as pd


def create_summary_and_save(
    input_file, output_file_concat, codec, target_vmaf, full_vmaf,
    original_size_bytes, encoded_size_bytes, duration_seconds,
    scene_results_list, overall_start_time,
    timings, scenes_df, results_csv, logging_enabled=True, scene_classifications_summary=None
    ):
    """Creates a summary dictionary and saves the results CSV with preprocessing timing info."""
    total_processing_time = (time.time() - overall_start_time) if overall_start_time else None
 
    
    # FIXED: Handle scene classifications properly
    if scene_classifications_summary is None:
        # Generate from scene_results_list if not provided
        if scene_results_list and len(scene_results_list) > 0:
            if len(scene_results_list) == 1:
                # Single scene - show the classification
                single_scene = scene_results_list[0]
                scene_type = single_scene.get('scene_type', 'Unknown')
                confidence = single_scene.get('confidence_score', 0)
                scene_classifications = f"{scene_type} (confidence: {confidence:.2f})"
              
            else:
                # Multiple scenes - show distribution
                scene_types = [scene.get('scene_type', 'Unknown') for scene in scene_results_list]
                scene_classifications = ", ".join(scene_types)
        else:
            scene_classifications = "N/A"
    else:
        # Use provided summary (ensure it's a string)
        scene_classifications = str(scene_classifications_summary)
    
    original_size_mb_calc = original_size_bytes / (1024 * 1024)
    encoded_size_mb_calc = encoded_size_bytes / (1024 * 1024)

    input_bitrate_kbps = 0
    output_bitrate_kbps = 0
    if duration_seconds and duration_seconds > 0:
        if original_size_bytes > 0:
            input_bitrate_kbps = (original_size_bytes * 8) / (duration_seconds * 1000)
        if encoded_size_bytes > 0:
            output_bitrate_kbps = (encoded_size_bytes * 8) / (duration_seconds * 1000)

    # ✅ Calculate preprocessing statistics
    preprocessing_enabled = any(scene.get('preprocessing_applied', False) for scene in scene_results_list) if scene_results_list else False
    
    # Calculate total preprocessing time across all scenes
    total_preprocessing_time = 0.0
    scenes_with_preprocessing = 0
    
    if scene_results_list:
        for scene in scene_results_list:
            scene_preprocessing_time = scene.get('preprocessing_total_time', 0.0)
            if isinstance(scene_preprocessing_time, (int, float)) and scene_preprocessing_time > 0:
                total_preprocessing_time += scene_preprocessing_time
                if scene.get('preprocessing_applied', False):
                    scenes_with_preprocessing += 1
    
    # Calculate average preprocessing time per scene (only for scenes that had preprocessing)
    avg_preprocessing_time_per_scene = 0.0
    if scenes_with_preprocessing > 0:
        avg_preprocessing_time_per_scene = total_preprocessing_time / scenes_with_preprocessing
    
    # Calculate preprocessing percentage of total processing time
    preprocessing_percentage_of_total = 0.0
    if total_processing_time and total_processing_time > 0 and total_preprocessing_time > 0:
        preprocessing_percentage_of_total = (total_preprocessing_time / total_processing_time) * 100
    
    print(f"Scene Classifications: {scene_classifications}")
    
    # ✅ Enhanced summary with preprocessing timing information
    summary = {
        'input_file': input_file,
        'output_file': output_file_concat,
        'codec': codec,
        'target_vmaf': target_vmaf,
        'actual_vmaf': full_vmaf,
        'original_size_mb': original_size_mb_calc,
        'encoded_size_mb': encoded_size_mb_calc,
        'compression_ratio': (100 * encoded_size_mb_calc / original_size_mb_calc) if original_size_mb_calc > 0 else 0,
        'duration_seconds': duration_seconds,
        'input_bitrate_kbps': input_bitrate_kbps,
        'output_bitrate_kbps': output_bitrate_kbps,
        'num_scenes': len(scene_results_list) if scene_results_list else 0,
        'scene_classifications': scene_classifications,
        'total_processing_time': total_processing_time,
        'timings': timings if logging_enabled else {},
        
        # ✅ NEW: Preprocessing timing fields
        'preprocessing_enabled': preprocessing_enabled,
        'scenes_with_preprocessing': scenes_with_preprocessing,
        'total_preprocessing_time': total_preprocessing_time,
        'avg_preprocessing_time_per_scene': avg_preprocessing_time_per_scene,
        'preprocessing_percentage_of_total': preprocessing_percentage_of_total,
        
        # Extract detailed preprocessing breakdown from timings if available
        'preprocessing_quality_analysis_time': timings.get('preprocessing_quality_analysis', 0.0),
        'preprocessing_filter_recommendation_time': timings.get('preprocessing_filter_recommendation', 0.0),
        'preprocessing_filter_application_time': timings.get('preprocessing_filter_application', 0.0),
        'preprocessing_total_all_scenes': timings.get('preprocessing_total_all_scenes', total_preprocessing_time),
    }

    if not scenes_df.empty:
        scenes_df.to_csv(results_csv, index=False)
        if logging_enabled: 
            print(f"Results saved to: {results_csv}")
    elif logging_enabled:
        print("No scene results to save to CSV.")
    
    if logging_enabled:
        print("\n--- Processing Complete ---")
        print(f"Input File: {summary.get('input_file', 'N/A')}")
        print(f"Output File: {summary.get('output_file', 'N/A')}")
        print(f"Codec: {summary.get('codec', 'N/A')}")
        print(f"Target VMAF: {summary.get('target_vmaf', 'N/A')}")
        if summary.get('actual_vmaf') is not None:
            print(f"Actual VMAF: {summary.get('actual_vmaf', 'N/A')}")
        print(f"Duration: {summary.get('duration_seconds', 0):.2f}s")
        print(f"Original Size: {summary.get('original_size_mb', 0):.2f} MB")
        print(f"Encoded Size: {summary.get('encoded_size_mb', 0):.2f} MB")
        print(f"Input Bitrate: {summary.get('input_bitrate_kbps', 0):.2f} kbps")
        print(f"Output Bitrate: {summary.get('output_bitrate_kbps', 0):.2f} kbps")
        print(f"Compression Ratio: {summary.get('compression_ratio', 0):.2f}%")
        print(f"Number of Scenes Processed: {summary.get('num_scenes', 0)}")
        print(f"Scene Classifications: {scene_classifications}")
        print(f"Total Processing Time: {total_processing_time:.2f}s")
        
        # ✅ NEW: Preprocessing timing breakdown
        if preprocessing_enabled:
            print("\n--- Preprocessing Timing Breakdown ---")
            print(f"Preprocessing Enabled: Yes")
            print(f"Scenes with Preprocessing: {scenes_with_preprocessing}/{summary.get('num_scenes', 0)}")
            print(f"Total Preprocessing Time: {total_preprocessing_time:.2f}s ({preprocessing_percentage_of_total:.1f}% of total)")
            if summary.get('preprocessing_quality_analysis_time', 0) > 0:
                print(f"  • Quality Analysis: {summary['preprocessing_quality_analysis_time']:.2f}s")
            if summary.get('preprocessing_filter_recommendation_time', 0) > 0:
                print(f"  • Filter Recommendation: {summary['preprocessing_filter_recommendation_time']:.2f}s")
            if summary.get('preprocessing_filter_application_time', 0) > 0:
                print(f"  • Filter Application: {summary['preprocessing_filter_application_time']:.2f}s")
            if avg_preprocessing_time_per_scene > 0:
                print(f"Average Preprocessing per Scene: {avg_preprocessing_time_per_scene:.2f}s")
        else:
            print("Preprocessing Enabled: No")

    return summary


def cleanup_temporary_items(original_scene_files, encoded_scene_files, frame_directories, logging_enabled=True):
    """Cleans up temporary files and directories created during processing."""
    if logging_enabled:
        print("\n--- Cleaning up temporary files and directories ---")

    items_to_delete = []
    if original_scene_files:
        items_to_delete.extend(original_scene_files)
    if encoded_scene_files:
        items_to_delete.extend(encoded_scene_files)

    for item_path in items_to_delete:
        if item_path and os.path.exists(item_path):
            try:
                os.remove(item_path)
                if logging_enabled:
                    print(f"  Deleted temporary file: {item_path}")
            except OSError as e:
                if logging_enabled:
                    print(f"  Error deleting temporary file {item_path}: {e}")
        elif logging_enabled and item_path:
             print(f"  Temporary file not found for deletion: {item_path}")

    if frame_directories:
        for dir_path in frame_directories:
            if dir_path and os.path.exists(dir_path) and os.path.isdir(dir_path):
                try:
                    shutil.rmtree(dir_path)
                    if logging_enabled:
                        print(f"  Deleted temporary directory: {dir_path}")
                except OSError as e:
                    if logging_enabled:
                        print(f"  Error deleting temporary directory {dir_path}: {e}")
            elif logging_enabled and dir_path:
                print(f"  Temporary directory not found for deletion: {dir_path}")
    if logging_enabled:
        print("--- Cleanup complete ---")