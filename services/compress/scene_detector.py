"""
Scene Detection Module

This module handles scene detection and splitting for video processing.
It supports multiple detection modes including adaptive, time-based, and forced splitting.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from utils.fast_scene_detect import adaptive_scene_detection_check
from utils.split_video_into_scenes import split_video_into_scenes
from utils.video_utils import get_keyframes


# ============================================================================
# Constants
# ============================================================================

# Default configuration values
DEFAULT_CONFIG = {
    'directories': {
        'temp_dir': './videos/temp_scenes',
        'output_dir': './output'
    },
    'video_processing': {
        'SHORT_VIDEO_THRESHOLD': 20,
        'target_vmaf': 93.0
    },
    'scene_detection': {
        'enable_time_based_fallback': True,
        'time_based_scene_duration': 60
    }
}


# ============================================================================
# Main Function
# ============================================================================

def scene_detection(video_metadata: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    Part 2: Scene detection based on video length using metadata from part 1.
    
    Args:
        video_metadata: Metadata from part1_pre_processing containing:
                       - 'path': video file path
                       - 'codec': current video codec
                       - 'original_codec': original codec (if reencoded)
                       - 'target_codec': target codec for encoding
                       - 'duration': video duration in seconds
                       - 'was_reencoded': boolean
                       - 'target_vmaf': target VMAF score
                       - 'target_quality': target quality level
                       - 'processing_info': processing details
        
    Returns:
        list: List of scene metadata dictionaries, each containing:
              - 'path': scene file path
              - 'scene_number': scene index (1-based)
              - 'start_time': scene start time in seconds
              - 'end_time': scene end time in seconds
              - 'duration': scene duration in seconds
              - 'original_video_metadata': complete metadata from part 1
    """
    # Load configuration
    config = _load_configuration()
    
    # Extract metadata
    video_path = video_metadata['path']
    duration = video_metadata['duration']
    codec = video_metadata['codec']
    original_codec = video_metadata.get('original_codec', codec)
    target_codec = video_metadata.get('target_codec', 'auto')
    was_reencoded = video_metadata.get('was_reencoded', False)
    target_vmaf = video_metadata.get('target_vmaf', 93.0)
    target_quality = video_metadata.get('target_quality', 'Medium')

    print(f"🎬 Processing video: {os.path.basename(video_path)}")
    print(f"   ⏱️ Duration: {duration:.1f}s, Codec: {codec}, Reencoded: {was_reencoded}")
    print(f"   🎯 Target: {target_quality} (VMAF: {target_vmaf})")
    print(f"   🎥 Codec flow: {original_codec} → {codec} → {target_codec}")
    
    # Get processing configuration
    video_processing_config = config.get('video_processing', {})
    short_video_threshold = video_processing_config.get('SHORT_VIDEO_THRESHOLD', 20)
    temp_dir = config.get('directories', {}).get('temp_dir', './videos/temp_scenes')
    
    # Handle short videos
    if duration <= short_video_threshold:
        print(f"📏 Video is shorter than {short_video_threshold}s. Treating as a single scene.")
        return _create_single_scene_metadata(video_metadata, duration)
    
    # Get scene detection configuration
    scene_config = config.get('scene_detection', {})
    detection_mode = scene_config.get('mode', 'adaptive').lower()
    
    print(f"   🔍 Scene detection mode: {detection_mode}")
    
    # Execute appropriate detection method
    if detection_mode == 'force_time_based':
        return _execute_force_time_based_splitting(
            video_path, temp_dir, scene_config, duration, video_metadata
        )
    elif detection_mode == 'adaptive':
        return _execute_adaptive_scene_detection(
            video_path, temp_dir, scene_config, duration, video_metadata, config
        )
    else:
        print(f"   ❌ Unknown scene detection mode: {detection_mode}. Falling back to adaptive.")
        return _execute_adaptive_scene_detection(
            video_path, temp_dir, scene_config, duration, video_metadata, config
        )


# ============================================================================
# Scene Detection Methods
# ============================================================================

def _execute_force_time_based_splitting(
    video_path: str, 
    temp_dir: str, 
    scene_config: Dict[str, Any], 
    duration: float, 
    video_metadata: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Execute forced time-based scene splitting."""
    force_duration = scene_config.get('force_time_based_duration', 10)
    print(f"   ⏰ Forcing time-based splitting with {force_duration}s segments")
    return force_time_based_splitting(
        video_path, temp_dir, force_duration, duration, video_metadata
    )


def _execute_adaptive_scene_detection(
    video_path: str, 
    temp_dir: str, 
    scene_config: Dict[str, Any], 
    duration: float, 
    video_metadata: Dict[str, Any], 
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Execute adaptive scene detection with fallback options."""
    print(f"   🔍 Running adaptive scene detection...")
    
    has_multiple_scenes, estimated_scenes, _ = adaptive_scene_detection_check(
        video_path, duration, logging_enabled=True
    )

    if has_multiple_scenes:
        print(f"   ✅ Multiple scenes detected ({estimated_scenes} estimated). Proceeding with full scene splitting.")
        scene_files, _ = split_video_into_scenes(video_path, temp_dir)
        
        if scene_files and len(scene_files) > 1:
            print(f"   📄 Created {len(scene_files)} scene files")
            return create_scene_metadata_from_files(scene_files, video_metadata, config)
        else:
            print(f"   ⚠️ Scene splitting failed or found only 1 scene. Falling back to time-based splitting.")
            return time_based_splitting(video_path, temp_dir, config, duration, video_metadata)
    else:
        print(f"   📏 Single scene detected.")
        enable_fallback = scene_config.get('enable_time_based_fallback', False)
        
        if enable_fallback:
            print(f"   🔄 Time-based fallback enabled. Attempting time-based scene splitting...")
            return time_based_splitting(video_path, temp_dir, config, duration, video_metadata)
        else:
            print(f"   ✅ Using single scene (time-based fallback disabled)")
            return _create_single_scene_metadata(video_metadata, duration)


# ============================================================================
# Time-based Splitting Functions
# ============================================================================

def force_time_based_splitting(
    video_path: str, 
    temp_dir: str, 
    segment_duration: float, 
    total_duration: float, 
    original_metadata: Dict[str, Any], 
    logging_enabled: bool = True
) -> List[Dict[str, Any]]:
    """
    Force time-based splitting with keyframe-aligned boundaries to prevent stuttering.
    
    Args:
        video_path: Path to the video file
        temp_dir: Temporary directory for scene files
        segment_duration: Target duration for each segment in seconds
        total_duration: Total video duration in seconds
        original_metadata: Complete metadata from Part 1
        logging_enabled: Enable detailed logging
    
    Returns:
        list: List of scene metadata dictionaries
    """
    if logging_enabled:
        print(f"\n--- Force time-based splitting: {segment_duration}s segments ---")
        print(f"   🎥 Source codec: {original_metadata.get('codec', 'unknown')}")
        print(f"   ⏱️ Total duration: {total_duration:.1f}s")
        print(f"   ✂️ Segment duration: {segment_duration}s")
    
    # Get keyframes for precise cutting
    keyframes = _get_video_keyframes(video_path, logging_enabled)
    
    # Calculate scene boundaries, preferring keyframe alignment
    scene_boundaries = _calculate_scene_boundaries(
        total_duration, segment_duration, keyframes, logging_enabled
    )
    
    # Create scene metadata
    scenes = []
    for i, (start_time, end_time) in enumerate(scene_boundaries):
        scene_number = i + 1
        scene_duration = end_time - start_time
        
        if logging_enabled:
            print(f"   Scene {scene_number}: {start_time:.1f}s - {end_time:.1f}s (duration: {scene_duration:.1f}s)")
        
        scenes.append({
            'path': video_path,  # Original video path for time-based scenes
            'scene_number': scene_number,
            'start_time': start_time,
            'end_time': end_time,
            'duration': scene_duration,
            'original_video_metadata': original_metadata,
            'splitting_method': 'force_time_based',
            'segment_duration': segment_duration
        })
    
    if logging_enabled:
        print(f"   ✅ Created {len(scenes)} time-based scenes")
    
    return scenes


def time_based_splitting(
    video_path: str, 
    temp_dir: str, 
    config: Dict[str, Any], 
    duration: float, 
    video_metadata: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Time-based scene splitting using configuration settings.
    
    Args:
        video_path: Path to the video file
        temp_dir: Temporary directory for scene files
        config: Configuration dictionary
        duration: Video duration in seconds
        video_metadata: Complete metadata from Part 1
        
    Returns:
        list: List of scene metadata dictionaries
    """
    scene_config = config.get('scene_detection', {})
    time_based_duration = scene_config.get('time_based_scene_duration', 90)
    
    print(f"   ⏰ Time-based splitting with {time_based_duration}s segments")
    
    return force_time_based_splitting(
        video_path, temp_dir, time_based_duration, duration, video_metadata
    )


# ============================================================================
# Helper Functions
# ============================================================================

def _load_configuration() -> Dict[str, Any]:
    """Load configuration from config.json or use defaults."""
    try:
        with open('services/compress/config.json', 'r') as f:
            config = json.load(f)
        print("✅ Configuration loaded successfully")
        return config
    except FileNotFoundError:
        print("⚠️ Config file not found, using default configuration")
        return DEFAULT_CONFIG


def _create_single_scene_metadata(
    video_metadata: Dict[str, Any], 
    duration: float
) -> List[Dict[str, Any]]:
    """Create metadata for a single scene video."""
    return [{
        'path': video_metadata['path'],
        'scene_number': 1,
        'start_time': 0.0,
        'end_time': duration,
        'duration': duration,
        'original_video_metadata': video_metadata
    }]


def _get_video_keyframes(video_path: str, logging_enabled: bool) -> List[float]:
    """Get video keyframes for precise scene cutting."""
    try:
        keyframes = get_keyframes(video_path)
        if logging_enabled:
            print(f"   🔑 Found {len(keyframes)} keyframes for alignment")
        return keyframes
    except Exception as e:
        if logging_enabled:
            print(f"   ⚠️ Could not get keyframes: {e}")
            print(f"   📐 Using time-based boundaries (may cause stuttering)")
        return []


def _calculate_scene_boundaries(
    total_duration: float, 
    segment_duration: float, 
    keyframes: List[float], 
    logging_enabled: bool
) -> List[Tuple[float, float]]:
    """Calculate scene boundaries with keyframe alignment when possible."""
    boundaries = []
    current_time = 0.0
    
    while current_time < total_duration:
        end_time = min(current_time + segment_duration, total_duration)
        
        # Try to align with nearest keyframe
        if keyframes:
            nearest_keyframe = _find_nearest_keyframe(current_time, keyframes)
            if nearest_keyframe and nearest_keyframe > current_time and nearest_keyframe < end_time:
                end_time = nearest_keyframe
                if logging_enabled:
                    print(f"      Aligned scene boundary with keyframe at {end_time:.1f}s")
        
        boundaries.append((current_time, end_time))
        current_time = end_time
    
    return boundaries


def _find_nearest_keyframe(current_time: float, keyframes: List[float]) -> Optional[float]:
    """Find the nearest keyframe after the current time."""
    for keyframe in keyframes:
        if keyframe > current_time:
            return keyframe
    return None


def create_scene_metadata_from_files(
    scene_files: List[str], 
    video_metadata: Dict[str, Any], 
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Create scene metadata from split scene files.
    
    Args:
        scene_files: List of scene file paths
        video_metadata: Original video metadata
        config: Configuration dictionary
        
    Returns:
        list: List of scene metadata dictionaries
    """
    scenes = []
    
    for i, scene_file in enumerate(scene_files):
        scene_number = i + 1
        
        # Get scene duration (you might want to implement this)
        scene_duration = 0.0  # Placeholder
        
        scenes.append({
            'path': scene_file,
            'scene_number': scene_number,
            'start_time': 0.0,  # Placeholder
            'end_time': scene_duration,
            'duration': scene_duration,
            'original_video_metadata': video_metadata,
            'splitting_method': 'adaptive_detection'
        })
    
    return scenes