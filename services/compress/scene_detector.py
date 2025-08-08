import os
import json
import subprocess
from utils.fast_scene_detect import adaptive_scene_detection_check
from utils.split_video_into_scenes import split_video_into_scenes

def scene_detection(video_metadata):
    """
    Part 2: Scene detection based on video length using metadata from part 1.
    
    Args:
        video_metadata (dict): Metadata from part1_pre_processing containing:
                              - 'path': video file path
                              - 'codec': current video codec
                              - 'original_codec': original codec (if reencoded)
                              - 'target_codec': target codec for encoding (NEW)
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
    try:
        with open('services/compress/config.json', 'r') as f:
            config = json.load(f)
        print("✅ Configuration loaded successfully")
    except FileNotFoundError:
        print("⚠️ Config file not found, using default configuration")
        config = {
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
    
    # Extract metadata
    video_path = video_metadata['path']
    duration = video_metadata['duration']
    codec = video_metadata['codec']
    original_codec = video_metadata.get('original_codec', codec)
    target_codec = video_metadata.get('target_codec', 'auto')  # ✅ NEW: Extract target codec
    was_reencoded = video_metadata.get('was_reencoded', False)
    target_vmaf = video_metadata.get('target_vmaf', 93.0)
    target_quality = video_metadata.get('target_quality', 'Medium')

    print(f"🎬 Processing video: {os.path.basename(video_path)}")
    print(f"   ⏱️ Duration: {duration:.1f}s, Codec: {codec}, Reencoded: {was_reencoded}")
    print(f"   🎯 Target: {target_quality} (VMAF: {target_vmaf})")
    print(f"   🎥 Codec flow: {original_codec} → {codec} → {target_codec}")  # ✅ NEW: Show codec flow
    
    # Get processing configuration
    video_processing_config = config.get('video_processing', {})
    short_video_threshold = video_processing_config.get('SHORT_VIDEO_THRESHOLD', 20)
    # Get directories from config only
    temp_dir = config.get('directories', {}).get('temp_dir', './videos/temp_scenes')
    
    if duration <= short_video_threshold:
        print(f"📏 Video is shorter than {short_video_threshold}s. Treating as a single scene.")
        return [{
            'path': video_metadata['path'],
            'scene_number': 1,
            'start_time': 0.0,
            'end_time': duration,
            'duration': duration,
            'original_video_metadata': video_metadata  # ✅ Pass complete metadata including target_codec
        }]

    # SCENE DETECTION: Check for multiple scenes
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
        scene_config = config.get('scene_detection', {})
        enable_fallback = scene_config.get('enable_time_based_fallback', False)
        
        if enable_fallback:
            print(f"   🔄 Time-based fallback enabled. Attempting time-based scene splitting...")
            return time_based_splitting(video_path, temp_dir, config, duration, video_metadata)
        else:
            print(f"   ✅ Using single scene (time-based fallback disabled)")
            return [{
                'path': video_path,
                'scene_number': 1,
                'start_time': 0.0,
                'end_time': duration,
                'duration': duration,
                'original_video_metadata': video_metadata  # ✅ Pass complete metadata including target_codec
            }]

def time_based_splitting(video_path, temp_dir, config, duration, original_metadata, logging_enabled=True):
    """
    Split video into scenes and return metadata for each scene.
    
    Args:
        video_path (str): Path to the video file
        temp_dir (str): Temporary directory for scene files
        config (dict): Configuration dictionary
        duration (float): Video duration in seconds
        original_metadata (dict): Complete metadata from Part 1 including codec info
        logging_enabled (bool): Enable detailed logging
    
    Returns:
        list: List of scene metadata dictionaries
    """
    if logging_enabled:
        print(f"\n--- Running time-based scene splitting with metadata ---")
        # ✅ NEW: Log codec information for scene splitting
        print(f"   🎥 Source codec: {original_metadata.get('codec', 'unknown')}")
        print(f"   🎯 Target codec: {original_metadata.get('target_codec', 'auto')}")
    
    try:
        total_duration = duration
        if logging_enabled:
            print(f"Using provided duration: {total_duration:.1f}s")
        
        if not total_duration:
            return [{
                'path': video_path,
                'scene_number': 1,
                'start_time': 0.0,
                'end_time': duration,
                'duration': duration,
                'original_video_metadata': original_metadata
            }]
        
        # Get scene duration from config
        scene_config = config.get('scene_detection', {})
        scene_duration = scene_config.get('time_based_scene_duration', None)
        
        # Auto-calculate scene duration if not specified
        if scene_duration is None:
            if total_duration <= 120:
                scene_duration = 60
            elif total_duration <= 600:
                scene_duration = 120
            elif total_duration <= 1800:
                scene_duration = 300
            else:
                scene_duration = 600
            
            if logging_enabled:
                print(f"Auto-calculated scene duration: {scene_duration}s for {total_duration:.1f}s video")
        
        # Calculate scene boundaries
        scene_boundaries = []
        current_start = 0
        
        while current_start < total_duration:
            scene_end = min(current_start + scene_duration, total_duration)
            scene_boundaries.append((current_start, scene_end))
            current_start = scene_end
        
        if logging_enabled:
            print(f"Calculated {len(scene_boundaries)} time-based scenes:")
            for i, (start, end) in enumerate(scene_boundaries):
                print(f"  Scene {i+1}: {start:.1f}s - {end:.1f}s (duration: {end-start:.1f}s)")
        
        # If only one scene, return original video
        if len(scene_boundaries) <= 1:
            return [{
                'path': video_path,
                'scene_number': 1,
                'start_time': 0.0,
                'end_time': total_duration,
                'duration': total_duration,
                'original_video_metadata': original_metadata
            }]
        
        # Create output directory
        os.makedirs(temp_dir, exist_ok=True)
        
        # ✅ IMPROVED: Choose scene file extension based on source codec
        source_codec = original_metadata.get('codec', 'h264').lower()
        if source_codec in ['ffv1', 'prores', 'dnxhd']:
            scene_extension = '.mkv'  # Use MKV for lossless codecs
        elif source_codec in ['av1', 'vp9']:
            scene_extension = '.webm'  # Use WebM for modern codecs
        else:
            scene_extension = '.mp4'  # Default to MP4 for compatibility
        
        if logging_enabled:
            print(f"   📦 Using {scene_extension} container for {source_codec} codec")
        
        # Split video and create metadata
        scene_metadata_list = []
        
        for i, (start_time_sec, end_time_sec) in enumerate(scene_boundaries):
            scene_number = i + 1
            scene_filename = f"scene_{scene_number:03d}{scene_extension}"
            scene_path = os.path.join(temp_dir, scene_filename)
            
            duration_scene = end_time_sec - start_time_sec
            
            # ✅ IMPROVED: FFmpeg command with codec-aware copy settings
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time_sec),
                '-i', video_path,
                '-t', str(duration_scene),
                '-c', 'copy',  # Stream copy for speed
                '-map', '0',   # Copy all streams
                '-avoid_negative_ts', 'make_zero',
                scene_path
            ]
            
            # ✅ NEW: Add codec-specific options if needed
            if source_codec == 'ffv1':
                # For FFV1, ensure proper MKV muxing
                cmd.extend(['-f', 'matroska'])
            elif source_codec in ['av1', 'vp9']:
                # For modern codecs, ensure proper WebM muxing
                cmd.extend(['-f', 'webm'])
            
            if logging_enabled:
                print(f"Extracting scene {scene_number}/{len(scene_boundaries)}: {start_time_sec:.1f}s-{end_time_sec:.1f}s")
            
            try:
                result = subprocess.run(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    text=True, 
                    check=True
                )
                
                if os.path.exists(scene_path) and os.path.getsize(scene_path) > 0:
                    # Create scene metadata
                    scene_metadata = {
                        'path': scene_path,
                        'scene_number': scene_number,
                        'start_time': start_time_sec,
                        'end_time': end_time_sec,
                        'duration': duration_scene,
                        'original_video_metadata': original_metadata,  # ✅ Complete metadata including codec info
                        'file_size_mb': os.path.getsize(scene_path) / (1024 * 1024),
                        'scene_codec': source_codec,  # ✅ NEW: Track scene codec
                        'scene_container': scene_extension[1:]  # ✅ NEW: Track container format
                    }
                    scene_metadata_list.append(scene_metadata)
                    
                    if logging_enabled:
                        print(f"  ✅ Created: {scene_filename} ({scene_metadata['file_size_mb']:.1f} MB)")
                else:
                    if logging_enabled:
                        print(f"  ❌ Failed: {scene_filename}")
                        
            except subprocess.CalledProcessError as e:
                if logging_enabled:
                    print(f"  ❌ FFmpeg error for scene {scene_number}: {e.stderr}")
                continue
            except Exception as e:
                if logging_enabled:
                    print(f"  ❌ Unexpected error for scene {scene_number}: {e}")
                continue
        
        if logging_enabled:
            print(f"Time-based splitting completed: {len(scene_metadata_list)} scenes created")
            if scene_metadata_list:
                total_size = sum(scene['file_size_mb'] for scene in scene_metadata_list)
                print(f"   📊 Total scenes size: {total_size:.1f} MB")
        
        return scene_metadata_list if scene_metadata_list else [{
            'path': video_path,
            'scene_number': 1,
            'start_time': 0.0,
            'end_time': total_duration,
            'duration': total_duration,
            'original_video_metadata': original_metadata
        }]
        
    except Exception as e:
        if logging_enabled:
            print(f"Time-based splitting failed: {e}")
        return [{
            'path': video_path,
            'scene_number': 1,
            'start_time': 0.0,
            'end_time': duration,
            'duration': duration,
            'original_video_metadata': original_metadata
        }]

def create_scene_metadata_from_files(scene_files, original_metadata, config):
    """
    Create metadata for scenes created by PySceneDetect.
    Note: PySceneDetect doesn't give us exact timing, so we estimate.
    
    Args:
        scene_files (list): List of scene file paths
        original_metadata (dict): Complete metadata from Part 1 including codec info
        config (dict): Configuration dictionary
    
    Returns:
        list: List of scene metadata dictionaries
    """
    scene_metadata_list = []
    total_duration = original_metadata['duration']
    source_codec = original_metadata.get('codec', 'unknown')
    
    if not scene_files:
        return [{
            'path': original_metadata['path'],
            'scene_number': 1,
            'start_time': 0.0,
            'end_time': total_duration,
            'duration': total_duration,
            'original_video_metadata': original_metadata
        }]
    
    # Estimate timing based on file count (not perfect, but reasonable)
    estimated_scene_duration = total_duration / len(scene_files)
    
    print(f"   🎥 Creating metadata for {len(scene_files)} scenes from {source_codec} source")
    
    for i, scene_file in enumerate(scene_files):
        scene_number = i + 1
        start_time = i * estimated_scene_duration
        end_time = min((i + 1) * estimated_scene_duration, total_duration)
        
        # Adjust last scene to cover remaining time
        if i == len(scene_files) - 1:
            end_time = total_duration
        
        # ✅ NEW: Extract scene file extension for container tracking
        scene_extension = os.path.splitext(scene_file)[1][1:] if os.path.splitext(scene_file)[1] else 'unknown'
        
        scene_metadata = {
            'path': scene_file,
            'scene_number': scene_number,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'original_video_metadata': original_metadata,  # ✅ Complete metadata including codec info
            'file_size_mb': os.path.getsize(scene_file) / (1024 * 1024) if os.path.exists(scene_file) else 0,
            'scene_codec': source_codec,  # ✅ NEW: Track scene codec
            'scene_container': scene_extension  # ✅ NEW: Track container format
        }
        scene_metadata_list.append(scene_metadata)
    
    return scene_metadata_list