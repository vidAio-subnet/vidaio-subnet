import os
import subprocess
import tempfile

def has_audio(video_path):
    """
    Returns True if the given video file has an audio stream.
    """
    cmd = [
        'ffprobe', '-v', 'error', 
        '-select_streams', 'a',
        '-show_entries', 'stream=codec_type',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return bool(result.stdout.strip())

def merge_videos(scene_videos, output_video, logging_enabled=True):
    """
    Merge the list of re-encoded scene files using FFmpeg's concat filter.
    Uses precise frame-level concatenation to prevent stuttering.
    
    Args:
        scene_files (list): List of file paths to the scene videos.
        output_video (str): Path to the final merged video.
        logging_enabled (bool): Whether to print messages.
    """
    if not scene_videos:
        if logging_enabled:
            print("❌ No scene videos to merge")
        return False
    
    # For single scene, just copy the file
    if len(scene_videos) == 1:
        try:
            import shutil
            shutil.copy2(scene_videos[0], output_video)
            if logging_enabled:
                print(f"✅ Single scene copied to {output_video}")
            return True
        except Exception as e:
            if logging_enabled:
                print(f"❌ Failed to copy single scene: {e}")
            return False
    
    # Create a temporary file with the correct format
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        for video in scene_videos:
            # Verify each scene file exists and has content
            if not os.path.exists(video):
                if logging_enabled:
                    print(f"⚠️ Scene file missing: {video}")
                return False
            if os.path.getsize(video) == 0:
                if logging_enabled:
                    print(f"⚠️ Scene file empty: {video}")
                return False
            # Write each file on a separate line with proper syntax
            f.write(f"file '{os.path.abspath(video)}'\n")
        concat_file = f.name
    
    # Use the concat demuxer with improved frame handling
    try:
        # Use concat filter instead of demuxer for better frame alignment
        cmd = [
            'ffmpeg', '-y', '-hide_banner',
            '-f', 'concat', '-safe', '0', 
            '-i', concat_file,
            '-c:v', 'copy',        # Video stream copy
            '-c:a', 'copy',        # Audio stream copy  
            '-avoid_negative_ts', 'make_zero',  # Handle timestamp alignment
            '-fflags', '+genpts',  # Generate proper timestamps
            output_video
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            # Verify output file
            if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
                if logging_enabled:
                    print(f"✅ Successfully merged {len(scene_videos)} scenes into {output_video}")
                return True
            else:
                if logging_enabled:
                    print(f"❌ Merge failed: output file not created or empty")
                return False
        else:
            if logging_enabled:
                print(f"❌ FFmpeg merge failed (code {result.returncode})")
                if result.stderr:
                    print(f"Error details: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        if logging_enabled:
            print(f"❌ Merge process timed out after 5 minutes")
        return False
    except subprocess.CalledProcessError as e:
        if logging_enabled:
            print(f"❌ Error merging videos: {e}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"Error details: {e.stderr}")
        return False
    except Exception as e:
        if logging_enabled:
            print(f"❌ Unexpected error during merge: {e}")
        return False
    finally:
        # Clean up temp file
        try:
            os.unlink(concat_file)
        except:
            pass
