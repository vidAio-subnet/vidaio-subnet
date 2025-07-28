

from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg, save_images
import datetime
import time
import os
import ffmpeg
import json
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
    This function checks whether the scene files have audio or not, and builds the
    concat filter accordingly.
    
    Args:
        scene_files (list): List of file paths to the scene videos.
        output_video (str): Path to the final merged video.
        logging_enabled (bool): Whether to print messages.
    """
    # Create a temporary file with the correct format
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        for video in scene_videos:
            # Write each file on a separate line with proper syntax
            f.write(f"file '{os.path.abspath(video)}'\n")
        concat_file = f.name
    
    # Use the concat demuxer with the file list
    try:
        cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', 
            '-i', concat_file, '-c', 'copy', output_video
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        if logging_enabled:
            print(f"Successfully merged videos into {output_video}")
        return True
    except subprocess.CalledProcessError as e:
        if logging_enabled:
            print(f"Error merging videos: {e.stderr}")
        return False
    finally:
        # Clean up temp file
        os.unlink(concat_file)
