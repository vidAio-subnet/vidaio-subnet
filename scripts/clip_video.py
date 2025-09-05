#!/usr/bin/env python3
"""
Video Clipping Script

This script clips a video file to exactly 19 seconds starting from the beginning.
Uses ffmpeg for efficient video processing.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import argparse

def _validate_video_file(file_path: str) -> bool:
    """
    Validate that the input file exists and is a readable video file.
    
    Args:
        file_path: Path to the video file
        
    Returns:
        True if file is valid, False otherwise
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return False
    
    if not os.path.isfile(file_path):
        print(f"Error: '{file_path}' is not a file.")
        return False
    
    if not os.access(file_path, os.R_OK):
        print(f"Error: File '{file_path}' is not readable.")
        return False
    
    return True

def _get_video_duration(file_path: str) -> Optional[float]:
    """
    Get the duration of a video file using ffprobe.
    
    Args:
        file_path: Path to the video file
        
    Returns:
        Duration in seconds, or None if ffprobe fails
    """
    try:
        cmd = [
            'ffprobe', 
            '-v', 'quiet', 
            '-show_entries', 'format=duration', 
            '-of', 'csv=p=0', 
            file_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        return duration
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        print(f"Error getting video duration: {e}")
        return None

def _generate_output_filename(input_path: str) -> str:
    """
    Generate output filename by adding '_19s' suffix before the extension.
    
    Args:
        input_path: Path to the input video file
        
    Returns:
        Output filename with '_19s' suffix
    """
    input_path_obj = Path(input_path)
    stem = input_path_obj.stem
    suffix = input_path_obj.suffix
    
    # Create output filename
    output_filename = f"{stem}_19s{suffix}"
    
    # If output would overwrite input, add timestamp
    if output_filename == input_path_obj.name:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{stem}_19s_{timestamp}{suffix}"
    
    return output_filename

def _clip_video(input_path: str, output_path: str, duration: int = 19) -> bool:
    """
    Clip the video to the specified duration using ffmpeg.
    
    Args:
        input_path: Path to the input video file
        output_path: Path for the output video file
        duration: Duration to clip to in seconds (default: 19)
        
    Returns:
        True if clipping was successful, False otherwise
    """
    try:
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-t', str(duration),
            '-c', 'copy',  # Copy streams without re-encoding for speed
            '-avoid_negative_ts', 'make_zero',
            '-y',  # Overwrite output file if it exists
            output_path
        ]
        
        print(f"Cliping video to {duration} seconds...")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        
        # Run ffmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("Video clipping completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error during video clipping: {e}")
        if e.stderr:
            print(f"FFmpeg error: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def clip_video_to_19s(video_path: str) -> bool:
    """
    Main function to clip a video file to 19 seconds.
    
    Args:
        video_path: Path to the input video file
        
    Returns:
        True if clipping was successful, False otherwise
    """
    # Validate input file
    if not _validate_video_file(video_path):
        return False
    
    # Get original video duration
    original_duration = _get_video_duration(video_path)
    if original_duration is None:
        return False
    
    print(f"Original video duration: {original_duration:.2f} seconds")
    
    # Check if video is already shorter than 19 seconds
    if original_duration <= 19:
        print(f"Video is already {original_duration:.2f} seconds long (≤ 19s). No clipping needed.")
        return True
    
    # Generate output filename
    output_filename = _generate_output_filename(video_path)
    output_path = os.path.join(os.path.dirname(video_path), output_filename)
    
    # Perform video clipping
    success = _clip_video(video_path, output_path, 19)
    
    if success:
        # Verify output file exists and get its duration
        if os.path.exists(output_path):
            output_duration = _get_video_duration(output_path)
            if output_duration:
                print(f"Output video duration: {output_duration:.2f} seconds")
                print(f"Output saved to: {output_path}")
            else:
                print("Warning: Could not verify output video duration")
        else:
            print("Warning: Output file was not created")
    
    return success

def main():
    """Main entry point for command line usage."""
    parser = argparse.ArgumentParser(
        description="Clip a video file to 19 seconds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clip_video.py video.mp4
  python clip_video.py /path/to/video.mov
  python clip_video.py "video with spaces.avi"
        """
    )
    
    parser.add_argument(
        'video_path',
        help='Path to the input video file'
    )
    #python clip_video.py --video_path /workspace/vidaio-subnet/2025-03-18 12-27-33.mov
    args = parser.parse_args()
    
    # Perform video clipping
    success = clip_video_to_19s(args.video_path)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 