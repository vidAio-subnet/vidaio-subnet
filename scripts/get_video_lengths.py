#!/usr/bin/env python3
"""
Video Length Scanner Script

This script scans the /videos directory and prints a list of all video files
along with their durations in a readable format.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional

def get_video_duration(file_path: str) -> Optional[float]:
    """
    Get the duration of a video file using ffprobe.
    
    Args:
        file_path: Path to the video file
        
    Returns:
        Duration in seconds, or None if ffprobe fails
    """
    try:
        # Use ffprobe to get video duration
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
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "1:23.45")
    """
    if seconds < 0:
        return "Invalid"
    
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    
    if minutes == 0:
        return f"{remaining_seconds:.2f}s"
    else:
        return f"{minutes}:{remaining_seconds:05.2f}"

def categorize_duration(seconds: float) -> str:
    """
    Categorize duration into chunks (e.g., 10s, 20s, 30s).
    Videos within ±0.5 seconds of a whole number are grouped together.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Category string (e.g., "10s", "20s", "1m", "2m")
    """
    if seconds < 0:
        return "Invalid"
    
    # Round to nearest second for categorization
    rounded_seconds = round(seconds)
    
    if rounded_seconds < 60:
        return f"{rounded_seconds}s"
    else:
        minutes = rounded_seconds // 60
        remaining_seconds = rounded_seconds % 60
        if remaining_seconds == 0:
            return f"{minutes}m"
        else:
            return f"{minutes}m{remaining_seconds}s"

def scan_videos_directory(videos_dir: str = "videos") -> List[Tuple[str, float]]:
    """
    Scan the videos directory and return a list of video files with their durations.
    
    Args:
        videos_dir: Path to the videos directory
        
    Returns:
        List of tuples containing (filename, duration_in_seconds)
    """
    video_files = []
    
    if not os.path.exists(videos_dir):
        print(f"Error: Directory '{videos_dir}' does not exist.")
        return []
    
    # Supported video extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    
    for filename in os.listdir(videos_dir):
        file_path = os.path.join(videos_dir, filename)
        
        # Check if it's a file and has a video extension
        if os.path.isfile(file_path) and Path(filename).suffix.lower() in video_extensions:
            duration = get_video_duration(file_path)
            if duration is not None:
                video_files.append((filename, duration))
            else:
                print(f"Warning: Could not get duration for {filename}")
    
    return sorted(video_files, key=lambda x: x[1])  # Sort by duration

def main():
    """Main function to run the video length scanner."""
    print("Video Length Scanner")
    print("=" * 50)
    
    # Check if ffprobe is available
    try:
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffprobe is not installed or not in PATH.")
        print("Please install FFmpeg to use this script.")
        print("On Ubuntu/Debian: sudo apt install ffmpeg")
        print("On CentOS/RHEL: sudo yum install ffmpeg")
        print("On macOS: brew install ffmpeg")
        sys.exit(1)
    
    # Scan for videos
    print("Scanning videos directory for video files...")
    video_files = scan_videos_directory()
    
    if not video_files:
        print("No video files found in /videos directory.")
        return
    
    # Print results
    print(f"\nFound {len(video_files)} video files:\n")
    print(f"{'Filename':<50} {'Duration':<15} {'Size':<10}")
    print("-" * 75)
    
    total_duration = 0
    
    for filename, duration in video_files:
        # Get file size
        file_path = os.path.join("videos", filename)
        file_size = os.path.getsize(file_path)
        size_mb = file_size / (1024 * 1024)
        
        formatted_duration = format_duration(duration)
        print(f"{filename:<50} {formatted_duration:<15} {size_mb:.1f}MB")
        total_duration += duration
    
    print("-" * 75)
    print(f"{'Total':<50} {format_duration(total_duration):<15} {len(video_files)} files")
    
    # Summary statistics
    if video_files:
        avg_duration = total_duration / len(video_files)
        min_duration = min(video_files, key=lambda x: x[1])[1]
        max_duration = max(video_files, key=lambda x: x[1])[1]
        
        print(f"\nSummary Statistics:")
        print(f"Average duration: {format_duration(avg_duration)}")
        print(f"Shortest video: {format_duration(min_duration)}")
        print(f"Longest video: {format_duration(max_duration)}")
        
        # Duration category summary
        print(f"\nDuration Categories:")
        print("=" * 40)
        
        # Group videos by duration category
        duration_categories = {}
        for filename, duration in video_files:
            category = categorize_duration(duration)
            if category not in duration_categories:
                duration_categories[category] = []
            duration_categories[category].append((filename, duration))
        
        # Sort categories by duration (convert to seconds for sorting)
        def category_sort_key(cat):
            if 'm' in cat:
                if 's' in cat:
                    # Format like "1m30s"
                    parts = cat.replace('m', ' ').replace('s', '').split()
                    return int(parts[0]) * 60 + int(parts[1])
                else:
                    # Format like "2m"
                    return int(cat.replace('m', '')) * 60
            else:
                # Format like "30s"
                return int(cat.replace('s', ''))
        
        sorted_categories = sorted(duration_categories.items(), key=lambda x: category_sort_key(x[0]))
        
        for category, videos in sorted_categories:
            total_category_duration = sum(duration for _, duration in videos)
            print(f"{category:<10} | {len(videos):>3} videos | Total: {format_duration(total_category_duration):<10} | Avg: {format_duration(total_category_duration/len(videos))}")
        
        print("=" * 40)
        print(f"Total categories: {len(duration_categories)}")

if __name__ == "__main__":
    main() 