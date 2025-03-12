import requests
import os
import random
import time
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional, Tuple
import uuid

load_dotenv()


def download_trim_downscale_video(
    clip_duration: int,
    vid: int,
    task_type : str,
    output_dir: str = "videos"
) -> Optional[Tuple[str, int]]:
    """
    Downloads a specified videos with video id from Pexels, trims it to the specified duration,
    and downscales it to required resolution.

    Args:
        clip_duration (int): Desired clip duration in seconds.
        vid (int): Pexels video ID to download.
        output_dir (str): Directory to save the processed videos.

    Returns:
        Optional[Tuple[str, int]]: Path to the downscaled video and the generated video ID, or None on failure.
    """

    downscale_height = 1080
    if task_type != "HD24K":
        downscale_height = 540

    api_key = os.getenv("PEXELS_API_KEY")
    
    if not api_key:
        raise ValueError("PEXELS_API_KEY not found in environment variables")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # API endpoint
    url = f"https://api.pexels.com/videos/videos/{vid}"
    headers = {"Authorization": api_key}
    
    try:
        start_time = time.time()
        # Get video details
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception for bad status codes
        
        data = response.json()
        if "video_files" not in data:
            raise ValueError("No video found or API error")
            
        # Get video download URL (first video file)
        video_url = data["video_files"][0]["link"]
        
        # Prepare output path
        video_id = uuid.uuid4()

        temp_path = Path(output_dir) / f"{video_id}_original.mp4"
        clipped_path = Path(output_dir) / f"{video_id}_trim.mp4"
        downscale_path = Path(output_dir) / f"{video_id}_downscale.mp4"
        
        # Download video with progress bar
        print(f"\nDownloading video ID: {vid}")
        response = requests.get(video_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(temp_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                size = f.write(chunk)
                pbar.update(size)
                
        print(f"\nVideo downloaded successfully to: {temp_path}")
        elapsed_time = time.time() - start_time
        print(f"Time taken to download video: {elapsed_time:.2f} seconds")
    
        # Trim video
        print("\nTrimming video...")
        start_time = time.time()

        print("\nChecking video resolution...")
        video_clip = VideoFileClip(str(temp_path))
        actual_width, actual_height = video_clip.size
        
        # expected_width, expected_height = 1920, 1080  # for SD2HD 
        expected_width, expected_height = 4096, 2160  # for HD24K
        
        if actual_width != expected_width or actual_height != expected_height:
            video_clip.close()
            error_msg = (f"Video resolution mismatch. Expected {expected_width}x{expected_height}, "
                        f"but got {actual_width}x{actual_height}")
            print(error_msg)
            raise ValueError(error_msg)
        
        # Continue with existing trimming code...
        print(f"\nVideo resolution verified: {actual_width}x{actual_height}")


        start_time_clip = 0 if video_clip.duration <= clip_duration else random.uniform(0, video_clip.duration - clip_duration)
        print(f"Trimming {clip_duration}s from position {start_time_clip:.1f}s")

        clipped_clip = video_clip.subclip(start_time_clip, start_time_clip + min(clip_duration, video_clip.duration))
        clipped_clip.write_videofile(str(clipped_path), codec='libx264', verbose=False, logger=None)

        elapsed_time = time.time() - start_time
        print(f"Time taken to clip video: {elapsed_time:.2f} seconds")

        # Downscaling
        print("\nSaving downscaled version...")
        start_time = time.time()
        downscale_clip = clipped_clip.resize(height=downscale_height)
        downscale_clip.write_videofile(str(downscale_path), codec='libx264', verbose=False, logger=None)

        elapsed_time = time.time() - start_time
        print(f"Time taken to save downscale version: {elapsed_time:.2f} seconds")

        # Cleanup
        video_clip.close()
        clipped_clip.close()
        downscale_clip.close()

        print(f"\nDone! Saved files:\n- {clipped_path}\n- {downscale_path}")

        return str(downscale_path), video_id
    except requests.exceptions.RequestException as e:
        print(f"Error downloading video: {e}")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def get_trim_video_path(file_id: int, dir_path: str = "videos") -> str:
    """Returns the path of the clipped trim video based on the file ID."""
    return str(Path(dir_path) / f"{file_id}_trim.mp4")


def delete_videos_with_fileid(file_id: int, dir_path: str = "videos") -> None:
    """Deletes all video files associated with the given file ID."""

    files_to_delete = [
        Path(dir_path) / f"{file_id}_original.mp4",
        Path(dir_path) / f"{file_id}_trim.mp4",
        Path(dir_path) / f"{file_id}_downscale.mp4",
    ]

    for file_path in files_to_delete:
        try:
            if file_path.exists():
                file_path.unlink()
                print(f"Deleted: {file_path}")
            else:
                print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


if __name__ == "__main__":
    
    CLIP_DURATION = 2
    vid = 2257054
    
    download_trim_downscale_video(CLIP_DURATION, vid)
