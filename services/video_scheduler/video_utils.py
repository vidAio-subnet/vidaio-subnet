import requests
import os
import random
import time
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional, Tuple, List
import uuid

load_dotenv()

def download_trim_downscale_video(
    clip_duration: int,
    vid: int,
    task_type: str,
    output_dir: str = "videos"
) -> Optional[Tuple[List[str], List[int]]]:
    """
    Downloads a specified video with video id from Pexels, extracts multiple non-overlapping chunks 
    of the specified duration, and downscales each chunk to required resolution.

    Args:
        clip_duration (int): Desired clip duration in seconds.
        vid (int): Pexels video ID to download.
        task_type (str): Type of task determining resolution requirements.
        output_dir (str): Directory to save the processed videos.

    Returns:
        Optional[Tuple[List[str], List[int]]]: Lists of paths to the downscaled videos and their generated IDs, 
        or None on failure.
    """

    DOWNSCALE_HEIGHTS = {
        "HD24K": 1080,
        "4K28K": 2160,
    }

    downscale_height = DOWNSCALE_HEIGHTS.get(task_type, 540)

    api_key = os.getenv("PEXELS_API_KEY")
    
    if not api_key:
        raise ValueError("PEXELS_API_KEY not found in environment variables")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    url = f"https://api.pexels.com/videos/videos/{vid}"
    headers = {"Authorization": api_key}
    
    try:
        start_time = time.time()

        EXPECTED_RESOLUTIONS = {
            "SD2HD": (1920, 1080),
            "4K28K": (7680, 4320),
        }
        expected_width, expected_height = EXPECTED_RESOLUTIONS.get(task_type, (3840, 2160))
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()  
        
        data = response.json()
        if "video_files" not in data:
            raise ValueError("No video found or API error")
            
        video_url = next(
            (video["link"] for video in data["video_files"] 
            if video["width"] == expected_width and video["height"] == expected_height), 
            None
        )
        
        if video_url is None:
            return None, None
        
        temp_path = Path(output_dir) / f"{vid}_original.mp4"
        
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
    
        print("\nChecking video resolution...")
        video_clip = VideoFileClip(str(temp_path))
        actual_width, actual_height = video_clip.size
        
        if actual_width != expected_width or actual_height != expected_height:
            video_clip.close()
            error_msg = (f"Video resolution mismatch. Expected {expected_width}x{expected_height}, "
                        f"but got {actual_width}x{actual_height}")
            print(error_msg)
            raise ValueError(error_msg)
        
        print(f"\nVideo resolution verified: {actual_width}x{actual_height}")
        
        total_duration = video_clip.duration
        num_chunks = int(total_duration // clip_duration)
        
        print(f"\nVideo duration: {total_duration:.2f}s")
        print(f"Extracting {num_chunks} non-overlapping chunks of {clip_duration}s each")
        
        downscale_paths = []
        video_ids = []
        
        for i in range(num_chunks):
            video_id = uuid.uuid4()
            video_ids.append(video_id)
            
            clipped_path = Path(output_dir) / f"{video_id}_trim.mp4"
            downscale_path = Path(output_dir) / f"{video_id}_downscale.mp4"
            downscale_paths.append(str(downscale_path))
            
            start_time_clip = i * clip_duration
            print(f"\nProcessing chunk {i+1}/{num_chunks}: {start_time_clip}s to {start_time_clip + clip_duration}s")
            
            print(f"Trimming chunk {i+1}...")
            chunk_start_time = time.time()
            
            clipped_clip = video_clip.subclip(start_time_clip, start_time_clip + clip_duration)
            clipped_clip.write_videofile(str(clipped_path), codec='libx264', verbose=False, logger=None)
            
            print(f"Downscaling chunk {i+1}...")
            downscale_clip = clipped_clip.resize(height=downscale_height)
            downscale_clip.write_videofile(str(downscale_path), codec='libx264', verbose=False, logger=None)
            
            clipped_clip.close()
            downscale_clip.close()
            
            chunk_elapsed_time = time.time() - chunk_start_time
            print(f"Time taken to process chunk {i+1}: {chunk_elapsed_time:.2f} seconds")
        
        video_clip.close()
        
        print("\nCleaning up original downloaded file...")
        os.remove(temp_path)
        print(f"Deleted: {temp_path}")

        print(f"\nDone! Processed {num_chunks} chunks.")
        
        return downscale_paths, video_ids
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading video: {e}")
        return None, None
    except Exception as e:
        print(f"Error: {str(e)}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"Cleaned up downloaded file after error: {temp_path}")
            except:
                pass
        return None, None


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
