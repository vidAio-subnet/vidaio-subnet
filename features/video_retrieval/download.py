import requests
import os
import random
import time
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from pathlib import Path
from typing import Optional, Tuple
import math
import json
import argparse
from modules.config import config

def load_video_ids_from_json(json_path: str) -> list[int]:
    with open(json_path, "r") as file:
        return json.load(file)

def download_trim_downscale_video(
    clip_duration: int,
    vid: int,
    task_type : str,
) -> Optional[Tuple[str, int]]:
    """
    Downloads a specified videos with video id from Pexels, trims it to the specified duration,
    and downscales it to required resolution.

    Args:
        clip_duration (int): Desired clip duration in seconds.
        vid (int): Pexels video ID to download.

    Returns:
        Optional[Tuple[str, int]]: Path to the downscaled video and the generated video ID, or None on failure.
    """

    DOWNSCALE_HEIGHTS = {
        "HD24K": 1080,
        "4K28K": 2160,
        
    }

    downscale_height = DOWNSCALE_HEIGHTS.get(task_type, 540)
    print(f"downscale_height: {downscale_height}")

    api_keys = [
        "WD9OjZcFWztgC6no5AQksf5ixmdrW1EfLqpiYqxOWozhLX2DKzCYVkcO",
        "IVVpGOKxM8vkqJQGziB7PN9FX8WhxWAFlusWe9UvMyPgPpMcAkNR2YNK",
        "nFfgK7tYZqdcBKPOQ3xhg1MNsofTa9KoWHfRtqFgtPT9TOMfg6uqV07R"
    ]
    api_key = random.choice(api_keys)
    # api_key = os.getenv("PEXELS_API_KEY")
    
    if not api_key:
        raise ValueError("PEXELS_API_KEY not found in environment variables")
    
    # API endpoint
    url = f"https://api.pexels.com/videos/videos/{vid}"
    headers = {"Authorization": api_key}
    
    try:
        video_id = f'{task_type}_{vid}'

        temp_path = Path(config['default_video_dir']) / f"{video_id}_original.mp4"
        trim_path = Path(config['default_video_dir']) / f"{video_id}_trim.mp4"
        origin_path = Path(config['default_video_dir']) / f"{video_id}.mp4"
        if temp_path.exists() and origin_path.exists() and trim_path.exists():
            print(f"✅ Already downloaded and processed: {temp_path}")
            return str(temp_path), video_id
        
        start_time = time.time()

        EXPECTED_RESOLUTIONS = {
            "SD2HD": (1920, 1080),
            "4K28K": (7680, 4320),
        }
        expected_width, expected_height = EXPECTED_RESOLUTIONS.get(task_type, (3840, 2160))
        print(f"expected_width, expected_height: {expected_width, expected_height}")
        # Get video details
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception for bad status codes
        data = response.json()
        if "video_files" not in data:
            raise ValueError("No video found or API error")
            
        if temp_path.exists() == False:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise exception for bad status codes
            data = response.json()
            if "video_files" not in data:
                raise ValueError("No video found or API error")
                
            # Get the highest resolution video file
            video_url = next(
                (video["link"] for video in data["video_files"] 
                if video["width"] == expected_width and video["height"] == expected_height), 
                None
            )
            
            if video_url is None:
                print(f"video url is None: id: {video_url}")
                return None, None
            
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
        
        if actual_width != expected_width or actual_height != expected_height:
            video_clip.close()
            error_msg = (f"Video resolution mismatch. Expected {expected_width}x{expected_height}, "
                        f"but got {actual_width}x{actual_height}")
            print(error_msg)
            return None, None
        
        downscale = video_clip.resize(height=downscale_height)
        if origin_path.exists() == False:
            downscale.write_videofile(str(origin_path), codec='libx264', verbose=False, logger=None)
        # Continue with existing trimming code...
        print(f"\nVideo resolution verified: {actual_width}x{actual_height}")

        start_time_clip = 0 if video_clip.duration <= clip_duration else random.uniform(0, video_clip.duration - clip_duration)
        print(f"Trimming {clip_duration}s from position {start_time_clip:.1f}s")
        clipped_clip = video_clip.subclip(start_time_clip, start_time_clip + min(clip_duration, video_clip.duration))
        clipped_clip.write_videofile(str(trim_path), codec='libx264', verbose=False, logger=None)

        elapsed_time = time.time() - start_time
        print(f"Time taken to clip video: {elapsed_time:.2f} seconds")

        # Downscaling
        print("\nSaving downscaled clip version...")
        start_time = time.time()
        
        downscale_clip = clipped_clip.resize(height=downscale_height)

        fps = round(downscale_clip.fps, 2)
        clip_frames = math.ceil(min(clip_duration, video_clip.duration) * fps)        
        scale_factor = 2
        if task_type == "SD24K":
            scale_factor = 4
        downscale_path = Path(config['default_video_dir']) / f"{video_id}_downscale-{scale_factor}x_FPS-{fps}_Frames-{clip_frames}.mp4"

        downscale_clip.write_videofile(str(downscale_path), codec='libx264', verbose=False, logger=None)

        elapsed_time = time.time() - start_time
        print(f"Time taken to save downscale version: {elapsed_time:.2f} seconds")

        downscale_clip.close()
            # Cleanup
        downscale.close()
        video_clip.close()
        clipped_clip.close()
        

        print(f"\nDone! Saved files:\n- {trim_path}\n- {downscale_path}")

        return str(downscale_path), video_id
    except requests.exceptions.RequestException as e:
        print(f"Error downloading video: {e}")
        return None, None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None

def run_download(task_type: str="SD24K", json_file: str="cache/pexels_SD24K.json"):
    video_ids = load_video_ids_from_json(json_file)

    for vid in video_ids['video_ids']:
        clip_duration_probabilities = {
            1: 0.6, 
            2: 0.05,
            3: 0.05,
            4: 0.05,
            5: 0.25 
        }

        # Generate a random number between 0 and 1 to determine the clip duration
        random_value = random.random()
        cumulative_probability = 0
        for clip_duration, probability in clip_duration_probabilities.items():
            cumulative_probability += probability
            if random_value <= cumulative_probability:
                break
        try:
            challenge_local_path, video_id = download_trim_downscale_video(
                clip_duration=clip_duration,
                vid=vid,
                task_type=task_type,
            )
            print(f"\nProcessing video ID: {vid}")
        except Exception as e:
            print(f"❌ Failed to process {vid}: {e}")
        
    print("---------end-----")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", help="Video Type", default="SD24K")
    parser.add_argument("--json", default="cache/pexels_SD24K.json", help="ID json file")
    args = parser.parse_args()
    run_download(args.task_type, args.json)

    