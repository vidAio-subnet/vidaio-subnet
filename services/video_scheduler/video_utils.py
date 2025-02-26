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
    output_dir: str = "videos"
) -> Optional[Tuple[str, int]]:
    """
    Downloads a specified DCI 4K video from Pexels, trims it to the specified duration,
    and downscales it to HD resolution.

    Args:
        clip_duration (int): Desired clip duration in seconds.
        vid (int): Pexels video ID to download.
        output_dir (str): Directory to save the processed videos.

    Returns:
        Optional[Tuple[str, int]]: Path to the HD video and the generated video ID, or None on failure.
    """

    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        print("PEXELS_API_KEY is missing in the environment variables.")
        return None

    os.makedirs(output_dir, exist_ok=True)

    headers = {"Authorization": api_key}
    url = f"https://api.pexels.com/videos/videos/{vid}"

    try:
        # Fetch video details
        start_time = time.time()
        response = requests.get(url, headers=headers)
        elapsed_time = time.time() - start_time
        print(f"Time taken to fetch video details: {elapsed_time:.2f} seconds")

        data = response.json()
        if "video_files" not in data:
            print("No video found or API error")
            return None

        # Find DCI 4K version of the video
        video_file = next(
            (vf for vf in data["video_files"] if vf["width"] == 4096 and vf["height"] == 2160),
            None
        )

        if not video_file:
            print("No DCI 4K version available for this video.")
            return None

        video_url = video_file["link"]
        video_id = uuid.uuid4()

        temp_path = Path(output_dir) / f"{video_id}_4k_original.mp4"
        clipped_path = Path(output_dir) / f"{video_id}_4k.mp4"
        hd_path = Path(output_dir) / f"{video_id}_hd.mp4"

        # Download video
        print(f"\nDownloading DCI 4K video (Resolution: {video_file['width']}x{video_file['height']})")
        
        start_time = time.time()
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

        elapsed_time = time.time() - start_time
        print(f"Time taken to download video: {elapsed_time:.2f} seconds")

        # Trim video
        print("\nTrimming video...")
        start_time = time.time()
        video_clip = VideoFileClip(str(temp_path))

        start_time_clip = 0 if video_clip.duration <= clip_duration else random.uniform(0, video_clip.duration - clip_duration)
        print(f"Trimming {clip_duration}s from position {start_time_clip:.1f}s")

        clipped_clip = video_clip.subclip(start_time_clip, start_time_clip + min(clip_duration, video_clip.duration))
        clipped_clip.write_videofile(str(clipped_path), codec='libx264', verbose=False, logger=None)

        elapsed_time = time.time() - start_time
        print(f"Time taken to clip video: {elapsed_time:.2f} seconds")

        # Downscale to HD
        print("\nSaving HD version...")
        start_time = time.time()
        hd_clip = clipped_clip.resize(height=1080)
        hd_clip.write_videofile(str(hd_path), codec='libx264', verbose=False, logger=None)

        elapsed_time = time.time() - start_time
        print(f"Time taken to save HD version: {elapsed_time:.2f} seconds")

        # Cleanup
        video_clip.close()
        clipped_clip.close()
        hd_clip.close()

        print(f"\nDone! Saved files:\n- {clipped_path}\n- {hd_path}")

        return str(hd_path), video_id

    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def get_4k_video_path(file_id: int, dir_path: str = "videos") -> str:
    """Returns the path of the clipped 4K video based on the file ID."""
    return str(Path(dir_path) / f"{file_id}_4k.mp4")


def delete_videos_with_fileid(file_id: int, dir_path: str = "videos") -> None:
    """Deletes all video files associated with the given file ID."""

    files_to_delete = [
        Path(dir_path) / f"{file_id}_4k_original.mp4",
        Path(dir_path) / f"{file_id}_4k.mp4",
        Path(dir_path) / f"{file_id}_hd.mp4",
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
