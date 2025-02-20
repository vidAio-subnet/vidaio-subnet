# import requests
# import os
# import random
# import time
# from tqdm import tqdm
# from moviepy.editor import VideoFileClip
# from dotenv import load_dotenv

# load_dotenv()

# def download_trim_downscale_video(clip_duration=1, min_video_len=10, max_video_len=20, output_dir="videos"):
#     """
#     Download a DCI 4K video and trim to specified duration from a random position.
#     clip_duration: desired clip duration in seconds
#     min_video_len: minimum video length in seconds
#     max_video_len: maximum video length in seconds
#     """
    
#     api_key = os.getenv("PEXELS_API_KEY")
    
#     os.makedirs(output_dir, exist_ok=True)
    
#     headers = {"Authorization": api_key}
#     url = "https://api.pexels.com/videos/search"
#     params = {
#         "query": "nature",
#         "per_page": 80,
#         "size": "large"
#     }
    
#     try:
#         # Step 1: Get video list
#         start_time = time.time()
#         response = requests.get(url, headers=headers, params=params)
#         data = response.json()
#         elapsed_time = time.time() - start_time
#         print(f"Time taken to fetch videos: {elapsed_time:.2f} seconds")
        
#         if "videos" not in data:
#             print("No videos found or API error")
#             return
        
#         # Step 2: Filter DCI 4K videos based on length
#         dci4k_videos = []
#         for video in data["videos"]:
#             video_length = video["duration"]
#             # Filter for DCI 4K videos (4096x2160)
#             dci4k_files = [f for f in video["video_files"] if f["width"] == 4096 and f["height"] == 2160]
#             if dci4k_files and min_video_len <= video_length <= max_video_len:
#                 dci4k_videos.append((video, dci4k_files[0]))
        
#         if not dci4k_videos:
#             print("No suitable DCI 4K videos found")
#             return
        
#         # Step 3: Randomly select one video
#         video, video_file = random.choice(dci4k_videos)
#         video_url = video_file["link"]
        
#         temp_path = f"{output_dir}/{video['id']}_4k_original.mp4"  # Downloaded DCI 4K file
#         clipped_path = f"{output_dir}/{video['id']}_4k.mp4"  # Clipped DCI 4K file
#         hd_path = f"{output_dir}/{video['id']}_hd.mp4"  # Downscaled HD file
#         hevc_path = f"{output_dir}/{video['id']}_hd.hevc"  # HEVC file
#         video_id = video['id']
        
#         # Step 4: Download video
#         print(f"\nDownloading DCI 4K video...")
#         print(f"Resolution: {video_file['width']}x{video_file['height']}")
#         print(f"Original duration: {video['duration']}s")
        
#         start_time = time.time()
#         response = requests.get(video_url, stream=True)
#         total_size = int(response.headers.get('content-length', 0))
        
#         with open(temp_path, 'wb') as f, tqdm(
#             desc="Downloading",
#             total=total_size,
#             unit='iB',
#             unit_scale=True,
#             unit_divisor=1024,
#         ) as pbar:
#             for data in response.iter_content(chunk_size=1024):
#                 size = f.write(data)
#                 pbar.update(size)
        
#         elapsed_time = time.time() - start_time
#         print(f"Time taken to download video: {elapsed_time:.2f} seconds")
        
#         # Step 5: Trim video
#         print("\nTrimming video...")
#         start_time = time.time()
#         video_clip = VideoFileClip(temp_path)
        
#         if video_clip.duration <= clip_duration:
#             print(f"Video is shorter than {clip_duration}s, keeping full length")
#             start_time_clip = 0
#         else:
#             max_start = video_clip.duration - clip_duration
#             start_time_clip = random.uniform(0, max_start)
#             print(f"Trimming {clip_duration}s from position {start_time_clip:.1f}s")
        
#         # Create clipped video
#         clipped_clip = video_clip.subclip(start_time_clip, start_time_clip + min(clip_duration, video_clip.duration))
#         clipped_clip.write_videofile(clipped_path, codec='libx264')
        
#         elapsed_time = time.time() - start_time
#         print(f"Time taken to clip video: {elapsed_time:.2f} seconds")
        
#         # Step 6: Downscale to HD and save as .mp4 using the clipped video
#         print("\nSaving HD version...")
#         start_time = time.time()
#         hd_clip = clipped_clip.resize(height=1080)  # Resize to HD (1080p) using the clipped video
#         hd_clip.write_videofile(hd_path, codec='libx264')  # Save as HD .mp4
        
#         elapsed_time = time.time() - start_time
#         print(f"Time taken to save HD version: {elapsed_time:.2f} seconds")
        
#         # Step 7: Convert to HEVC
#         # print("\nConverting to HEVC...")
#         # start_time = time.time()
#         # hd_clip.write_videofile(hevc_path, codec='libx265')  # Convert to HEVC
        
#         elapsed_time = time.time() - start_time
#         print(f"Time taken to convert to HEVC: {elapsed_time:.2f} seconds")
        
#         # Cleanup
#         video_clip.close()
#         clipped_clip.close()
#         hd_clip.close()
        
#         print(f"\nDone! Saved to: {temp_path}, {clipped_path}, {hd_path}, and {hevc_path}")
        
#         # return hevc_path, video_id
#         return hd_path, video_id
        
#     except Exception as e:
#         print(f"Error: {str(e)}")



# def get_4k_vide_path(file_id, dir_path="videos"):
    
#     clipped_path = f"{dir_path}/{file_id}_4k.mp4"
    
#     return clipped_path


# def delete_videos_with_fileid(file_id, dir_path="videos"):
#     # Define file paths
#     temp_path = f"{dir_path}/{file_id}_4k_original.mp4"  # Downloaded 4K file
#     clipped_path = f"{dir_path}/{file_id}_4k.mp4"  # Clipped 4K file
#     hd_path = f"{dir_path}/{file_id}_hd.mp4"  # Downscaled HD file
#     # hevc_path = f"{dir_path}/{file_id}_hd.hevc"  # HEVC file

#     # List of files to delete
#     files_to_delete = [temp_path, clipped_path, hd_path]

#     # Delete each file if it exists
#     for file_path in files_to_delete:
#         try:
#             if os.path.exists(file_path):
#                 os.remove(file_path)
#                 print(f"Deleted: {file_path}")
#             else:
#                 print(f"File not found: {file_path}")
#         except Exception as e:
#             print(f"Error deleting {file_path}: {e}")


# if __name__ == "__main__":
#     PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
#     print(PEXELS_API_KEY)
#     CLIP_DURATION = 2
#     MIN_VIDEO_LEN = 10  # Set minimum video length in seconds
#     MAX_VIDEO_LEN = 20  # Set maximum video length in seconds
    
#     download_trim_downscale_video(CLIP_DURATION, MIN_VIDEO_LEN, MAX_VIDEO_LEN)




import requests
import os
import random
import time
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional, Tuple

# Load environment variables
load_dotenv()

def download_trim_downscale_video(
    clip_duration: int = 1,
    min_video_len: int = 10,
    max_video_len: int = 20,
    output_dir: str = "videos"
) -> Optional[Tuple[str, int]]:
    """
    Downloads a DCI 4K video from Pexels, trims it to the specified duration, 
    and downscales it to HD resolution.

    Args:
        clip_duration (int): Desired clip duration in seconds.
        min_video_len (int): Minimum video length in seconds.
        max_video_len (int): Maximum video length in seconds.
        output_dir (str): Directory to save the processed videos.

    Returns:
        Optional[Tuple[str, int]]: Path to the HD video and the video ID, or None on failure.
    """

    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        print("PEXELS_API_KEY is missing in the environment variables.")
        return None

    os.makedirs(output_dir, exist_ok=True)

    headers = {"Authorization": api_key}
    url = "https://api.pexels.com/videos/search"
    params = {
        "query": "nature",
        "per_page": 80,
        "size": "large"
    }

    try:
        # Fetch video list
        start_time = time.time()
        response = requests.get(url, headers=headers, params=params)
        elapsed_time = time.time() - start_time
        print(f"Time taken to fetch videos: {elapsed_time:.2f} seconds")

        data = response.json()
        if "videos" not in data:
            print("No videos found or API error")
            return None

        # Filter DCI 4K videos based on length
        dci4k_videos = [
            (video, video_file)
            for video in data["videos"]
            for video_file in video["video_files"]
            if video_file["width"] == 4096 and video_file["height"] == 2160 and min_video_len <= video["duration"] <= max_video_len
        ]

        if not dci4k_videos:
            print("No suitable DCI 4K videos found")
            return None

        # Select a random video
        video, video_file = random.choice(dci4k_videos)
        video_url = video_file["link"]
        video_id = video['id']

        temp_path = Path(output_dir) / f"{video_id}_4k_original.mp4"
        clipped_path = Path(output_dir) / f"{video_id}_4k.mp4"
        hd_path = Path(output_dir) / f"{video_id}_hd.mp4"

        # Download video
        print(f"\nDownloading DCI 4K video (Resolution: {video_file['width']}x{video_file['height']}, Duration: {video['duration']}s)")
        
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
    print(f"PEXELS_API_KEY: {os.getenv('PEXELS_API_KEY')}")
    CLIP_DURATION = 2
    MIN_VIDEO_LEN = 10
    MAX_VIDEO_LEN = 20

    download_trim_downscale_video(CLIP_DURATION, MIN_VIDEO_LEN, MAX_VIDEO_LEN)
