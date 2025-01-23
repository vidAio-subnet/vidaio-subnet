import requests
import os
import random
import time
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv

load_dotenv()

def download_and_trim_video(api_key, clip_duration=10, min_video_len=0, max_video_len=600, output_dir="4k_videos"):
    """
    Download a 4K video and trim to specified duration from random position
    api_key: Pexels API key
    clip_duration: desired clip duration in seconds
    min_video_len: minimum video length in seconds
    max_video_len: maximum video length in seconds
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    headers = {"Authorization": api_key}
    url = "https://api.pexels.com/videos/search"
    params = {
        "query": "nature",
        "per_page": 80,
        "size": "large"
    }
    
    try:
        # Step 1: Get video list
        start_time = time.time()
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        elapsed_time = time.time() - start_time
        print(f"Time taken to fetch videos: {elapsed_time:.2f} seconds")
        
        if "videos" not in data:
            print("No videos found or API error")
            return
        
        # Step 2: Filter 4K videos based on length
        k4_videos = []
        for video in data["videos"]:
            video_length = video["duration"]
            k4_files = [f for f in video["video_files"] if f["height"] >= 2160]
            if k4_files and min_video_len <= video_length <= max_video_len:
                k4_videos.append((video, k4_files[0]))
        
        if not k4_videos:
            print("No suitable 4K videos found")
            return
        
        # Step 3: Randomly select one video
        video, video_file = random.choice(k4_videos)
        video_url = video_file["link"]
        
        temp_path = f"{output_dir}/temp_{video['id']}_4k.mp4"  # Downloaded 4K file
        clipped_path = f"{output_dir}/{video['id']}_{clip_duration}sec_clipped.mp4"  # Clipped 4K file
        hd_path = f"{output_dir}/{video['id']}_{clip_duration}sec_hd.mp4"  # Downscaled HD file
        hevc_path = f"{output_dir}/{video['id']}_{clip_duration}sec_hd.hevc"  # HEVC file
        
        # Step 4: Download video
        print(f"\nDownloading 4K video...")
        print(f"Resolution: {video_file['width']}x{video_file['height']}")
        print(f"Original duration: {video['duration']}s")
        
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
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        
        elapsed_time = time.time() - start_time
        print(f"Time taken to download video: {elapsed_time:.2f} seconds")
        
        # Step 5: Trim video
        print("\nTrimming video...")
        start_time = time.time()
        video_clip = VideoFileClip(temp_path)
        
        if video_clip.duration <= clip_duration:
            print(f"Video is shorter than {clip_duration}s, keeping full length")
            start_time_clip = 0
        else:
            max_start = video_clip.duration - clip_duration
            start_time_clip = random.uniform(0, max_start)
            print(f"Trimming {clip_duration}s from position {start_time_clip:.1f}s")
        
        # Create clipped video
        clipped_clip = video_clip.subclip(start_time_clip, start_time_clip + min(clip_duration, video_clip.duration))
        clipped_clip.write_videofile(clipped_path, codec='libx264')
        
        elapsed_time = time.time() - start_time
        print(f"Time taken to clip video: {elapsed_time:.2f} seconds")
        
        # Step 6: Downscale to HD and save as .mp4
        print("\nSaving HD version...")
        start_time = time.time()
        hd_clip = video_clip.resize(height=720)  # Resize to HD (720p)
        hd_clip.write_videofile(hd_path, codec='libx264')  # Save as HD .mp4
        
        elapsed_time = time.time() - start_time
        print(f"Time taken to save HD version: {elapsed_time:.2f} seconds")
        
        # Step 7: Convert to HEVC
        print("\nConverting to HEVC...")
        start_time = time.time()
        hd_clip.write_videofile(hevc_path, codec='libx265')  # Convert to HEVC
        
        elapsed_time = time.time() - start_time
        print(f"Time taken to convert to HEVC: {elapsed_time:.2f} seconds")
        
        # Cleanup
        video_clip.close()
        clipped_clip.close()
        hd_clip.close()
        
        print(f"\nDone! Saved to: {temp_path}, {clipped_path}, {hd_path}, and {hevc_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
    print(PEXELS_API_KEY)
    CLIP_DURATION = 5
    MIN_VIDEO_LEN = 23  # Set minimum video length in seconds
    MAX_VIDEO_LEN = 24  # Set maximum video length in seconds
    
    download_and_trim_video(PEXELS_API_KEY, CLIP_DURATION, MIN_VIDEO_LEN, MAX_VIDEO_LEN)
