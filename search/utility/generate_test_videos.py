from pymongo import MongoClient
from search.modules.search_config import search_config
import asyncio
import os
import json
from moviepy.editor import VideoFileClip
import cv2
import random
import time
import subprocess
from tqdm import tqdm

async def main():
    # Initialize MongoDB client
    test_video_dir = search_config['TEST_VIDEO_DIR']
    video_dir = search_config['VIDEO_DIR']

    try:
        if not os.path.exists(test_video_dir):
            os.makedirs(test_video_dir)
            print(f"Created test video directory at {test_video_dir}")
            
        test_files = os.listdir(test_video_dir)
        print(f"Found {len(test_files)} files in test video directory")

        # Loop through all files in video_dir
        for filename in tqdm(os.listdir(video_dir), desc="Processing videos"):
            try:
                file_path = os.path.join(video_dir, filename)
                # Check if it's a file (not a directory)
                if not os.path.isfile(file_path):
                    continue
                
                filename_without_ext = os.path.splitext(filename)[0]
                if filename_without_ext.endswith('_original'):
                    filename_without_ext = filename_without_ext[:-9]  # Remove '_original' suffix
                # Check if there's already a test file for this video
                existing_test_files = [f for f in test_files if filename_without_ext in f]
                if existing_test_files:
                    print(f"Skipping {filename} - test files already exist: {existing_test_files}")
                    continue
                cap = cv2.VideoCapture(file_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                start_time = time.time()
                
                chunk_duration = random.choice([5, 10, 20])
                if chunk_duration == 5:
                    chunk_duration = random.choice([1, 2, 3, 4, 5])
                    
                start_time_clip = 0
                if chunk_duration < duration:
                    start_time_clip = random.uniform(0, duration - chunk_duration)
                start_frame = int(start_time_clip * fps)
                start_time_clip = start_frame / fps
                filename_without_ext = os.path.splitext(filename)[0]
                if filename_without_ext.endswith('_original'):
                    filename_without_ext = filename_without_ext[:-9]  # Remove '_original' suffix
                trimmed_path = os.path.join(test_video_dir, f"{filename_without_ext}_trimmed_{start_frame}_{chunk_duration}.mp4")
                downscale_path = os.path.join(test_video_dir, f"{filename_without_ext}_downscale_{start_frame}_{chunk_duration}.mp4")

                query_scale = 2
                if filename.startswith("SD24K"):
                    query_scale = 4
                downscale_height = height // query_scale
                
                trim_cmd = [
                    "taskset", "-c", "0,1,2,3,4,5",
                    "ffmpeg", "-y", "-i", str(file_path), "-ss", str(start_time_clip), 
                    "-t", str(chunk_duration), "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", str(trimmed_path), "-hide_banner", "-loglevel", "error"
                ]
                
                scale_cmd = [
                    "taskset", "-c", "0,1,2,3,4,5",
                    "ffmpeg", "-y", "-i", str(trimmed_path), "-vf", f"scale=-1:{downscale_height}", 
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", 
                    str(downscale_path), "-hide_banner", "-loglevel", "error"
                ]
            
            
                subprocess.run(trim_cmd, check=True)
                subprocess.run(scale_cmd, check=True)
                
                elapsed_time = time.time() - start_time
                print(f"Time taken to process {filename}: {elapsed_time:.2f} seconds")
                
            except subprocess.SubprocessError as e:
                print(f"Error processing {filename}: {e}")
                continue  # Skip to next file instead of exiting
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue  # Skip to next file instead of exiting

    except Exception as e:
        print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
