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
    Downloads a specified video with video id from Pexels, extracts multiple chunks 
    of the specified duration with adaptive overlapping strategies, and downscales 
    each chunk to required resolution using multi-threading.

    Args:
        clip_duration (int): Desired clip duration in seconds (5, 10, or 20).
        vid (int): Pexels video ID to download.
        task_type (str): Type of task determining resolution requirements.
        output_dir (str): Directory to save the processed videos.

    Returns:
        Optional[Tuple[List[str], List[int]]]: Lists of paths to the downscaled videos and their generated IDs, 
        or None on failure.
    """
    import os
    import time
    import uuid
    import requests
    from pathlib import Path
    from typing import List, Tuple, Optional, Dict, Any
    from tqdm import tqdm
    import concurrent.futures
    import subprocess
    import tempfile
    import threading
    from functools import partial

    MAX_WORKERS = 5  
    DOWNSCALE_HEIGHTS = {
        "HD24K": 1080,
        "4K28K": 2160,
    }
    EXPECTED_RESOLUTIONS = {
        "SD2HD": (1920, 1080),
        "4K28K": (7680, 4320),
    }

    print_lock = threading.Lock()
    
    def safe_print(message):
        with print_lock:
            print(message)

    downscale_height = DOWNSCALE_HEIGHTS.get(task_type, 540)
    expected_width, expected_height = EXPECTED_RESOLUTIONS.get(task_type, (3840, 2160))

    os.makedirs(output_dir, exist_ok=True)
    
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        raise ValueError("PEXELS_API_KEY not found in environment variables")

    def download_video(video_url, output_path):
        """Download video from URL with progress bar"""
        safe_print(f"\nDownloading video from Pexels...")
        response = requests.get(video_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                size = f.write(chunk)
                pbar.update(size)
                
        safe_print(f"Video downloaded successfully to: {output_path}")
        return output_path

    def get_video_info(video_path):
        """Get video resolution and duration using FFprobe"""
        probe_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0", 
            "-show_entries", "stream=width,height,duration", "-of", "csv=p=0", str(video_path)
        ]
        
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        width, height, duration = map(float, result.stdout.strip().split(','))
        return int(width), int(height), duration

    def process_chunk(chunk_info, source_path):
        """Process a single video chunk using FFmpeg directly"""
        i, start_time_clip, end_time_clip, video_id = chunk_info
        
        clipped_path = Path(output_dir) / f"{video_id}_trim.mp4"
        downscale_path = Path(output_dir) / f"{video_id}_downscale.mp4"
        
        chunk_start_time = time.time()
        safe_print(f"Processing chunk {i+1}: {start_time_clip:.1f}s to {end_time_clip:.1f}s")
        
        actual_duration = end_time_clip - start_time_clip
        
        trim_cmd = [
            "ffmpeg", "-y", "-i", str(source_path), "-ss", str(start_time_clip), 
            "-t", str(actual_duration), "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", str(clipped_path), "-hide_banner", "-loglevel", "error"
        ]
        
        scale_cmd = [
            "ffmpeg", "-y", "-i", str(clipped_path), "-vf", f"scale=-1:{downscale_height}", 
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", 
            str(downscale_path), "-hide_banner", "-loglevel", "error"
        ]
        
        try:
            subprocess.run(trim_cmd, check=True)
            subprocess.run(scale_cmd, check=True)
            
            chunk_elapsed_time = time.time() - chunk_start_time
            safe_print(f"Time taken to process chunk {i+1}: {chunk_elapsed_time:.2f} seconds")
            
            return (i, str(downscale_path), video_id)
        except subprocess.SubprocessError as e:
            safe_print(f"Error processing chunk {i+1}: {e}")
            return None

    def generate_chunk_timestamps(total_duration, clip_duration):
        """
        Generate chunk timestamps based on clip_duration with different strategies:
        - 5s: Non-overlapping chunks
        - 10s: Chunks with strategic overlapping (no more than 3s overlap)
        - 20s: Chunks with minimal overlapping for longer videos
        """
        chunks = []
        
        if clip_duration <= 5:
            num_chunks = int(total_duration // clip_duration)
            for i in range(num_chunks):
                start_time = i * clip_duration
                end_time = start_time + clip_duration
                chunks.append((i, start_time, end_time))
                
        elif clip_duration == 10:
            if total_duration <= 10:
                chunks.append((0, 0, min(total_duration, clip_duration)))
            elif total_duration <= 20:
                chunks.append((0, 0, clip_duration))
                chunks.append((1, total_duration - clip_duration, total_duration))
            else:
                chunks.append((0, 0, clip_duration))
                
                position = 8  
                i = 1
                
                while position + clip_duration < total_duration - 2:
                    chunks.append((i, position, position + clip_duration))
                    position += 7 
                    i += 1
                
                if position < total_duration - 5: 
                    chunks.append((i, total_duration - clip_duration, total_duration))
                
        elif clip_duration == 20:
            if total_duration <= 20:
                chunks.append((0, 0, min(total_duration, clip_duration)))
            elif total_duration <= 30:
                chunks.append((0, 0, clip_duration))
                chunks.append((1, total_duration - clip_duration, total_duration))
            else:
                position = 0
                i = 0
                
                while position + clip_duration <= total_duration:
                    chunks.append((i, position, position + clip_duration))
                    position += 15  
                    i += 1
                    
                if position < total_duration and position + 5 < total_duration:
                    chunks.append((i, total_duration - clip_duration, total_duration))
        
        return chunks

    def process_video_20s(vid, task_type, output_dir):
        """
        Process 20s clips with fully parallel approach:
        Each thread handles download, trim, and downscale for a separate video
        """
        try:
            url = f"https://api.pexels.com/videos/videos/{vid}"
            headers = {"Authorization": api_key}
            
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
                safe_print("No video found with required resolution")
                return None, None
                
            video_duration = data.get("duration", 30)  
            
            chunks = generate_chunk_timestamps(video_duration, 20)
            
            results = []
            
            def process_single_chunk(chunk_info):
                i, start_time, end_time = chunk_info
                video_id = uuid.uuid4()
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir) / f"{vid}_original_{i}.mp4"
                    download_video(video_url, temp_path)
                    
                    result = process_chunk((i, start_time, end_time, video_id), temp_path)
                    
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    return result
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_chunk = {executor.submit(process_single_chunk, chunk): chunk for chunk in chunks}
                
                for future in concurrent.futures.as_completed(future_to_chunk):
                    result = future.result()
                    if result:
                        results.append(result)
            
            results.sort(key=lambda x: x[0])
            
            downscale_paths = [result[1] for result in results]
            final_video_ids = [result[2] for result in results]
            
            safe_print(f"\nDone! Successfully processed {len(results)} out of {len(chunks)} chunks.")
            
            return downscale_paths, final_video_ids
            
        except Exception as e:
            safe_print(f"Error in 20s processing: {str(e)}")
            return None, None

    def process_video_standard(vid, clip_duration, task_type, output_dir):
        """
        Process 5s or 10s clips with standard approach:
        1. Download video once
        2. Process chunks in parallel
        """
        try:
            start_time = time.time()
            
            url = f"https://api.pexels.com/videos/videos/{vid}"
            headers = {"Authorization": api_key}
            
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
            
            download_video(video_url, temp_path)
            elapsed_time = time.time() - start_time
            safe_print(f"Time taken to download video: {elapsed_time:.2f} seconds")
        
            safe_print("\nChecking video resolution...")
            width, height, total_duration = get_video_info(temp_path)
            
            if width != expected_width or height != expected_height:
                error_msg = (f"Video resolution mismatch. Expected {expected_width}x{expected_height}, "
                            f"but got {width}x{height}")
                safe_print(error_msg)
                raise ValueError(error_msg)
            
            safe_print(f"\nVideo resolution verified: {width}x{height}")
            safe_print(f"Video duration: {total_duration:.2f}s")
            
            chunks_info = generate_chunk_timestamps(total_duration, clip_duration)
            safe_print(f"Extracting {len(chunks_info)} chunks of {clip_duration}s each")
            
            video_ids = [uuid.uuid4() for _ in range(len(chunks_info))]
            
            chunks_with_ids = [(i, start, end, video_ids[idx]) 
                              for idx, (i, start, end) in enumerate(chunks_info)]
            
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_chunk = {executor.submit(process_chunk, chunk, temp_path): chunk 
                                  for chunk in chunks_with_ids}
                
                for future in concurrent.futures.as_completed(future_to_chunk):
                    result = future.result()
                    if result:
                        results.append(result)
            
            results.sort(key=lambda x: x[0])
            
            downscale_paths = [result[1] for result in results]
            final_video_ids = [result[2] for result in results]
            
            safe_print("\nCleaning up original downloaded file...")
            os.remove(temp_path)
            safe_print(f"Deleted: {temp_path}")

            safe_print(f"\nDone! Successfully processed {len(results)} out of {len(chunks_info)} chunks.")
            
            return downscale_paths, final_video_ids
            
        except requests.exceptions.RequestException as e:
            safe_print(f"Error downloading video: {e}")
            return None, None
        except Exception as e:
            safe_print(f"Error: {str(e)}")
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    safe_print(f"Cleaned up downloaded file after error: {temp_path}")
                except:
                    pass
            return None, None

    try:
        if clip_duration == 20:
            return process_video_20s(vid, task_type, output_dir)
        else:
            return process_video_standard(vid, clip_duration, task_type, output_dir)
            
    except Exception as e:
        safe_print(f"Unexpected error: {str(e)}")
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
