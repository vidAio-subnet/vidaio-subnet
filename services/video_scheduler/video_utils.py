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

    # Constants
    MAX_WORKERS = 5  # Use 5 CPU cores
    DOWNSCALE_HEIGHTS = {
        "HD24K": 1080,
        "4K28K": 2160,
    }
    EXPECTED_RESOLUTIONS = {
        "SD2HD": (1920, 1080),
        "4K28K": (7680, 4320),
    }

    # Create a lock for print statements to avoid messy output
    print_lock = threading.Lock()
    
    # Safe print function
    def safe_print(message):
        with print_lock:
            print(message)

    downscale_height = DOWNSCALE_HEIGHTS.get(task_type, 540)
    expected_width, expected_height = EXPECTED_RESOLUTIONS.get(task_type, (3840, 2160))

    # Create output directory if it doesn't exist
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
        
        # Calculate actual duration
        actual_duration = end_time_clip - start_time_clip
        
        # Use FFmpeg directly for trimming
        trim_cmd = [
            "ffmpeg", "-y", "-i", str(source_path), "-ss", str(start_time_clip), 
            "-t", str(actual_duration), "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", str(clipped_path), "-hide_banner", "-loglevel", "error"
        ]
        
        # Use FFmpeg directly for downscaling
        scale_cmd = [
            "ffmpeg", "-y", "-i", str(clipped_path), "-vf", f"scale=-1:{downscale_height}", 
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", 
            str(downscale_path), "-hide_banner", "-loglevel", "error"
        ]
        
        try:
            # Execute commands
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
        
        if clip_duration == 5:
            # Strategy for 5s clips: No overlap
            num_chunks = int(total_duration // clip_duration)
            for i in range(num_chunks):
                start_time = i * clip_duration
                end_time = start_time + clip_duration
                chunks.append((i, start_time, end_time))
                
        elif clip_duration == 10:
            # Strategy for 10s clips: Strategic overlapping
            # For videos longer than 25s, create overlapping chunks
            if total_duration <= 10:
                # Just one chunk if video is shorter than clip duration
                chunks.append((0, 0, min(total_duration, clip_duration)))
            elif total_duration <= 20:
                # Two chunks with minimal overlap for videos between 10-20s
                chunks.append((0, 0, clip_duration))
                chunks.append((1, total_duration - clip_duration, total_duration))
            else:
                # For longer videos, create strategically overlapped chunks
                # First chunk always starts at 0
                chunks.append((0, 0, clip_duration))
                
                # Middle chunks with strategic positioning
                position = 8  # Start next chunk at 8s (2s overlap with first)
                i = 1
                
                while position + clip_duration < total_duration - 2:
                    chunks.append((i, position, position + clip_duration))
                    position += 7  # Move forward by 7s (3s overlap)
                    i += 1
                
                # Last chunk always ends at total_duration
                if position < total_duration - 5:  # Only add if we have at least 5s of content
                    chunks.append((i, total_duration - clip_duration, total_duration))
                
        elif clip_duration == 20:
            # Strategy for 20s clips: Minimal overlapping for longer videos
            if total_duration <= 20:
                # Just one chunk if video is shorter than clip duration
                chunks.append((0, 0, min(total_duration, clip_duration)))
            elif total_duration <= 30:
                # Two chunks with overlap for videos between 20-30s
                chunks.append((0, 0, clip_duration))
                chunks.append((1, total_duration - clip_duration, total_duration))
            else:
                # For longer videos (>30s), create chunks with minimal overlap
                position = 0
                i = 0
                
                while position + clip_duration <= total_duration:
                    chunks.append((i, position, position + clip_duration))
                    position += 15  # Move forward by 15s (5s overlap)
                    i += 1
                    
                # Ensure the last chunk captures the end of the video
                if position < total_duration and position + 5 < total_duration:
                    chunks.append((i, total_duration - clip_duration, total_duration))
        
        return chunks

    def process_video_20s(vid, task_type, output_dir):
        """
        Process 20s clips with fully parallel approach:
        Each thread handles download, trim, and downscale for a separate video
        """
        try:
            # Get video info from API
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
                
            # Get video duration from API if available, otherwise estimate
            video_duration = data.get("duration", 30)  # Default to 30s if not provided
            
            # Generate chunk timestamps
            chunks = generate_chunk_timestamps(video_duration, 20)
            
            # Process each chunk in parallel (download, trim, downscale)
            results = []
            
            def process_single_chunk(chunk_info):
                i, start_time, end_time = chunk_info
                video_id = uuid.uuid4()
                
                # Create temporary directory for this chunk
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Download video to temp directory
                    temp_path = Path(temp_dir) / f"{vid}_original_{i}.mp4"
                    download_video(video_url, temp_path)
                    
                    # Process the chunk
                    result = process_chunk((i, start_time, end_time, video_id), temp_path)
                    
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    return result
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_chunk = {executor.submit(process_single_chunk, chunk): chunk for chunk in chunks}
                
                for future in concurrent.futures.as_completed(future_to_chunk):
                    result = future.result()
                    if result:
                        results.append(result)
            
            # Sort results by chunk index
            results.sort(key=lambda x: x[0])
            
            # Extract paths and IDs
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
            
            # 1. Download and verify video
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
            
            # Download the video
            download_video(video_url, temp_path)
            elapsed_time = time.time() - start_time
            safe_print(f"Time taken to download video: {elapsed_time:.2f} seconds")
        
            # 2. Verify video resolution and get duration
            safe_print("\nChecking video resolution...")
            width, height, total_duration = get_video_info(temp_path)
            
            if width != expected_width or height != expected_height:
                error_msg = (f"Video resolution mismatch. Expected {expected_width}x{expected_height}, "
                            f"but got {width}x{height}")
                safe_print(error_msg)
                raise ValueError(error_msg)
            
            safe_print(f"\nVideo resolution verified: {width}x{height}")
            safe_print(f"Video duration: {total_duration:.2f}s")
            
            # 3. Generate chunk timestamps based on clip_duration
            chunks_info = generate_chunk_timestamps(total_duration, clip_duration)
            safe_print(f"Extracting {len(chunks_info)} chunks of {clip_duration}s each")
            
            # Generate UUIDs for all chunks
            video_ids = [uuid.uuid4() for _ in range(len(chunks_info))]
            
            # Prepare chunk information with video IDs
            chunks_with_ids = [(i, start, end, video_ids[idx]) 
                              for idx, (i, start, end) in enumerate(chunks_info)]
            
            # 4. Process chunks in parallel
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_chunk = {executor.submit(process_chunk, chunk, temp_path): chunk 
                                  for chunk in chunks_with_ids}
                
                for future in concurrent.futures.as_completed(future_to_chunk):
                    result = future.result()
                    if result:
                        results.append(result)
            
            # Sort results by chunk index
            results.sort(key=lambda x: x[0])
            
            # Extract paths and IDs
            downscale_paths = [result[1] for result in results]
            final_video_ids = [result[2] for result in results]
            
            # 5. Clean up
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

    # Main processing logic
    try:
        # Choose processing strategy based on clip_duration
        if clip_duration == 20:
            # For 20s clips, use fully parallel approach
            return process_video_20s(vid, task_type, output_dir)
        else:
            # For 5s and 10s clips, use standard approach
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
