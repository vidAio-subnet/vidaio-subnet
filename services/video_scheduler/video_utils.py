import requests
import os
import time
from tqdm import tqdm
from pathlib import Path
import uuid
from typing import List, Tuple, Optional
import concurrent.futures
import subprocess
import tempfile
import threading
from dotenv import load_dotenv
import random

load_dotenv()

MAX_WORKERS = 5


print_lock = threading.Lock()

def safe_print(message):
    with print_lock:
        print(message)


def apply_color_space_transformation(video_path: str, output_path: str = None) -> str:
    """
    Apply color space transformation to make videos less identifiable while maintaining quality.
    
    This function applies various color transformations to reduce the recognizability of 
    Pexels videos while preserving visual quality for training purposes.
    
    Args:
        video_path (str): Path to the input video file
        output_path (str, optional): Path for the output video. If None, creates a new path.
    
    Returns:
        str: Path to the transformed video file
    """
    if not os.path.exists(video_path):
        print(f"Input video file does not exist: {video_path}")
        return video_path
    
    if output_path is None:
        # Create output path by adding "_transformed" before the extension
        video_path_obj = Path(video_path)
        output_path = str(video_path_obj.parent / f"{video_path_obj.stem}_transformed{video_path_obj.suffix}")
    
    # Define various color transformation options
    transformations = [
        # Slight hue shift
        "hue=h=15:s=1.1",
        "hue=h=-10:s=1.05", 
        "hue=h=25:s=0.95",
        
        # Color balance adjustments
        "colorbalance=rs=0.1:gs=-0.05:bs=0.05:rm=0.05:gm=0.1:bm=-0.05",
        "colorbalance=rs=-0.08:gs=0.1:bs=-0.02:rm=-0.05:gm=0.05:bm=0.1",
        
        # Curves adjustments (subtle gamma and contrast changes)
        "curves=r='0/0 0.5/0.48 1/1':g='0/0 0.5/0.52 1/1':b='0/0 0.5/0.50 1/1'",
        "curves=r='0/0 0.5/0.52 1/1':g='0/0 0.5/0.50 1/1':b='0/0 0.5/0.48 1/1'",
        
        # Exposure and vibrance
        "eq=brightness=0.02:contrast=1.05:saturation=1.1",
        "eq=brightness=-0.015:contrast=1.03:saturation=0.95",
        "eq=brightness=0.01:contrast=0.98:saturation=1.08",
        
        # Temperature shifts (subtle)
        "colortemperature=temperature=5800:mix=0.3",
        "colortemperature=temperature=6500:mix=0.25",
        "colortemperature=temperature=5200:mix=0.35",
        
        # Combination transformations for more variation
        "hue=h=12:s=1.02,eq=brightness=0.01:contrast=1.02:saturation=1.05",
        "colorbalance=rs=0.05:bs=-0.03:gm=0.08,eq=brightness=-0.01:contrast=1.01",
        "curves=r='0/0 0.5/0.51 1/1':g='0/0 0.5/0.49 1/1',hue=h=-8:s=1.03",
    ]
    
    # Randomly select a transformation
    selected_transformation = random.choice(transformations)
    
    # Build FFmpeg command for color space transformation
    transform_cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", selected_transformation,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",  # Faster preset, slightly higher CRF for speed
        "-c:a", "aac", "-b:a", "128k",
        str(output_path),
        "-hide_banner", "-loglevel", "error"
    ]
    
    try:
        print(f"Applying color transformation: {selected_transformation}")
        start_time = time.time()
        
        result = subprocess.run(transform_cmd, check=True, capture_output=True, text=True)
        
        elapsed_time = time.time() - start_time
        print(f"Color transformation completed in {elapsed_time:.2f}s: {video_path} -> {output_path}")
        
        # Verify the output file was created and is not empty
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            print(f"Warning: Output file is empty or doesn't exist: {output_path}")
            return video_path
        
        # Remove original file to save space only if transformation was successful
        if os.path.exists(video_path) and os.path.exists(output_path):
            try:
                os.remove(video_path)
                print(f"Removed original file: {video_path}")
            except OSError as e:
                print(f"Warning: Could not remove original file {video_path}: {e}")
            
        return output_path
        
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error applying color transformation: {e}")
        if e.stderr:
            print(f"FFmpeg stderr: {e.stderr}")
        # If transformation fails, return the original path
        return video_path
    except Exception as e:
        print(f"Unexpected error applying color transformation: {e}")
        return video_path


def generate_chunk_timestamps(total_duration, clip_duration):
    """
    Generate chunk timestamps using sliding window approach to maximize chunks per video.
    This reduces API requests by extracting many overlapping chunks from each downloaded video.

    Strategy:
    - 5s clips: Slide by 1s intervals (1-6s, 2-7s, 3-8s, etc.)
    - 10s clips: Slide by 2s intervals (0-10s, 2-12s, 4-14s, etc.)
    - 20s clips: Slide by 3s intervals (0-20s, 3-23s, 6-26s, etc.)
    """
    chunks = []

    # Define slide intervals for each clip duration
    slide_intervals = {
        1: 0.5,  # For 1s clips, slide by 0.5s
        2: 0.5,  # For 2s clips, slide by 0.5s
        3: 1,  # For 3s clips, slide by 1s
        4: 1,  # For 4s clips, slide by 1s
        5: 1,  # For 5s clips, slide by 1s
        10: 2,  # For 10s clips, slide by 2s
        20: 3  # For 20s clips, slide by 3s
    }

    slide_interval = slide_intervals.get(clip_duration, 1)

    # Calculate maximum number of chunks we can extract
    max_chunks_possible = int((total_duration - clip_duration) / slide_interval)

    # Set reasonable limits to avoid over-extraction from one video
    max_chunks_per_video = {
        1: max(20, max_chunks_possible),  # Max 20 chunks for 1s clips
        2: max(15, max_chunks_possible),  # Max 15 chunks for 2s clips
        3: max(12, max_chunks_possible),  # Max 12 chunks for 3s clips
        4: max(10, max_chunks_possible),  # Max 10 chunks for 4s clips
        5: max(8, max_chunks_possible),  # Max 8 chunks for 5s clips
        10: max(6, max_chunks_possible),  # Max 6 chunks for 10s clips
        20: max(4, max_chunks_possible)  # Max 4 chunks for 20s clips
    }

    max_chunks = max_chunks_per_video.get(clip_duration, max_chunks_possible)

    # Ensure we have enough video duration
    if max_chunks_possible <= 0:
        return [(0, 0, min(total_duration, clip_duration))]

    # Generate sliding window chunks
    current_start = random.uniform(0, slide_interval)
    chunk_index = 0

    while (current_start + clip_duration <= total_duration and
           chunk_index < max_chunks):
        end_time = current_start + clip_duration
        chunks.append((chunk_index, current_start, end_time))

        current_start += slide_interval
        chunk_index += 1

    # Add one final chunk from the end if we have remaining duration
    if (chunk_index < max_chunks and
            total_duration > clip_duration and
            current_start < total_duration - 2):  # At least 2s remaining

        final_start = total_duration - clip_duration
        # Only add if it doesn't overlap too much with the last chunk
        if not chunks or final_start - chunks[-1][1] >= slide_interval:
            chunks.append((chunk_index, final_start, total_duration))

    safe_print(f"Generated {len(chunks)} chunks for {clip_duration}s duration from {total_duration:.1f}s video")
    safe_print(f"Chunk extraction details: slide_interval={slide_interval}s, max_chunks={max_chunks}")

    return chunks


def generate_youtube_chunk_timestamps(total_duration, clip_duration):
    """
    Generate chunk timestamps using sliding window approach.
    Similar to Pexels but adapted for potentially longer YouTube videos.
    """
    chunks = []

    # Define slide intervals for each clip duration
    slide_intervals = {
        5: 2,     # For 5s clips, slide by 2s
        10: 4,    # For 10s clips, slide by 4s
        20: 6     # For 20s clips, slide by 6s
    }

    slide_interval = slide_intervals.get(clip_duration, 2)

    # Calculate maximum number of chunks we can extract
    max_chunks_possible = int((total_duration - clip_duration) / slide_interval)

    # Set reasonable limits for YouTube videos (can be longer)
    max_chunks_per_video = {
        5: max(30, max_chunks_possible),   # Max 30 chunks for 5s clips
        10: max(20, max_chunks_possible),  # Max 20 chunks for 10s clips
        20: max(15, max_chunks_possible)   # Max 15 chunks for 20s clips
    }

    max_chunks = max_chunks_per_video.get(clip_duration, max_chunks_possible)

    # Ensure we have enough video duration
    if max_chunks_possible <= 0:
        return [(0, 0, min(total_duration, clip_duration))]

    # Generate sliding window chunks
    current_start = random.uniform(0, slide_interval)
    chunk_index = 0

    while (current_start + clip_duration <= total_duration and
           chunk_index < max_chunks):

        end_time = current_start + clip_duration
        chunks.append((chunk_index, current_start, end_time))

        current_start += slide_interval
        chunk_index += 1

    safe_print(f"Generated {len(chunks)} chunks for {clip_duration}s duration from {total_duration:.1f}s YouTube video")

    return chunks


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

    DOWNSCALE_HEIGHTS = {
        "HD24K": 1080,
        "4K28K": 2160,
    }
    EXPECTED_RESOLUTIONS = {
        "SD2HD": (1920, 1080),
        "4K28K": (7680, 4320),
    }

    downscale_height = DOWNSCALE_HEIGHTS.get(task_type, 540)
    expected_width, expected_height = EXPECTED_RESOLUTIONS.get(task_type, (3840, 2160))

    os.makedirs(output_dir, exist_ok=True)
    load_dotenv()
    
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
            "taskset", "-c", "0,1,2,3,4,5,6,7,8,9,10,11",
            "ffmpeg", "-y", "-i", str(source_path), "-ss", str(start_time_clip), 
            "-t", str(actual_duration), "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", str(clipped_path), "-hide_banner", "-loglevel", "error"
        ]
        
        scale_cmd = [
            "taskset", "-c", "0,1,2,3,4,5,6,7,8,9,10,11",
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

def download_trim_downscale_youtube_video(
    clip_duration: int,
    youtube_video_id: str,
    task_type: str,
    output_dir: str = "videos",
    youtube_handler=None
) -> Optional[Tuple[List[str], List[int]]]:
    """
    Downloads a YouTube video, extracts multiple chunks of the specified duration
    with sliding window approach, and downscales each chunk to required resolution.

    Args:
        clip_duration (int): Desired clip duration in seconds (5, 10, or 20).
        youtube_video_id (str): YouTube video ID to download.
        task_type (str): Type of task determining resolution requirements.
        output_dir (str): Directory to save the processed videos.
        youtube_handler: Optional YouTubeHandler instance for downloading.

    Returns:
        Optional[Tuple[List[str], List[int]]]: Lists of paths to the downscaled videos and their generated IDs,
        or None on failure.
    """
    try:
        from .youtube_requests import YouTubeHandler, RESOLUTIONS
    except ImportError:
        try:
            from youtube_requests import YouTubeHandler, RESOLUTIONS
        except ImportError:
            print("Error: Could not import YouTubeHandler and RESOLUTIONS")
            return None, None
    
    if youtube_handler is None:
        youtube_handler = YouTubeHandler()

    DOWNSCALE_HEIGHTS = {
        "HD24K": 1080,
        "4K28K": 2160,
    }
    
    TASK_TO_RESOLUTION = {
        "SD2HD": RESOLUTIONS.HD_1080,
        "HD24K": RESOLUTIONS.HD_2160,
        "SD24K": RESOLUTIONS.HD_2160,
        "4K28K": RESOLUTIONS.HD_4320,
    }

    print_lock = threading.Lock()
    
    def safe_print(message):
        with print_lock:
            print(message)

    downscale_height = DOWNSCALE_HEIGHTS.get(task_type, 1080)
    target_resolution = TASK_TO_RESOLUTION.get(task_type, RESOLUTIONS.HD_2160)

    os.makedirs(output_dir, exist_ok=True)

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
        safe_print(f"Processing YouTube chunk {i+1}: {start_time_clip:.1f}s to {end_time_clip:.1f}s")
        
        actual_duration = end_time_clip - start_time_clip
        
        trim_cmd = [
            "ffmpeg", "-y", "-i", str(source_path), "-ss", str(start_time_clip), 
            "-t", str(actual_duration), "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac", str(clipped_path), "-hide_banner", "-loglevel", "error"
        ]
        
        scale_cmd = [
            "ffmpeg", "-y", "-i", str(clipped_path), "-vf", f"scale=-1:{downscale_height}", 
            "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", 
            str(downscale_path), "-hide_banner", "-loglevel", "error"
        ]
        
        try:
            subprocess.run(trim_cmd, check=True)
            subprocess.run(scale_cmd, check=True)
            
            # Clean up intermediate file
            if os.path.exists(clipped_path):
                os.remove(clipped_path)
            
            chunk_elapsed_time = time.time() - chunk_start_time
            safe_print(f"Time taken to process YouTube chunk {i+1}: {chunk_elapsed_time:.2f} seconds")
            
            return (i, str(downscale_path), video_id)
        except subprocess.SubprocessError as e:
            safe_print(f"Error processing YouTube chunk {i+1}: {e}")
            return None

    try:
        start_time = time.time()
        
        # Download YouTube video to temporary path
        temp_path = Path(output_dir) / f"{youtube_video_id}_original.mp4"
        
        safe_print(f"Downloading YouTube video: {youtube_video_id}")
        
        # Download video using YouTubeHandler
        youtube_handler.download_video(
            youtube_video_id, 
            target_resolution, 
            output_path=str(temp_path)
        )
        
        elapsed_time = time.time() - start_time
        safe_print(f"Time taken to download YouTube video: {elapsed_time:.2f} seconds")
    
        safe_print("\nChecking YouTube video resolution...")
        width, height, total_duration = get_video_info(temp_path)
        
        safe_print(f"YouTube video resolution: {width}x{height}")
        safe_print(f"YouTube video duration: {total_duration:.2f}s")
        
        # Skip very short videos
        if total_duration < clip_duration * 2:
            safe_print(f"YouTube video too short ({total_duration:.2f}s), skipping...")
            os.remove(temp_path)
            return None, None
        
        chunks_info = generate_youtube_chunk_timestamps(total_duration, clip_duration)
        safe_print(f"Extracting {len(chunks_info)} chunks of {clip_duration}s each from YouTube video")
        
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
        
        safe_print("\nCleaning up original downloaded YouTube file...")
        os.remove(temp_path)
        safe_print(f"Deleted: {temp_path}")

        safe_print(f"\nDone! Successfully processed {len(results)} out of {len(chunks_info)} YouTube chunks.")
        
        return downscale_paths, final_video_ids
        
    except Exception as e:
        safe_print(f"Error processing YouTube video: {str(e)}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                safe_print(f"Cleaned up YouTube file after error: {temp_path}")
            except:
                pass
        return None, None

if __name__ == "__main__":
    
    CLIP_DURATION = 2
    vid = 2257054
    
    download_trim_downscale_video(CLIP_DURATION, vid)
