import os
import requests
import time
from tqdm import tqdm
import traceback
from pathlib import Path
import uuid
from typing import List, Tuple, Optional
import concurrent.futures
import subprocess
import tempfile
import threading
import functools
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import yt_dlp

load_dotenv()
MAX_WORKERS = 8

def cookies_to_netscape(cookies, max_age=48 * 3600) -> str:
    """
    Converts Selenium Chrome driver cookies to Netscape format
    Returns the cookies as a string
    """
    expiry = int(time.time()) + max_age
    lines = ["# Netscape HTTP Cookie File"]
    for c in cookies:
        domain = c["domain"]
        include_subdomains = "TRUE" if domain.startswith(".") else "FALSE"
        path = c["path"]
        secure = "TRUE" if c.get("secure", False) else "FALSE"
        expiry = c.get("expiry", expiry)
        name = c["name"]
        value = c["value"]
        lines.append(f"{domain}\t{include_subdomains}\t{path}\t{secure}\t{expiry}\t{name}\t{value}")
    return "\n".join(lines)

def fetch_cookies() -> os.PathLike:
    """
    Gets cookies from YouTube using Selenium
    Converts them to Netscape format for yt dlp
    Saves them to a tempfile
    Returns the tempfile path.
    """
    start = time.time()
    options = [
        "--headless=new",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--window-size=64,64",
        "--disable-extensions",
        "--disable-background-networking",
        "--disable-sync",
        "--disable-default-apps",
        "--disable-translate",
        "--disable-features=TranslateUI",
    ]
    chrome_options = Options()
    for o in options:
        chrome_options.add_argument(o)
    driver = webdriver.Chrome(options=chrome_options)
    driver.get("https://www.youtube.com")
    driver.implicitly_wait(5)
    cookies = driver.get_cookies()
    driver.quit()
    cookies = cookies_to_netscape(cookies)
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.cookies') as f:
        f.write(cookies)
        cookie_file = f.name
    print(f"Cookie fetch took {time.time() - start} seconds")
    return cookie_file

def with_tempdir(func):
    """Provides a contextual temp dir to a function"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with tempfile.TemporaryDirectory() as temp:
            return func(*args, temp_path=temp, **kwargs)
    return wrapper

def download_trim_downscale_video(
    clip_duration: int,
    vid: int,
    task_type: str,
    output_dir: str = "/videos"
) -> Optional[Tuple[List[str], List[int]]]:
    """
    Downloads a specified video with video id from YouTube, extracts multiple chunks 
    of the specified duration with adaptive overlapping strategies, and downscales 
    each chunk to required resolution using multi-threading.

    Args:
        clip_duration (int): Desired clip duration in seconds (5, 10, or 20).
        vid (int): YouTube video ID to download.
        task_type (str): Type of task determining resolution requirements.
        output_dir (str): Directory to save the processed videos.

    Returns:
        Optional[Tuple[List[str], List[int]]]: Lists of paths to the downscaled videos and their generated IDs, 
        or None on failure.
    """
    DOWNSCALE_HEIGHTS = {
        "SD2HD": 540,
        "HD24K": 1080,
        "4K28K": 2160,
    }
    EXPECTED_RESOLUTIONS = {
        "SD2HD": (1920, 1080),
        "HD24K": (3840, 2160),
        "4K28K": (7680, 4320),
    }

    print_lock = threading.Lock()
    
    def safe_print(message):
        with print_lock:
            print(message)

    os.makedirs(output_dir, exist_ok=True)

    downscale_height = DOWNSCALE_HEIGHTS.get(task_type, 540)
    expected_width, expected_height = EXPECTED_RESOLUTIONS.get(task_type, (3840, 2160))
    

    def download_video(video_url:str, video_format:dict, output_path:os.PathLike) -> os.PathLike:
        """Download video from YouTube"""
        print(f"Downloading video {video_url} to {output_path}")
        
        print(f"Getting YouTube cookies from https://youtube.com")
        cookie_file = fetch_cookies()

        format_id = video_format['format_id']
        actual_width = video_format.get('width')
        actual_height = video_format.get('height')
        print(f"Selected video format: {format_id}")

        ydl_opts = {
            "cookiefile": cookie_file,
            "format": format_id,
            "outtmpl": str(output_path),
            "quiet": False,
            "no_warnings": True,
            "fixup": "never",
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        
    
    def get_video_info(video_path):
        """Get video resolution and duration using FFprobe"""
        
        probe_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0", 
            "-show_entries", "stream=width,height,duration", "-of", "csv=p=0", str(video_path)
        ]
        
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        width, height, duration = map(float, result.stdout.strip().replace("\n", "").split(',')[:3])
        return int(width), int(height), duration


    def process_chunk(chunk_info, source_path):
        """Process a single video chunk using FFmpeg directly"""
        i, start_time_clip, end_time_clip, video_id = chunk_info
        
        clipped_path = Path(output_dir) / f"{video_id}_trim.mp4"
        downscale_path = Path(output_dir) / f"{video_id}_downscale.mp4"
        
        chunk_start_time = time.time()
        safe_print(f"Processing chunk {i+1}: {start_time_clip:.1f}s to {end_time_clip:.1f}s")
        
        actual_duration = end_time_clip - start_time_clip
        source_path = os.path.abspath(source_path)
        clipped_path = os.path.abspath(clipped_path)
        downscale_path = os.path.abspath(downscale_path)
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
            start_time = time.time()

            # Fetch video metadata
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(vid, download=False)
                formats = info.get('formats', [])

            # Get video matches
            matching_formats = [
                f for f in formats
                if f.get('vcodec') != 'none' and f.get('acodec') == 'none'
                and f.get('width') == expected_width and f.get('height') == expected_height
            ]

            # Select highest bitrate version
            vid_format = max(matching_formats, key=lambda f: f.get('tbr') or 0)

            url = f"https://youtube.com/watch?v={vid}"
            temp_path = str(Path(output_dir) / f"{vid}_original.mp4")
            download_video(url, vid_format, temp_path)

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
            
            chunks = generate_chunk_timestamps(total_duration, 20)
            
            results = []

            def process_single_chunk(chunk_info):
                i, start_time, end_time = chunk_info
                video_id = uuid.uuid4()             
                result = process_chunk((i, start_time, end_time, video_id), temp_path)
                return result
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_chunk = {executor.submit(process_single_chunk, chunk): chunk for chunk in chunks}
                
                for future in concurrent.futures.as_completed(future_to_chunk):
                    result = future.result()
                    if result:
                        results.append(result)

            if os.path.exists(temp_path):
                os.remove(temp_path)
            
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

            # Fetch video metadata
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(vid, download=False)
                formats = info.get('formats', [])

            # Get video matches
            matching_formats = [
                f for f in formats
                if f.get('vcodec') != 'none' and f.get('acodec') == 'none'
                and f.get('width') == expected_width and f.get('height') == expected_height
            ]

            # Select highest bitrate version
            vid_format = max(matching_formats, key=lambda f: f.get('tbr') or 0)

            temp_path = str(Path(output_dir) / f"{vid}_original.mp4")

            url = f"https://youtube.com/watch?v={vid}"
            download_video(url, vid_format, temp_path)

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
            
            chunks = generate_chunk_timestamps(total_duration, clip_duration)
            safe_print(f"Extracting {len(chunks)} chunks of {clip_duration}s each")
            
            video_ids = [uuid.uuid4() for _ in range(len(chunks))]
            
            chunks_with_ids = [(i, start, end, video_ids[idx]) 
                                for idx, (i, start, end) in enumerate(chunks)]
            
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

            safe_print(f"\nDone! Successfully processed {len(results)} out of {len(chunks)} chunks.")
            
            return downscale_paths, final_video_ids
            
        except requests.exceptions.RequestException as e:
            safe_print(f"Error downloading video: {e}")
            return None, None
        except Exception as e:
            print(traceback.print_exc())
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
        print(traceback.print_exc())
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


if __name__ == "__main__":

    CLIP_DURATION = 20
    vid = "LrSvfI2hMJc" # 4k/HD YouTube video
    download_trim_downscale_video(CLIP_DURATION, vid, "HD24K")
