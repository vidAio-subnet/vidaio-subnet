import asyncio
import os
import yaml
from typing import List, Dict, Optional
import httpx
from dotenv import load_dotenv
from loguru import logger
import random
import requests
import time
import yaml
import redis
import shutil

from redis_utils import (
    get_redis_connection,
    get_organic_queue_size,
    get_synthetic_queue_size,
    push_synthetic_chunks,
    push_pexels_video_ids,
    get_pexels_queue_size,
    pop_pexels_video_id,
)
from video_utils import download_trim_downscale_video
from services.google_drive.google_drive_manager import GoogleDriveManager
from vidaio_subnet_core import CONFIG
from vidaio_subnet_core.utilities.minio_client import minio_client

load_dotenv()

script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_file_path = os.path.join(script_dir, "pexels_categories.yaml")

def clear_queues(redis_conn) -> None:
    """Clear both organic and synthetic queues before starting."""
    logger.info("Clearing queues")
    redis_conn.delete(CONFIG.redis.organic_queue_key)
    redis_conn.delete(CONFIG.redis.synthetic_queue_key)
    redis_conn.delete(CONFIG.redis.pexels_video_ids_key)

def read_synthetic_urls(config_path: str) -> List[str]:
    """Read synthetic URLs from a YAML configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config.get("synthetic_urls", [])

def purge_cached_videos():
    
    videos_dir = os.path.join(os.getcwd(), "videos")

    try:
        if os.path.exists(videos_dir):
            shutil.rmtree(videos_dir)
            os.makedirs(videos_dir)
            print(f"Successfully purged all files from {videos_dir}")
        else:
            print(f"Directory not found: {videos_dir}")
            os.makedirs(videos_dir)
            print(f"Created new videos directory at {videos_dir}")
        
            
    except Exception as e:
        print(f"Error while purging videos: {str(e)}")

async def get_synthetic_urls_with_retry(hotkey: str, max_retries: int = 2, initial_delay: float = 5.0, num_needed: int = 2) -> Optional[List[str]]:
    """Attempt to fetch synthetic request URLs with exponential backoff retry."""
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Fetching synthetic URLs (Attempt {attempt + 1}/{max_retries + 1})...")
            fetched_urls = await get_synthetic_urls(hotkey, num_needed)
            
            if fetched_urls and len(fetched_urls) == num_needed:
                logger.info(f"Successfully fetched {len(fetched_urls)} URLs")
                return fetched_urls
            else:
                logger.warning(f"Insufficient URLs returned: got {len(fetched_urls) if fetched_urls else 0}, needed {num_needed}")
        except Exception as e:
            logger.error(f"Error fetching synthetic URLs: {str(e)}", exc_info=True)
        
        if attempt < max_retries:
            delay = initial_delay * (2 ** attempt)
            logger.info(f"Retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)
    
    logger.error("Failed to fetch synthetic URLs after retries.")
    return None

async def get_synthetic_urls(hotkey: str, num_needed: int) -> Optional[List[str]]:
    """Fetch synthetic request URLs from the API."""
    api_url = f"{os.getenv('VIDAIO_API_ENDPOINT')}/api/synthetic_urls"
    params = {"validator_hotkey": hotkey, "num_needed": num_needed}
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data, dict) or "synthetic_urls" not in data:
                logger.error(f"Unexpected API response format: {data}")
                return None
            
            urls = data["synthetic_urls"]
            if not isinstance(urls, list):
                logger.error(f"Invalid URL format in response: {urls}")
                return None
            
            return urls
    except httpx.HTTPError as e:
        logger.error(f"HTTP Error {e.response.status_code}: {e.response.text if e.response else 'No response'}")
    except httpx.TimeoutException:
        logger.error("API request timed out")
    except Exception as e:
        logger.error(f"Unexpected error fetching synthetic URLs: {str(e)}", exc_info=True)
    return None

def get_pexels_random_vids(
    num_needed: int, 
    min_len: int, 
    max_len: int, 
    width: int = None, 
    height: int = None, 
    max_results: int = None,
    task_type: str = None
):
    """
    Fetch video IDs of a specific resolution from Pexels API with randomized selection.

    Args:
        num_needed (int): Number of video IDs required.
        min_len (int): Minimum video length in seconds.
        max_len (int): Maximum video length in seconds.
        width (int): Required video width (default: 4096).
        height (int): Required video height (default: 2160).
        max_results (int, optional): Max videos to fetch before selecting randomly (default: num_needed * 10).
        task_type: The type of task
    Returns:
        list: A shuffled list of `num_needed` video IDs.
    """

    DCI_4K_WIDTH = 4096
    DCI_4K_HEIGHT = 2160
    HD_WIDTH = 1920
    HD_HEIGHT = 1080

    if not width or not height:  
        if task_type == "SD2HD":
            width = HD_WIDTH
            height = HD_HEIGHT
        else:  
            width = DCI_4K_WIDTH
            height = DCI_4K_HEIGHT

    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        logger.error("[ERROR] Missing Pexels API Key")
        return []
    
    start_time = time.time()
    headers = {"Authorization": api_key}

    with open(yaml_file_path, "r") as file:
        yaml_data = yaml.safe_load(file)
        query_list = yaml_data.get("pexels_categories", [])

    random.shuffle(query_list) 
    
    max_results = max_results or num_needed * 7
    valid_video_ids = []
    
    per_page = 80
    
    logger.info(f"[INFO] Fetching {num_needed} video IDs with resolution {width}x{height}")
    logger.info(f"[INFO] Searching through a maximum of {max_results} potential videos")
    
    for query in query_list:
        page = random.randint(1, 10)  
        # logger.info(f"[INFO] Searching for query: '{query}', starting from page {page}")
        while len(valid_video_ids) < max_results:
            params = {
                "query": query,
                "per_page": per_page,
                "page": page,
                "size": "large",
            }
            
            try:
                response = requests.get("https://api.pexels.com/videos/search", headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                if "videos" not in data or not data["videos"]:
                    logger.info(f"[WARNING] No videos found for query '{query}' on page {page}")
                    break  
                
                logger.info(f"[INFO] Found {len(data['videos'])} videos for query '{query}' on page {page}")
                
                for video in data["videos"]:
                    if min_len <= video["duration"] <= max_len:
                        # Check if the first video file matches the required resolution
                        if (video["video_files"] and 
                            video["video_files"][0]["width"] == width and 
                            video["video_files"][0]["height"] == height):
                            valid_video_ids.append(video["id"])
                            logger.info(f"[INFO] Added video ID {video['id']} ({video['video_files'][0]['width']}x{video['video_files'][0]['height']})")

                
                if len(data["videos"]) < per_page:
                    logger.info(f"[INFO] No more pages available for query '{query}'")
                    break 
                
                page += random.randint(1, 3)  
            
            except requests.exceptions.RequestException as e:
                logger.info(f"[ERROR] Error fetching videos for '{query}': {e}")
                break  
    
    logger.info(f"[INFO] Total matching videos found: {len(valid_video_ids)}")
    elapsed_time = time.time() - start_time
    logger.info(f"Time taken to get {num_needed} vids: {elapsed_time:.2f} seconds")
    
    random.shuffle(valid_video_ids)
    return valid_video_ids[:num_needed]
    

async def get_synthetic_requests_paths(num_needed: int, redis_conn: redis.Redis) -> List[Dict[str, str]]:
    """Generate synthetic Google Drive URLs by uploading trimmed videos."""
    uploaded_video_chunks = []
    remaining_count = num_needed

    while remaining_count > 0:
        
        video_id_data = pop_pexels_video_id(redis_conn)
        
        if video_id_data == None:
            time.sleep(10)
            continue

        video_id = video_id_data["vid"]
        task_type = video_id_data["task_type"]

        challenge_local_path, video_id = download_trim_downscale_video(
            clip_duration=CONFIG.video_scheduler.clip_duration,
            vid = video_id,
            task_type = task_type,
        )

        if not challenge_local_path:
            logger.info("Failed to download and trim video. Retrying...")
            continue

        uploaded_file_id = video_id
        object_name = f"{uploaded_file_id}.mp4"
        
        await minio_client.upload_file(object_name, challenge_local_path)
        sharing_link = await minio_client.get_presigned_url(object_name)

        if not sharing_link:
            logger.info("Upload failed. Retrying...")
            continue
        logger.info("Uploading success!")
        uploaded_video_chunks.append({
            "video_id": str(video_id),
            "uploaded_object_name": object_name,
            "sharing_link": sharing_link,
            "task_type": task_type,
        })
        remaining_count -= 1

    return uploaded_video_chunks

async def main():
    """Main function to manage video processing and synthetic queue handling."""
    
    logger.info("Starting worker...")
    redis_conn = get_redis_connection()
    
    clear_queues(redis_conn)
    purge_cached_videos()
    await minio_client.delete_all_items()


    scheduler_threshold = CONFIG.video_scheduler.refill_threshold
    fill_target = CONFIG.video_scheduler.refill_target

    pexels_queue_threshold = CONFIG.video_scheduler.pexels_threshold
    pexels_queue_max_size = CONFIG.video_scheduler.pexels_max_size
    video_min_len = CONFIG.video_scheduler.min_video_len
    video_max_len = CONFIG.video_scheduler.max_video_len
    
    threshold_hd_to_4k = CONFIG.video_scheduler.weight_hd_to_4k
    threshold_sd_to_hd = CONFIG.video_scheduler.weight_sd_to_hd + threshold_hd_to_4k

    while True:
        
        pexels_queue_size = get_pexels_queue_size(redis_conn)
        logger.info(f"Pexels video ids queue size: {pexels_queue_size}")

        if pexels_queue_size <= pexels_queue_threshold:

            needed = pexels_queue_max_size - pexels_queue_size

            if pexels_queue_size == 0:
                needed = 5
            
            ran_num = random.random()
            logger.info(f"Seleted random number: {ran_num}")
            ran_num = 0.3
            if ran_num < threshold_hd_to_4k:
                task_type = "HD24K"
            elif ran_num < threshold_sd_to_hd:
                task_type = "SD2HD"
            else:
                task_type = "SD24K"
            
            logger.info(f"Need {needed} pexels video ids with task type: {task_type}")

            needed_vids = get_pexels_random_vids(num_needed = needed, min_len = video_min_len, max_len = video_max_len, task_type = task_type)

            needed_vids_dict = []
            for vid in needed_vids:
                needed_vids_dict.append({
                    "vid": vid,
                    "task_type": task_type,
                })

            push_pexels_video_ids(redis_conn, needed_vids_dict)
        
        organic_size = get_organic_queue_size(redis_conn)
        synthetic_size = get_synthetic_queue_size(redis_conn)
        total_size = organic_size + synthetic_size
       
        logger.info(f"Organic queue size: {organic_size}")
        logger.info(f"Synthetic queue size: {synthetic_size}")

        if total_size <= scheduler_threshold:
            
            needed = fill_target - total_size

            if total_size == 0:
                needed = 3
        
            logger.info(f"Need {needed} chunks...")
            
            needed_urls = await get_synthetic_requests_paths(num_needed=needed, redis_conn = redis_conn)
            push_synthetic_chunks(redis_conn, needed_urls)
        
        time.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())