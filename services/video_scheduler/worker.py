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
    get_5s_queue_size,
    get_10s_queue_size,
    get_20s_queue_size,
    push_5s_chunks,
    push_10s_chunks,
    push_20s_chunks,
    push_pexels_video_ids,
    get_pexels_queue_size,
    pop_pexels_video_id,
)
from video_utils import download_trim_downscale_video
from services.google_drive.google_drive_manager import GoogleDriveManager
from vidaio_subnet_core import CONFIG
from vidaio_subnet_core.utilities.storage_client import storage_client

load_dotenv()

script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_file_path = os.path.join(script_dir, "pexels_categories.yaml")

def clear_queues(redis_conn) -> None:
    """Clear both organic and synthetic queues before starting."""
    logger.info("Clearing queues")
    redis_conn.delete(CONFIG.redis.organic_queue_key)
    redis_conn.delete(CONFIG.redis.synthetic_5s_clip_queue_key)
    redis_conn.delete(CONFIG.redis.synthetic_10s_clip_queue_key)
    redis_conn.delete(CONFIG.redis.synthetic_20s_clip_queue_key)
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
            logger.info(f"Successfully purged all files from {videos_dir}")
        else:
            logger.info(f"Directory not found: {videos_dir}")
            os.makedirs(videos_dir)
            logger.info(f"Created new videos directory at {videos_dir}")
        
            
    except Exception as e:
        logger.info(f"Error while purging videos: {str(e)}")

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
        width (int): Required video width (default: 3840).
        height (int): Required video height (default: 2160).
        max_results (int, optional): Max videos to fetch before selecting randomly (default: num_needed * 10).
        task_type: The type of task
    Returns:
        list: A shuffled list of `num_needed` video IDs.
    """

    RESOLUTIONS = {
        "HD24K": (3840, 2160),
        "SD2HD": (1920, 1080),
        "SD24K": (3840, 2160),
        "4K28K": (7680, 4320),
        "HD28K": (7680, 4320),
    }


    width, height = (width, height) if width and height else RESOLUTIONS.get(task_type, (3840, 2160))

    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        logger.error("[ERROR] Missing Pexels API Key")
        return []
    
    start_time = time.time()
    headers = {"Authorization": api_key}

    with open(yaml_file_path, "r") as file:
        yaml_data = yaml.safe_load(file)
        query_list = yaml_data.get("pexels_categories", [])
    query_list = ["nature"]
    random.shuffle(query_list) 
    
    max_results = max_results or num_needed * 3

    if task_type == "4K28K":
        max_results = num_needed

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

            if task_type == "SD2HD":
                params = {
                    "query": query,
                    "per_page": per_page,
                    "page": page,
                }

            try:
                response = requests.get("https://api.pexels.com/videos/search", headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                import json
                with open("output.json", "w") as file:
                    # writing the JSON data to the file with indentation for better readability
                    json.dump(data, file, indent=4)
                if "videos" not in data or not data["videos"]:
                    logger.info(f"[WARNING] No videos found for query '{query}' on page {page}")
                    break  
                
                logger.info(f"[INFO] Found {len(data['videos'])} videos for query '{query}' on page {page}")
                
                for video in data["videos"]:
                    if min_len <= video["duration"] <= max_len and video["width"] == width and video["height"] == height:
                        valid_video_ids.append(video["id"])
                        logger.info(f"[INFO] Added video ID {video['id']} ({video['width']}x{video['height']})")
                
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

    return_val = valid_video_ids[:num_needed]

    logger.info(return_val)

    return return_val
    

async def get_synthetic_requests_paths(num_needed: int, redis_conn: redis.Redis, chunk_duration: int) -> List[Dict[str, str]]:
    """Generate synthetic Google Drive URLs by uploading trimmed videos."""
    uploaded_video_chunks = []
    remaining_count = num_needed

    while remaining_count > 0:
        
        video_id_data = pop_pexels_video_id(redis_conn)
        
        logger.info("downloading video data: ")
        logger.info(video_id_data)

        if video_id_data == None:
            time.sleep(10)
            continue

        video_id = video_id_data["vid"]
        task_type = video_id_data["task_type"]

        clip_duration_probabilities = {
            1: 0.35, 
            2: 0.05,
            3: 0.05,
            4: 0.05,
            5: 0.5
        }

        if chunk_duration == 5:
            random_value = random.random()
            cumulative_probability = 0
            for clip_duration, probability in clip_duration_probabilities.items():
                cumulative_probability += probability
                if random_value <= cumulative_probability:
                    break
        else:
            clip_duration = chunk_duration
        challenge_local_paths, video_ids = download_trim_downscale_video(
            clip_duration=clip_duration,
            vid=video_id,
            task_type=task_type,
            chunk_duration=chunk_duration
        )

        if challenge_local_paths is None:
            logger.info("Failed to download and trim video. Retrying...")
            continue

        for video_id, challenge_local_path in zip(video_ids, challenge_local_paths):
            uploaded_file_id = video_id
            object_name = f"{uploaded_file_id}.mp4"
        
            await storage_client.upload_file(object_name, challenge_local_path)
            sharing_link = await storage_client.get_presigned_url(object_name)

            print(f"Sharing_link:{sharing_link} ")

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
    """
    Main service function that manages video processing workflow and queue maintenance.
    
    Handles:
    - Queue initialization and monitoring
    - Dynamic video source replenishment based on configurable thresholds
    - Multi-duration synthetic chunk generation and distribution
    - Continuous system health monitoring
    """
    logger.info("Initializing video processing worker...")
    
    try:
        redis_conn = get_redis_connection()
        await initialize_environment(redis_conn)
        
        scheduler_config = CONFIG.video_scheduler
        queue_thresholds = {
            "refill": scheduler_config.refill_threshold,
            "target": scheduler_config.refill_target,
            "pexels": scheduler_config.pexels_threshold,
            "pexels_max": scheduler_config.pexels_max_size
        }
        
        video_constraints = {
            "min_length": scheduler_config.min_video_len,
            "max_length": scheduler_config.max_video_len
        }
        
        task_thresholds = calculate_task_thresholds(scheduler_config)
        
        while True:

            try:
                
                cycle_start_time = time.time()

                await manage_pexels_queue(
                    redis_conn, 
                    queue_thresholds, 
                    video_constraints, 
                    task_thresholds
                )
                
                log_queue_status(redis_conn)
                
                for duration in [5, 10, 20]:
                    await replenish_synthetic_queue(
                        redis_conn,
                        duration,
                        queue_thresholds["refill"],
                        queue_thresholds["target"]
                    )
                
                processed_time = time.time() - cycle_start_time

                logger.info(f"✳️✳️✳️ One cycle processed in {processed_time:.2f} ✳️✳️✳️")

                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error in main service loop: {str(e)}")
            
    except Exception as e:
        logger.error(f"Critical error in main service loop: {str(e)}")
        logger.exception("Exception details:")
        raise


async def initialize_environment(redis_conn):
    """Initialize the processing environment by clearing queues and cached data."""
    logger.info("Clearing queues and cached content...")
    clear_queues(redis_conn)
    purge_cached_videos()
    await storage_client.delete_all_items()
    logger.info("Environment initialized successfully")


def calculate_task_thresholds(config):
    """Calculate cumulative thresholds for task type selection."""
    return {
        "HD24K": config.weight_hd_to_4k,
        "SD2HD": config.weight_sd_to_hd + config.weight_hd_to_4k,
        "SD24K": config.weight_sd_to_4k + config.weight_sd_to_hd + config.weight_hd_to_4k,
        "4K28K": config.weight_4k_to_8k + config.weight_sd_to_4k + config.weight_sd_to_hd + config.weight_hd_to_4k
    }


async def manage_pexels_queue(redis_conn, thresholds, video_constraints, task_thresholds):
    """Manage the Pexels video ID queue, replenishing when below threshold."""
    pexels_queue_size = get_pexels_queue_size(redis_conn)
    logger.info(f"Pexels video IDs queue size: {pexels_queue_size}")
    
    if pexels_queue_size <= thresholds["pexels"]:
        needed = thresholds["pexels_max"] - pexels_queue_size
        needed = 5 if pexels_queue_size == 0 else needed
        
        task_type = select_task_type(task_thresholds)
        logger.info(f"Replenishing queue with {needed} videos for task type: {task_type}")
        
        try:
            video_ids = get_pexels_random_vids(
                num_needed=needed,
                min_len=video_constraints["min_length"],
                max_len=video_constraints["max_length"],
                task_type=task_type
            )
            
            video_entries = [{"vid": vid, "task_type": task_type} for vid in video_ids]
            push_pexels_video_ids(redis_conn, video_entries)
            
            logger.info(f"Added {len(video_entries)} new video IDs to Pexels queue")
        except Exception as e:
            logger.error(f"Failed to replenish Pexels queue: {str(e)}")


def select_task_type(thresholds):
    """Select a task type based on weighted probability thresholds."""
    random_value = random.random()
    logger.debug(f"Task selection random value: {random_value:.4f}")
    
    for task_type, threshold in thresholds.items():
        if random_value <= threshold:
            return task_type
    
    return "HD28K"


def log_queue_status(redis_conn):
    """Log the current status of all processing queues."""
    queue_sizes = {
        "organic": get_organic_queue_size(redis_conn),
        "synthetic_5s": get_5s_queue_size(redis_conn),
        "synthetic_10s": get_10s_queue_size(redis_conn),
        "synthetic_20s": get_20s_queue_size(redis_conn)
    }
    
    for queue_name, size in queue_sizes.items():
        logger.info(f"{queue_name.replace('_', ' ').title()} queue size: {size}")
    
    return queue_sizes


async def replenish_synthetic_queue(redis_conn, duration, threshold, target):
    """Replenish a specific synthetic chunk queue if below threshold."""
    queue_size = get_queue_size_by_duration(redis_conn, duration)
    
    if queue_size < threshold:
        needed = target - queue_size
        logger.info(f"Replenishing {duration}s chunk queue with {needed} items")
        
        try:
            chunk_data = await get_synthetic_requests_paths(
                num_needed=needed,
                redis_conn=redis_conn,
                chunk_duration=duration
            )
            
            push_chunks_by_duration(redis_conn, chunk_data, duration)
            logger.info(f"Successfully added {len(chunk_data)} chunks to {duration}s queue")
        except Exception as e:
            logger.error(f"Failed to replenish {duration}s queue: {str(e)}")


def get_queue_size_by_duration(redis_conn, duration):
    """Get queue size based on duration."""
    if duration == 5:
        return get_5s_queue_size(redis_conn)
    elif duration == 10:
        return get_10s_queue_size(redis_conn)
    elif duration == 20:
        return get_20s_queue_size(redis_conn)
    else:
        raise ValueError(f"Unsupported duration: {duration}")


def push_chunks_by_duration(redis_conn, chunk_data, duration):
    """Push chunks to the appropriate queue based on duration."""
    if duration == 5:
        push_5s_chunks(redis_conn, chunk_data)
    elif duration == 10:
        push_10s_chunks(redis_conn, chunk_data)
    elif duration == 20:
        push_20s_chunks(redis_conn, chunk_data)
    else:
        raise ValueError(f"Unsupported duration: {duration}")


if __name__ == "__main__":
    asyncio.run(main())