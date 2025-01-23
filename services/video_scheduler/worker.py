import time
import random
from typing import Optional, List
import httpx
import asyncio
import os
from redis_utils import (
    get_redis_connection,
    get_organic_queue_size,
    get_synthetic_queue_size,
    push_synthetic_chunks,
)
from video_utils import download_and_trim_video
from google_drive.google_drive_manager import GoogleDriveManager
from video_subnet_core import CONFIG
from loguru import logger
import yaml
from dotenv import load_dotenv

load_dotenv()

def clear_queues(redis_conn):
    """Clear both organic and synthetic queues before starting."""
    logger.info("Clearing queues")
    redis_conn.delete(CONFIG.redis.organic_queue_key)
    redis_conn.delete(CONFIG.redis.synthetic_queue_key)

def read_synthetic_urls(config_path: str) -> list[str]:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config.get("synthetic_urls", [])

async def get_synthetic_urls_with_retry(
    hotkey: str, 
    max_retries: int = 2, 
    initial_delay: float = 5.0, 
    num_needed: int = 2
) -> Optional[List[str]]:
    """
    Attempt to fetch synthetic request urls from the API with exponential backoff retry.
    
    Args:
        hotkey (str): The validator's hotkey
        max_retries (int): Maximum number of retry attempts
        initial_delay (float): Initial delay in seconds between retries
        num_needed (int): Number of synthetic urls needed
    
    Returns:
        Optional[List[str]]: List of video urls if successful, None otherwise
    """
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Fetching synthetic urls from API (Attempt {attempt + 1}/{max_retries + 1})...")
            fetched_urls = await get_synthetic_urls(hotkey, num_needed)
            
            if fetched_urls and len(fetched_urls) == num_needed:
                logger.info(f"Successfully fetched {len(fetched_urls)} urls")
                return fetched_urls
            else:
                logger.warning(f"Insufficient urls returned: got {len(fetched_urls) if fetched_urls else 0}, needed {num_needed}")
        
        except Exception as e:
            logger.error(f"Error fetching synthetic urls: {str(e)}", exc_info=True)
        
        if attempt < max_retries:
            delay = initial_delay * (2 ** attempt)  # Exponential backoff
            logger.info(f"Retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)
    
    logger.error(f"Failed to fetch synthetic urls after {max_retries + 1} attempts")
    return None

async def get_synthetic_urls(hotkey: str, num_needed: int) -> Optional[List[str]]:
    """
    Fetch synthetic request urls from the API.
    
    Args:
        hotkey (str): The validator's hotkey
        num_needed (int): Number of synthetic urls needed
    
    Returns:
        Optional[List[str]]: List of video urls if successful, None otherwise
    
    Raises:
        httpx.HTTPError: If the API request fails
    """
    vidaio_api_endpoint = os.getenv("VIDAIO_API_ENDPOINT")
    api_url = f"{vidaio_api_endpoint}/api/synthetic_urls"
    params = {
        "validator_hotkey": hotkey,
        "num_needed": num_needed
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(api_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"API response: {data}")
            
            if not isinstance(data, dict) or "synthetic_urls" not in data:
                logger.error(f"Unexpected API response format: {data}")
                return None
            
            urls = data["synthetic_urls"]
            if not isinstance(urls, list):
                logger.error(f"Invalid urls format in response: {urls}")
                return None
                
            return urls
            
    except httpx.TimeoutException:
        logger.error("API request timed out")
        return None
    except httpx.HTTPError as e:
        logger.error(f"HTTP Error {e.response.status_code}: {e.response.text if hasattr(e, 'response') else 'No response'}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching synthetic urls: {str(e)}", exc_info=True)
        return None

async def get_synthetic_gdrive_urls(num_needed):

    sharing_video_urls = []
    sharing_file_ids = []
    
    count = num_needed
    
    while count:
        
        challenge_local_path = download_and_trim_video(
            clip_duration=CONFIG.video_scheduler.clip_duration,
            min_video_len=CONFIG.video_scheduler.min_video_len, 
            max_video_len=CONFIG.video_scheduler.max_video_len)
        
        if challenge_local_path is None:
            continue
        
        gdrive = GoogleDriveManager()
        uploaded_file_id, sharing_link = gdrive.upload_file(challenge_local_path)
        
        if uploaded_file_id is None or sharing_link is None:
            continue
        
        sharing_video_urls.append(sharing_link)
        sharing_file_ids.append(uploaded_file_id)
        
        count = count - 1
        
    return sharing_video_urls, sharing_file_ids

def main():
    r = get_redis_connection()
    logger.info("Starting worker")
    clear_queues(r)
    synthetic_urls = read_synthetic_urls("video_samples.yaml")
    logger.info(f"Synthetic URLs: {synthetic_urls}")

    while True:
        organic_size = get_organic_queue_size(r)
        synthetic_size = get_synthetic_queue_size(r)
        total_size = organic_size + synthetic_size

        # If total queue is below some threshold, push synthetic chunks
        # Adjust threshold as needed. Example: If queue < 500, fill it up to 1000 with synthetic.
        threshold = CONFIG.video_scheduler.refill_threshold
        fill_target = CONFIG.video_scheduler.refill_target

        if total_size < threshold:
            # Fill with synthetic chunks
            needed = fill_target - total_size
            # needed_urls = asyncio.run(get_synthetic_urls_with_retry(hotkey = hotkey, num_needed = needed))
            needed_urls = asyncio.run(get_synthetic_gdrive_urls(num_needed = needed))
            push_synthetic_chunks(r, needed_urls)

        # Sleep for some time, e.g. 5 seconds, then re-check
        time.sleep(5)


if __name__ == "__main__":
    main()
