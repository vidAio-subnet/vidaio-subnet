import asyncio
import os
import uuid
import yaml
from typing import List, Dict, Optional
import httpx
from dotenv import load_dotenv
from loguru import logger

from redis_utils import (
    get_redis_connection,
    get_organic_queue_size,
    get_synthetic_queue_size,
    push_synthetic_chunks,
)
from video_utils import download_trim_downscale_video
from services.google_drive.google_drive_manager import GoogleDriveManager
from vidaio_subnet_core import CONFIG
from vidaio_subnet_core.utilities.minio_client import minio_client

# Load environment variables
load_dotenv()

def clear_queues(redis_conn) -> None:
    """Clear both organic and synthetic queues before starting."""
    logger.info("Clearing queues")
    redis_conn.delete(CONFIG.redis.organic_queue_key)
    redis_conn.delete(CONFIG.redis.synthetic_queue_key)

def read_synthetic_urls(config_path: str) -> List[str]:
    """Read synthetic URLs from a YAML configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config.get("synthetic_urls", [])

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

async def get_synthetic_requests_paths(num_needed: int) -> List[Dict[str, str]]:
    """Generate synthetic Google Drive URLs by uploading trimmed videos."""
    uploaded_video_chunks = []
    remaining_count = num_needed

    while remaining_count > 0:
        challenge_local_path, video_id = download_trim_downscale_video(
            clip_duration=CONFIG.video_scheduler.clip_duration,
            min_video_len=CONFIG.video_scheduler.min_video_len,
            max_video_len=CONFIG.video_scheduler.max_video_len
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

        uploaded_video_chunks.append({
            "video_id": video_id,
            "uploaded_object_name": object_name,
            "sharing_link": sharing_link
        })
        remaining_count -= 1

    return uploaded_video_chunks

async def main():
    """Main function to manage video processing and synthetic queue handling."""
    logger.info("Starting worker...")
    redis_conn = get_redis_connection()
    clear_queues(redis_conn)

    while True:
        organic_size = get_organic_queue_size(redis_conn)
        synthetic_size = get_synthetic_queue_size(redis_conn)
        total_size = organic_size + synthetic_size

        logger.info(f"Organic queue size: {organic_size}")
        logger.info(f"Synthetic queue size: {synthetic_size}")
        
        threshold = CONFIG.video_scheduler.refill_threshold
        fill_target = CONFIG.video_scheduler.refill_target

        if total_size < threshold:
            needed = fill_target - total_size
            logger.info(f"Need {needed} chunks...")
            needed_urls = await get_synthetic_requests_paths(num_needed=needed)
            push_synthetic_chunks(redis_conn, needed_urls)

        await asyncio.sleep(20)

if __name__ == "__main__":
    asyncio.run(main())
