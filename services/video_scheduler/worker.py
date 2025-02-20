# import time
# import random
# from typing import List, Dict, Optional
# import httpx
# import asyncio
# import os
# from redis_utils import (
#     get_redis_connection,
#     get_organic_queue_size,
#     get_synthetic_queue_size,
#     push_synthetic_chunks,
# )
# from video_utils import download_trim_downscale_video
# from services.google_drive.google_drive_manager import GoogleDriveManager
# from vidaio_subnet_core import CONFIG
# from loguru import logger
# import yaml
# from dotenv import load_dotenv
# from vidaio_subnet_core.utilities.minio_client import minio_client
# import uuid
# from loguru import logger
# load_dotenv()

# def clear_queues(redis_conn):
#     """Clear both organic and synthetic queues before starting."""
#     logger.info("Clearing queues")
#     redis_conn.delete(CONFIG.redis.organic_queue_key)
#     redis_conn.delete(CONFIG.redis.synthetic_queue_key)

# def read_synthetic_urls(config_path: str) -> list[str]:
#     with open(config_path, "r") as file:
#         config = yaml.safe_load(file)
#     return config.get("synthetic_urls", [])

# async def get_synthetic_urls_with_retry(
#     hotkey: str, 
#     max_retries: int = 2, 
#     initial_delay: float = 5.0, 
#     num_needed: int = 2
# ) -> Optional[List[str]]:
#     """
#     Attempt to fetch synthetic request urls from the API with exponential backoff retry.
    
#     Args:
#         hotkey (str): The validator's hotkey
#         max_retries (int): Maximum number of retry attempts
#         initial_delay (float): Initial delay in seconds between retries
#         num_needed (int): Number of synthetic urls needed
    
#     Returns:
#         Optional[List[str]]: List of video urls if successful, None otherwise
#     """
#     for attempt in range(max_retries + 1):
#         try:
#             logger.info(f"Fetching synthetic urls from API (Attempt {attempt + 1}/{max_retries + 1})...")
#             fetched_urls = await get_synthetic_urls(hotkey, num_needed)
            
#             if fetched_urls and len(fetched_urls) == num_needed:
#                 logger.info(f"Successfully fetched {len(fetched_urls)} urls")
#                 return fetched_urls
#             else:
#                 logger.warning(f"Insufficient urls returned: got {len(fetched_urls) if fetched_urls else 0}, needed {num_needed}")
        
#         except Exception as e:
#             logger.error(f"Error fetching synthetic urls: {str(e)}", exc_info=True)
        
#         if attempt < max_retries:
#             delay = initial_delay * (2 ** attempt)  # Exponential backoff
#             logger.info(f"Retrying in {delay:.2f} seconds...")
#             await asyncio.sleep(delay)
    
#     logger.error(f"Failed to fetch synthetic urls after {max_retries + 1} attempts")
#     return None

# async def get_synthetic_urls(hotkey: str, num_needed: int) -> Optional[List[str]]:
#     """
#     Fetch synthetic request urls from the API.
    
#     Args:
#         hotkey (str): The validator's hotkey
#         num_needed (int): Number of synthetic urls needed
    
#     Returns:
#         Optional[List[str]]: List of video urls if successful, None otherwise
    
#     Raises:
#         httpx.HTTPError: If the API request fails
#     """
#     vidaio_api_endpoint = os.getenv("VIDAIO_API_ENDPOINT")
#     api_url = f"{vidaio_api_endpoint}/api/synthetic_urls"
#     params = {
#         "validator_hotkey": hotkey,
#         "num_needed": num_needed
#     }
    
#     try:
#         async with httpx.AsyncClient(timeout=30.0) as client:
#             response = await client.get(api_url, params=params)
#             response.raise_for_status()
            
#             data = response.json()
#             logger.debug(f"API response: {data}")
            
#             if not isinstance(data, dict) or "synthetic_urls" not in data:
#                 logger.error(f"Unexpected API response format: {data}")
#                 return None
            
#             urls = data["synthetic_urls"]
#             if not isinstance(urls, list):
#                 logger.error(f"Invalid urls format in response: {urls}")
#                 return None
                
#             return urls
            
#     except httpx.TimeoutException:
#         logger.error("API request timed out")
#         return None
#     except httpx.HTTPError as e:
#         logger.error(f"HTTP Error {e.response.status_code}: {e.response.text if hasattr(e, 'response') else 'No response'}")
#         return None
#     except Exception as e:
#         logger.error(f"Unexpected error fetching synthetic urls: {str(e)}", exc_info=True)
#         return None

# from typing import List, Dict

# async def get_synthetic_requests_urls(num_needed: int) -> List[Dict[str, str]]:
#     """
#     Generate synthetic Google Drive URLs by uploading trimmed videos.

#     Args:
#         num_needed (int): The number of synthetic URLs needed.

#     Returns:
#         List[Dict[str, str]]: A list of dictionaries mapping file IDs to sharing links.
#     """
#     uploaded_video_chunks: List[Dict[str, str]] = []
#     remaining_count: int = num_needed

#     while remaining_count > 0:
        
#         # Download and trim video
#         challenge_local_path, video_id = download_trim_downscale_video(
#             clip_duration=CONFIG.video_scheduler.clip_duration,
#             min_video_len=CONFIG.video_scheduler.min_video_len,
#             max_video_len=CONFIG.video_scheduler.max_video_len
#         )

#         if challenge_local_path is None:
#             logger.info("Failed to download and trim video. Retrying...")
#             continue

#         uploaded_file_id = uuid.uuid4()
#         object_name = f"{uploaded_file_id}.mp4"
        
#         await minio_client.upload_file(object_name, challenge_local_path)
#         sharing_link = await minio_client.get_presigned_url(object_name)

#         if uploaded_file_id is None or sharing_link is None:
#             logger.info("Upload failed. Retrying...")
#             continue

#         # Append result to the list
#         uploaded_video_chunks.append({
#             "video_id": video_id,
#             "uploaded_object_name": object_name,
#             "sharing_link": sharing_link
#         })

#         remaining_count -= 1

#     return uploaded_video_chunks


# async def main():
#     logger.info("starting")
#     r = get_redis_connection()
#     logger.info("Starting worker")
#     clear_queues(r)

#     while True:
#         organic_size = get_organic_queue_size(r)
#         synthetic_size = get_synthetic_queue_size(r)
#         total_size = organic_size + synthetic_size
#         logger.info(f"The organic queue size is {organic_size}")
#         logger.info(f"The synthetic queue size is {synthetic_size}")
#         # If total queue is below some threshold, push synthetic chunks
#         # Adjust threshold as needed. Example: If queue < 500, fill it up to 1000 with synthetic.
#         threshold = CONFIG.video_scheduler.refill_threshold
#         fill_target = CONFIG.video_scheduler.refill_target

#         if total_size < threshold:
#             # Fill with synthetic chunks
#             needed = fill_target - total_size
#             # needed_urls = asyncio.run(get_synthetic_urls_with_retry(hotkey = hotkey, num_needed = needed))
#             logger.info(f"need {needed} chunks.....")
#             needed_urls = await get_synthetic_requests_urls(num_needed = needed)
#             push_synthetic_chunks(r, needed_urls)

#         # Sleep for some time, e.g. 5 seconds, then re-check
#         time.sleep(20)


# if __name__ == "__main__":
#     asyncio.run(main())



import time
import random
import asyncio
import os
import uuid
import yaml
from typing import List, Dict, Optional
import httpx
from dotenv import load_dotenv
from loguru import logger
from pathlib import Path

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

async def get_synthetic_requests_urls(num_needed: int) -> List[Dict[str, str]]:
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

        uploaded_file_id = uuid.uuid4()
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
            needed_urls = await get_synthetic_requests_urls(num_needed=needed)
            push_synthetic_chunks(redis_conn, needed_urls)

        await asyncio.sleep(20)

if __name__ == "__main__":
    asyncio.run(main())
