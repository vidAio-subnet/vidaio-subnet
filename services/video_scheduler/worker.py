import time
import random
from typing import Optional, Dict, Any, List
import httpx
import asyncio




from redis_utils import (
    get_redis_connection,
    get_organic_queue_size,
    get_synthetic_queue_size,
    push_synthetic_chunk,
)
from video_subnet_core import CONFIG
from loguru import logger
import yaml


def clear_queues(redis_conn):
    """Clear both organic and synthetic queues before starting."""
    logger.info("Clearing queues")
    redis_conn.delete(CONFIG.redis.organic_queue_key)
    redis_conn.delete(CONFIG.redis.synthetic_queue_key)


def read_synthetic_urls(config_path: str) -> list[str]:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config.get("synthetic_urls", [])

async def get_next_challenge_with_retry(hotkey: str, max_retries: int = 2, initial_delay: float = 5.0, num_needed: int = 2) -> Optional[List[str]]:
    """
    Attempt to fetch the next synthetic request urls from the API with retries.
    
    Args:
        hotkey (str): The validator's hotkey.
        max_retries (int): Maximum number of retry attempts.
        initial_delay (float): Initial delay in seconds between retries.
    
    Returns:
        Optional[dict]: video url if successful, None otherwise.
    """
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Fetching synthetic urls from API (Attempt {attempt + 1}/{max_retries + 1})...")
            fetched_urls = await get_synthetic_urls(hotkey, num_needed)
            if fetched_urls:
                logger.info(f"Successfully fetched {num_needed} urls.")
                return fetched_urls
            else:
                logger.warning("No urls available from API")
        except Exception as e:
            logger.error(f"Error fetching synthetic urls: {str(e)}")
        
        if attempt < max_retries:
            delay = initial_delay * (2 ** attempt)  # Exponential backoff
            logger.info(f"Retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)
    
    logger.warning("Failed to fetch challenge after all retry attempts")
    return None

async def get_synthetic_urls(validator_address: str, num_needed: int) -> Optional[List[int]]:
    """
    Fetch the synthetic request urls from the API.
    
    Args:
        validator_address: The validator's ss58 address
    
    Returns:
        video_urls list, or None if no challenge available
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{VIDAIO_API}/api/synthetic_urls/?validator_hotkey={validator_address}/num_needed={num_needed}")
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"Got synthetic urls from API: {data}")
            
            # Only return the fields we need
            return data["synthetic_urls"]
            
    except Exception as e:
        logger.error(f"Error fetching synthetic urls: {str(e)}")
        if isinstance(e, httpx.HTTPError):
            logger.error(f"HTTP Error response: {e.response.text if hasattr(e, 'response') else 'No response'}")
        return None

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
            for _ in range(needed):
                url = random.choice(synthetic_urls)
                push_synthetic_chunk(r, url)

        # Sleep for some time, e.g. 5 seconds, then re-check
        time.sleep(5)


if __name__ == "__main__":
    main()
