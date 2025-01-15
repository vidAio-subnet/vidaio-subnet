import time
import random

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
