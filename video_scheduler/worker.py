import time
import random

from redis_utils import (
    get_redis_connection,
    get_organic_queue_size,
    get_synthetic_queue_size,
    push_synthetic_chunk,
    MAX_CHUNK_IN_QUEUE,
    ORGANIC_QUEUE_KEY,
    SYNTHETIC_QUEUE_KEY,
)
from chunk_logic import read_synthetic_urls
from loguru import logger

logger.add("logs/worker.log")

# Where is your YAML config?
CONFIG_PATH = "config.yaml"


def clear_queues(redis_conn):
    """Clear both organic and synthetic queues before starting."""
    logger.info("Clearing queues")
    redis_conn.delete(ORGANIC_QUEUE_KEY)
    redis_conn.delete(SYNTHETIC_QUEUE_KEY)


def main():
    r = get_redis_connection()
    logger.info("Starting worker")
    # Add queue clearing at startup
    clear_queues(r)
    synthetic_urls = read_synthetic_urls(CONFIG_PATH)
    logger.info(f"Synthetic URLs: {synthetic_urls}")

    while True:
        organic_size = get_organic_queue_size(r)
        synthetic_size = get_synthetic_queue_size(r)
        total_size = organic_size + synthetic_size

        # If total queue is below some threshold, push synthetic chunks
        # Adjust threshold as needed. Example: If queue < 500, fill it up to 1000 with synthetic.
        threshold = 500
        fill_target = 1000

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
