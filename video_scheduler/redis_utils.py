import os
import redis

# Constants
MAX_CHUNK_IN_QUEUE = 4096

ORGANIC_QUEUE_KEY = "video_scheduler:organic_queue"
SYNTHETIC_QUEUE_KEY = "video_scheduler:synthetic_queue"


def get_redis_connection():
    """
    Create and return a Redis connection.
    Adjust host, port, and password if needed.
    """
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=int(os.getenv("REDIS_DB", "0")),
        password=os.getenv("REDIS_PASSWORD", None),
        decode_responses=True,
    )


def push_organic_chunk(r: redis.Redis, url: str):
    """
    Push an organic chunk URL to the queue (FIFO).
    """
    r.rpush(ORGANIC_QUEUE_KEY, url)


def push_synthetic_chunk(r: redis.Redis, url: str):
    """
    Push a synthetic chunk URL to the queue (FIFO).
    """
    r.rpush(SYNTHETIC_QUEUE_KEY, url)


def pop_organic_chunk(r: redis.Redis) -> str:
    """
    Pop the oldest organic chunk URL (FIFO).
    Returns a URL or None if queue is empty.
    """
    return r.lpop(ORGANIC_QUEUE_KEY)


def pop_synthetic_chunk(r: redis.Redis) -> str:
    """
    Pop the oldest synthetic chunk URL (FIFO).
    Returns a URL or None if queue is empty.
    """
    return r.lpop(SYNTHETIC_QUEUE_KEY)


def get_organic_queue_size(r: redis.Redis) -> int:
    return r.llen(ORGANIC_QUEUE_KEY)


def get_synthetic_queue_size(r: redis.Redis) -> int:
    return r.llen(SYNTHETIC_QUEUE_KEY)
