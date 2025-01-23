import redis
from video_subnet_core import CONFIG
from typing import List
REDIS_CONFIG = CONFIG.redis


def get_redis_connection():
    """
    Create and return a Redis connection.
    Adjust host, port, and password if needed.
    """
    return redis.Redis(
        host=REDIS_CONFIG.host,
        port=REDIS_CONFIG.port,
        db=REDIS_CONFIG.db,
        decode_responses=True,
    )


def push_organic_chunk(r: redis.Redis, url: str):
    """
    Push an organic chunk URL to the queue (FIFO).
    """
    r.rpush(REDIS_CONFIG.organic_queue_key, url)


def push_synthetic_chunk(r: redis.Redis, url: str):
    """
    Push a synthetic chunk URL to the queue (FIFO).
    """
    r.rpush(REDIS_CONFIG.synthetic_queue_key, url)


def push_synthetic_chunks(r: redis.Redis, urls: List[str]):
    """
    Push multiple synthetic chunk URLs to the queue (FIFO).
    
    Args:
        r (redis.Redis): Redis connection
        urls (List[str]): List of synthetic chunk URLs to push
    """
    r.rpush(REDIS_CONFIG.synthetic_queue_key, *urls)


def pop_organic_chunk(r: redis.Redis) -> str:
    """
    Pop the oldest organic chunk URL (FIFO).
    Returns a URL or None if queue is empty.
    """
    return r.lpop(REDIS_CONFIG.organic_queue_key)


def pop_synthetic_chunk(r: redis.Redis) -> str:
    """
    Pop the oldest synthetic chunk URL (FIFO).
    Returns a URL or None if queue is empty.
    """
    return r.lpop(REDIS_CONFIG.synthetic_queue_key)


def get_organic_queue_size(r: redis.Redis) -> int:
    return r.llen(REDIS_CONFIG.organic_queue_key)


def get_synthetic_queue_size(r: redis.Redis) -> int:
    return r.llen(REDIS_CONFIG.synthetic_queue_key)
