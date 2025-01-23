import redis
from video_subnet_core import CONFIG
from typing import Dict, List
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


def push_organic_chunk(r: redis.Redis, data: Dict[str, str]):
    """
    Push an organic chunk dictionary to the queue (FIFO).
    """
    r.rpush(REDIS_CONFIG.organic_queue_key, str(data))


def push_synthetic_chunk(r: redis.Redis, data: Dict[str, str]):
    """
    Push a synthetic chunk dictionary to the queue (FIFO).
    """
    r.rpush(REDIS_CONFIG.synthetic_queue_key, str(data))


def push_synthetic_chunks(r: redis.Redis, data_list: List[Dict[str, str]]):
    """
    Push multiple synthetic chunk dictionaries to the queue (FIFO).
    
    Args:
        r (redis.Redis): Redis connection
        data_list (List[Dict[str, str]]): List of synthetic chunk dictionaries to push
    """
    r.rpush(REDIS_CONFIG.synthetic_queue_key, *[str(data) for data in data_list])


def pop_organic_chunk(r: redis.Redis) -> Dict[str, str]:
    """
    Pop the oldest organic chunk dictionary (FIFO).
    Returns a dictionary or None if queue is empty.
    """
    data = r.lpop(REDIS_CONFIG.organic_queue_key)
    return eval(data) if data else None


def pop_synthetic_chunk(r: redis.Redis) -> Dict[str, str]:
    """
    Pop the oldest synthetic chunk dictionary (FIFO).
    Returns a dictionary or None if queue is empty.
    """
    data = r.lpop(REDIS_CONFIG.synthetic_queue_key)
    return eval(data) if data else None


def get_organic_queue_size(r: redis.Redis) -> int:
    return r.llen(REDIS_CONFIG.organic_queue_key)


def get_synthetic_queue_size(r: redis.Redis) -> int:
    return r.llen(REDIS_CONFIG.synthetic_queue_key)
