import redis
import json
from vidaio_subnet_core import CONFIG
from typing import Dict, List, Optional

REDIS_CONFIG = CONFIG.redis

def get_redis_connection() -> redis.Redis:
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

def push_organic_chunk(r: redis.Redis, data: Dict[str, str]) -> None:
    """
    Push an organic chunk dictionary to the queue (FIFO).
    
    Args:
        r (redis.Redis): Redis connection
        data (Dict[str, str]): Organic chunk dictionary to push
    """
    r.rpush(REDIS_CONFIG.organic_queue_key, json.dumps(data))

def push_synthetic_chunk(r: redis.Redis, data: Dict[str, str]) -> None:
    """
    Push a synthetic chunk dictionary to the queue (FIFO).
    
    Args:
        r (redis.Redis): Redis connection
        data (Dict[str, str]): Synthetic chunk dictionary to push
    """
    r.rpush(REDIS_CONFIG.synthetic_queue_key, json.dumps(data))

def push_synthetic_chunks(r: redis.Redis, data_list: List[Dict[str, str]]) -> None:
    """
    Push multiple synthetic chunk dictionaries to the queue (FIFO).
    
    Args:
        r (redis.Redis): Redis connection
        data_list (List[Dict[str, str]]): List of synthetic chunk dictionaries to push
    """
    r.rpush(REDIS_CONFIG.synthetic_queue_key, *[json.dumps(data) for data in data_list])
    print("Pushed all URLs correctly in the Redis queue")

def pop_organic_chunk(r: redis.Redis) -> Optional[Dict[str, str]]:
    """
    Pop the oldest organic chunk dictionary (FIFO).
    Returns a dictionary or None if queue is empty.
    
    Args:
        r (redis.Redis): Redis connection

    Returns:
        Optional[Dict[str, str]]: The popped organic chunk or None if empty.
    """
    data = r.lpop(REDIS_CONFIG.organic_queue_key)
    chunk = json.loads(data) if data else None
    r.rpush(REDIS_CONFIG.synthetic_queue_key, json.dumps(chunk))
    return chunk

def pop_synthetic_chunk(r: redis.Redis) -> Optional[Dict[str, str]]:
    """
    Pop the oldest synthetic chunk dictionary (FIFO).
    Returns a dictionary or None if queue is empty.
    
    Args:
        r (redis.Redis): Redis connection

    Returns:
        Optional[Dict[str, str]]: The popped synthetic chunk or None if empty.
    """
    data = r.lpop(REDIS_CONFIG.synthetic_queue_key)
    return json.loads(data) if data else None

def get_organic_queue_size(r: redis.Redis) -> int:
    """
    Get the size of the organic queue.
    
    Args:
        r (redis.Redis): Redis connection
        
    Returns:
        int: Size of the organic queue.
    """
    return r.llen(REDIS_CONFIG.organic_queue_key)

def get_synthetic_queue_size(r: redis.Redis) -> int:
    """
    Get the size of the synthetic queue.
    
    Args:
        r (redis.Redis): Redis connection
        
    Returns:
        int: Size of the synthetic queue.
    """
    return r.llen(REDIS_CONFIG.synthetic_queue_key)
