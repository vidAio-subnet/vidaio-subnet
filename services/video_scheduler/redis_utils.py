import redis
import json
from vidaio_subnet_core import CONFIG
from typing import Dict, List, Optional

REDIS_CONFIG = CONFIG.redis
REDIS_TTL = REDIS_CONFIG.redis_ttl #TTL in seconds

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

def push_organic_upscaling_chunk(r: redis.Redis, data: Dict[str, str]) -> None:
    """
    Push an organic upscaling chunk dictionary to the queue (FIFO).
    
    Args:
        r (redis.Redis): Redis connection
        data (Dict[str, str]): Organic upscaling chunk dictionary to push
    """
    r.rpush(REDIS_CONFIG.organic_upscaling_queue_key, json.dumps(data))
    r.expire(REDIS_CONFIG.organic_upscaling_queue_key, REDIS_TTL)
    print("Pushed organic upscaling chunk correctly in the Redis queue")

def push_organic_compression_chunk(r: redis.Redis, data: Dict[str, str]) -> None:
    """
    Push an organic compression chunk dictionary to the queue (FIFO).
    
    Args:
        r (redis.Redis): Redis connection
        data (Dict[str, str]): Organic compression chunk dictionary to push
    """
    r.rpush(REDIS_CONFIG.organic_compression_queue_key, json.dumps(data))
    r.expire(REDIS_CONFIG.organic_compression_queue_key, REDIS_TTL)
    print("Pushed organic compression chunk correctly in the Redis queue")

def push_5s_chunks(r: redis.Redis, data_list: List[Dict[str, str]]) -> None:

    r.rpush(REDIS_CONFIG.synthetic_5s_clip_queue_key, *[json.dumps(data) for data in data_list])
    print("Pushed all URLs correctly in the Redis queue")

def push_10s_chunks(r: redis.Redis, data_list: List[Dict[str, str]]) -> None:

    r.rpush(REDIS_CONFIG.synthetic_10s_clip_queue_key, *[json.dumps(data) for data in data_list])
    print("Pushed all URLs correctly in the Redis queue")

def push_20s_chunks(r: redis.Redis, data_list: List[Dict[str, str]]) -> None:

    r.rpush(REDIS_CONFIG.synthetic_20s_clip_queue_key, *[json.dumps(data) for data in data_list])
    print("Pushed all URLs correctly in the Redis queue")

def push_compression_chunks(r: redis.Redis, data_list: List[Dict[str, str]]) -> None:

    r.rpush(REDIS_CONFIG.synthetic_compression_queue_key, *[json.dumps(data) for data in data_list])
    print("Pushed all compression chunks correctly in the Redis queue")

def pop_organic_upscaling_chunk(r: redis.Redis) -> Optional[Dict[str, str]]:
    """
    Pop the oldest organic upscaling chunk dictionary (FIFO).
    Returns a dictionary or None if queue is empty.
    
    Args:
        r (redis.Redis): Redis connection

    Returns:
        Optional[Dict[str, str]]: The popped organic upscaling chunk or None if empty.
    """
    data = r.lpop(REDIS_CONFIG.organic_upscaling_queue_key)
    return json.loads(data) if data else None

def pop_organic_compression_chunk(r: redis.Redis) -> Optional[Dict[str, str]]:
    """
    Pop the oldest organic compression chunk dictionary (FIFO).
    Returns a dictionary or None if queue is empty.
    
    Args:
        r (redis.Redis): Redis connection

    Returns:
        Optional[Dict[str, str]]: The popped organic compression chunk or None if empty.
    """
    data = r.lpop(REDIS_CONFIG.organic_compression_queue_key)
    return json.loads(data) if data else None

# def pop_synthetic_chunk(r: redis.Redis) -> Optional[Dict[str, str]]:
#     """
#     Pop the oldest synthetic chunk dictionary (FIFO), and push it back to the end of the queue
#     to maintain the queue size. Returns a dictionary or None if the queue is empty.
    
#     Args:
#         r (redis.Redis): Redis connection

#     Returns:
#         Optional[Dict[str, str]]: The popped synthetic chunk or None if the queue is empty.
#     """
#     # Pop the oldest item from the queue
#     data = r.lpop(REDIS_CONFIG.synthetic_queue_key)
    
#     if data:
#         chunk = json.loads(data)
#         # Push the chunk back to maintain queue size
#         push_synthetic_chunk(r, chunk)
#         return chunk
        
#     return None

def pop_5s_chunk(r: redis.Redis) -> Optional[Dict[str, str]]:
    """
    Pop the oldest 5s chunk dictionary (FIFO).
    Returns a dictionary or None if queue is empty.
    
    Args:
        r (redis.Redis): Redis connection

    Returns:
        Optional[Dict[str, str]]: The popped 5s chunk or None if empty.
    """
    data = r.lpop(REDIS_CONFIG.synthetic_5s_clip_queue_key)
    return json.loads(data) if data else None

# def pop_5s_chunk(r: redis.Redis) -> Optional[Dict[str, str]]:
#     """
#     Pop the oldest 5s chunk dictionary (FIFO), and push it back to the end of the queue
#     to maintain the queue size. Returns a dictionary or None if the queue is empty.
    
#     Args:
#         r (redis.Redis): Redis connection

#     Returns:
#         Optional[Dict[str, str]]: The popped 5s chunk or None if the queue is empty.
#     """
#     # Pop the oldest item from the queue
#     data = r.lpop(REDIS_CONFIG.synthetic_5s_clip_queue_key)
    
#     if data:
#         chunk = json.loads(data)
#         # Push the chunk back to maintain queue size
#         push_5s_chunks(r, [chunk])
#         return chunk
        
#     return None

def pop_10s_chunk(r: redis.Redis) -> Optional[Dict[str, str]]:
    """
    Pop the oldest 10s chunk dictionary (FIFO).
    Returns a dictionary or None if queue is empty.
    
    Args:
        r (redis.Redis): Redis connection

    Returns:
        Optional[Dict[str, str]]: The popped 10s chunk or None if empty.
    """
    data = r.lpop(REDIS_CONFIG.synthetic_10s_clip_queue_key)
    return json.loads(data) if data else None

def pop_20s_chunk(r: redis.Redis) -> Optional[Dict[str, str]]:

    data = r.lpop(REDIS_CONFIG.synthetic_20s_clip_queue_key)
    return json.loads(data) if data else None

def pop_compression_chunk(r: redis.Redis) -> Optional[Dict[str, str]]:

    data = r.lpop(REDIS_CONFIG.synthetic_compression_queue_key)
    return json.loads(data) if data else None

def get_organic_upscaling_queue_size(r: redis.Redis) -> int:
    """
    Get the size of the organic upscaling queue.
    
    Args:
        r (redis.Redis): Redis connection
        
    Returns:
        int: Size of the organic upscaling queue.
    """
    return r.llen(REDIS_CONFIG.organic_upscaling_queue_key)

def get_organic_compression_queue_size(r: redis.Redis) -> int:
    """
    Get the size of the organic compression queue.
    
    Args:
        r (redis.Redis): Redis connection
    """
    return r.llen(REDIS_CONFIG.organic_compression_queue_key)

def get_5s_queue_size(r: redis.Redis) -> int:

    return r.llen(REDIS_CONFIG.synthetic_5s_clip_queue_key)

def get_10s_queue_size(r: redis.Redis) -> int:

    return r.llen(REDIS_CONFIG.synthetic_10s_clip_queue_key)

def get_20s_queue_size(r: redis.Redis) -> int:

    return r.llen(REDIS_CONFIG.synthetic_20s_clip_queue_key)

def get_compression_queue_size(r: redis.Redis) -> int:
    """
    Get the size of the compression queue.
    
    Args:
        r (redis.Redis): Redis connection
    """

    return r.llen(REDIS_CONFIG.synthetic_compression_queue_key)

def push_pexels_video_ids(r: redis.Redis, data_list: List[Dict[str, str]]) -> None:
    """
    Push multiple Pexels video IDs to the queue (FIFO).

    Args:
        r (redis.Redis): Redis connection instance.
        id_list (List[int]): List of Pexels video IDs to push.
    """
    r.rpush(REDIS_CONFIG.pexels_video_ids_key, *[json.dumps(data) for data in data_list])
    print("Pushed all Pexels video IDs with task_type correctly in the Redis queue")

def pop_pexels_video_id(r: redis.Redis) -> int:
    """
    Pop pexels video id from queue
    """
    data = r.lpop(REDIS_CONFIG.pexels_video_ids_key)
    return json.loads(data) if data else None

def get_pexels_queue_size(r: redis.Redis) -> int:
    """
    Get the size of the pexels video ids queue.
    
    Args:
        r (redis.Redis): Redis connection
        
    Returns:
        int: Size of the pexels video ids queue.
    """
    return r.llen(REDIS_CONFIG.pexels_video_ids_key)

def push_youtube_video_ids(r: redis.Redis, data_list: List[Dict[str, str]]) -> None:
    """
    Push multiple YouTube video IDs to the queue (FIFO).

    Args:
        r (redis.Redis): Redis connection instance.
        data_list (List[Dict[str, str]]): List of YouTube video data to push.
    """
    r.rpush(REDIS_CONFIG.youtube_video_ids_key, *[json.dumps(data) for data in data_list])
    print("Pushed all YouTube video IDs with task_type correctly in the Redis queue")

def pop_youtube_video_id(r: redis.Redis) -> Optional[Dict[str, str]]:
    """
    Pop YouTube video id from queue
    
    Args:
        r (redis.Redis): Redis connection
        
    Returns:
        Optional[Dict[str, str]]: YouTube video data or None if empty
    """
    data = r.lpop(REDIS_CONFIG.youtube_video_ids_key)
    return json.loads(data) if data else None

def get_youtube_queue_size(r: redis.Redis) -> int:
    """
    Get the size of the youtube video ids queue.
    
    Args:
        r (redis.Redis): Redis connection
        
    Returns:
        int: Size of the youtube video ids queue.
    """
    return r.llen(REDIS_CONFIG.youtube_video_ids_key)

def set_scheduler_ready(r: redis.Redis, is_ready: bool) -> None:
    """
    Set the scheduler readiness flag in Redis.
    
    Args:
        r (redis.Redis): Redis connection
        is_ready (bool): Whether the scheduler is ready
    """
    r.set("scheduler_ready", "true" if is_ready else "false")

def is_scheduler_ready(r: redis.Redis) -> bool:
    """
    Check if the scheduler is ready by checking the flag in Redis.
    
    Args:
        r (redis.Redis): Redis connection
        
    Returns:
        bool: True if scheduler is ready, False otherwise
    """
    flag = r.get("scheduler_ready")
    return flag == "true" if flag else False