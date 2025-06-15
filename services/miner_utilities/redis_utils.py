import redis
import json
import time
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

def schedule_file_deletion(object_name: str) -> bool:
    """
    Schedule a file for deletion after the specified time period.
    
    Args:
        object_name: The name of the object in MinIO to delete
        
    Returns:
        bool: True if scheduling was successful, False otherwise
    """
    try:
        redis_conn = get_redis_connection()
        deletion_time = int(time.time()) + REDIS_CONFIG.delete_after_second
        
        file_info = {
            "object_name": object_name,
            "deletion_time": deletion_time
        }
        
        # Add to a sorted set with score as deletion time
        redis_conn.zadd(REDIS_CONFIG.file_deletioin_key, {json.dumps(file_info): deletion_time})
        return True
    except Exception as e:
        print(f"Error scheduling file deletion: {e}")
        return False

def get_files_to_delete() -> List[Dict]:
    """
    Get all files that are due for deletion (current time >= deletion time)
    
    Returns:
        List of dictionaries containing file information
    """
    try:
        redis_conn = get_redis_connection()
        current_time = int(time.time())
        
        # Get all items with score <= current_time
        items = redis_conn.zrangebyscore(REDIS_CONFIG.file_deletioin_key, 0, current_time)
        
        files_to_delete = []
        for item in items:
            file_info = json.loads(item)
            files_to_delete.append(file_info)
            
        # Remove these items from the queue
        if items:
            redis_conn.zrem(REDIS_CONFIG.file_deletioin_key, *items)
            
        return files_to_delete
    except Exception as e:
        print(f"Error getting files to delete: {e}")
        return []

def get_gpu_count():
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        return device_count
    except Exception as e:
        print(f"Error getting available GPUs: {e}")
        return 0

TOTAL_GPU_COUNT = get_gpu_count()

def init_gpus():
    """
    Initialize GPU states in Redis.
    Set all GPUs to available (0).
    """

    redis_conn = get_redis_connection()
    for i in range(TOTAL_GPU_COUNT):
        redis_conn.set(f"gpu:{i}", 0)  # 0 = available, 1 = busy

def get_total_gpu_count():
    return TOTAL_GPU_COUNT

def acquire_gpu(timeout=1) -> Optional[int]:
    """
    Acquire a GPU for processing.
    
    Args:
        timeout: Time to wait before retrying to acquire a GPU.
        
    Returns:
        int: Index of the acquired GPU, or None if no GPU is available.
    """
    redis_conn = get_redis_connection()
    while True:
        for i in range(TOTAL_GPU_COUNT):
            if redis_conn.get(f"gpu:{i}") == "0":
                if redis_conn.setnx(f"gpu:{i}:lock", 1):  # atomic lock
                    redis_conn.set(f"gpu:{i}", 1)
                    redis_conn.delete(f"gpu:{i}:lock")
                    # if i > 0:
                    #     time.sleep(3)
                    return i
        time.sleep(timeout)

def release_gpu(gpu_index: int):
    """
    Release a GPU after processing.
    
    Args:
        gpu_index: Index of the GPU to release.
    """
    redis_conn = get_redis_connection()
    redis_conn.set(f"gpu:{gpu_index}", 0)

def schedule_local_file_deletion(filepath: str, delay_seconds : int) -> bool:
    try:
        redis_conn = get_redis_connection()
        deletion_time = int(time.time()) + delay_seconds
        
        file_info = {
            "filepath": filepath,
            "deletion_time": deletion_time
        }
        
        # Add to a sorted set with score as deletion time
        redis_conn.zadd("local_file_deletion_queue", {json.dumps(file_info): deletion_time})
        return True
    except Exception as e:
        print(f"Error scheduling file deletion: {e}")
        return False

def get_local_files_to_delete() -> List[Dict]:
    try:
        redis_conn = get_redis_connection()
        current_time = int(time.time())
        
        # Get all items with score <= current_time
        items = redis_conn.zrangebyscore("local_file_deletion_queue", 0, current_time)
        
        files_to_delete = []
        for item in items:
            file_info = json.loads(item)
            files_to_delete.append(file_info)
            
        # Remove these items from the queue
        if items:
            redis_conn.zrem("local_file_deletion_queue", *items)
            
        return files_to_delete
    except Exception as e:
        print(f"Error getting files to delete: {e}")
        return []