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
