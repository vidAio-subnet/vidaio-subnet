import redis
import httpx
import asyncio
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException
from typing import Dict, Optional
from vidaio_subnet_core import CONFIG
from config import get_settings, logger
from models import TaskStatus, InsertOrganicUpscalingRequest, InsertOrganicCompressionRequest

redis_ttl = CONFIG.redis.redis_ttl

# Redis connection
def get_redis_connection(settings=Depends(get_settings)):
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD,
        decode_responses=True  # Automatically decode bytes to strings
    )

# Task management service
class TaskService:
    def __init__(self, redis_conn):
        self.redis = redis_conn
    
    def create_task(self, task_id: str, chunk_id: str, chunk_url: str, resolution_type: Optional[str], compression_type: Optional[str]):
        """Create a new task and store in Redis"""
        now = datetime.utcnow().isoformat()
        task_data = {
            "task_id": task_id,
            "chunk_id": chunk_id,
            "chunk_url": chunk_url,
            "resolution_type": resolution_type or "",
            "compression_type": compression_type or "",
            "status": TaskStatus.QUEUED,
            "created_at": now,
            "updated_at": now
        }
        
        # Store task data
        task_key = f"task:{task_id}"
        self.redis.hset(task_key, mapping=task_data)
        self.redis.expire(task_key, redis_ttl)
        
        # Add to task list for easier lookup
        self.redis.sadd("tasks", task_id)
        self.redis.expire("tasks", redis_ttl)
        
        return task_data
    
    def get_task(self, task_id: str):
        """Get task data from Redis"""
        task_key = f"task:{task_id}"
        task_data = self.redis.hgetall(task_key)
        
        if not task_data:
            return None
        
        return task_data
    
    def update_task_status(self, task_id: str, status: TaskStatus, progress: float = None):
        """Update task status in Redis"""
        task_key = f"task:{task_id}"
        
        # Check if task exists
        if not self.redis.exists(task_key):
            return False
        
        # Update status and timestamp
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if progress is not None:
            update_data["progress"] = progress
        
        self.redis.hset(task_key, mapping=update_data)
        self.redis.expire(task_key, redis_ttl)
        return True
    
    def link_task_to_result(self, task_id: str, original_video_url: str):
        """Link task to result for easy lookup"""
        self.redis.set(f"task_result:{task_id}", original_video_url, ex=redis_ttl)
    
    def get_result_url_for_task(self, task_id: str):
        """Get the original video URL associated with a task"""
        return self.redis.get(f"task_result:{task_id}")
    
    def cleanup_old_data(self, retention_days: int):
        """Clean up data older than the specified number of days"""
        logger.info(f"Starting cleanup of data older than {retention_days} days")
        cutoff_date = (datetime.utcnow() - timedelta(days=retention_days)).isoformat()
        
        # Get all task IDs
        all_tasks = self.redis.smembers("tasks")
        deleted_count = 0
        
        for task_id in all_tasks:
            task_key = f"task:{task_id}"
            task_data = self.redis.hgetall(task_key)
            
            # Skip if task doesn't exist
            if not task_data:
                continue
                
            # Check if task is old enough to delete
            created_at = task_data.get("created_at", "")
            if created_at < cutoff_date:
                # Get original video URL before deleting
                original_url = self.get_result_url_for_task(task_id)
                
                # Delete task data
                self.redis.delete(task_key)
                self.redis.srem("tasks", task_id)
                self.redis.delete(f"task_result:{task_id}")
                
                # Delete result data if exists
                if original_url:
                    self.redis.delete(f"result:{original_url}")
                
                deleted_count += 1
                
        logger.info(f"Cleanup completed: {deleted_count} old tasks deleted")
        return deleted_count

# Redis service client
class RedisServiceClient:
    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()
        self.endpoint = settings.REDIS_SERVICE_ENDPOINT
        self.timeout = settings.REQUEST_TIMEOUT
        self.max_retries = settings.MAX_RETRIES
        self.retry_delay = settings.RETRY_DELAY
    
    async def insert_organic_upscaling_chunk(self, url: str, chunk_id: str, task_id: str, resolution_type: str):
        """Insert chunk into organic upscaling queue via Redis service"""
        api_url = f"{self.endpoint}/api/insert_organic_upscaling_chunk"
        payload = InsertOrganicUpscalingRequest(
            url=url,
            chunk_id=chunk_id,
            task_id=task_id,
            resolution_type=resolution_type
        )
        
        return await self._make_request("POST", api_url, payload.dict())

    async def insert_organic_compression_chunk(self, url: str, chunk_id: str, task_id: str, compression_type: str):
        """Insert chunk into organic compression queue via Redis service"""
        api_url = f"{self.endpoint}/api/insert_organic_compression_chunk"
        payload = InsertOrganicCompressionRequest(
            url=url,
            chunk_id=chunk_id,
            task_id=task_id,
            compression_type=compression_type
        )
        
        return await self._make_request("POST", api_url, payload.dict())
    
    async def get_result(self, original_video_url: str):
        """Get processing result from Redis service"""
        api_url = f"{self.endpoint}/api/get_result"
        
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    # Use POST method to send JSON data properly
                    response = await client.post(
                        api_url,
                        json={"original_video_url": original_video_url},
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        return response.json()
                    else:
                        logger.error(f"Request failed: {response.status_code}, {response.text}")
                        raise HTTPException(status_code=response.status_code, detail=response.text)
                    
            except Exception as e:
                retry_count += 1
                if retry_count >= self.max_retries:
                    logger.error(f"Request failed after {self.max_retries} attempts: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Service request failed: {str(e)}")
                
                logger.warning(f"Retry {retry_count}/{self.max_retries}: {str(e)}")
                await asyncio.sleep(self.retry_delay)
        
        raise HTTPException(status_code=500, detail="Request failed after max retries")

    async def get_organic_compression_queue_size(self):
        """Get the size of the organic compression queue"""
        api_url = f"{self.endpoint}/api/get_organic_compression_queue_size"
        return await self._make_request("GET", api_url)

    async def get_organic_upscaling_queue_size(self):
        """Get the size of the organic upscaling queue"""
        api_url = f"{self.endpoint}/api/get_organic_upscaling_queue_size"
        return await self._make_request("GET", api_url)

    async def _make_request(self, method: str, url: str, json_data: Dict = None):
        """Make HTTP request with retry logic"""
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    if method == "GET":
                        response = await client.get(url)
                    else:
                        response = await client.post(url, json=json_data)
                    
                    if response.status_code == 200:
                        return response.json()
                    else:
                        logger.error(f"Request failed: {response.status_code}, {response.text}")
                        raise HTTPException(status_code=response.status_code, detail=response.text)
                    
            except Exception as e:
                retry_count += 1
                if retry_count >= self.max_retries:
                    logger.error(f"Request failed after {self.max_retries} attempts: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Service request failed: {str(e)}")
                
                logger.warning(f"Retry {retry_count}/{self.max_retries}: {str(e)}")
                await asyncio.sleep(self.retry_delay)
        
        raise HTTPException(status_code=500, detail="Request failed after max retries")

# Dependencies
def get_task_service(redis_conn=Depends(get_redis_connection)):
    return TaskService(redis_conn)

def get_redis_service(settings=Depends(get_settings)):
    return RedisServiceClient(settings)
