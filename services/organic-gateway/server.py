from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, Any, List, Literal
import redis
import uuid
import httpx
import asyncio
import logging
import json
from datetime import datetime, timedelta
from functools import lru_cache
import os
from enum import Enum
import time
from apscheduler.schedulers.background import BackgroundScheduler
from vidaio_subnet_core import CONFIG

REDIS_CONFIG = CONFIG.redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("validator-api")

# Create FastAPI app
app = FastAPI(
    title="Video Upscaling Validator API",
    description="API for handling video upscaling tasks",
    version="1.0.0"
)

# Configuration
class Settings:
    def __init__(self):
        self.REDIS_HOST = REDIS_CONFIG.host
        self.REDIS_PORT = REDIS_CONFIG.port
        self.REDIS_DB = REDIS_CONFIG.db
        self.REDIS_PASSWORD = None
        self.REQUEST_TIMEOUT = 30
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 2
        self.REDIS_SERVICE_ENDPOINT = CONFIG.video_scheduler.host + ':' + str(CONFIG.video_scheduler.port)
        self.DATA_RETENTION_DAYS = 3
        self.CLEANUP_INTERVAL_HOURS = 72 # hours

@lru_cache()
def get_settings():
    return Settings()

# Task status enum
class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Models
class UpscaleRequest(BaseModel):
    chunk_id: str
    chunk_url: str
    resolution_type: Literal["HD", "4K", "8K"]

class UpscaleResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str

class StatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str
    created_at: str
    updated_at: str
    progress: Optional[float] = None

class ResultResponse(BaseModel):
    task_id: str
    status: TaskStatus
    original_video_url: str
    processed_video_url: Optional[str] = None
    score: Optional[float] = None
    message: str

class InsertOrganicRequest(BaseModel):
    url: str
    chunk_id: str
    task_id: str
    resolution_type: str

class InsertResultRequest(BaseModel):
    processed_video_url: str
    original_video_url: str
    score: float
    task_id: str

# Redis connection
def get_redis_connection(settings: Settings = Depends(get_settings)):
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
    
    def create_task(self, task_id: str, chunk_id: str, chunk_url: str, resolution_type: str):
        """Create a new task and store in Redis"""
        now = datetime.utcnow().isoformat()
        task_data = {
            "task_id": task_id,
            "chunk_id": chunk_id,
            "chunk_url": chunk_url,
            "resolution_type": resolution_type,
            "status": TaskStatus.QUEUED,
            "created_at": now,
            "updated_at": now
        }
        
        # Store task data
        task_key = f"task:{task_id}"
        self.redis.hset(task_key, mapping=task_data)
        
        # Add to task list for easier lookup
        self.redis.sadd("tasks", task_id)
        
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
        return True
    
    def link_task_to_result(self, task_id: str, original_video_url: str):
        """Link task to result for easy lookup"""
        self.redis.set(f"task_result:{task_id}", original_video_url)
    
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
    def __init__(self, settings: Settings = Depends(get_settings)):
        self.endpoint = settings.REDIS_SERVICE_ENDPOINT
        self.timeout = settings.REQUEST_TIMEOUT
        self.max_retries = settings.MAX_RETRIES
        self.retry_delay = settings.RETRY_DELAY
    
    async def insert_organic_chunk(self, url: str, chunk_id: str, task_id: str, resolution_type: str):
        """Insert chunk into organic queue via Redis service"""
        api_url = f"{self.endpoint}/api/insert_organic_chunk"
        payload = InsertOrganicRequest(
            url=url,
            chunk_id=chunk_id,
            task_id=task_id,
            resolution_type=resolution_type
        )
        
        return await self._make_request("POST", api_url, payload.dict())
    
    async def get_result(self, original_video_url: str):
        """Get processing result from Redis service"""
        api_url = f"{self.endpoint}/api/get_result/{original_video_url}"
        
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

def get_redis_service(settings: Settings = Depends(get_settings)):
    return RedisServiceClient(settings)

# Data cleanup scheduler
scheduler = BackgroundScheduler()

def setup_data_cleanup_job(settings: Settings = Depends(get_settings)):
    """Setup scheduled job for data cleanup"""
    redis_conn = get_redis_connection(settings)
    task_service = TaskService(redis_conn)
    
    def cleanup_job():
        try:
            task_service.cleanup_old_data(settings.DATA_RETENTION_DAYS)
        except Exception as e:
            logger.error(f"Error during scheduled data cleanup: {str(e)}")
    
    # Schedule job to run at the specified interval
    scheduler.add_job(
        cleanup_job, 
        'interval', 
        hours=settings.CLEANUP_INTERVAL_HOURS,
        id='data_cleanup_job',
        replace_existing=True
    )
    
    # Run once at startup
    cleanup_job()
    
    if not scheduler.running:
        scheduler.start()

# Endpoints
@app.get("/ping")
async def ping():
    """Health check endpoint"""
    return {"status": "ok", "message": "Validator service is running"}

@app.post("/upscale", response_model=UpscaleResponse)
async def upscale(
    request: UpscaleRequest,
    background_tasks: BackgroundTasks,
    task_service: TaskService = Depends(get_task_service),
    redis_service: RedisServiceClient = Depends(get_redis_service)
):
    """
    Submit a video chunk for upscaling
    
    This endpoint receives a chunk ID and URL, generates a task ID,
    and submits the chunk to the processing queue.
    """
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Create task in Redi00s
    task_service.create_task(
        task_id=task_id,
        chunk_id=request.chunk_id,
        chunk_url=request.chunk_url,
        resolution_type=request.resolution_type
    )
    
    # Submit to Redis service in background to avoid blocking
    background_tasks.add_task(
        redis_service.insert_organic_chunk,
        url=request.chunk_url,
        chunk_id=request.chunk_id,
        task_id=task_id,
        resolution_type=request.resolution_type
    )
    
    return UpscaleResponse(
        task_id=task_id,
        status=TaskStatus.QUEUED,
        message="Task has been queued for processing"
    )

@app.get("/status/{task_id}", response_model=StatusResponse)
async def get_task_status(
    task_id: str,
    task_service: TaskService = Depends(get_task_service)
):
    """
    Get the status of a processing task
    
    This endpoint checks Redis for the current status of a task.
    """
    task_data = task_service.get_task(task_id)
    
    if not task_data:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    progress = float(task_data.get("progress", 0)) if "progress" in task_data else None
    
    return StatusResponse(
        task_id=task_id,
        status=TaskStatus(task_data["status"]),
        message=f"Task is {task_data['status']}",
        created_at=task_data["created_at"],
        updated_at=task_data["updated_at"],
        progress=progress
    )

@app.get("/result/{task_id}", response_model=ResultResponse)
async def get_task_result(
    task_id: str,
    task_service: TaskService = Depends(get_task_service),
    redis_service: RedisServiceClient = Depends(get_redis_service)
):
    """
    Get the result of a completed task
    
    This endpoint retrieves the processing result for a completed task.
    """
    # First check if task exists and get its status
    task_data = task_service.get_task(task_id)
    
    if not task_data:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    status = TaskStatus(task_data["status"])
    
    # If task is not completed, return current status
    if status != TaskStatus.COMPLETED:
        return ResultResponse(
            task_id=task_id,
            status=status,
            original_video_url=task_data["chunk_url"],
            message=f"Task is {status}. Results not available yet."
        )
    
    # Get the original video URL associated with this task
    original_video_url = task_service.get_result_url_for_task(task_id)
    
    if not original_video_url:
        original_video_url = task_data["chunk_url"]
    
    # Get result from Redis service
    try:
        result = await redis_service.get_result(original_video_url)
        
        if "message" in result and result.get("message") == "No result found for this video":
            return ResultResponse(
                task_id=task_id,
                status=status,
                original_video_url=original_video_url,
                message="Task is completed but result data is not available"
            )
        
        return ResultResponse(
            task_id=task_id,
            status=status,
            original_video_url=result["original_video_url"],
            processed_video_url=result["processed_video_url"],
            score=result["score"],
            message="Task completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Error retrieving result for task {task_id}: {str(e)}")
        return ResultResponse(
            task_id=task_id,
            status=status,
            original_video_url=original_video_url,
            message=f"Error retrieving result: {str(e)}"
        )

# Admin endpoints for updating task status (would be used by the validator process)
@app.post("/admin/task/{task_id}/status")
async def update_task_status(
    task_id: str,
    status: TaskStatus,
    original_video_url: Optional[str] = None,
    progress: Optional[float] = None,
    task_service: TaskService = Depends(get_task_service)
):
    """
    Update task status (admin endpoint)
    
    This endpoint allows the validator process to update task status.
    """
    success = task_service.update_task_status(task_id, status, progress)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    # If task is completed and original URL is provided, link it for result lookup
    if status == TaskStatus.COMPLETED and original_video_url:
        task_service.link_task_to_result(task_id, original_video_url)
    
    return {"message": f"Task {task_id} status updated to {status}"}

# Manual trigger for data cleanup
@app.post("/admin/cleanup")
async def trigger_cleanup(
    task_service: TaskService = Depends(get_task_service),
    settings: Settings = Depends(get_settings)
):
    """
    Manually trigger cleanup of old data
    
    This endpoint allows administrators to manually trigger cleanup of old data.
    """
    try:
        deleted_count = task_service.cleanup_old_data(settings.DATA_RETENTION_DAYS)
        return {
            "message": f"Cleanup completed successfully",
            "deleted_count": deleted_count,
            "retention_days": settings.DATA_RETENTION_DAYS
        }
    except Exception as e:
        logger.error(f"Error during manual data cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

# Error handlers
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return {"detail": "An internal server error occurred"}

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    settings = get_settings()
    setup_data_cleanup_job(settings)
    logger.info(f"Data cleanup scheduled to run every {settings.CLEANUP_INTERVAL_HOURS} hours")
    logger.info(f"Data retention period set to {settings.DATA_RETENTION_DAYS} days")

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Cleanup scheduler shut down")


if __name__ == "__main__":
    
    import uvicorn
    host = CONFIG.organic_gateway.host
    port = CONFIG.organic_gateway.port
    uvicorn.run(app, host=host, port=port)