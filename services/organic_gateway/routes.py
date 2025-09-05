import uuid
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Optional

from config import logger
from models import (
    UpscaleRequest, UpscaleResponse, CompressionRequest, CompressionResponse, StatusResponse, 
    ResultResponse, TaskStatus, UpdateTaskStatusRequest, TaskCountResponse
)
from services import get_task_service, get_redis_service, TaskService
from config import get_settings

router = APIRouter()

@router.get("/ping")
async def ping():
    """Health check endpoint"""
    return {"status": "ok", "message": "Video processing service is running"}

@router.post("/upscale", response_model=UpscaleResponse)
async def upscale(
    request: UpscaleRequest,
    background_tasks: BackgroundTasks,
    task_service: TaskService = Depends(get_task_service),
    redis_service = Depends(get_redis_service)
):
    """
    Submit a video chunk for upscaling
    
    This endpoint receives a chunk ID and URL, generates a task ID,
    and submits the chunk to the processing queue.
    """
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Create task in Redis
    task_service.create_task(
        task_id=task_id,
        chunk_id=request.chunk_id,
        chunk_url=request.chunk_url,
        resolution_type=request.resolution_type,
        compression_type=None
    )
    
    # Submit to Redis service in background to avoid blocking
    background_tasks.add_task(
        redis_service.insert_organic_upscaling_chunk,
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

@router.get("/task_count", response_model=TaskCountResponse)
async def get_task_count(
    redis_service = Depends(get_redis_service)
):
    """
    Get the number of tasks in the queue
    """

    compression_count =await redis_service.get_organic_compression_queue_size()
    upscaling_count = await redis_service.get_organic_upscaling_queue_size()
    total_count = compression_count + upscaling_count

    return TaskCountResponse(compression_count=compression_count, upscaling_count=upscaling_count, total_count=total_count)

@router.post("/compress", response_model=CompressionResponse)
async def compression(
    request: CompressionRequest,
    background_tasks: BackgroundTasks,
    task_service: TaskService = Depends(get_task_service),
    redis_service = Depends(get_redis_service)
):
    """
    Submit a video chunk for compression
    
    This endpoint receives a chunk ID and URL, generates a task ID,
    and submits the chunk to the compression processing queue.
    """
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Create task in Redis
    task_service.create_task(
        task_id=task_id,
        chunk_id=request.chunk_id,
        chunk_url=request.chunk_url,
        resolution_type=None,
        compression_type=request.compression_type
    )
    
    # Submit to Redis service in background to avoid blocking
    background_tasks.add_task(
        redis_service.insert_organic_compression_chunk,
        url=request.chunk_url,
        chunk_id=request.chunk_id,
        task_id=task_id,
        compression_type=request.compression_type
    )
    
    return CompressionResponse(
        task_id=task_id,
        status=TaskStatus.QUEUED,
        message="Compression task has been queued for processing"
    )

@router.get("/status/{task_id}", response_model=StatusResponse)
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

@router.get("/result/{task_id}", response_model=ResultResponse)
async def get_task_result(
    task_id: str,
    task_service: TaskService = Depends(get_task_service),
    redis_service = Depends(get_redis_service)
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
        
    except HTTPException as e:
        logger.error(f"HTTP error retrieving result for task {task_id}: {str(e.detail)}")
        return ResultResponse(
            task_id=task_id,
            status=status,
            original_video_url=original_video_url,
            message=f"Service error: {str(e.detail)}"
        )
        
    except Exception as e:
        logger.error(f"Error retrieving result for task {task_id}: {str(e)}")
        return ResultResponse(
            task_id=task_id,
            status=status,
            original_video_url=original_video_url,
            message=f"Error retrieving result: {str(e)}"
        )

# Admin routes
admin_router = APIRouter(prefix="/admin")

@admin_router.post("/task/{task_id}/status")
async def update_task_status(
    task_id: str,
    request: UpdateTaskStatusRequest,
    task_service: TaskService = Depends(get_task_service)
):
    """
    Update task status (admin endpoint)
    
    This endpoint allows the validator process to update task status.
    """

    status = request.status
    progress = request.progress
    original_video_url = request.original_video_url

    success = task_service.update_task_status(task_id, status, progress)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    # If task is completed and original URL is provided, link it for result lookup
    if status == TaskStatus.COMPLETED and original_video_url:
        task_service.link_task_to_result(task_id, original_video_url)
    
    return {"message": f"Task {task_id} status updated to {status}"}

@admin_router.post("/cleanup")
async def trigger_cleanup(
    task_service: TaskService = Depends(get_task_service),
    settings = Depends(get_settings)
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
