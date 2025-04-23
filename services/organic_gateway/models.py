from enum import Enum
from typing import Optional, Literal
from pydantic import BaseModel

# Task status enum
class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# API Models
class UpscaleRequest(BaseModel):
    chunk_id: str
    chunk_url: str
    resolution_type: Literal["SD2HD", "SD24K", "HD24K"]

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

# Redis Service Models
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

class UpdateTaskStatusRequest(BaseModel):
    status: TaskStatus
    original_video_url: Optional[str] = None
    progress: Optional[float] = None
