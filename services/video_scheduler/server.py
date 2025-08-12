from fastapi import FastAPI, HTTPException, Request
import time
from pydantic import BaseModel
from vidaio_subnet_core import CONFIG
from typing import Optional, List
from fastapi.responses import JSONResponse

from redis_utils import (
    get_redis_connection,
    push_organic_chunk,
    pop_organic_chunk,
    get_organic_queue_size,
    get_5s_queue_size,
    get_10s_queue_size,
    get_20s_queue_size,
    pop_5s_chunk,
    pop_10s_chunk,
    pop_20s_chunk,
    is_scheduler_ready,
    pop_compression_chunk,
)

app = FastAPI()

class InsertOrganicRequest(BaseModel):
    url: str
    chunk_id: str
    task_id: str
    resolution_type: str

class InsertResultRequest(BaseModel):
    processed_video_url: str
    original_video_url: str
    score: Optional[float] = None
    task_id: str 

class ResultRequest(BaseModel):
    original_video_url: str

class CompressionChunkRequest(BaseModel):
    num_needed: int

class SyntheticChunkRequest(BaseModel):
    content_lengths: Optional[List[int]] = []

@app.post("/api/insert_organic_chunk")
def api_insert_organic_chunk(payload: InsertOrganicRequest):
    """
    Insert an organic video URL into the organic queue.
    """
    r = get_redis_connection()
    data = {
        "url": payload.url,
        "chunk_id": payload.chunk_id,
        "task_id": payload.task_id,
        "resolution_type": payload.resolution_type
    }
    push_organic_chunk(r, data)
    return {"message": "Organic chunk inserted"}


# @app.get("/api/get_prioritized_chunk")
# def api_get_prioritized_chunk():
#     """
#     Retrieve the highest-priority chunk:
#     1) Pop the oldest organic chunk, if available.
#     2) Otherwise, pop the oldest synthetic chunk.
#     """
#     print("got the request correctly")
#     r = get_redis_connection()
#     chunk = pop_organic_chunk(r)
#     print("organic queue is empty, checking synthetic queue...")
#     if not chunk:
#         chunk = pop_synthetic_chunk(r)
#         print(f"processing synthetic queue. {chunk}")
#     if not chunk:
#         return {"message": "No chunks available"}
#     return {"chunk": chunk}


@app.post("/api/get_synthetic_chunks")
def api_get_synthetic_chunks(request_data: SyntheticChunkRequest):

    print(f"Processing synthetic chunk request for durations: {request_data.content_lengths}")
    
    redis_conn = get_redis_connection()
    chunks = []
    
    for content_length in request_data.content_lengths:
        chunk = retrieve_chunk_with_retry(redis_conn, content_length)
        chunks.append(chunk)
    
    # Filter out None values
    valid_chunks = [chunk for chunk in chunks if chunk is not None]
    
    if not valid_chunks:
        print("No valid chunks available after retries")
        return JSONResponse(
            status_code=404,
            content={"message": "No chunks available", "status": "error"}
        )
    
    print(f"Successfully retrieved {len(valid_chunks)} chunks")
    return {"chunks": chunks, "status": "success"}

@app.post("/api/get_compression_chunks")
def api_get_compression_chunks(request_data: CompressionChunkRequest):
    """
    Retrieve compression chunks from the Redis queue.
    """
    redis_conn = get_redis_connection()
    chunks = []

    for i in range(request_data.num_needed):
        chunk = pop_compression_chunk(redis_conn)
        chunks.append(chunk)
    
    valid_chunks = [chunk for chunk in chunks if chunk is not None]
    
    if not valid_chunks:
        print("No valid chunks available after retries")
        return JSONResponse(
            status_code=404,
            content={"message": "No chunks available", "status": "error"}
        )
    
    print(f"Successfully retrieved {len(valid_chunks)} chunks")
    return {"chunks": valid_chunks, "status": "success"}

def retrieve_chunk_with_retry(redis_conn, content_length: int, max_retries: int = 3, retry_delay: int = 20):
   
    chunk_type_map = {
        5: ("5-second", pop_5s_chunk),
        10: ("10-second", pop_10s_chunk),
        20: ("20-second", pop_20s_chunk)
    }
    
    if content_length not in chunk_type_map:
        print(f"Unsupported content length requested: {content_length}")
        return None
    
    chunk_name, pop_function = chunk_type_map[content_length]
    
    for attempt in range(1, max_retries + 1):
        chunk = pop_function(redis_conn)
        
        if chunk:
            print(f"Retrieved {chunk_name} chunk on attempt {attempt}")
            return chunk
        
        if attempt < max_retries:
            print(f"{chunk_name} chunk unavailable, retry {attempt}/{max_retries} after {retry_delay}s")
            time.sleep(retry_delay)
        else:
            print(f"Failed to retrieve {chunk_name} chunk after {max_retries} attempts")
    
    return None


@app.get("/api/get_organic_chunks")
def api_get_organic_chunks(needed: int):
    print("Received request for organic chunks")
    
    try:
        r = get_redis_connection()
    except Exception as e:
        print(f"Error connecting to Redis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
    chunks = []
    for i in range(needed):
        try:
            chunk = pop_organic_chunk(r)
            if chunk is None:
                print("No more organic chunks available")
                break
            chunks.append(chunk)
        except Exception as e:
            print(f"Error popping chunk: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    if len(chunks) == 0:
        print("No organic chunks in the queue")
        return {"message": "No organic chunks available", "chunks": chunks}
    
    return {"chunks": chunks}

@app.get("/api/scheduler_ready")
def api_scheduler_ready():
    """
    Check if the scheduler is ready to process synthetic requests.
    Returns True if all synthetic queues are above their thresholds.
    """
    try:
        r = get_redis_connection()
        ready = is_scheduler_ready(r)
        return {
            "ready": ready,
            "status": "ready" if ready else "not_ready",
            "message": "All synthetic queues are ready" if ready else "Waiting for synthetic queues to be populated"
        }
    except Exception as e:
        print(f"Error checking scheduler readiness: {e}")
        return {
            "ready": False,
            "status": "error",
            "message": f"Error checking readiness: {str(e)}"
        }


@app.get("/api/queue_sizes")
def api_queue_sizes():
    """
    Debug endpoint to view current queue sizes.
    """
    r = get_redis_connection()
    return {
        "organic_queue_size": get_organic_queue_size(r),
        "synthetic_5s_queue_size": get_5s_queue_size(r),
        "synthetic_10s_queue_size": get_10s_queue_size(r),
        "synthetic_20s_queue_size": get_20s_queue_size(r),
    }


@app.post("/api/push_result")
def api_push_result(payload: InsertResultRequest):
    """
    Save video processing result to Redis.
    """
    r = get_redis_connection()
    result_key = f"result:{payload.original_video_url}"
    result_data = {
        "processed_video_url": payload.processed_video_url,
        "original_video_url": payload.original_video_url,
        "score": payload.score,
        "task_id": payload.task_id,
    }
    r.hmset(result_key, result_data)
    return {"message": "Result saved successfully"}


@app.get("/api/get_result")
def api_get_result(payload: ResultRequest):
    """
    Retrieve processing result for a specific video URL.
    """
    # Decode the URL once since FastAPI will have already decoded it once
    
    original_video_url = payload.original_video_url

    r = get_redis_connection()
    
    result_key = f"result:{original_video_url}"
    
    result = r.hgetall(result_key)

    if not result:
        return {"message": "No result found for this video"}

    # Only try to access these fields if result is not empty
    print(result, result["processed_video_url"], result["original_video_url"])
    
    # Return the result as a proper JSON response
    return {
        "processed_video_url": result["processed_video_url"] if "processed_video_url" in result else None,
        "original_video_url": result["original_video_url"] if "original_video_url" in result else None,
        "task_id": result["task_id"] if "task_id" in result else None,
        "score": float(result["score"]) if "score" in result else None
    }

if __name__ == "__main__":
    
    import uvicorn
    host = CONFIG.video_scheduler.host
    port = CONFIG.video_scheduler.port
    uvicorn.run(app, host=host, port=port)