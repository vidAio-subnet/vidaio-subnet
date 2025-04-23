from fastapi import FastAPI
from pydantic import BaseModel
from vidaio_subnet_core import CONFIG
from typing import Optional, Literal
import urllib

from redis_utils import (
    get_redis_connection,
    push_organic_chunk,
    pop_organic_chunk,
    pop_synthetic_chunk,
    get_organic_queue_size,
    get_synthetic_queue_size,
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


@app.get("/api/get_synthetic_chunk")
def api_get_synthetic_chunk():
    print("got the get_synthetic_chunk request correctly")
    r = get_redis_connection()
    chunk = pop_synthetic_chunk(r)
    if not chunk:
        return {"message": "No chunks available"}
    return {"chunk": chunk}


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

@app.get("/api/queue_sizes")
def api_queue_sizes():
    """
    Debug endpoint to view current queue sizes.
    """
    r = get_redis_connection()
    return {
        "organic_queue_size": get_organic_queue_size(r),
        "synthetic_queue_size": get_synthetic_queue_size(r),
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