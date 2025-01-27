from fastapi import FastAPI
from pydantic import BaseModel
from video_subnet_core import CONFIG

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


class InsertResultRequest(BaseModel):
    compressed_video_url: str
    original_video_url: str
    score: float


@app.post("/api/insert_organic_chunk")
def api_insert_organic_chunk(payload: InsertOrganicRequest):
    """
    Insert an organic video URL into the organic queue.
    """
    r = get_redis_connection()
    push_organic_chunk(r, payload.url)
    return {"message": "Organic chunk inserted"}


@app.get("/api/get_prioritized_chunk")
def api_get_prioritized_chunk():
    """
    Retrieve the highest-priority chunk:
    1) Pop the oldest organic chunk, if available.
    2) Otherwise, pop the oldest synthetic chunk.
    """
    print("got the request correctoy")
    r = get_redis_connection()
    chunk = pop_organic_chunk(r)
    print("organic queue is empty, checking synthetic queue...")
    if not chunk:
        chunk = pop_synthetic_chunk(r)
        print(f"processing synthetic queue. {chunk}")
    if not chunk:
        return {"message": "No chunks available"}
    return {"chunk": chunk}


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
        "compressed_video_url": payload.compressed_video_url,
        "original_video_url": payload.original_video_url,
        "score": payload.score,
    }
    r.hmset(result_key, result_data)
    return {"message": "Result saved successfully"}


@app.get("/api/get_result/{original_video_url:path}")
def api_get_result(original_video_url: str):
    """
    Retrieve processing result for a specific video URL.
    """
    r = get_redis_connection()
    result_key = f"result:{original_video_url}"
    result = r.hgetall(result_key)

    if not result:
        return {"message": "No result found for this video"}

    return {
        "compressed_video_url": result[b"compressed_video_url"].decode(),
        "original_video_url": result[b"original_video_url"].decode(),
        "score": float(result[b"score"]),
    }



if __name__ == "__main__":
    
    import uvicorn
    host = CONFIG.video_scheduler.host
    port = CONFIG.video_scheduler.port
    uvicorn.run(app, host=host, port=port)