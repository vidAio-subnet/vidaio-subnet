from fastapi import FastAPI
from pydantic import BaseModel

from redis_utils import (
    get_redis_connection,
    push_organic_chunk,
    push_synthetic_chunk,
    pop_organic_chunk,
    pop_synthetic_chunk,
    get_organic_queue_size,
    get_synthetic_queue_size,
    MAX_CHUNK_IN_QUEUE,
)
from chunk_logic import read_synthetic_urls

app = FastAPI()

# Load synthetic URLs from config
SYNTHETIC_URLS = read_synthetic_urls("config.yaml")


class InsertOrganicRequest(BaseModel):
    url: str


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
    r = get_redis_connection()
    chunk_url = pop_organic_chunk(r)
    if not chunk_url:
        chunk_url = pop_synthetic_chunk(r)

    if not chunk_url:
        return {"message": "No chunks available"}
    return {"chunk_url": chunk_url}


@app.get("/api/queue_sizes")
def api_queue_sizes():
    """
    Debug endpoint to view current queue sizes.
    """
    r = get_redis_connection()
    return {
        "organic_queue_size": get_organic_queue_size(r),
        "synthetic_queue_size": get_synthetic_queue_size(r),
        "max_chunk_in_queue": MAX_CHUNK_IN_QUEUE,
    }
