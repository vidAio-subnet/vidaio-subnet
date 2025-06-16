from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from search.modules.search_config import search_config
from search.modules.video_search_engine import VideoSearchEngine, get_ramfs_path
from services.miner_utilities.redis_utils import schedule_local_file_deletion
import os
import json
import aiohttp
import logging
from fastapi.responses import FileResponse
import time
from vidaio_subnet_core.utilities import download_video

logger = logging.getLogger(__name__)

app = FastAPI()
search_engine = VideoSearchEngine()

class SearchRequest(BaseModel):
    payload_url: str
    task_type: str

class VideoInfo(BaseModel):
    filename: str
    fps: int
    width: int
    height: int
    frame_count: int
    hashes: list[str]

@app.post("/search")
async def search(request: SearchRequest):
    try:
        start_time = time.time()
        logger.info("📻 Downloading video...")
        payload_video_path: str = await download_video(request.payload_url)
        elapsed_time = time.time() - start_time
        logger.info(f"📻 Download video finished, Path: {payload_video_path}, Time: {elapsed_time:.2f}s")
    except Exception as e:
        logger.error(f"❌ Failed to download video: {e}")
        return {
            "success": False,
            "error": str(e)
        }

    try:
        clip_path, max_vmaf_idx = search_engine.search_and_clip(payload_video_path, request.task_type)
        schedule_local_file_deletion(clip_path, 2 * 60)
        clip_filename = os.path.basename(clip_path)
        if max_vmaf_idx >= 0:
            return {
                "success": True,
                "filename": clip_filename,
            }
        else:
            return {
                "success": False,
                "error": "No clip found"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        if payload_video_path and os.path.exists(payload_video_path):
            os.remove(payload_video_path)

@app.get("/download/{filename}")
async def download_file(filename: str):
    try:
        file_path = os.path.join(get_ramfs_path(), filename)
        if not os.path.exists(file_path):
            file_path = os.path.join(search_config['VIDEO_DIR'], filename)
        if not os.path.exists(file_path):
            file_path = os.path.join(search_config['TEST_VIDEO_DIR'], filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
            
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='video/mp4'
        )
    except Exception as e:
        logger.error(f"Error downloading file {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    host = search_config['SEARCH_SERVICE_HOST']
    port = search_config['SEARCH_SERVICE_PORT']

    uvicorn.run(app, host=host, port=port)
