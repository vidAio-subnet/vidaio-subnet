from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from search.modules.config import config
from search.modules.video_search_engine import VideoSearchEngine

app = FastAPI()
search_engine = VideoSearchEngine()

class SearchRequest(BaseModel):
    payload_url: str
    payload_path: str
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
        clip_path, max_vmaf_idx = search_engine.search_and_clip(request.payload_path, request.task_type)
        return {
            "clip_path": clip_path,
            "max_vmaf_idx": max_vmaf_idx
        }
    except Exception as e:
        return {
            "clip_path": None,
            "max_vmaf_idx": None,
            "error": str(e)
        }

@app.post("/put_video_info")
async def put_video_info(video_info: VideoInfo):
    try:
        logger.info(f"✅ Video search service received video info {video_info.filename}")
        search_engine.put_video_info({
            'filename': video_info.filename,
            'fps': video_info.fps,
            'width': video_info.width,
            'height': video_info.height,
            'frame_count': video_info.frame_count,
            'hashes': video_info.hashes
        })
        return {
            "success": True
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

async def search_video(payload_url: str, payload_path : str, task_type: str) -> str | None:
    url = f"http://{config.search_service_host}:{config.search_service_port}/search"
    headers = {"Content-Type": "application/json"}
    data = {
        "payload_url": payload_url,
        "payload_path": payload_path,
        "task_type": task_type,
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=json.dumps(data)) as response:
            if response.status == 200:
                result = await response.json()
                clip_path = result.get("clip_path")
                max_vmaf_idx = result.get("max_vmaf_idx")
                if clip_path is None:
                    logger.info("🩸 Received None response from video search service 🩸")
                    return None
                logger.info("✈️ Received response from video search service correctly ✈️")
                return clip_path
            logger.error(f"Video search service error: {response.status}")
            return None

if __name__ == "__main__":
    import uvicorn

    host = config['search_service_host']
    port = config['search_service_port']

    uvicorn.run(app, host=host, port=port)
