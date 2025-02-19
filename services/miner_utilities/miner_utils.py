import uuid
import requests
from pathlib import Path
from requests.exceptions import RequestException
import httpx
import asyncio
from fastapi import HTTPException
from loguru import logger
import aiohttp
import json


async def download_video(url: str) -> Path:
    """
    Download video with retries and proper redirect handling.
    
    Args:
        url: URL of the video to download
        
    Returns:
        Path: Path to the downloaded video file
        
    Raises:
        HTTPException: If download fails
    """
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url)
            
            if "drive.google.com" in url:
                if "drive.usercontent.google.com" in response.url.path:
                    download_url = str(response.url)
                else:
                    file_id = url.split("id=")[1].split("&")[0]
                    download_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
                response = await client.get(download_url)
            
            response.raise_for_status()
            
            # Define video directory and create it if necessary
            video_dir = Path(__file__).parent.parent / "upscaling" / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate a UUID filename
            filename = f"{uuid.uuid4()}.mp4"
            output_path = video_dir / filename
            output_path.unlink(missing_ok=True)
            # Save the video content to the output path
            with open(output_path, "wb") as video_file:
                video_file.write(response.content)
            
            logger.info(f"Video downloaded successfully to {output_path}")
            return output_path
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error downloading video: {str(e)}")
        logger.error(f"Response status code: {e.response.status_code}")
        logger.error(f"Response headers: {e.response.headers}")
        raise HTTPException(status_code=500, detail=f"Failed to download video: {str(e)}")
    except Exception as e:
        logger.error(f"Error downloading video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download video: {str(e)}")


async def video_upscaler(input_file_path):
    url = "http://localhost:29115/upscale-video"
    headers = {"Content-Type": "application/json"}
    data = {"task_file_path": str(input_file_path)}
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=json.dumps(data)) as response:
            if response.status == 200:
                result = await response.json()
                upscaled_video_path = result.get("upscaled_video_path")
                print("upscaled_video_path:", upscaled_video_path)
                return upscaled_video_path
            else:
                print("Error:", response.status)
                return None


if __name__ == "__main__":
    video_url = "/root/workspace/vidaio-subnet/services/upscaling/videos/7a4297c6-970b-4f62-8aef-f4fb4f40156f.mp4"
    # # asyncio.run(download_video(video_url))
    # response = asyncio.run(video_upscaler(video_url))
    # print(response)
    asyncio.run(video_upscaler(video_url))