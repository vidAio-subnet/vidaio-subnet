import uuid
import json
import asyncio
from pathlib import Path
import aiohttp
from fastapi import HTTPException
from loguru import logger
from vidaio_subnet_core import CONFIG
import os
import time

async def download_video(video_url: str) -> Path:
    """
    Downloads a video from the given URL with retries and redirect handling.
    
    Args:
        url (str): The URL of the video to download.
    
    Returns:
        Path: The local path of the downloaded video.
    
    Raises:
        HTTPException: If the download fails.
    """
    try:
        video_dir = Path(__file__).parent.parent / "upscaling" / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate a unique filename
        filename = f"{uuid.uuid4()}.mp4"
        output_path = video_dir / filename
        
        logger.info(f"Downloading video from {video_url} to {output_path}")
        start_time = time.time()
        # Download the file using aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(video_url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download video. HTTP status: {response.status}")

                # Write the content to the temp file in chunks
                with open(output_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(3 * 1024 * 1024):  # 3 MB chunks
                        f.write(chunk)
        elapsed_time = time.time() - start_time
        logger.info(f"Chunk download time: {elapsed_time:.2f} seconds")

        # Verify the file was successfully downloaded
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise Exception(f"Download failed or file is empty: {output_path}")

        logger.info(f"File successfully downloaded to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to download video from {video_url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading video: {str(e)}")


async def video_upscaler(payload_url: str, task_type: str) -> str | None:
    """
    Sends a video file path to the upscaling service and retrieves the processed video path.
    
    Args:
        payload_url (str): The url of the video to be upscaled.
    
    Returns:
        str | None: The path of the upscaled video or None if an error occurs.
    """
    url = f"http://{CONFIG.video_upscaler.host}:{CONFIG.video_upscaler.port}/upscale-video"
    headers = {"Content-Type": "application/json"}
    data = {
        "payload_url": payload_url,
        "task_type": task_type,
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=json.dumps(data)) as response:
            if response.status == 200:
                result = await response.json()
                uploaded_video_url = result.get("uploaded_video_url")
                # logger.info(f"Processed video URL: {uploaded_video_url}")
                if uploaded_video_url is None:
                    logger.info("🩸 Received None response from video upscaler 🩸")
                    return None
                logger.info("✈️ Received response from video upscaler correctly ✈️")
                return uploaded_video_url
            logger.error(f"Upscaling service error: {response.status}")
            return None

async def video_compressor(payload_url: str, vmaf_threshold: float) -> str | None:
    """
    Sends a video file path to the compression service and retrieves the processed video path.
    """
    url = f"http://{CONFIG.video_compressor.host}:{CONFIG.video_compressor.port}/compress-video"
    headers = {"Content-Type": "application/json"}
    data = {
        "payload_url": payload_url,
        "vmaf_threshold": vmaf_threshold,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=json.dumps(data)) as response:
            if response.status == 200:
                result = await response.json()
                uploaded_video_url = result.get("uploaded_video_url")
                if uploaded_video_url is None:
                    logger.info("🩸 Received None response from video compressor 🩸")
                    return None
                logger.info("✈️ Received response from video compressor correctly ✈️")
                return uploaded_video_url
            logger.error(f"Compression service error: {response.status}")
            return None