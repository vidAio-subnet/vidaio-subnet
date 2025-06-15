from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import subprocess
import os
from fastapi.responses import JSONResponse
import time
import asyncio
from vidaio_subnet_core import CONFIG
import re
from pydantic import BaseModel
from typing import Optional
from services.miner_utilities.redis_utils import schedule_file_deletion, schedule_local_file_deletion
from vidaio_subnet_core.utilities import storage_client, download_video
from loguru import logger
import traceback
import shutil
from services.miner_utilities.redis_utils import init_gpus, acquire_gpu, release_gpu, get_gpu_count
import cv2
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import json
from search.modules.search_config import search_config

app = FastAPI()

# initialize GPU
@app.on_event("startup")
def startup_event():
    init_gpus()

class UpscaleRequest(BaseModel):
    payload_url: str
    task_type: str
    # output_file_upscaled: Optional[str] = None
    

def get_frame_rate(input_file: Path) -> float:
    """
    Extracts the frame rate of the input video using FFmpeg.

    Args:
        input_file (Path): The path to the video file.

    Returns:
        float: The frame rate of the video.
    """
    frame_rate_command = [
        "ffmpeg",
        "-i", str(input_file),
        "-hide_banner"
    ]
    process = subprocess.run(frame_rate_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = process.stderr  # Frame rate is usually in stderr

    # Extract frame rate using regex
    match = re.search(r"(\d+(?:\.\d+)?) fps", output)
    if match:
        return float(match.group(1))
    else:
        raise HTTPException(status_code=500, detail="Unable to determine frame rate of the video.")


async def upscale_video2x(payload_url: str, task_type: str):
    """
    Upscales a video using the video2x tool and returns the full paths of the upscaled video and the converted mp4 file.

    Args:
        payload_video_path (str): The path to the video to upscale.
        task_type (str): The type of upscaling task to perform.

    Returns:
        str: The full path to the upscaled video.
    """
    try:
        start_time = time.time()
        logger.info("📻 Downloading video...")
        payload_video_path: str = await download_video(payload_url)
        elapsed_time = time.time() - start_time
        logger.info(f"📻 Download video finished, Path: {payload_video_path}, Time: {elapsed_time:.2f}s")
    except Exception as e:
        logger.error(f"❌ Failed to download video: {e}")
        return None
    
    gpu_index = -1
    try:
        # Acquire GPU
        gpu_index = acquire_gpu()
        if gpu_index is None:
            logger.error("1️❌ No GPU available for processing.")
            return None

        input_file = Path(payload_video_path)
        scale_factor = "2"
        if task_type == "SD24K":
            scale_factor = "4"

        # Validate input file
        if not input_file.exists() or not input_file.is_file():
            return None

        # Generate output file paths
        output_file_upscaled = f"{input_file.stem}_video2x_upscaled.mp4"
        if os.path.exists(output_file_upscaled):
            logger.info(f"1️ Output file already exists: {output_file_upscaled}")
            os.remove(output_file_upscaled)
            logger.info(f"1️ Output file removed: {output_file_upscaled}")

        if get_gpu_count() > 1:
            MULTI_GPU_INFO = f"{gpu_index * 2}, {gpu_index * 2 + 1}"
        else:
            MULTI_GPU_INFO = f"{gpu_index}"

        video2x_command = []
        if scale_factor == "2":
            model = "realesr-animevideov3"
            preset = "fast"
            filter_option = "5:5:1.5"

            video2x_command = [
                "video2x",
                "-i", str(input_file),
                "-o", str(output_file_upscaled),
                "-p", "realesrgan",
                "--realesrgan-model", model,
                "-s", scale_factor,
                "-c", "libx264",
                "-D", str(MULTI_GPU_INFO),
                "-f", filter_option,
                "-e", f"preset={preset}",  
                "-e", "crf=18",
                "-e", "tune=stillimage",
            ]
        else:
            preset = "fast"
            model = "realesr-generalv3"
            filter_option = "7:7:2.32"

            video2x_command = [
                "video2x",
                "-i", str(input_file),
                "-o", str(output_file_upscaled),
                "-p", "realesrgan",
                "--realesrgan-model", model,
                "-s", scale_factor,
                "-c", "libx264",
                "-D", str(MULTI_GPU_INFO),
                "-f", filter_option,
                "-e", f"preset={preset}",  
                "-e", "crf=18",
                "-e", "tune=stillimage",
            ]

        logger.info(f"1️ video2x: command: {' '.join(video2x_command)}")
        video2x_process = subprocess.run(video2x_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if video2x_process.returncode != 0:
            logger.info(f"1️❌ video2x failed: {video2x_process.stderr.strip()}")

            # Release the GPU
            if gpu_index != -1:
                release_gpu(gpu_index)
            else:
                logger.info("1️❌ Failed to release GPU index, it was never acquired.")

            return None
        if not os.path.exists(str(output_file_upscaled)):
            logger.info("1️❌ video2x: Upscaled MP4 video file was not created.")

            # Release the GPU
            if gpu_index != -1:
                release_gpu(gpu_index)
            else:
                logger.info("1️❌ Failed to release GPU index, it was never acquired.")

            return None
        
        logger.info(f"1️✅ Returning from FastAPI: {output_file_upscaled}")
        schedule_local_file_deletion(output_file_upscaled, 2 * 60)
        return str(output_file_upscaled)
    except Exception as e:
        logger.info(f"1️❌ Error: {str(e)}")

        # Release the GPU
        if gpu_index != -1:
            release_gpu(gpu_index)
        else:
            logger.info("1️❌ Failed to release GPU index, it was never acquired.")

        return None
    finally:
        # Release the GPU
        if gpu_index != -1:
            release_gpu(gpu_index)
        else:
            logger.info("1️❌ Failed to release GPU index, it was never acquired.")
        if payload_video_path and os.path.exists(payload_video_path):
            os.remove(payload_video_path)  

async def search_video(payload_url: str, task_type: str) -> str | None:
    url = f"http://{search_config['SEARCH_SERVICE_HOST']}:{search_config['SEARCH_SERVICE_PORT']}/search"
    headers = {"Content-Type": "application/json"}
    data = {
        "payload_url": payload_url,
        "task_type": task_type,
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=json.dumps(data)) as response:
            if response.status == 200:
                result = await response.json()
                if result.get("success"):
                    return result.get("filename")
                else:
                    return None
            logger.error(f"Video search service error: {response.status}")
            return None

async def upscale_search(payload_url: str, task_type: str) -> str | None:
    filename = await search_video(payload_url, task_type)
    if filename is None:
        return None
    try:
        start_time = time.time()
        logger.info("📻 Downloading video...")
        upscaled_video_path: str = await download_video(f"http://{search_config['SEARCH_SERVICE_HOST']}:{search_config['SEARCH_SERVICE_PORT']}/download/{filename}")
        elapsed_time = time.time() - start_time
        logger.info(f"📻 Download video finished, Path: {upscaled_video_path}, Time: {elapsed_time:.2f}s")
        schedule_local_file_deletion(upscaled_video_path, 2 * 60)
        return upscaled_video_path
    except Exception as e:
        logger.error(f"❌ Failed to download video: {e}")
        return None
    
async def upscale_video_wrapper(payload_url: str, task_type: str):
    """
    Wrapper function for upscale_video.
    """
    try:
        start_time = time.time()
        
        async def run_upscale_methods():
            # Create a ThreadPoolExecutor for running the synchronous functions
            with ThreadPoolExecutor() as executor:
                # Create tasks for both upscaling methods
                loop = asyncio.get_event_loop()
                task1 = loop.run_in_executor(executor, upscale_video2x, payload_url, task_type)
                task2 = loop.run_in_executor(executor, upscale_search, payload_url, task_type)
                
                # Wait for the first task to complete
                done, pending = await asyncio.wait(
                    [task1, task2],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Print which task completed
                timeout_delay = 35.0
                for task in done:
                    if task == task1:
                        logger.info("🔴 video2x completed first")
                        if (time.time() - start_time) > timeout_delay:
                            logger.info(f"🔴 video2x finished, {time.time() - start_time:.2f}s passed, use video2x as final result")
                            task2.cancel()
                            return task1.result()
                        
                        logger.info(f"⏰ Wait for search_engine to be completed {(time.time() - start_time):.2f}s passed")
                        task2_result = await asyncio.wait_for(task2, timeout=timeout_delay-(time.time() - start_time))
                        if task2_result is not None:
                            logger.info("✅ Using search_engine as result")
                            return task2_result
                        else:
                            logger.info(f"🔴 search_engine timed out after {timeout_delay} seconds, using video2x result")
                            return task1.result()
                    elif task == task2:
                        logger.info("✅ search_engine completed first")
                        task2_result = task2.result()
                        if task2_result is not None:
                            task1.cancel()
                            return task2_result
                # Wait for task1 to complete and use its result as fallback
                result = await task1
                logger.info("🔴 Using video2x result as fallback")
                return result
        
        processed_video_path = await run_upscale_methods()

        if processed_video_path is not None:
            processed_video_name = Path(processed_video_path).name
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Upscaled video: {processed_video_name}, duration: {elapsed_time:.2f}s")
        else:
            logger.info(f"❌ Upscaled video: {processed_video_name} failed")
            return None

        return processed_video_path
    except Exception as e:
        logger.error(f"❌ Failed to upscale video: {payload_video_path}, error: {e}")
        traceback.print_exc()
        return None

@app.post("/upscale-video")
async def video_upscaler(request: UpscaleRequest):
    logger.info(f"📻 upscale-video request payload: {request.json()}")

    try:
        start_time = time.time()
        payload_url = request.payload_url
        task_type = request.task_type
        
        processed_video_path = await upscale_video_wrapper(payload_url, task_type)

        if processed_video_path is not None:
            object_name: str = Path(processed_video_path).name
            
            await storage_client.upload_file(object_name, processed_video_path)
            logger.info("📻 Video uploaded successfully.")
                
            sharing_link: str | None = await storage_client.get_presigned_url(object_name)
            if not sharing_link:
                logger.error("❌ Upload failed")
                return {"uploaded_video_url": None}
            # Schedule the file for deletion after 10 minutes (600 seconds)
            deletion_scheduled = schedule_file_deletion(object_name)
            if deletion_scheduled:
                logger.info(f"Scheduled deletion of {object_name} after 10 minutes")
            else:
                logger.warning(f"Failed to schedule deletion of {object_name}")
            
            logger.info(f"Public download link: {sharing_link}")
            logger.info(f"Total processing time: {time.time() - start_time:.2f}s")

            return {"uploaded_video_url": sharing_link}

    except Exception as e:
        logger.error(f"❌ Failed to process upscaling request: {e}")
        traceback.print_exc()
        return {"uploaded_video_url": None}


if __name__ == "__main__":
    
    import uvicorn
    
    host = CONFIG.video_upscaler.host
    port = CONFIG.video_upscaler.port
    
    uvicorn.run(app, host=host, port=port)