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
from services.miner_utilities.redis_utils import schedule_file_deletion
from vidaio_subnet_core.utilities import storage_client, download_video
from loguru import logger
import traceback
from features.modules.search_engine import VideoSearchEngine
import shutil

from services.miner_utilities.redis_utils import init_gpus, acquire_gpu, release_gpu, get_gpu_count
import cv2
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()
engine = VideoSearchEngine(logger)

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


def upscale_video(payload_video_path: str, task_type: str, delete_input_file: bool = True):
    """
    Upscales a video using the video2x tool and returns the full paths of the upscaled video and the converted mp4 file.

    Args:
        payload_video_path (str): The path to the video to upscale.
        task_type (str): The type of upscaling task to perform.

    Returns:
        str: The full path to the upscaled video.
    """
    gpu_index = -1
    try:
        # Acquire GPU
        gpu_index = acquire_gpu()
        if gpu_index is None:
            logger.error("✅ No GPU available for processing.")
            return JSONResponse(status_code=503, content={"error": "No GPU available for processing."})
        logger.info(f"✅ <<<<<<<<<<< Acquired GPU index: {gpu_index} >>>>>>>>>>")

        input_file = Path(payload_video_path)

        scale_factor = "2"

        if task_type == "SD24K":
            scale_factor = "4"

        # Validate input file
        if not input_file.exists() or not input_file.is_file():
            raise HTTPException(status_code=400, detail="Input file does not exist or is not a valid file.")

        # Get the frame rate of the video
        frame_rate = get_frame_rate(input_file)
        logger.info(f"✅ Frame rate detected:💜 {frame_rate} fps")

        # Generate output file paths
        output_file_upscaled = f"{input_file.stem}_video2x_upscaled.mp4"
        if os.path.exists(output_file_upscaled):
            logger.info(f"✅ Output file already exists: {output_file_upscaled}")
            os.remove(output_file_upscaled)
            logger.info(f"✅ Output file removed: {output_file_upscaled}")

        # Step 1: 
        logger.info("✅ Step 1: Start upscaling ...")
        start_time = time.time()
        
        # ===================
        dist_cap = cv2.VideoCapture(input_file)

        dist_total_frames = 0
        if not dist_cap.isOpened():
            logger.info(f"✅ ====>>>> Error opening distorted video file from {input_file}.")
            dist_total_frames = 0
        else:
            dist_total_frames = int(dist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"✅ Input video has 💜💜 {dist_total_frames} frames. 💜💜 scale factor={scale_factor}")
        # ===================

        if get_gpu_count() > 1:
            MULTI_GPU_INFO = f"{gpu_index * 2}, {gpu_index * 2 + 1}"
        else:
            MULTI_GPU_INFO = f"{gpu_index}"

        video2x_command = []
        if scale_factor == "2":
            model = "realesr-animevideov3"
            preset = "slow"
            filter_option = "5:5:1.5"

            # if dist_total_frames < 70:
            #     preset = "fast"
            # else:
            #     preset = "slow"

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
            preset = "slow"
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

        dist_cap.release()
        logger.info(f"✅ video2x: command: {' '.join(video2x_command)}")
        video2x_process = subprocess.run(video2x_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        elapsed_time = time.time() - start_time
        if video2x_process.returncode != 0:
            logger.info(f"❌ video2x failed: {video2x_process.stderr.strip()}")

            # Release the GPU
            if gpu_index != -1:
                release_gpu(gpu_index)
                logger.info(f"✅ <<<<<<<<<  Released GPU index: {gpu_index}  >>>>>>>>>")
            else:
                logger.info("❌ Failed to release GPU index, it was never acquired.")

            raise HTTPException(status_code=500, detail=f"video2x: Upscaling failed: {video2x_process.stderr.strip()}")
        if not os.path.exists(str(output_file_upscaled)):
            logger.info("❌ video2x: Upscaled MP4 video file was not created.")

            # Release the GPU
            if gpu_index != -1:
                release_gpu(gpu_index)
                logger.info(f"✅ <<<<<<<<<  Released GPU index: {gpu_index}  >>>>>>>>>")
            else:
                logger.info("❌ Failed to release GPU index, it was never acquired.")

            raise HTTPException(status_code=500, detail="video2x: Upscaled MP4 video file was not created.")
        logger.info(f"✅ Step 2 completed in ✅ {elapsed_time:.2f} seconds. Upscaled MP4 file: {output_file_upscaled}")
        total_time = time.time() - start_time

        logger.info(f"✅ Total completed in ✅✅ {total_time:.2f} seconds.")

        if input_file.exists() and delete_input_file:
            input_file.unlink()
            logger.info(f"✅ Original file {input_file} deleted.")
        
        logger.info(f"✅ Returning from FastAPI: {output_file_upscaled}")
        return str(output_file_upscaled)
    except Exception as e:
        logger.info(f"✅ Error: {str(e)}")

        # Release the GPU
        if gpu_index != -1:
            release_gpu(gpu_index)
            logger.info(f"✅ <<<<<<<<<  Released GPU index: {gpu_index}  >>>>>>>>>")
        else:
            logger.info("✅ Failed to release GPU index, it was never acquired.")

        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        # Release the GPU
        if gpu_index != -1:
            release_gpu(gpu_index)
            logger.info(f"✅ <<<<<<<<<  Released GPU index: {gpu_index}  >>>>>>>>>")
        else:
            logger.info("✅ Failed to release GPU index, it was never acquired.")

def upscale_query_based(payload_video_path: str, task_type: str, delete_input_file: bool = True):
    """
    Upscales a video based on a query.
    """
    try:
        query_scale = 2
        if task_type == "SD24K":
            query_scale = 4

        result = engine.search(query_path=payload_video_path, query_scale=query_scale, top_n=1)
        if os.path.exists(payload_video_path) and delete_input_file:
            logger.info(f"🔴 Deleting input file: {payload_video_path}")
            os.remove(payload_video_path)

        if len(result) > 0:
            upscaled_video_path = result[0][0]
            return upscaled_video_path
            # if (result[0][1] < 98):
            #     logger.info(f"VMAF score is less than 98, returning original upscaled video: {upscaled_video_path}")
            #     return upscaled_video_path

            # start_time = time.time()
            # logger.info(f"started unsharpping upscaled_video_path: {upscaled_video_path}")
            # sharp_factor = "7:7:-0.01"  # for scale_factor 2
            # upscaled_post_proced_video_path = f"{os.path.splitext(upscaled_video_path)[0]}_unsharp.mp4"
            # if os.path.exists(upscaled_post_proced_video_path):
            #     os.remove(upscaled_post_proced_video_path)
            #     logger.info(f"Deleted existing unsharp file: {upscaled_post_proced_video_path}")
            # # ffmpeg_command = [
            # #     "ffmpeg",
            # #     "-i", str(upscaled_video_path),
            # #     "-vf", f"unsharp={sharp_factor}",
            # #     "-threads", "0",
            # #     "-preset", "fast",
            # #     str(upscaled_post_proced_video_path)
            # # ]

            # ffmpeg_command = [
            #     "ffmpeg",
            #     "-i", str(upscaled_video_path),
            #     "-c:v", "libx264",
            #     "-c:a", "copy",
            #     "-crf", "18",
            #     "-preset", "fast",
            #     str(upscaled_post_proced_video_path)
            # ]

            # ffmpeg_process = subprocess.run(
            #     ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            # )
            # logger.info(f"unsharpping completed in ✅ {time.time() - start_time:.2f} seconds. Upscaled Unsharp MP4 file: {upscaled_post_proced_video_path}")
            # if ffmpeg_process.returncode != 0 or not os.path.exists(upscaled_post_proced_video_path):
            #     logger.info("❌ Upscaled afterprocess MP4 video file was not created.")
            #     return upscaled_video_path

            # if os.path.exists(upscaled_video_path):
            #     os.remove(upscaled_video_path)
            #     logger.info(f"Intermediate upscaled file {upscaled_video_path} deleted.")
            # return upscaled_post_proced_video_path

        else:
            return None
    except Exception as e:
        logger.error(f"🔴 Failed to process upscaling request: {e}")
        traceback.print_exc()
        return None

async def upscale_video_wrapper(payload_video_path: str, task_type: str, delete_input_file: bool = True):
    """
    Wrapper function for upscale_video.
    """
    try:
        start_time = time.time()
        # Copy the video file to /home/online_data
        if delete_input_file:
            target_path = f"/home/online_data/{Path(payload_video_path).name}"
            shutil.copy2(payload_video_path, target_path)
            logger.info(f"Copied video file to {target_path}")

        logger.info(f"upscale_video_wrapper: start for {payload_video_path}")
        async def run_upscale_methods():
            # Create a ThreadPoolExecutor for running the synchronous functions
            with ThreadPoolExecutor() as executor:
                # Create tasks for both upscaling methods
                loop = asyncio.get_event_loop()
                task1 = loop.run_in_executor(executor, upscale_video, payload_video_path, task_type, False)
                task2 = loop.run_in_executor(executor, upscale_query_based, payload_video_path, task_type, False)
                
                # Wait for the first task to complete
                done, pending = await asyncio.wait(
                    [task1, task2],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Print which task completed
                timeout_delay = 35.0
                for task in done:
                    if task == task1:
                        logger.info("✅ Upscale video method completed first")
                        if (time.time() - start_time) > timeout_delay:
                            logger.info(f"⏰ Upscale video method finished, if more than {timeout_delay}s passed, use its result")
                            task2.cancel()
                            return task1.result()
                        
                        logger.info(f"🔴 ⏰ Wait for query based upscaling to be completed, as {timeout_delay}s not passed still")
                        task2_result = await asyncio.wait_for(task2, timeout=timeout_delay-(time.time() - start_time))
                        if task2_result is not None:
                            logger.info("🔴 Using query-based upscale result as both methods completed")
                            return task2_result
                        else:
                            logger.info(f"🔴 ⏰ Query-based upscale timed out after {timeout_delay} seconds, using direct upscale result")
                            return task1.result()
                    elif task == task2:
                        logger.info("🔴 Query-based upscale method completed first")
                        task2_result = task2.result()
                        if task2_result is not None:
                            logger.info("🔴 Using query-based upscale result as both methods completed")
                            task1.cancel()
                            return task2_result
                # Wait for task1 to complete and use its result as fallback
                result = await task1
                logger.info("✅ Using direct upscale result as fallback")
                return result

        processed_video_path = await run_upscale_methods()
        logger.info(f"upscale_video_wrapper: end for {payload_video_path}, result: {processed_video_path}")
        
        if delete_input_file and os.path.exists(payload_video_path):
            os.remove(payload_video_path)
            logger.info(f"Original file {payload_video_path} deleted.")

        if processed_video_path is not None:
            processed_video_name = Path(processed_video_path).name
            elapsed_time = time.time() - start_time
            logger.info(f"Upscaling completed in ✅ {elapsed_time:.2f} seconds. Upscaled video: {processed_video_name}")
        else:
            logger.info(f"Failed to upscale video, returning None. {payload_video_path}")
            return None

        if delete_input_file:
            target_path = f"/home/online_data/{Path(payload_video_path).name}_outscaled.mp4"
            shutil.copy2(processed_video_path, target_path)
            logger.info(f"Copied video file to {target_path}")

        return processed_video_path
    except Exception as e:
        logger.error(f"Failed to process upscaling request: {e}")
        traceback.print_exc()
        return None

@app.post("/upscale-video")
async def video_upscaler(request: UpscaleRequest):
    logger.info("Received request to upscale video.")
    logger.info(f"Request payload: {request.json()}")

    try:
        payload_url = request.payload_url
        task_type = request.task_type

        start_time = time.time()
        logger.info("📻 Downloading video....")
        payload_video_path: str = await download_video(payload_url)
        elapsed_time = time.time() - start_time
        logger.info(f"Download video finished, Path: {payload_video_path}, Time: {elapsed_time:.2f} seconds")

        processed_video_path = await upscale_video_wrapper(payload_video_path, task_type, True)

        if processed_video_path is not None:
            object_name: str = Path(processed_video_path).name
            
            await storage_client.upload_file(object_name, processed_video_path)
            logger.info("Video uploaded successfully.")
            
            # Delete the local file since we've already uploaded it to MinIO
            if os.path.exists(processed_video_path):
                os.remove(processed_video_path)
                logger.info(f"{processed_video_path} has been deleted.")
            else:
                logger.info(f"{processed_video_path} does not exist.")
                
            sharing_link: str | None = await storage_client.get_presigned_url(object_name)
            if not sharing_link:
                logger.error("Upload failed")
                return {"uploaded_video_url": None}
            # Schedule the file for deletion after 10 minutes (600 seconds)
            deletion_scheduled = schedule_file_deletion(object_name)
            if deletion_scheduled:
                logger.info(f"Scheduled deletion of {object_name} after 10 minutes")
            else:
                logger.warning(f"Failed to schedule deletion of {object_name}")
            
            logger.info(f"Public download link: {sharing_link}")
            logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")

            return {"uploaded_video_url": sharing_link}

    except Exception as e:
        logger.error(f"Failed to process upscaling request: {e}")
        traceback.print_exc()
        return {"uploaded_video_url": None}


if __name__ == "__main__":
    
    import uvicorn
    
    host = CONFIG.video_upscaler.host
    port = CONFIG.video_upscaler.port
    
    uvicorn.run(app, host=host, port=port)