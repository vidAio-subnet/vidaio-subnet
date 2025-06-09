from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import cv2
import numpy as np
from firerequests import FireRequests
import tempfile
import os
import random
from moviepy.editor import VideoFileClip
import aiohttp
import logging
import time
import math
import subprocess
from vidaio_subnet_core import CONFIG
from pathlib import Path
import sys
from services.miner_utilities.redis_utils import init_gpus, acquire_gpu, release_gpu
import re
from services.upscaling.server import upscale_video_wrapper
import asyncio
from vidaio_subnet_core.utilities import storage_client, download_video
import traceback
import requests

# Set up logging
from loguru import logger
import httpx

# Initialize score client
score_client = None

async def init_score_client():
    global score_client
    score_client = httpx.AsyncClient(
        base_url=f"http://{CONFIG.score.host}:{CONFIG.score.port}"
    )
    logger.info(
        f"💧 Initialized score client with base URL: http://{CONFIG.score.host}:{CONFIG.score.port} 💧"
    )

async def score_video(downscaled_video_path, ref_video_path):
    start_time = time.time()

    try:

        input_file = Path(downscaled_video_path)
        task_type = "SD24K" if str(input_file.name).startswith("SD24K") else "NOT-SD24K"

        start_time = time.time()

        processed_video_path = await upscale_video_wrapper(downscaled_video_path, task_type, False)
        if processed_video_path is not None:
            object_name: str = Path(processed_video_path).name
            
            await storage_client.upload_file(object_name, processed_video_path)
            #await storage_client.upload_file(object_name, ref_video_path)
            
            logger.info("Video uploaded successfully.")
                
            sharing_link: str | None = await storage_client.get_presigned_url(object_name)
            if not sharing_link:
                logger.error("Upload failed")
                return (None, None)
           
            logger.info(f"Public download link: {sharing_link}")
            logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
            video_duration = VideoFileClip(ref_video_path).duration
            uids = [1]
            
            post_json = {
                "uids": uids,
                "distorted_urls": [sharing_link],
                "reference_paths": [ref_video_path],
                "video_ids": [input_file.name],
                "uploaded_object_names": [object_name],
                "content_lengths": [math.floor(video_duration)]
            }

            logger.info(f"Post JSON: {post_json}")

            score_response = await score_client.post(
                "/score_synthetics",
                json = post_json,
                timeout=240
            )

            response_data = score_response.json()

            quality_scores = response_data.get("quality_scores", [])
            length_scores = response_data.get("length_scores", [])
            final_scores = response_data.get("final_scores", [])
            vmaf_scores = response_data.get("vmaf_scores", [])
            pieapp_scores = response_data.get("pieapp_scores", [])
            reasons = response_data.get("reasons", [])
            
            max_length = max(len(uids), len(quality_scores), len(length_scores), len(final_scores), len(vmaf_scores), len(pieapp_scores), len(reasons))
            uids.extend([0] * (max_length - len(uids)))
            quality_scores.extend([0.0] * (max_length - len(quality_scores)))
            length_scores.extend([0.0] * (max_length - len(length_scores)))
            final_scores.extend([0.0] * (max_length - len(final_scores)))
            vmaf_scores.extend([0.0] * (max_length - len(vmaf_scores)))
            pieapp_scores.extend([0.0] * (max_length - len(pieapp_scores)))
            reasons.extend(["No reason provided"] * (max_length - len(reasons)))
        
            return (vmaf_scores[0], final_scores[0])

    except Exception as e:
        logger.error(f"Failed to process upscaling request: {e}")
        traceback.print_exc()
        return (None, None)


async def do_score_check(file_list):
    vmaf_scores = []
    final_scores = []
    for file in file_list:
        try:
            vmaf_score, final_score = await score_video(file[0], file[1])
            file_name = os.path.splitext(os.path.basename(file[0]))[0]
            logger.info(f"🔯🔯🔯 calculated score: VMAF: {vmaf_score} Final: {final_score} for {file_name} 🔯🔯🔯")
            if vmaf_score is not None:
                vmaf_scores.append(vmaf_score)  
            if final_score is not None:
                final_scores.append(final_score)
        except Exception as e:
            logger.info(f"Error: {e}")
            continue

    avg_vmaf = 0
    avg_final = 0

    if vmaf_scores:
        avg_vmaf = sum(vmaf_scores) / len(vmaf_scores)
        logger.info(f"\n📊 Average VMAF score: {avg_vmaf:.2f}")
    if final_scores:
        avg_final = sum(final_scores) / len(final_scores)
        logger.info(f"📊 Average Final score: {avg_final:.2f}")
    return avg_vmaf, avg_final
    
if __name__ == "__main__":
    if len(sys.argv) <= 1:
        logger.info("Error: Please provide a video file directory or video file path as argument")
        sys.exit(1)
    path = sys.argv[1]
    file_list = []
    asyncio.run(init_score_client())
    time.sleep(1)

    init_gpus()
    if os.path.isdir(path): 
        # Get all files in directory
        for file in os.listdir(path):
            # Check for common video file extensions
            file_path = os.path.join(path, file)
            if 'downscale' in file_path:
                ref_file = file.split('downscale')[0] + 'trim.mp4'
                file_list.append((file_path, os.path.join(path, ref_file)))
        if not file_list:
            logger.info("No video files found in directory")
            sys.exit(1)
    else:
        # Single file - convert to list for consistent handling
        ref_file = path.split('downscale')[0] + 'trim.mp4'
        file_list = [(path, ref_file)]
    logger.info(f"🔯🔯🔯 file_list length: {len(file_list)} 🔯🔯🔯")
    asyncio.run(do_score_check(file_list))

