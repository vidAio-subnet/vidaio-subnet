from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from loguru import logger
import cv2
import numpy as np
from firerequests import FireRequests
import tempfile
import os
import random
from vidaio_subnet_core import CONFIG
from vmaf_metric import calculate_vmaf
from lpips_metric import calculate_lpips
import asyncio

app = FastAPI()
fire_requests = FireRequests()

class ScoringRequest(BaseModel):
    distorted_urls: List[str]
    reference_path: str
    fps: Optional[float] = None
    subsample: Optional[int] = 1
    verbose: Optional[bool] = False
    progress: Optional[bool] = False

class ScoringResponse(BaseModel):
    scores: List[float]

async def download_video(video_url, verbose) -> str:
    """Download reference and distorted videos to temporary files."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as vid_temp:
            file_path = vid_temp.name  # Path to the temporary file

        logger.info(f"Downloading video from {video_url}")
        
        file_path = "2.mp4"
        
        # Attempt to download the file
        await fire_requests.download_file(
            video_url,
            file_path,
            max_files=1,
            chunk_size=2 * 1024 * 1024,  # 2 MB chunks
            show_progress=verbose,
        )

        # Check if the file was successfully downloaded
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            raise Exception(f"Download failed or file is empty: {file_path}")

        logger.info(f"File successfully downloaded to: {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"Failed to download video from {video_url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading video: {str(e)}")


def calculate_psnr(ref_frame: np.ndarray, dist_frame: np.ndarray) -> float:
    """Calculate PSNR between reference and distorted frames."""
    mse = np.mean((ref_frame - dist_frame) ** 2)
    if mse == 0:
        return 1000
    return 10 * np.log10((255.0**2) / mse)

# @app.post("/score", response_model=ScoringResponse)
async def score(request: ScoringRequest):
    ref_path = request.reference_path
    ref_cap = cv2.VideoCapture(ref_path)

    if not ref_cap.isOpened():
        raise HTTPException(status_code=500, detail="Error opening reference video file")

    ref_total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(ref_total_frames)
    if ref_total_frames <= 0:
        raise HTTPException(status_code=500, detail="Invalid reference video: no frames found")

    scores = []
    for dist_url in request.distorted_urls:
        ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # dist_path = await download_video(dist_url, request.verbose)
        dist_path = "/root/workspace/vidaio-subnet/videos/4k_4887282_hd.mp4"
        dist_cap = cv2.VideoCapture(dist_path)

        if not dist_cap.isOpened():
            logger.warning(f"Error opening distorted video file from {dist_url}. Assigning score of 0.")
            scores.append(0.0)
            continue  # Skip to the next distorted video

        dist_total_frames = int(dist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(dist_total_frames)
        if dist_total_frames != ref_total_frames:
            logger.warning(
                f"Video length mismatch for {dist_url}: ref({ref_total_frames}) != dist({dist_total_frames}). Assigning score of 0."
            )
            scores.append(0.0)
            dist_cap.release()
            os.unlink(dist_path)
            continue  # Skip to the next distorted video

        # Calculate VMAF
        try:
            vmaf_score = calculate_vmaf(ref_path, dist_path)
        except Exception as e:
            logger.warning(f"Failed to calculate VMAF for {dist_url}: {str(e)}. Assigning score of 0.")
            scores.append(0.0)
            dist_cap.release()
            # os.unlink(dist_path)
            continue  # Skip to the next distorted video

        # Select two random frames for LPIPS calculation
        frame_indices = random.sample(range(ref_total_frames), 2)
        lpips_scores = []

        for idx in frame_indices:
            ref_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret_ref, ref_frame = ref_cap.read()
            dist_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret_dist, dist_frame = dist_cap.read()

            if not ret_ref or not ret_dist:
                logger.warning(f"Frames cannot be read for {dist_url} at index {idx}. Assigning score of 0.")
                lpips_scores.append(1.0)  # Assign a high LPIPS score (bad quality)
                continue

            lpips_score = calculate_lpips(ref_frame, dist_frame)
            lpips_scores.append(lpips_score)
        print(lpips_score)
        if lpips_scores:
            average_lpips = sum(lpips_scores) / len(lpips_scores)
        else:
            average_lpips = 1.0  # No valid LPIPS scores, assume worst case

        # Final score calculation
        final_score = vmaf_score * 0.6 + (1 - average_lpips) * 100 * 0.4
        scores.append(final_score)

        dist_cap.release()
        # os.unlink(dist_path)

    # Cleanup
    ref_cap.release()

    return ScoringResponse(scores=scores)

if __name__ == "__main__":

    # import uvicorn
    
    # host = CONFIG.score.host
    # port = CONFIG.score.port
    
    # uvicorn.run(app, host=host, port=port)
    
    
    #testing
    urls = ScoringRequest(
        distorted_urls=["/root/workspace/vidaio-subnet/videos/4k_4887282_hd.mp4",],
        reference_path="/root/workspace/vidaio-subnet/videos/4887282_dci4k.mp4"
    )

    scores = asyncio.run(score(urls))  
    print(scores)