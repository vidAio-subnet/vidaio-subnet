from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import cv2
import numpy as np
from firerequests import FireRequests
import tempfile
import os
import random
from vidaio_subnet_core import CONFIG
from vmaf_metric import calculate_vmaf
from lpips_metric import calculate_lpips
from pieapp_metric import calculate_pieapp_score
import aiohttp
import logging

# Set up logging
logger = logging.getLogger(__name__)
app = FastAPI()
fire_requests = FireRequests()

VMAF_THRESHOLD = 0.5


class ScoringRequest(BaseModel):
    """
    Request model for scoring. Contains URLs for distorted videos and the reference video path.
    """
    distorted_urls: List[str]
    reference_path: str
    fps: Optional[float] = None
    subsample: Optional[int] = 1
    verbose: Optional[bool] = False
    progress: Optional[bool] = False

class ScoringResponse(BaseModel):
    """
    Response model for scoring. Contains the list of calculated scores for each distorted video.
    """
    scores: List[float]


async def download_video(video_url: str, verbose: bool) -> str:
    """
    Download a video from the given URL and save it to a temporary file.

    Args:
        video_url (str): The URL of the video to download.
        verbose (bool): Whether to show download progress.

    Returns:
        str: The path to the downloaded video file.

    Raises:
        HTTPException: If the download fails.
    """
    try:
        # Create a temporary file for the video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as vid_temp:
            file_path = vid_temp.name  # Path to the temporary file
        logger.info(f"Downloading video from {video_url} to {file_path}")

        # Download the file using aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(video_url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download video. HTTP status: {response.status}")

                # Write the content to the temp file in chunks
                with open(file_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(2 * 1024 * 1024):  # 2 MB chunks
                        f.write(chunk)

        # Verify the file was successfully downloaded
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            raise Exception(f"Download failed or file is empty: {file_path}")

        logger.info(f"File successfully downloaded to: {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"Failed to download video from {video_url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading video: {str(e)}")


def calculate_psnr(ref_frame: np.ndarray, dist_frame: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between reference and distorted frames.

    Args:
        ref_frame (np.ndarray): The reference video frame.
        dist_frame (np.ndarray): The distorted video frame.

    Returns:
        float: The PSNR value between the reference and distorted frames.
    """
    mse = np.mean((ref_frame - dist_frame) ** 2)
    if mse == 0:
        return 1000  # Maximum PSNR value (perfect similarity)
    return 10 * np.log10((255.0**2) / mse)


@app.post("/score")
async def score(request: ScoringRequest) -> ScoringResponse:
    """
    Scores the distorted videos by comparing them with the reference video using VMAF and LPIPS.

    Args:
        request (ScoringRequest): The request object containing URLs for distorted videos 
                                  and the reference video path.

    Returns:
        ScoringResponse: The response object containing a list of scores for each distorted video.
    """
    logger.info("🤖 Start scoring..........")
    logger.debug(f"Request data: {request.__dict__}")
    
    ref_path = request.reference_path
    ref_cap = cv2.VideoCapture(ref_path)

    if not ref_cap.isOpened():
        raise HTTPException(status_code=500, detail="Error opening reference video file")

    ref_total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Reference video has {ref_total_frames} frames.")
    
    if ref_total_frames <= 0:
        raise HTTPException(status_code=500, detail="Invalid reference video: no frames found")

    scores = []
    for dist_url in request.distorted_urls:
        ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        logger.info("Attempting to download processed video....")
        dist_path = await download_video(dist_url, request.verbose)
        dist_cap = cv2.VideoCapture(dist_path)

        if not dist_cap.isOpened():
            logger.error(f"Error opening distorted video file from {dist_url}. Assigning score of 0.")
            scores.append(0.0)
            continue  # Skip to the next distorted video

        dist_total_frames = int(dist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Distorted video has {dist_total_frames} frames.")
        
        if dist_total_frames != ref_total_frames:
            logger.warning(
                f"Video length mismatch for {dist_url}: ref({ref_total_frames}) != dist({dist_total_frames}). Assigning score of 0."
            )
            scores.append(0.0)
            dist_cap.release()
            os.unlink(dist_path)
            continue  # Skip to the next distorted video

        vmaf_score = 0
        # Calculate VMAF
        try:
            vmaf_score = calculate_vmaf(ref_path, dist_path)
            logger.info(f"🎾 vmaf_score is {vmaf_score}")

        except Exception as e:
            logger.error(f"Failed to calculate VMAF for {dist_url}: {str(e)}. Assigning score of 0.")
            scores.append(0.0)
            dist_cap.release()
            os.unlink(dist_path)
            continue  # Skip to the next distorted video
        
        # # Select two random frames for LPIPS calculation
        # frame_indices = random.sample(range(ref_total_frames), 2)
        # lpips_scores = []

        # for idx in frame_indices:
        #     ref_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        #     ret_ref, ref_frame = ref_cap.read()
        #     dist_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        #     ret_dist, dist_frame = dist_cap.read()

        #     if not ret_ref or not ret_dist:
        #         logger.error(f"Frames cannot be read for {dist_url} at index {idx}. Assigning score of 0.")
        #         lpips_scores.append(1.0)  # Assign a high LPIPS score (bad quality)
        #         continue

        #     lpips_score = calculate_lpips(ref_frame, dist_frame)
        #     lpips_scores.append(lpips_score)

        # logger.info(f"LPIPS scores: {lpips_scores}")
        # average_lpips = sum(lpips_scores) / len(lpips_scores) if lpips_scores else 1.0  # Worst case
        # logger.info(f"Average LPIPS: {average_lpips}")

        # # Final score calculation
        # final_score = vmaf_score * 0.6 / 100 + (1 - average_lpips) * 0.4
        # scores.append(final_score)

        # Calculate pieapp score
        
        pieapp_score = calculate_pieapp_score(ref_cap, dist_cap)
        logger.info(f"🎾 Pieapp_score is {pieapp_score}")

        # Final score calculation
        if vmaf_score / 100 < VMAF_THRESHOLD:
            logger.info(f"vmaf score is too low, giving zero score, current vmaf score: {vmaf_score}")
            scores.append(0)
        else:
            final_score = 1 - pieapp_score**2
            logger.info(f"🏀 final_score is {final_score}")
            scores.append(final_score)

        dist_cap.release()
        os.unlink(dist_path)

    # Cleanup
    ref_cap.release()

    return ScoringResponse(scores=scores)


if __name__ == "__main__":
    import uvicorn
    
    host = CONFIG.score.host
    port = CONFIG.score.port
    
    uvicorn.run(app, host=host, port=port)
    
    # Testing
    # urls = ScoringRequest(
    #     distorted_urls=[...],
    #     reference_path="..."
    # )
    # scores = asyncio.run(score(urls))  
    # print(scores)
