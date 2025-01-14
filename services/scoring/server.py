from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from loguru import logger
import cv2
import numpy as np
from firerequests import FireRequests
import tempfile
import os

app = FastAPI()
fire_requests = FireRequests()


class ScoringRequest(BaseModel):
    disorted_url: str
    reference_url: str
    fps: Optional[float] = None
    subsample: Optional[int] = 1
    verbose: Optional[bool] = False
    progress: Optional[bool] = False


class ScoringResponse(BaseModel):
    score: float


async def download_videos(request: ScoringRequest) -> tuple[str, str]:
    """Download reference and distorted videos to temporary files.
    
    Args:
        request: The scoring request containing video URLs
        
    Returns:
        Tuple of paths to downloaded reference and distorted videos
    """
    with (
        tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as ref_temp,
        tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as dist_temp,
    ):
        ref_path = ref_temp.name
        dist_path = dist_temp.name

    logger.info(f"Downloading videos from {request.reference_url} and {request.disorted_url}")
    await fire_requests.download_file(
        request.reference_url,
        ref_path,
        max_files=10,
        chunk_size=2 * 1024 * 1024,
        show_progress=request.verbose,
    )
    await fire_requests.download_file(
        request.disorted_url,
        dist_path,
        max_files=10,
        chunk_size=2 * 1024 * 1024,
        show_progress=request.verbose,
    )
    
    return ref_path, dist_path


def calculate_psnr(ref_frame: np.ndarray, dist_frame: np.ndarray) -> float:
    """Calculate PSNR between reference and distorted frames.
    
    Args:
        ref_frame: Reference video frame
        dist_frame: Distorted video frame
        
    Returns:
        PSNR value in dB
    """
    mse = np.mean((ref_frame - dist_frame) ** 2)
    if mse == 0:
        return 1000
    return 10 * np.log10((255.0**2) / mse)


@app.post("/score", response_model=ScoringResponse)
async def score(request: ScoringRequest):
    try:
        ref_path, dist_path = await download_videos(request)

        # Open both videos
        logger.info("Opening downloaded video files")
        ref_cap = cv2.VideoCapture(ref_path)
        dist_cap = cv2.VideoCapture(dist_path)

        if not ref_cap.isOpened() or not dist_cap.isOpened():
            raise HTTPException(status_code=500, detail="Error opening video files")

        psnr_values = []
        frame_count = 0

        while True:
            ret_ref, ref_frame = ref_cap.read()
            ret_dist, dist_frame = dist_cap.read()

            if not ret_ref or not ret_dist:
                break

            # Skip frames based on subsample rate
            if frame_count % request.subsample != 0:
                frame_count += 1
                continue

            if ref_frame.shape != dist_frame.shape:
                raise HTTPException(
                    status_code=500, detail="Video dimensions do not match"
                )

            psnr = calculate_psnr(ref_frame, dist_frame)
            psnr_values.append(psnr)
            frame_count += 1
            logger.info(f"Processing frame {frame_count}, PSNR: {psnr}")

            if request.progress and frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames...")

        # Release video captures and clean up temp files
        ref_cap.release()
        dist_cap.release()
        os.unlink(ref_path)
        os.unlink(dist_path)

        if not psnr_values:
            raise HTTPException(status_code=500, detail="No frames were processed")

        average_psnr = np.mean(psnr_values)

        if request.verbose:
            logger.info(f"Average PSNR: {average_psnr:.2f} dB")
            logger.info(f"Total frames processed: {frame_count}")

        return ScoringResponse(score=float(average_psnr))

    except Exception as e:
        # Clean up temp files in case of error
        if "ref_path" in locals():
            try:
                os.unlink(ref_path)
            except:
                pass
        if "dist_path" in locals():
            try:
                os.unlink(dist_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")
