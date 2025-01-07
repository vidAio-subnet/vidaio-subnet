from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import json
import re
from typing import Optional
from loguru import logger
import cv2
import numpy as np

app = FastAPI()


class ScoringRequest(BaseModel):
    distorted_path: str
    reference_path: str
    fps: Optional[float] = None
    subsample: Optional[int] = 1
    verbose: Optional[bool] = False
    progress: Optional[bool] = False


class ScoringResponse(BaseModel):
    score: float


@app.post("/score", response_model=ScoringResponse)
async def score(request: ScoringRequest):
    try:
        # Open both videos
        logger.info(
            f"Opening video files: {request.reference_path} and {request.distorted_path}"
        )
        ref_cap = cv2.VideoCapture(request.reference_path)
        dist_cap = cv2.VideoCapture(request.distorted_path)

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

            mse = np.mean((ref_frame - dist_frame) ** 2)
            if mse == 0:
                psnr = 1000
            else:
                psnr = 10 * np.log10((255.0**2) / mse)

            psnr_values.append(psnr)
            frame_count += 1
            logger.info(f"Processing frame {frame_count}, PSNR: {psnr}, MSE: {mse}")

            if request.progress and frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames...")

        # Release video captures
        ref_cap.release()
        dist_cap.release()

        if not psnr_values:
            raise HTTPException(status_code=500, detail="No frames were processed")

        average_psnr = np.mean(psnr_values)

        if request.verbose:
            logger.info(f"Average PSNR: {average_psnr:.2f} dB")
            logger.info(f"Total frames processed: {frame_count}")

        return ScoringResponse(score=float(average_psnr))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")
