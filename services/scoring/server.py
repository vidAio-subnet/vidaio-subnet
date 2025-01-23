from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from loguru import logger
import cv2
import numpy as np
from firerequests import FireRequests
import tempfile
import os

app = FastAPI()
fire_requests = FireRequests()


class ScoringRequest(BaseModel):
    disorted_urls: List[str]
    reference_url: str
    fps: Optional[float] = None
    subsample: Optional[int] = 1
    verbose: Optional[bool] = False
    progress: Optional[bool] = False


class ScoringResponse(BaseModel):
    score: List[float]


async def download_video(video_url, verbose) -> str:
    """Download reference and distorted videos to temporary files.
    
    Args:
        request: The scoring request containing video URLs
        
    Returns:
        Tuple of paths to downloaded reference and distorted videos
    """
    with (
        tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as vid_temp,
    ):
        file_path = vid_temp.name

    logger.info(f"Downloading videos from {video_url}")
    await fire_requests.download_file(
        video_url,
        file_path,
        max_files=10,
        chunk_size=2 * 1024 * 1024,
        show_progress=verbose,
    )
    
    return file_path


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
    ref_path = None
    dist_path = None
    ref_cap = None
    dist_cap = None
    
    try:
        # Download reference video only once since all references are same
        ref_path = await download_video(request.reference_url, request.verbose)
        ref_cap = cv2.VideoCapture(ref_path)
        
        if not ref_cap.isOpened():
            raise HTTPException(status_code=500, detail="Error opening reference video file")
            
        ref_total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if ref_total_frames <= 0:
            raise HTTPException(status_code=500, detail="Invalid reference video: no frames found")
        
        scores = []
        for dist_url in request.disorted_urls:
            # Reset reference video position for each comparison
            ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            dist_path = await download_video(dist_url, request.verbose)
            dist_cap = cv2.VideoCapture(dist_path)
            
            if not dist_cap.isOpened():
                raise HTTPException(status_code=500, detail="Error opening distorted video file")
                
            dist_total_frames = int(dist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if dist_total_frames != ref_total_frames:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Video length mismatch: ref({ref_total_frames}) != dist({dist_total_frames})"
                )
            
            psnr_values = []
            frame_count = 0
            
            while True:
                ret_ref, ref_frame = ref_cap.read()
                ret_dist, dist_frame = dist_cap.read()
                
                if not ret_ref and not ret_dist:
                    break
                elif not ret_ref or not ret_dist:
                    raise HTTPException(
                        status_code=500, 
                        detail="Videos have different lengths or are corrupted"
                    )
                
                if frame_count % request.subsample != 0:
                    frame_count += 1
                    continue
                
                if ref_frame.shape != dist_frame.shape:
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Frame dimensions do not match at frame {frame_count}"
                    )
                
                psnr = calculate_psnr(ref_frame, dist_frame)
                psnr_values.append(psnr)
                frame_count += 1
                
                if request.verbose:
                    logger.info(f"Processing frame {frame_count}, PSNR: {psnr}")
                
                if request.progress and frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames...")
            
            if not psnr_values:
                raise HTTPException(status_code=500, detail="No frames were processed")
            
            average_psnr = float(np.mean(psnr_values))
            
            if request.verbose:
                logger.info(f"Average PSNR: {average_psnr:.2f} dB")
                logger.info(f"Total frames processed: {frame_count}")
            
            scores.append(average_psnr)
            
            if dist_cap is not None:
                dist_cap.release()
            if dist_path is not None:
                os.unlink(dist_path)
                dist_path = None
        
        return ScoringResponse(scores=scores)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")
    finally:
        if ref_cap is not None:
            ref_cap.release()
        if dist_cap is not None:
            dist_cap.release()
        if ref_path is not None:
            try:
                os.unlink(ref_path)
            except OSError:
                logger.warning(f"Failed to delete reference file: {ref_path}")
        if dist_path is not None:
            try:
                os.unlink(dist_path)
            except OSError:
                logger.warning(f"Failed to delete distorted file: {dist_path}")
