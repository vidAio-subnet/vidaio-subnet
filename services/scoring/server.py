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
import time

# Set up logging
logger = logging.getLogger(__name__)
app = FastAPI()
fire_requests = FireRequests()

VMAF_THRESHOLD = CONFIG.score.vmaf_threshold
SAMPLE_FRAME_COUNT = CONFIG.score.sample_count

class ScoringRequest(BaseModel):
    """
    Request model for scoring. Contains URLs for distorted videos and the reference video path.
    """
    distorted_urls: List[str]
    reference_path: str
    uids: List[int]
    fps: Optional[float] = None
    subsample: Optional[int] = 1
    verbose: Optional[bool] = False
    progress: Optional[bool] = False

class ScoringResponse(BaseModel):
    """
    Response model for scoring. Contains the list of calculated scores for each d ofistorted video.
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
        print(f"Downloading video from {video_url} to {file_path}")

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

        print(f"File successfully downloaded to: {file_path}")
        return file_path

    except Exception as e:
        print(f"Failed to download video from {video_url}: {str(e)}")
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

def calculate_final_score(pieapp_score):
    return (1 - (np.log10(pieapp_score + 1) / np.log10(3.5))) ** 2.5

def get_sample_frames(ref_cap, dist_cap, total_frames):
    """
    Get sample frames from both reference and distorted videos.
    
    Args:
        ref_cap: Reference video capture
        dist_cap: Distorted video capture
        total_frames: Total number of frames in the videos
        
    Returns:
        tuple: (ref_frames, dist_frames) - Lists of sampled frames
    """
    # Determine how many frames to sample
    frames_to_sample = min(SAMPLE_FRAME_COUNT, total_frames)
    
    # Generate a random starting point that ensures we can get consecutive frames
    # without exceeding the total number of frames
    max_start_frame = total_frames - frames_to_sample
    if max_start_frame <= 0:
        start_frame = 0
    else:
        start_frame = random.randint(0, max_start_frame)
    
    print(f"Sampling {frames_to_sample} consecutive frames starting from frame {start_frame}")
    
    # Extract frames from reference video
    ref_frames = []
    ref_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(frames_to_sample):
        ret, frame = ref_cap.read()
        if not ret:
            break
        ref_frames.append(frame)
    
    # Extract frames from distorted video
    dist_frames = []
    dist_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(frames_to_sample):
        ret, frame = dist_cap.read()
        if not ret:
            break
        dist_frames.append(frame)
    
    return ref_frames, dist_frames

def calculate_pieapp_score_on_samples(ref_frames, dist_frames):
    """
    Calculate PIE-APP score on sampled frames without creating temporary files.
    
    Args:
        ref_frames: List of reference frames
        dist_frames: List of distorted frames
        
    Returns:
        float: Average PIE-APP score
    """
    if not ref_frames or not dist_frames:
        print("No frames to process")
        return 2.0  # Return max penalty if no frames
    
    # Create a custom frame provider class that mimics cv2.VideoCapture
    class FrameProvider:
        def __init__(self, frames):
            self.frames = frames
            self.current_frame = 0
            self.frame_count = len(frames)
        
        def read(self):
            if self.current_frame < self.frame_count:
                frame = self.frames[self.current_frame]
                self.current_frame += 1
                return True, frame
            return False, None
        
        def get(self, prop_id):
            if prop_id == cv2.CAP_PROP_FRAME_COUNT:
                return self.frame_count
            # Add other properties as needed
            return 0
        
        def set(self, prop_id, value):
            if prop_id == cv2.CAP_PROP_POS_FRAMES:
                self.current_frame = int(value) if value < self.frame_count else self.frame_count
                return True
            return False
        
        def release(self):
            # Nothing to release
            pass
        
        def isOpened(self):
            return self.frame_count > 0
    
    # Create frame providers that will act as cv2.VideoCapture objects
    ref_provider = FrameProvider(ref_frames)
    dist_provider = FrameProvider(dist_frames)
    
    # Calculate PieAPP score using the original function
    try:
        # Use frame_interval=1 to process all frames since we're already working with sampled frames
        score = calculate_pieapp_score(ref_provider, dist_provider, frame_interval=1)
        return score
    except Exception as e:
        print(f"Error calculating PieAPP score on frames: {str(e)}")
        return 2.0  # Return max penalty on error

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
    print("#################### 🤖 Start scoring ####################")

    start_time = time.time()

    ref_path = request.reference_path
    ref_cap = cv2.VideoCapture(ref_path)

    if not ref_cap.isOpened():
        raise HTTPException(status_code=500, detail="Error opening reference video file")

    ref_total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Reference video has {ref_total_frames} frames.")
    
    if ref_total_frames <= 0:
        raise HTTPException(status_code=500, detail="Invalid reference video: no frames found")

    # Generate a consistent sample frame position for all videos
    sample_size = min(SAMPLE_FRAME_COUNT, ref_total_frames)
    max_start_frame = ref_total_frames - sample_size
    if max_start_frame <= 0:
        start_frame = 0
    else:
        start_frame = random.randint(0, max_start_frame)
    
    print(f"Selected frame range for all videos: {start_frame} to {start_frame + sample_size - 1}")
    
    # Extract reference frames once
    ref_frames = []
    ref_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(sample_size):
        ret, frame = ref_cap.read()
        if not ret:
            break
        ref_frames.append(frame)

    scores = []
    for dist_url, uid in zip(request.distorted_urls, request.uids):
        print(f"🧩 Processing {uid}.... Attempting to download processed video.... 🧩")
        try:
            if len(dist_url) < 10:
                print(f"Wrong download url: {dist_url}. Assigning score of 0.")
                scores.append(0.0)
                continue
            dist_path = await download_video(dist_url, request.verbose)
        except Exception as e:
            print(f"Failed to download video from {dist_url}: {str(e)}. Assigning score of 0.")
            scores.append(0.0)
            continue
         
        dist_cap = cv2.VideoCapture(dist_path)

        if not dist_cap.isOpened():
            print(f"Error opening distorted video file from {dist_url}. Assigning score of 0.")
            scores.append(0.0)
            continue  # Skip to the next distorted video

        dist_total_frames = int(dist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Distorted video has {dist_total_frames} frames.")
        
        if dist_total_frames != ref_total_frames:
            print(
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
            print(f"🎾 vmaf_score is {vmaf_score}")

            if vmaf_score / 100 < VMAF_THRESHOLD:
                print(f"vmaf score is too low, giving zero score, current vmaf score: {vmaf_score}")
                scores.append(0)
                dist_cap.release()
                os.unlink(dist_path)
                continue

        except Exception as e:
            print(f"Failed to calculate VMAF for {dist_url}: {str(e)}. Assigning score of 0.")
            scores.append(0.0)
            dist_cap.release()
            os.unlink(dist_path)
            continue  # Skip to the next distorted video
        
        # Extract distorted frames from the same position
        dist_frames = []
        dist_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(sample_size):
            ret, frame = dist_cap.read()
            if not ret:
                break
            dist_frames.append(frame)
        
        # Calculate PieAPP score on the sampled frames
        pieapp_score = calculate_pieapp_score_on_samples(ref_frames, dist_frames)
        if pieapp_score > 2.0:
            pieapp_score = 2.0
        if pieapp_score < 0.1:
            pieapp_score = 0.1
        print(f"🎾 Pieapp_score is {pieapp_score}")

        # Final score calculation
        final_score = calculate_final_score(pieapp_score)
        print(f"🏀 final_score is {final_score}")
        scores.append(final_score)

        dist_cap.release()
        os.unlink(dist_path)

    # Cleanup
    ref_cap.release()

    print(f"🔯🔯🔯 calculated score: {scores} 🔯🔯🔯")
    processed_time = time.time() - start_time
    print(f"Completed one batch scoring within {processed_time:.2f} seconds")
    return ScoringResponse(scores=scores)


if __name__ == "__main__":
    import uvicorn
    
    host = CONFIG.score.host
    port = CONFIG.score.port
    
    uvicorn.run(app, host=host, port=port)
