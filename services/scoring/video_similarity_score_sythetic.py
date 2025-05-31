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

from vmaf_metric import calculate_vmaf, convert_mp4_to_y4m, trim_video
from lpips_metric import calculate_lpips
from pieapp_metric import calculate_pieapp_score
from vidaio_subnet_core import CONFIG

# Set up logging
logger = logging.getLogger(__name__)
app = FastAPI()
fire_requests = FireRequests()

VMAF_THRESHOLD = CONFIG.score.vmaf_threshold
PIEAPP_SAMPLE_COUNT = CONFIG.score.pieapp_sample_count
PIEAPP_THRESHOLD = CONFIG.score.pieapp_threshold
VMAF_SAMPLE_COUNT = CONFIG.score.vmaf_sample_count

class SyntheticsScoringRequest(BaseModel):
    """
    Request model for scoring. Contains URLs for distorted videos and the reference video path.
    """
    distorted_paths: List[str]
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
    vmaf_scores: List[float]
    pieapp_scores: List[float]
    reasons: List[str]

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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_final_score(pieapp_score):
    sigmoid_normalized_score = sigmoid(pieapp_score)
    
    original_at_zero = (1 - (np.log10(sigmoid(0) + 1) / np.log10(3.5))) ** 2.5
    original_at_two = (1 - (np.log10(sigmoid(2.0) + 1) / np.log10(3.5))) ** 2.5
    
    original_value = (1 - (np.log10(sigmoid_normalized_score + 1) / np.log10(3.5))) ** 2.5
    
    scaled_value = 1 - ((original_value - original_at_zero) / (original_at_two - original_at_zero))
    
    return scaled_value


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
    frames_to_sample = min(PIEAPP_SAMPLE_COUNT, total_frames)
    
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

async def score(request: SyntheticsScoringRequest) -> ScoringResponse:
    print("#################### 🤖 start scoring ####################")

    start_time = time.time()
    scores = []
    vmaf_scores = []
    pieapp_scores = []
    reasons = []

    ref_path = request.reference_path
    ref_cap = cv2.VideoCapture(ref_path)

    if not ref_cap.isOpened():
        raise HTTPException(status_code=500, detail="error opening reference video file")

    ref_total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"reference video has {ref_total_frames} frames.")

    if ref_total_frames <= 0:
        raise HTTPException(status_code=500, detail="invalid reference video: no frames found")

    sample_size = min(PIEAPP_SAMPLE_COUNT, ref_total_frames)
    max_start_frame = ref_total_frames - sample_size
    start_frame = 0 if max_start_frame <= 0 else random.randint(0, max_start_frame)

    print(f"selected frame range for all videos: {start_frame} to {start_frame + sample_size - 1}")

    ref_frames = []
    ref_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(sample_size):
        ret, frame = ref_cap.read()
        if not ret:
            break
        ref_frames.append(frame)


    if ref_total_frames < 10:
        raise ValueError("Video must contain at least 10 frames.")
    
    random_frames = sorted(random.sample(range(ref_total_frames), VMAF_SAMPLE_COUNT))
    print(f"randomly selected {VMAF_SAMPLE_COUNT}frames for vmaf score: frame list: {random_frames}")

    ref_y4m_path = convert_mp4_to_y4m(ref_path, random_frames)
    print("the reference video has been successfully trimmed and converted to y4m format.")

    url_cache = {}

    for dist_path, uid in zip(request.distorted_paths, request.uids):
        print(f"🧩 processing {uid}.... attempting to download processed video.... 🧩")
        cache_entry = url_cache.get(dist_path)

        try:
            if cache_entry is not None:
                if cache_entry["status"] == "fail":
                    vmaf_scores.append(0.0)
                    pieapp_scores.append(0.0)
                    reasons.append(cache_entry["reason"])
                    scores.append(0.0)
                    print(f"Using cached failure result for {uid} from previous identical URL")
                    continue
                elif "score" in cache_entry:
                    vmaf_scores.append(cache_entry["vmaf_score"])
                    pieapp_scores.append(cache_entry["pieapp_score"])
                    reasons.append(cache_entry["reason"])
                    scores.append(cache_entry["score"])
                    print(f"Using cached scoring result for {uid} from previous identical URL")
                    continue
                else:
                    dist_path = cache_entry["path"]
            else:

                dist_cap = cv2.VideoCapture(dist_path)

                if not dist_cap.isOpened():
                    print(f"error opening distorted video file from {dist_path}. assigning score of 0.")
                    vmaf_scores.append(0.0)
                    pieapp_scores.append(0.0)
                    reasons.append("error opening distorted video file")
                    scores.append(0.0)
                    url_cache[dist_path] = {"status": "fail", "reason": "error opening distorted video file"}
                    continue

                dist_total_frames = int(dist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"distorted video has {dist_total_frames} frames.")

                if dist_total_frames != ref_total_frames:
                    print(
                        f"video length mismatch for {dist_path}: ref({ref_total_frames}) != dist({dist_total_frames}). assigning score of 0."
                    )
                    vmaf_scores.append(0.0)
                    pieapp_scores.append(0.0)
                    reasons.append("video length mismatch")
                    scores.append(0.0)
                    url_cache[dist_path] = {"status": "fail", "reason": "video length mismatch"}
                    dist_cap.release()
                    if dist_path and os.path.exists(dist_path):
                        os.unlink(dist_path)
                    continue

                url_cache[dist_path] = {"status": "ok", "path": dist_path}
                dist_cap.release()

            dist_cap = cv2.VideoCapture(dist_path)

            # calculate vmaf
            try:
                vmaf_score = calculate_vmaf(ref_y4m_path, dist_path, random_frames)
                if vmaf_score is not None:
                    vmaf_scores.append(vmaf_score)
                else:
                    vmaf_score = 0.0
                    vmaf_scores.append(vmaf_score)
                print(f"🎾 vmaf_score is {vmaf_score}")
            except Exception as e:
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                reasons.append("failed to calculate vmaf score due to video dimension mismatch")
                scores.append(0.0)
                dist_cap.release()
                print(f"error calculating vmaf score: {e}")
                # Cache the failure
                url_cache[dist_path] = {
                    "status": "fail", 
                    "reason": "failed to calculate vmaf score due to video dimension mismatch"
                }
                continue

            if vmaf_score / 100 < VMAF_THRESHOLD:
                print(f"vmaf score is too low, giving zero score, current vmaf score: {vmaf_score}")
                pieapp_scores.append(0.0)
                reasons.append(f"vmaf score is too low, current vmaf score: {vmaf_score}")
                scores.append(0.0)
                dist_cap.release()
                # Cache this result
                url_cache[dist_path] = {
                    "status": "ok",
                    "path": dist_path,
                    "vmaf_score": vmaf_score,
                    "pieapp_score": 0.0,
                    "score": 0.0,
                    "reason": f"vmaf score is too low, current vmaf score: {vmaf_score}"
                }
                continue

            # extract distorted frames
            dist_frames = []
            dist_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(sample_size):
                ret, frame = dist_cap.read()
                if not ret:
                    break
                dist_frames.append(frame)

            pieapp_score = calculate_pieapp_score_on_samples(ref_frames, dist_frames)
            pieapp_scores.append(pieapp_score)
            print(f"🎾 pieapp_score is {pieapp_score}")

            final_score = calculate_final_score(pieapp_score)
            print(f"🏀 final_score is {final_score}")

            reasons.append("success")
            scores.append(final_score)
            dist_cap.release()
            
            # Cache the complete successful result
            url_cache[dist_path] = {
                "status": "ok",
                "path": dist_path,
                "vmaf_score": vmaf_score,
                "pieapp_score": pieapp_score,
                "score": final_score,
                "reason": "success"
            }

        except Exception as e:
            error_msg = f"failed to process video from {dist_path}: {str(e)}"
            print(f"{error_msg}. assigning score of 0.")
            vmaf_scores.append(0.0)
            pieapp_scores.append(0.0)
            reasons.append("failed to process video")
            scores.append(0.0)
            url_cache[dist_path] = {"status": "fail", "reason": "failed to process video"}

        finally:
            pass
            # if dist_path and os.path.exists(dist_path):
            #     remaining_uses = request.distorted_urls[request.distorted_urls.index(dist_url)+1:].count(dist_url)
            #     if remaining_uses == 0:
            #         os.unlink(dist_path)

    # cleanup
    ref_cap.release()
    if os.path.exists(ref_y4m_path):
        os.unlink(ref_y4m_path)

    print(f"🔯🔯🔯 calculated score: {scores} 🔯🔯🔯")

    processed_time = time.time() - start_time
    print(f"completed one batch scoring within {processed_time:.2f} seconds")
    return ScoringResponse(
        scores=scores,
        vmaf_scores=vmaf_scores,
        pieapp_scores=pieapp_scores,
        reasons=reasons
    )

if __name__ == "__main__":
    import asyncio

    # Define the paths for the reference and distorted videos
    reference_path = ""
    distorted_paths = [
        "",
    ]
    uids = [1, ]

    async def main():
        request = SyntheticsScoringRequest(
            distorted_paths=distorted_paths,
            reference_path=reference_path,
            uids=uids,
            fps=None,  # Optional
            subsample=1,  # Optional
            verbose=True,  # Enable verbose logging if needed
            progress=False  # Disable progress tracking
        )

        try:
            response = await score(request)
            print("Scoring Results:")
            print(f"Scores: {response.scores}")
            print(f"VMAF Scores: {response.vmaf_scores}")
            print(f"PIE-APP Scores: {response.pieapp_scores}")
            print(f"Reasons: {response.reasons}")
        except Exception as e:
            print(f"Error during scoring: {str(e)}")

    asyncio.run(main())