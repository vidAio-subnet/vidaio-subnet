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
from vmaf_metric import calculate_vmaf, convert_mp4_to_y4m, trim_video
from lpips_metric import calculate_lpips
from pieapp_metric import calculate_pieapp_score
from vidaio_subnet_core import CONFIG

# Set up logging
logger = logging.getLogger(__name__)
app = FastAPI()
fire_requests = FireRequests()

VMAF_THRESHOLD = CONFIG.score.vmaf_threshold
SAMPLE_FRAME_COUNT = CONFIG.score.pieapp_sample_count
PIEAPP_THRESHOLD = CONFIG.score.pieapp_threshold
VMAF_SAMPLE_COUNT = CONFIG.score.vmaf_sample_count

class SyntheticsScoringRequest(BaseModel):
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

class OrganicsScoringRequest(BaseModel):
    """
    Request model for scoring. Contains URLs for distorted videos and the reference video path.
    """
    distorted_urls: List[str]
    reference_urls: List[str]
    task_types: List[str]
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

def get_frame_count(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return int(result.stdout.strip())

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

def upscale_video(input_path, scale_factor=2):
    """
    Upscales a video using FFmpeg by the specified scale factor.
    
    Args:
        input_path (str): Path to the input video file
        scale_factor (int): Factor by which to upscale the video (default: 2)
        
    Returns:
        str: Path to the upscaled video file
    """
    # Create output filename with scale factor indication
    filename, extension = os.path.splitext(input_path)
    output_path = f"{filename}_upscaled{extension}"
    
    # Get video dimensions
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Calculate new dimensions
    new_width = width * scale_factor
    new_height = height * scale_factor
    
    # FFmpeg command to upscale the video
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vf", f"scale={new_width}:{new_height}",
        "-c:v", "libx264",
        "-preset", "medium",  # Balance between speed and quality
        "-crf", "18",         # High quality
        "-c:a", "copy",       # Copy audio stream without re-encoding
        output_path
    ]
    
    # Execute the command
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Successfully upscaled video to {new_width}x{new_height}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error upscaling video: {e}")
        # If upscaling fails, return the original path
        return input_path

@app.post("/score_synthetics")
async def score_synthetics(request: SyntheticsScoringRequest) -> ScoringResponse:
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

    sample_size = min(SAMPLE_FRAME_COUNT, ref_total_frames)
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

    video_duration = VideoFileClip(ref_path).duration
    start_point = random.uniform(0, video_duration - 1)
    print(f"randomly selected 1s video clip for vmaf score from {start_point}s.")

    if ref_total_frames < 10:
        raise ValueError("Video must contain at least 10 frames.")
    
    random_frames = sorted(random.sample(range(ref_total_frames), VMAF_SAMPLE_COUNT))

    ref_y4m_path = convert_mp4_to_y4m(ref_path, random_frames)
    print("the reference video has been successfully trimmed and converted to y4m format.")

    url_cache = {}

    for dist_url, uid in zip(request.distorted_urls, request.uids):
        print(f"🧩 processing {uid}.... attempting to download processed video.... 🧩")
        dist_path = None
        cache_entry = url_cache.get(dist_url)

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
                # process and cache
                if len(dist_url) < 10:
                    print(f"wrong download url: {dist_url}. assigning score of 0.")
                    vmaf_scores.append(0.0)
                    pieapp_scores.append(0.0)
                    reasons.append("wrong download url")
                    scores.append(0.0)
                    url_cache[dist_url] = {"status": "fail", "reason": "wrong download url"}
                    continue

                dist_path = await download_video(dist_url, request.verbose)
                dist_cap = cv2.VideoCapture(dist_path)

                if not dist_cap.isOpened():
                    print(f"error opening distorted video file from {dist_url}. assigning score of 0.")
                    vmaf_scores.append(0.0)
                    pieapp_scores.append(0.0)
                    reasons.append("error opening distorted video file")
                    scores.append(0.0)
                    url_cache[dist_url] = {"status": "fail", "reason": "error opening distorted video file"}
                    continue

                dist_total_frames = int(dist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"distorted video has {dist_total_frames} frames.")

                if dist_total_frames != ref_total_frames:
                    print(
                        f"video length mismatch for {dist_url}: ref({ref_total_frames}) != dist({dist_total_frames}). assigning score of 0."
                    )
                    vmaf_scores.append(0.0)
                    pieapp_scores.append(0.0)
                    reasons.append("video length mismatch")
                    scores.append(0.0)
                    url_cache[dist_url] = {"status": "fail", "reason": "video length mismatch"}
                    dist_cap.release()
                    if dist_path and os.path.exists(dist_path):
                        os.unlink(dist_path)
                    continue

                url_cache[dist_url] = {"status": "ok", "path": dist_path}
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
                url_cache[dist_url] = {
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
                url_cache[dist_url] = {
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
            url_cache[dist_url] = {
                "status": "ok",
                "path": dist_path,
                "vmaf_score": vmaf_score,
                "pieapp_score": pieapp_score,
                "score": final_score,
                "reason": "success"
            }

        except Exception as e:
            error_msg = f"failed to process video from {dist_url}: {str(e)}"
            print(f"{error_msg}. assigning score of 0.")
            vmaf_scores.append(0.0)
            pieapp_scores.append(0.0)
            reasons.append("failed to process video")
            scores.append(0.0)
            url_cache[dist_url] = {"status": "fail", "reason": "failed to process video"}

        finally:
            if dist_path and os.path.exists(dist_path):
                remaining_uses = request.distorted_urls[request.distorted_urls.index(dist_url)+1:].count(dist_url)
                if remaining_uses == 0:
                    os.unlink(dist_path)

    # cleanup
    ref_cap.release()
    if os.path.exists(ref_trim_path):
        os.unlink(ref_trim_path)
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

@app.post("/score_organics")
async def score_organics(request: OrganicsScoringRequest) -> ScoringResponse:
    print("#################### 🤖 start scoring ####################")
    start_time = time.time()
    scores = []
    vmaf_scores = []
    pieapp_scores = []
    reasons = []

    # step 1: download all distorted videos first
    distorted_video_paths = []
    for dist_url in request.distorted_urls:
        if len(dist_url) < 10:
            distorted_video_paths.append(None)
            continue
        try:
            path = await download_video(dist_url, request.verbose)
            distorted_video_paths.append(path)
        except Exception as e:
            print(f"failed to download distorted video: {dist_url}, error: {e}")
            distorted_video_paths.append(None)

    # step 2: process each pair
    for idx, (ref_url, dist_path, uid, task_type) in enumerate(
        zip(request.reference_urls, distorted_video_paths, request.uids, request.task_types)
    ):
        print(f"🧩 processing {uid}.... downloading reference video.... 🧩")
        ref_path = None
        ref_cap = None
        dist_cap = None
        ref_trim_path = None
        ref_y4m_path = None

        scale_factor = 2
        if task_type == "SD24K":
            scale_factor = 4

        try:
            # download reference video
            if len(ref_url) < 10:
                print(f"invalid reference download url: {ref_url}. skipping...")
                vmaf_scores.append(-1)
                pieapp_scores.append(-1)
                reasons.append("wrong reference download url")
                scores.append(-100)
                continue

            original_video_path = await download_video(ref_url, request.verbose)
            ref_path = upscale_video(original_video_path, scale_factor)
            ref_cap = cv2.VideoCapture(ref_path)

            if not ref_cap.isOpened():
                print(f"error opening reference video file from {ref_url}. skipping...")
                vmaf_scores.append(-1)
                pieapp_scores.append(-1)
                reasons.append("error opening reference video file")
                scores.append(-100)
                continue

            ref_total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"reference video has {ref_total_frames} frames.")

            if ref_total_frames <= 0:
                print(f"invalid reference video from {ref_url}: no frames found. skipping...")
                vmaf_scores.append(-1)
                pieapp_scores.append(-1)
                reasons.append("invalid reference video: no frames found")
                scores.append(-100)
                ref_cap.release()
                continue

            # check if distorted video failed to download
            if dist_path is None:
                print(f"failed to download distorted video for uid {uid}. assigning score of 0.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                reasons.append("failed to download distorted video")
                scores.append(0.0)
                ref_cap.release()
                continue

            dist_cap = cv2.VideoCapture(dist_path)
            if not dist_cap.isOpened():
                print(f"error opening distorted video file from {dist_path}. assigning score of 0.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                reasons.append("error opening distorted video file")
                scores.append(0.0)
                ref_cap.release()
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
                ref_cap.release()
                dist_cap.release()
                continue

            sample_size = min(SAMPLE_FRAME_COUNT, ref_total_frames)
            max_start_frame = ref_total_frames - sample_size
            start_frame = 0 if max_start_frame <= 0 else random.randint(0, max_start_frame)
            print(f"selected frame range for video pair: {start_frame} to {start_frame + sample_size - 1}")

            ref_frames = []
            ref_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(sample_size):
                ret, frame = ref_cap.read()
                if not ret:
                    break
                ref_frames.append(frame)

            video_duration = VideoFileClip(ref_path).duration
            start_point = random.uniform(0, video_duration - 1)
            print(f"randomly selected 1s video clip for vmaf score from {start_point}s.")

            ref_trim_path = trim_video(ref_path, start_point)
            ref_y4m_path = convert_mp4_to_y4m(ref_trim_path)
            print("the reference video has been successfully trimmed and converted to y4m format.")

            # calculate vmaf
            try:
                vmaf_score = calculate_vmaf(ref_y4m_path, dist_path, start_point)
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
                ref_cap.release()
                dist_cap.release()
                print(f"error calculating vmaf score: {e}")
                continue

            if vmaf_score / 100 < VMAF_THRESHOLD:
                print(f"vmaf score is too low, giving zero score, current vmaf score: {vmaf_score}")
                pieapp_scores.append(0.0)
                reasons.append(f"vmaf score is too low, current vmaf score: {vmaf_score}")
                scores.append(0.0)
                ref_cap.release()
                dist_cap.release()
                continue

            dist_frames = []
            dist_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(sample_size):
                ret, frame = dist_cap.read()
                if not ret:
                    break
                dist_frames.append(frame)

            pieapp_score = calculate_pieapp_score_on_samples(ref_frames, dist_frames)
            pieapp_scores.append(pieapp_score)
            print(f"🎾 pieapp score is {pieapp_score}")

            if pieapp_score > PIEAPP_THRESHOLD:
                print(f"pieapp score is too low, giving zero score, current pieapp score: {pieapp_score}")
                pieapp_scores.append(0.0)
                reasons.append(f"pieapp score is too low, current pieapp score: {pieapp_score}")
                scores.append(0.0)
                ref_cap.release()
                dist_cap.release()
                continue

            reasons.append("success")
            scores.append(1)

        except Exception as e:
            error_msg = f"failed to process video pair (ref: {ref_url}, dist: {dist_path}): {str(e)}. failed to download video"
            print(f"{error_msg}. assigning score of 0.")
            vmaf_scores.append(0.0)
            pieapp_scores.append(0.0)
            reasons.append("failed to process video pair")
            scores.append(0.0)

        finally:
            if ref_cap:
                ref_cap.release()
            if dist_cap:
                dist_cap.release()
            if ref_path and os.path.exists(ref_path):
                os.unlink(ref_path)
            if dist_path and os.path.exists(dist_path):
                os.unlink(dist_path)
            if ref_trim_path and os.path.exists(ref_trim_path):
                os.unlink(ref_trim_path)
            if ref_y4m_path and os.path.exists(ref_y4m_path):
                os.unlink(ref_y4m_path)

    processed_time = time.time() - start_time
    print(f"🔯🔯🔯 calculated score: {scores} 🔯🔯🔯")
    print(f"completed one batch scoring within {processed_time:.2f} seconds")
    return ScoringResponse(
        scores=scores,
        vmaf_scores=vmaf_scores,
        pieapp_scores=pieapp_scores,
        reasons=reasons
    )

if __name__ == "__main__":
    import uvicorn
    
    host = CONFIG.score.host
    port = CONFIG.score.port
    
    uvicorn.run(app, host=host, port=port)
