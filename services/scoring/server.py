import os
import cv2
import glob
import time
import math
import random
import asyncio
import aiohttp
import logging
import tempfile
import subprocess
import numpy as np
from pydantic import BaseModel
from typing import Optional, List
from firerequests import FireRequests
from vidaio_subnet_core import CONFIG
from lpips_metric import calculate_lpips
from moviepy.editor import VideoFileClip
from fastapi import FastAPI, HTTPException
from pieapp_metric import calculate_pieapp_score
from vmaf_metric import calculate_vmaf, convert_mp4_to_y4m, trim_video
from vidaio_subnet_core.utilities.storage_client import storage_client
from services.video_scheduler.video_utils import get_trim_video_path, delete_videos_with_fileid

# Compression scoring constants
COMPRESSION_RATE_WEIGHT = 0.8  # w_c
COMPRESSION_VMAF_WEIGHT = 0.2  # w_vmaf

# Set up logging
logger = logging.getLogger(__name__)
app = FastAPI()
fire_requests = FireRequests()

VMAF_THRESHOLD = CONFIG.score.vmaf_threshold
PIEAPP_SAMPLE_COUNT = CONFIG.score.pieapp_sample_count
PIEAPP_THRESHOLD = CONFIG.score.pieapp_threshold
VMAF_SAMPLE_COUNT = CONFIG.score.vmaf_sample_count

class UpscalingScoringRequest(BaseModel):
    """
    Request model for upscaling scoring. Contains URLs for distorted videos and the reference video path.
    """
    distorted_urls: List[str]
    reference_paths: List[str]
    uids: List[int]
    video_ids: List[str]
    uploaded_object_names: List[str]
    content_lengths: List[int]
    fps: Optional[float] = None
    subsample: Optional[int] = 1
    verbose: Optional[bool] = False
    progress: Optional[bool] = False

class CompressionScoringRequest(BaseModel):
    """
    Request model for compression scoring. Contains URLs for distorted videos and the reference video path.
    """
    distorted_urls: List[str]
    reference_paths: List[str]
    uids: List[int]
    video_ids: List[str]
    uploaded_object_names: List[str]
    vmaf_threshold: float
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

class UpscalingScoringResponse(BaseModel):
    """
    Response model for upscaling scoring. Contains the list of calculated scores for each distorted video.
    """
    vmaf_scores: List[float]
    pieapp_scores: List[float]
    quality_scores: List[float]
    length_scores: List[float]
    final_scores: List[float]
    reasons: List[str]

class CompressionScoringResponse(BaseModel):
    """
    Response model for compression scoring. Contains the list of calculated scores for each distorted video.
    """
    vmaf_scores: List[float]
    compression_rates: List[float]
    final_scores: List[float]
    reasons: List[str]

class OrganicsScoringResponse(BaseModel):
    """
    Response model for organics scoring. Contains the list of calculated scores for each distorted video.
    """
    vmaf_scores: List[float]
    pieapp_scores: List[float]
    quality_scores: List[float]
    length_scores: List[float]
    final_scores: List[float]
    reasons: List[str]

async def download_video(video_url: str, verbose: bool) -> tuple[str, float]:
    """
    Download a video from the given URL and save it to a temporary file.

    Args:
        video_url (str): The URL of the video to download.
        verbose (bool): Whether to show download progress.

    Returns:
        tuple[str, float]: A tuple containing the path to the downloaded video file
                           and the time taken to download it.

    Raises:
        Exception: If the download fails or takes longer than the timeout.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as vid_temp:
            file_path = vid_temp.name  # Path to the temporary file
        if verbose:
            print(f"Downloading video from {video_url} to {file_path}")

        timeout = aiohttp.ClientTimeout(sock_connect=0.5, total=7.5)

        start_time = time.time()  # Record start time

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(video_url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download video. HTTP status: {response.status}")

                with open(file_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(2 * 1024 * 1024):
                        f.write(chunk)

        end_time = time.time()  # Record end time
        download_time = end_time - start_time  # Calculate download duration

        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            raise Exception(f"Download failed or file is empty: {file_path}")

        if verbose:
            print(f"File successfully downloaded to: {file_path}")
            print(f"Download time: {download_time:.2f} seconds")

        return file_path, download_time

    except aiohttp.ClientError as e:
        raise Exception(f"Download failed due to a network error: {e}")
    except asyncio.TimeoutError:
        raise Exception("Download timed out")

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

def calculate_length_score(content_length):
    """
    Convert content length in seconds to a normalized length score.
    
    Args:
        content_length (float): Video duration in seconds (5-320s)
        
    Returns:
        float: Normalized length score (0-1)
    """
    return math.log(1 + content_length) / math.log(1 + 320)

def calculate_preliminary_score(quality_score, length_score, quality_weight=0.5, length_weight=0.5):
    """
    Calculate the preliminary score from quality and length scores.
    
    Args:
        quality_score (float): Normalized quality score (0-1)
        length_score (float): Normalized length score (0-1)
        quality_weight (float): Weight for quality component (default: 0.5)
        length_weight (float): Weight for length component (default: 0.5)
        
    Returns:
        float: Preliminary combined score (0-1)
    """
    return (quality_score * quality_weight) + (length_score * length_weight)

def calculate_final_score(s_pre):
    """
    Transform preliminary score into final score using exponential function.
    
    Args:
        s_pre (float): Preliminary score (0-1)
        
    Returns:
        float: Final exponentially-transformed score
    """
    return 0.1 * math.exp(6.979 * (s_pre - 0.5))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_quality_score(pieapp_score):
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
    if not ref_frames:
        print("No ref frames to process")
        return -100
    
    if not dist_frames:
        print("No dist frames to process")
        return 2.0
    
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
    
    ref_provider = FrameProvider(ref_frames)
    dist_provider = FrameProvider(dist_frames)
    
    try:
        score = calculate_pieapp_score(ref_provider, dist_provider, frame_interval=1)
        return score
    except Exception as e:
        print(f"Error calculating PieAPP score on frames: {str(e)}")
        return -100

def upscale_video(input_path, scale_factor=2):
    """
    Upscales a video using FFmpeg by the specified scale factor.
    
    Args:
        input_path (str): Path to the input video file
        scale_factor (int): Factor by which to upscale the video (default: 2)
        
    Returns:
        str: Path to the upscaled video file
    """
    filename, extension = os.path.splitext(input_path)
    output_path = f"{filename}_upscaled{extension}"
    
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    new_width = width * scale_factor
    new_height = height * scale_factor
    
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vf", f"scale={new_width}:{new_height}",
        "-c:v", "libx264",
        "-preset", "medium",  
        "-crf", "18",         
        "-an",
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Successfully upscaled video to {new_width}x{new_height}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error upscaling video: {e}")
        return input_path

@app.post("/score_upscaling_synthetics")
async def score_upscaling_synthetics(request: UpscalingScoringRequest) -> UpscalingScoringResponse:
    print("#################### 🤖 start upscaling request scoring ####################")

    start_time = time.time()
    quality_scores = []
    length_scores = []
    final_scores = []
    vmaf_scores = []
    pieapp_scores = []
    reasons = []

    if len(request.reference_paths) != len(request.distorted_urls):
        raise HTTPException(
            status_code=400, 
            detail="Number of reference paths must match number of distorted URLs"
        )
    
    if len(request.uids) != len(request.distorted_urls):
        raise HTTPException(
            status_code=400, 
            detail="Number of UIDs must match number of distorted URLs"
        )

    for idx, (ref_path, dist_url, uid, video_id, uploaded_object_name, content_length) in enumerate(zip(
        request.reference_paths, 
        request.distorted_urls, 
        request.uids,
        request.video_ids,
        request.uploaded_object_names,
        request.content_lengths
    )):
        try:
            print(f"🧩 Processing pair {idx+1}/{len(request.distorted_urls)}: UID {uid} 🧩")
            
            uid_start_time = time.time()  # Start time for this UID

            ref_cap = None
            dist_cap = None
            ref_y4m_path = None
            dist_path = None

            ref_cap = cv2.VideoCapture(ref_path)
            
            # Check if the reference video was actually opened successfully
            if not ref_cap.isOpened():
                # Add diagnostic information
                file_exists = os.path.exists(ref_path)
                file_size = os.path.getsize(ref_path) if file_exists else 0
                
                print(f"Error opening reference video file {ref_path}.")
                print(f"  File exists: {file_exists}")
                print(f"  File size: {file_size} bytes")
                print(f"  Current working directory: {os.getcwd()}")
                print(f"  Assigning score of 0.")
                
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(-100)
                reasons.append(f"error opening reference video file: {ref_path} (exists: {file_exists}, size: {file_size})")
                continue
            
            # Only log success after confirming the file was opened
            step_time = time.time() - uid_start_time
            print(f"♎️ 1. Opened reference video in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            ref_total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step_time = time.time() - uid_start_time
            print(f"♎️ 2. Retrieved reference video frame count ({ref_total_frames}) in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            if ref_total_frames < 10:
                print(f"Video must contain at least 10 frames. Assigning score of 0.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(-100)
                reasons.append("reference video has fewer than 10 frames")
                ref_cap.release()
                continue

            sample_size = min(PIEAPP_SAMPLE_COUNT, ref_total_frames)
            max_start_frame = ref_total_frames - sample_size
            start_frame = 0 if max_start_frame <= 0 else random.randint(0, max_start_frame)

            print(f"Selected frame range for pieapp score {idx+1}: {start_frame} to {start_frame + sample_size - 1}")
            step_time = time.time() - uid_start_time
            print(f"♎️ 3. Selected frame range in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            ref_frames = []
            ref_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(sample_size):
                ret, frame = ref_cap.read()
                if not ret:
                    break
                ref_frames.append(frame)
            step_time = time.time() - uid_start_time
            print(f"♎️ 4. Extracted sampled frames from reference video in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            random_frames = sorted(random.sample(range(ref_total_frames), VMAF_SAMPLE_COUNT))
            print(f"Randomly selected {VMAF_SAMPLE_COUNT} frames for VMAF score: frame list: {random_frames}")
            step_time = time.time() - uid_start_time
            print(f"♎️ 5. Selected random frames for VMAF in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            ref_y4m_path = convert_mp4_to_y4m(ref_path, random_frames)
            print("The reference video has been successfully converted to Y4M format.")
            step_time = time.time() - uid_start_time
            print(f"♎️ 6. Converted reference video to Y4M in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            if len(dist_url) < 10:
                print(f"Wrong dist download URL: {dist_url}. Assigning score of 0.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append("Invalid download URL: the distorted video download URL must be at least 10 characters long.")
                continue

            try:
                dist_path, download_time = await download_video(dist_url, request.verbose)
            except Exception as e:
                error_msg = f"Failed to download video from {dist_url}: {str(e)}"
                print(f"{error_msg}. Assigning score of 0.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append("failed to download video file from url")
                continue

            step_time = time.time() - uid_start_time
            print(f"♎️ 7. Downloaded distorted video in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            dist_cap = cv2.VideoCapture(dist_path)
            step_time = time.time() - uid_start_time
            print(f"♎️ 8. Opened distorted video in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            if not dist_cap.isOpened():
                print(f"Error opening distorted video file from {dist_url}. Assigning score of 0.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append("error opening distorted video file")
                dist_cap.release()
                if dist_path and os.path.exists(dist_path):
                    os.unlink(dist_path)
                continue

            dist_total_frames = int(dist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Distorted video has {dist_total_frames} frames.")
            step_time = time.time() - uid_start_time
            print(f"♎️ 9. Retrieved distorted video frame count in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            if dist_total_frames != ref_total_frames:
                print(
                    f"Video length mismatch for pair {idx+1}: ref({ref_total_frames}) != dist({dist_total_frames}). Assigning score of 0."
                )
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append("video length mismatch")
                dist_cap.release()
                if dist_path and os.path.exists(dist_path):
                    os.unlink(dist_path)
                continue

            # Calculate VMAF
            try:
                vmaf_start = time.time()
                vmaf_score = calculate_vmaf(ref_y4m_path, dist_path, random_frames)
                vmaf_calc_time = time.time() - vmaf_start
                print(f"☣️☣️ VMAF calculation took {vmaf_calc_time:.2f} seconds.")

                step_time = time.time() - uid_start_time
                print(f"♎️ 10. Completed VMAF calculation in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

                if vmaf_score is not None:
                    vmaf_scores.append(vmaf_score)
                else:
                    vmaf_score = 0.0
                    vmaf_scores.append(vmaf_score)
                print(f"🎾 VMAF score is {vmaf_score}")
            except Exception as e:
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append("failed to calculate VMAF score due to video dimension mismatch")
                dist_cap.release()
                print(f"Error calculating VMAF score: {e}")
                continue

            if vmaf_score / 100 < VMAF_THRESHOLD:
                print(f"VMAF score is too low, giving zero score, current VMAF score: {vmaf_score}")
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append(f"VMAF score is too low, current VMAF score: {vmaf_score}")
                dist_cap.release()
                continue

            # Extract distorted frames
            dist_frames = []
            dist_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(sample_size):
                ret, frame = dist_cap.read()
                if not ret:
                    break
                dist_frames.append(frame)
            step_time = time.time() - uid_start_time
            print(f"♎️ 11. Extracted sampled frames from distorted video in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            # Calculate PieAPP score
            pieapp_score = calculate_pieapp_score_on_samples(ref_frames, dist_frames)
            step_time = time.time() - uid_start_time
            print(f"♎️ 12. Calculated PieAPP score in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")
            print(f"🎾 PieAPP score is {pieapp_score}")

            if pieapp_score == -100:
                print(f"Uncertain error in pieapp calculation")
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(-100)
                reasons.append("Uncertain error in pieapp calculation")
                dist_cap.release()
                continue

            pieapp_scores.append(pieapp_score)
            s_q = calculate_quality_score(pieapp_score)
            print(f"🏀 Quality score is {s_q}")

            dist_cap.release()

            s_l = calculate_length_score(content_length)

            s_pre = calculate_preliminary_score(s_q, s_l)

            s_f = calculate_final_score(s_pre)

            quality_scores.append(s_q)
            length_scores.append(s_l)
            final_scores.append(s_f)
            reasons.append("success")

            step_time = time.time() - uid_start_time
            print(f"🛑 Processed one UID in {step_time:.2f} seconds.")

        except Exception as e:
            error_msg = f"Failed to process video from {dist_url}: {str(e)}"
            print(f"{error_msg}. Assigning score of 0.")
            vmaf_scores.append(0.0)
            pieapp_scores.append(0.0)
            quality_scores.append(0.0)
            length_scores.append(0.0)
            final_scores.append(0.0)
            reasons.append("failed to process video")

        finally:
            # Clean up resources for this pair
            ref_cap.release()
            if ref_y4m_path and os.path.exists(ref_y4m_path):
                os.unlink(ref_y4m_path)
            if dist_path and os.path.exists(dist_path):
                os.unlink(dist_path)
            # if ref_path and os.path.exists(ref_path):
            #     os.unlink(ref_path)

            # Delete the uploaded object
            storage_client.delete_file(uploaded_object_name)

    for ref_path in request.reference_paths:
        if os.path.exists(ref_path):
            os.unlink(ref_path)

    tmp_directory = "/tmp"
    try:
        print("🧹 Cleaning up temporary files in /tmp...")
        for file_path in glob.glob(os.path.join(tmp_directory, "*.mp4")):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        for file_path in glob.glob(os.path.join(tmp_directory, "*.y4m")):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"⚠️ Error during cleanup: {e}")

    processed_time = time.time() - start_time
    print(f"Completed batch scoring of {len(request.distorted_urls)} pairs within {processed_time:.2f} seconds")
    print(f"🔯🔯🔯 Calculated final scores: {final_scores} 🔯🔯🔯")
    
    return UpscalingScoringResponse(
        vmaf_scores=vmaf_scores,
        pieapp_scores=pieapp_scores,
        quality_scores=quality_scores,
        length_scores=length_scores,
        final_scores=final_scores,
        reasons=reasons
    )

@app.post("/score_compression_synthetics")
async def score_compression_synthetics(request: CompressionScoringRequest) -> CompressionScoringResponse:
    print("#################### 🤖 start compression request scoring ####################")
    start_time = time.time()
    compression_rates = []
    final_scores = []
    vmaf_scores = []
    reasons = []

    if len(request.reference_paths) != len(request.distorted_urls):
        raise HTTPException(
            status_code=400, 
            detail="Number of reference paths must match number of distorted URLs"
        )
    
    if len(request.uids) != len(request.distorted_urls):
        raise HTTPException(
            status_code=400, 
            detail="Number of UIDs must match number of distorted URLs"
        )

    vmaf_threshold = request.vmaf_threshold
    for idx, (ref_path, dist_url, uid, video_id, uploaded_object_name) in enumerate(zip(
        request.reference_paths, 
        request.distorted_urls, 
        request.uids,
        request.video_ids,
        request.uploaded_object_names,
    )):
        try:
            print(f"🧩 Processing pair {idx+1}/{len(request.distorted_urls)}: UID {uid} 🧩")
            
            uid_start_time = time.time()  # Start time for this UID

            ref_cap = None
            dist_cap = None
            ref_y4m_path = None
            dist_path = None

            ref_cap = cv2.VideoCapture(ref_path)
            
            # Check if the reference video was actually opened successfully
            if not ref_cap.isOpened():
                # Add diagnostic information
                file_exists = os.path.exists(ref_path)
                file_size = os.path.getsize(ref_path) if file_exists else 0
                
                print(f"Error opening reference video file {ref_path}.")
                print(f"  File exists: {file_exists}")
                print(f"  File size: {file_size} bytes")
                print(f"  Current working directory: {os.getcwd()}")
                print(f"  Assigning score of 0.")
                
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)  # No compression achieved
                final_scores.append(-100)
                reasons.append(f"error opening reference video file: {ref_path} (exists: {file_exists}, size: {file_size})")
                continue
            
            # Only log success after confirming the file was opened
            step_time = time.time() - uid_start_time
            print(f"♎️ 1. Opened reference video in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            ref_total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step_time = time.time() - uid_start_time
            print(f"♎️ 2. Retrieved reference video frame count ({ref_total_frames}) in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            if ref_total_frames < 10:
                print(f"Video must contain at least 10 frames. Assigning score of 0.")
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)  # No compression achieved
                final_scores.append(-100)
                reasons.append("reference video has fewer than 10 frames")
                ref_cap.release()
                continue

            # Get reference video file size for compression rate calculation
            ref_file_size = os.path.getsize(ref_path)
            print(f"Reference video file size: {ref_file_size} bytes")

            if len(dist_url) < 10:
                print(f"Wrong dist download URL: {dist_url}. Assigning score of 0.")
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)  # No compression achieved
                final_scores.append(0.0)
                reasons.append("Invalid download URL: the distorted video download URL must be at least 10 characters long.")
                continue

            try:
                dist_path, download_time = await download_video(dist_url, request.verbose)
            except Exception as e:
                error_msg = f"Failed to download video from {dist_url}: {str(e)}"
                print(f"{error_msg}. Assigning score of 0.")
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)  # No compression achieved
                final_scores.append(0.0)
                reasons.append("failed to download video file from url")
                continue

            step_time = time.time() - uid_start_time
            print(f"♎️ 3. Downloaded distorted video in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            dist_cap = cv2.VideoCapture(dist_path)
            step_time = time.time() - uid_start_time
            print(f"♎️ 4. Opened distorted video in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            if not dist_cap.isOpened():
                print(f"Error opening distorted video file from {dist_url}. Assigning score of 0.")
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)  # No compression achieved
                final_scores.append(0.0)
                reasons.append("error opening distorted video file")
                dist_cap.release()
                if dist_path and os.path.exists(dist_path):
                    os.unlink(dist_path)
                continue

            dist_total_frames = int(dist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Distorted video has {dist_total_frames} frames.")
            step_time = time.time() - uid_start_time
            print(f"♎️ 5. Retrieved distorted video frame count in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            if dist_total_frames != ref_total_frames:
                print(
                    f"Video length mismatch for pair {idx+1}: ref({ref_total_frames}) != dist({dist_total_frames}). Assigning score of 0."
                )
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)  # No compression achieved
                final_scores.append(0.0)
                reasons.append("video length mismatch")
                dist_cap.release()
                if dist_path and os.path.exists(dist_path):
                    os.unlink(dist_path)
                continue

            # Get distorted video file size for compression rate calculation
            dist_file_size = os.path.getsize(dist_path)
            print(f"Distorted video file size: {dist_file_size} bytes")

            # Sample frames for VMAF calculation
            random_frames = sorted(random.sample(range(ref_total_frames), VMAF_SAMPLE_COUNT))
            print(f"Randomly selected {VMAF_SAMPLE_COUNT} frames for VMAF score: frame list: {random_frames}")
            step_time = time.time() - uid_start_time
            print(f"♎️ 6. Selected random frames for VMAF in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            ref_y4m_path = convert_mp4_to_y4m(ref_path, random_frames)
            print("The reference video has been successfully converted to Y4M format.")
            step_time = time.time() - uid_start_time
            print(f"♎️ 7. Converted reference video to Y4M in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            # Calculate VMAF
            try:
                vmaf_start = time.time()
                vmaf_score = calculate_vmaf(ref_y4m_path, dist_path, random_frames)
                vmaf_calc_time = time.time() - vmaf_start
                print(f"☣️☣️ VMAF calculation took {vmaf_calc_time:.2f} seconds.")

                step_time = time.time() - uid_start_time
                print(f"♎️ 8. Completed VMAF calculation in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

                if vmaf_score is not None:
                    vmaf_scores.append(vmaf_score)
                else:
                    vmaf_score = 0.0
                    vmaf_scores.append(vmaf_score)
                print(f"🎾 VMAF score is {vmaf_score}")
            except Exception as e:
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)  # No compression achieved
                final_scores.append(0.0)
                reasons.append("failed to calculate VMAF score due to video dimension mismatch")
                dist_cap.release()
                print(f"Error calculating VMAF score: {e}")
                continue

            # Calculate compression score using the proper formula
            # C = compressed_file_size / original_file_size
            # S_c = w_c × (1 - C^1.5) + w_vmaf × (VMAF_score - VMAF_threshold) / (100 - VMAF_threshold)
            
            if ref_file_size > 0 and dist_file_size < ref_file_size:
                compression_rate = dist_file_size / ref_file_size
                print(f"Compression rate: {compression_rate:.4f} ({dist_file_size}/{ref_file_size})")
            else:
                compression_rate = 1.0
                print("Reference file size is 0 or distorted file size is greater than reference file size, setting compression rate to 1.0")

            # Check VMAF threshold first
            if vmaf_score < vmaf_threshold:
                print(f"VMAF score is lower than threshold, giving zero score, current VMAF score: {vmaf_score}, vmaf threshold: {vmaf_threshold}")
                compression_rates.append(compression_rate)  # Keep the compression rate even if VMAF fails
                final_scores.append(0.0)
                reasons.append(f"VMAF score is too low, current VMAF score: {vmaf_score}")
                dist_cap.release()
                continue

            # Calculate compression rate component: w_c × (1 - C^1.5)
            compression_rate_component = COMPRESSION_RATE_WEIGHT * (1 - compression_rate ** 1.5)
            print(f"Compression rate component: {compression_rate_component:.4f}")

            # Calculate VMAF quality component: w_vmaf × (VMAF_score - VMAF_threshold) / (100 - VMAF_threshold)
            vmaf_quality_component = COMPRESSION_VMAF_WEIGHT * (vmaf_score - vmaf_threshold) / (100 - vmaf_threshold)
            print(f"VMAF quality component: {vmaf_quality_component:.4f}")

            # Calculate final compression score
            compression_score = compression_rate_component + vmaf_quality_component
            print(f"🎯 Compression score is {compression_score:.4f}")

            # Calculate final score (for compression, this is the same as compression_score)
            print(f"🎯 Final score is {compression_score:.4f}")

            compression_rates.append(compression_rate)
            final_scores.append(compression_score)
            reasons.append("success")

            step_time = time.time() - uid_start_time
            print(f"🛑 Processed one UID in {step_time:.2f} seconds.")

        except Exception as e:
            error_msg = f"Failed to process video from {dist_url}: {str(e)}"
            print(f"{error_msg}. Assigning score of 0.")
            vmaf_scores.append(0.0)
            compression_rates.append(1.0)  # No compression achieved
            final_scores.append(0.0)
            reasons.append("failed to process video")

        finally:
            # Clean up resources for this pair
            if ref_cap:
                ref_cap.release()
            if ref_y4m_path and os.path.exists(ref_y4m_path):
                os.unlink(ref_y4m_path)
            if dist_path and os.path.exists(dist_path):
                os.unlink(dist_path)
            # if ref_path and os.path.exists(ref_path):
            #     os.unlink(ref_path)

            # Delete the uploaded object
            storage_client.delete_file(uploaded_object_name)
            
    for ref_path in request.reference_paths:
        if os.path.exists(ref_path):
            os.unlink(ref_path)

    tmp_directory = "/tmp"
    try:
        print("🧹 Cleaning up temporary files in /tmp...")
        for file_path in glob.glob(os.path.join(tmp_directory, "*.mp4")):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        for file_path in glob.glob(os.path.join(tmp_directory, "*.y4m")):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"⚠️ Error during cleanup: {e}")

    processed_time = time.time() - start_time
    print(f"Completed batch scoring of {len(request.distorted_urls)} pairs within {processed_time:.2f} seconds")
    print(f"🔯🔯🔯 Calculated final scores: {final_scores} 🔯🔯🔯")
    
    return CompressionScoringResponse(
        vmaf_scores=vmaf_scores,
        compression_rates=compression_rates,
        final_scores=final_scores,
        reasons=reasons
    )

@app.post("/score_organics")
async def score_organics(request: OrganicsScoringRequest) -> OrganicsScoringResponse:
    print("#################### 🤖 start scoring ####################")
    start_time = time.time()
    scores = []
    vmaf_scores = []
    pieapp_scores = []
    reasons = []

    distorted_video_paths = []
    for dist_url in request.distorted_urls:
        if len(dist_url) < 10:
            distorted_video_paths.append(None)
            continue
        try:
            path, download_time = await download_video(dist_url, request.verbose)
            distorted_video_paths.append(path)
        except Exception as e:
            print(f"failed to download distorted video: {dist_url}, error: {e}")
            distorted_video_paths.append(None)

    for idx, (ref_url, dist_path, uid, task_type) in enumerate(
        zip(request.reference_urls, distorted_video_paths, request.uids, request.task_types)
    ):
        print(f"🧩 processing {uid}.... downloading reference video.... 🧩")
        ref_path = None
        ref_cap = None
        dist_cap = None
        ref_upscaled_y4m_path = None

        scale_factor = 2
        if task_type == "SD24K":
            scale_factor = 4

        print(f"scale factor: {scale_factor}")

        # === REFERENCE VIDEO VALIDATION (NOT MINER'S FAULT) ===
        try:
            # download reference video
            if len(ref_url) < 10:
                print(f"invalid reference download url: {ref_url}. skipping scoring")
                vmaf_scores.append(-1)
                pieapp_scores.append(-1)
                reasons.append("invalid reference url - skipped")
                scores.append(-1)  # -1 means skip, don't penalize
                continue

            ref_path, download_time = await download_video(ref_url, request.verbose)
            ref_cap = cv2.VideoCapture(ref_path)
            
            if not ref_cap.isOpened():
                file_exists = os.path.exists(ref_path)
                file_size = os.path.getsize(ref_path) if file_exists else 0
                
                print(f"error opening reference video from {ref_url}. skipping scoring")
                print(f"  File path: {ref_path}")
                print(f"  File exists: {file_exists}")
                print(f"  File size: {file_size} bytes")
                
                vmaf_scores.append(-1)
                pieapp_scores.append(-1)
                reasons.append("corrupted reference video - skipped")
                scores.append(-1)  # -1 means skip, don't penalize
                continue
            
            ref_total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if ref_total_frames <= 0:
                print(f"invalid reference video from {ref_url}: no frames found. skipping...")
                vmaf_scores.append(-1)
                pieapp_scores.append(-1)
                reasons.append("invalid reference video - skipped")
                scores.append(-1)  # -1 means skip, don't penalize
                ref_cap.release()
                continue

            if ref_total_frames < 10:
                print(f"reference video too short (<10 frames). skipping scoring")
                vmaf_scores.append(-1)
                pieapp_scores.append(-1)
                reasons.append("reference video too short - skipped")
                scores.append(-1)  # -1 means skip, don't penalize
                ref_cap.release()
                continue

        except Exception as e:
            print(f"system error processing reference video: {e}. skipping scoring")
            vmaf_scores.append(-1)
            pieapp_scores.append(-1)
            reasons.append("system error with reference - skipped")
            scores.append(-1)  # -1 means skip, don't penalize
            if ref_cap:
                ref_cap.release()
            continue

        # === MINER OUTPUT VALIDATION (MINER'S FAULT) ===
        try:
            # check if distorted video failed to download
            if dist_path is None:
                print(f"failed to download distorted video for uid {uid}. penalizing miner.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                reasons.append("MINER FAILURE: failed to download distorted video, invalid url")
                scores.append(0.0)  # 0 = miner penalty
                ref_cap.release()
                continue

            # Check if miner's output can be opened
            dist_cap = cv2.VideoCapture(dist_path)
            if not dist_cap.isOpened():
                print(f"error opening distorted video from {dist_path}. penalizing miner.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                reasons.append("MINER FAILURE: corrupted output video")
                scores.append(0.0)  # 0 = miner penalty
                ref_cap.release()
                continue

            dist_total_frames = int(dist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"distorted video has {dist_total_frames} frames.")

            # Check frame count match
            if dist_total_frames != ref_total_frames:
                print(f"video length mismatch: ref({ref_total_frames}) != dist({dist_total_frames}). penalizing miner.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                reasons.append("MINER FAILURE: video length mismatch")
                scores.append(0.0)  # 0 = miner penalty
                ref_cap.release()
                dist_cap.release()
                continue

            # Check minimum frame count for miner output
            if dist_total_frames < 10:
                print(f"miner output too short (<10 frames). penalizing miner.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                reasons.append("MINER FAILURE: output video too short")
                scores.append(0.0)  # 0 = miner penalty
                ref_cap.release()
                dist_cap.release()
                continue

        except Exception as e:
            print(f"error validating miner output: {e}. penalizing miner.")
            vmaf_scores.append(0.0)
            pieapp_scores.append(0.0)
            reasons.append(f"MINER FAILURE: {e}")
            scores.append(0.0)  # 0 = miner penalty
            if ref_cap:
                ref_cap.release()
            if dist_cap:
                dist_cap.release()
            continue

        # === QUALITY CONTROL SCORING ===
        try:
            # Sample frames for basic quality checks
            sample_size = min(PIEAPP_SAMPLE_COUNT, ref_total_frames)
            max_start_frame = ref_total_frames - sample_size
            start_frame = 0 if max_start_frame <= 0 else random.randint(0, max_start_frame)
            print(f"selected frame range for quality checks: {start_frame} to {start_frame + sample_size - 1}")

            # Extract frames from both videos for comparison
            ref_frames = []
            ref_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(sample_size):
                ret, frame = ref_cap.read()
                if not ret:
                    break
                ref_frames.append(frame)

            dist_frames = []
            dist_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(sample_size):
                ret, frame = dist_cap.read()
                if not ret:
                    break
                dist_frames.append(frame)

            # Ensure we have frames to compare
            if not ref_frames or not dist_frames:
                print(f"insufficient frames for comparison. penalizing miner.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                reasons.append("MINER FAILURE: insufficient frames for comparison")
                scores.append(0.0)  # 0 = miner penalty
                ref_cap.release()
                dist_cap.release()
                continue

            # === BASIC QUALITY CHECKS (NO ARTIFICIAL REFERENCE) ===
            
            # Check 1: Resolution validation
            ref_height, ref_width = ref_frames[0].shape[:2]
            dist_height, dist_width = dist_frames[0].shape[:2]
            
            expected_width = ref_width * scale_factor
            expected_height = ref_height * scale_factor
            
            if dist_width != expected_width or dist_height != expected_height:
                print(f"resolution mismatch: expected {expected_width}x{expected_height}, got {dist_width}x{dist_height}. penalizing miner.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                reasons.append(f"MINER FAILURE: wrong resolution (expected {expected_width}x{expected_height}, got {dist_width}x{dist_height})")
                scores.append(0.0)  # 0 = miner penalty
                ref_cap.release()
                dist_cap.release()
                continue

            # Check 2: Basic quality validation (no obvious corruption)
            quality_issues = []
            
            for i, (ref_frame, dist_frame) in enumerate(zip(ref_frames, dist_frames)):
                # Check for completely black or white frames (likely corruption)
                if np.mean(dist_frame) < 5 or np.mean(dist_frame) > 250:
                    quality_issues.append(f"frame {i}: suspicious brightness")
                
                # Check for extreme noise (std dev too high/low)
                if np.std(dist_frame) < 1 or np.std(dist_frame) > 100:
                    quality_issues.append(f"frame {i}: suspicious noise levels")
                
                # Check for obvious artifacts (extreme pixel values)
                if np.min(dist_frame) < 0 or np.max(dist_frame) > 255:
                    quality_issues.append(f"frame {i}: invalid pixel values")

            if quality_issues:
                print(f"quality issues detected: {quality_issues}. penalizing miner.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                reasons.append(f"MINER FAILURE: quality issues detected ({', '.join(quality_issues[:3])})")
                scores.append(0.0)  # 0 = miner penalty
                ref_cap.release()
                dist_cap.release()
                continue

            # Check 3: Basic upscaling validation (should be different from original)
            # Compare a few frames to ensure upscaling actually happened
            upscaling_detected = False
            
            def calculate_frame_quality(frame):
                """Calculate frame quality metrics for adaptive thresholding"""
                # Convert to grayscale for edge detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate edge strength using Sobel
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
                edge_strength = np.mean(edge_magnitude)
                
                # Calculate contrast (standard deviation of pixel values)
                contrast = np.std(gray)
                
                # Calculate brightness
                brightness = np.mean(gray)
                
                # Calculate noise level (using Laplacian variance)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                noise_level = np.var(laplacian)
                
                return {
                    'edge_strength': edge_strength,
                    'contrast': contrast,
                    'brightness': brightness,
                    'noise_level': noise_level
                }
            
            def calculate_edge_difference(frame1, frame2):
                """Calculate edge-based difference between frames"""
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                
                # Calculate edges for both frames
                edges1 = cv2.Canny(gray1, 50, 150)
                edges2 = cv2.Canny(gray2, 50, 150)
                
                # Calculate edge difference
                edge_diff = np.mean(np.abs(edges1.astype(float) - edges2.astype(float)))
                return edge_diff
            
            for i, (ref_frame, dist_frame) in enumerate(zip(ref_frames[:3], dist_frames[:3])):
                # Calculate original frame quality metrics
                ref_quality = calculate_frame_quality(ref_frame)
                
                # Downscale the upscaled frame back to original size for comparison
                downscaled_dist = cv2.resize(dist_frame, (ref_width, ref_height), interpolation=cv2.INTER_LINEAR)
                
                # Calculate multiple difference metrics
                pixel_diff = np.mean(np.abs(ref_frame.astype(float) - downscaled_dist.astype(float)))
                edge_diff = calculate_edge_difference(ref_frame, downscaled_dist)
                
                # Adaptive threshold based on original frame quality
                # Higher quality originals need higher thresholds
                base_threshold = 5.0
                quality_factor = min(2.0, max(0.5, ref_quality['edge_strength'] / 50.0))  # Normalize edge strength
                contrast_factor = min(1.5, max(0.8, ref_quality['contrast'] / 30.0))  # Normalize contrast
                
                adaptive_threshold = base_threshold * quality_factor * contrast_factor
                
                print(f"Frame {i} quality metrics:")
                print(f"  Edge strength: {ref_quality['edge_strength']:.2f}")
                print(f"  Contrast: {ref_quality['contrast']:.2f}")
                print(f"  Brightness: {ref_quality['brightness']:.2f}")
                print(f"  Noise level: {ref_quality['noise_level']:.2f}")
                print(f"  Pixel difference: {pixel_diff:.2f}")
                print(f"  Edge difference: {edge_diff:.2f}")
                print(f"  Adaptive threshold: {adaptive_threshold:.2f}")
                
                # Check if upscaling is detected using multiple criteria
                pixel_upscaling = pixel_diff > adaptive_threshold
                edge_upscaling = edge_diff > (adaptive_threshold * 0.5)  # Lower threshold for edge differences
                
                # Additional check: ensure the upscaled frame has more detail than downscaled
                dist_quality = calculate_frame_quality(dist_frame)
                downscaled_quality = calculate_frame_quality(downscaled_dist)
                
                detail_improvement = (
                    dist_quality['edge_strength'] > downscaled_quality['edge_strength'] * 1.1 and
                    dist_quality['contrast'] > downscaled_quality['contrast'] * 1.05
                )
                
                print(f"  Pixel upscaling detected: {pixel_upscaling}")
                print(f"  Edge upscaling detected: {edge_upscaling}")
                print(f"  Detail improvement detected: {detail_improvement}")
                
                # Upscaling is detected if any two criteria are met
                upscaling_criteria_met = sum([pixel_upscaling, edge_upscaling, detail_improvement])
                
                if upscaling_criteria_met >= 2:
                    upscaling_detected = True
                    print(f"✅ Frame {i}: Upscaling detected with {upscaling_criteria_met}/3 criteria")
                    break
                else:
                    print(f"❌ Frame {i}: Insufficient upscaling detected ({upscaling_criteria_met}/3 criteria)")
            
            if not upscaling_detected:
                print(f"no meaningful upscaling detected across all checked frames. penalizing miner.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                reasons.append("MINER FAILURE: no meaningful upscaling detected")
                scores.append(0.0)  # 0 = miner penalty
                ref_cap.release()
                dist_cap.release()
                continue

            # SUCCESS: Miner passed all quality control checks
            print(f"✅ miner {uid} passed quality control")
            reasons.append("SUCCESS: passed quality control")
            scores.append(1.0)  # 1 = miner success
            
            # Set dummy values for compatibility
            vmaf_scores.append(50.0)  # Dummy VMAF score for successful miners
            pieapp_scores.append(1.0)  # Dummy PIE-APP score for successful miners

        except Exception as e:
            print(f"system error during quality scoring: {e}. skipping scoring...")
            vmaf_scores.append(-1)
            pieapp_scores.append(-1)
            reasons.append("system error during scoring - skipped")
            scores.append(-1)  # -1 means skip, don't penalize

        finally:
            if ref_cap:
                ref_cap.release()
            if dist_cap:
                dist_cap.release()
            if ref_path and os.path.exists(ref_path):
                os.unlink(ref_path)
            if dist_path and os.path.exists(dist_path):
                os.unlink(dist_path)
            if ref_upscaled_y4m_path and os.path.exists(ref_upscaled_y4m_path):
                os.unlink(ref_upscaled_y4m_path)

    processed_time = time.time() - start_time
    print(f"🔯🔯🔯 calculated scores: {scores} 🔯🔯🔯")
    print(f"completed one batch scoring within {processed_time:.2f} seconds")
    
    # Summary statistics
    successful_miners = sum(1 for score in scores if score == 1.0)
    failed_miners = sum(1 for score in scores if score == 0.0)
    skipped_miners = sum(1 for score in scores if score == -1)
    
    print(f"📊 SCORING SUMMARY:")
    print(f"  ✅ Successful miners: {successful_miners}")
    print(f"  ❌ Failed miners: {failed_miners}")
    print(f"  ⏭️ Skipped miners: {skipped_miners}")
    
    quality_scores = [0.0] * len(request.uids)
    length_scores = [0.0] * len(request.uids)
    
    return OrganicsScoringResponse(
        vmaf_scores=vmaf_scores,
        pieapp_scores=pieapp_scores,
        quality_scores=quality_scores,
        length_scores=length_scores,
        final_scores=scores,
        reasons=reasons
    )

if __name__ == "__main__":
    import uvicorn
    
    host = CONFIG.score.host
    port = CONFIG.score.port
    
    uvicorn.run(app, host=host, port=port)