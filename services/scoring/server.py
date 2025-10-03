import os
import sys
import cv2
import glob
import time
import math
import torch
import random
import asyncio
import aiohttp
import logging
import tempfile
import subprocess
import numpy as np
from loguru import logger
from pydantic import BaseModel
from typing import Optional, List
from firerequests import FireRequests
from vidaio_subnet_core import CONFIG
from torchvision.models import resnet50
from fastapi import FastAPI, HTTPException
import torchvision.transforms as transforms
from pieapp_metric import calculate_pieapp_score
from vidaio_subnet_core.utilities.storage_client import storage_client
from vmaf_metric import convert_mp4_to_y4m, trim_video, calculate_vmaf_neg_segments
from services.video_scheduler.video_utils import get_trim_video_path, delete_videos_with_fileid

# Configuration constants for calculating compression score
COMPRESSION_WEIGHT = 0.75  # Weight for compression component (0-1)
QUALITY_WEIGHT = 0.25      # Weight for quality bonus component (0-1)
SOFT_THRESHOLD_MARGIN = 5.0  # Points below threshold before hard cutof



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

class OrganicsUpscalingScoringRequest(BaseModel):
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

class OrganicsCompressionScoringRequest(BaseModel):
    """
    Request model for scoring. Contains URLs for distorted videos and the reference video path.
    """
    distorted_urls: List[str]
    reference_urls: List[str]
    vmaf_thresholds: List[float]
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

class OrganicsUpscalingScoringResponse(BaseModel):
    """
    Response model for organics scoring. Contains the list of calculated scores for each distorted video.
    """
    vmaf_scores: List[float]
    pieapp_scores: List[float]
    quality_scores: List[float]
    length_scores: List[float]
    final_scores: List[float]
    reasons: List[str]

class OrganicsCompressionScoringResponse(BaseModel):
    """
    Response model for organics scoring. Contains the list of calculated scores for each distorted video.
    """
    vmaf_scores: List[float]
    compression_rates: List[float]
    final_scores: List[float]
    reasons: List[str]

# Load pre-trained model for feature extraction
def load_quality_model():
    """Load ResNet50 model for quality assessment"""
    from torchvision.models import ResNet50_Weights
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    return model

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
            logger.info(f"Downloading video from {video_url} to {file_path}")

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
            logger.info(f"File successfully downloaded to: {file_path}")
            logger.info(f"Download time: {download_time:.2f} seconds")

        return file_path, download_time

    except aiohttp.ClientError as e:
        raise Exception(f"Download failed due to a network error: {e}")
    except asyncio.TimeoutError:
        raise Exception("Download timed out")

# Function to get ClipIQA+ score programmatically
def get_clipiqa_score(video_path, num_frames=3):
    """Get ClipIQA+ score for a video file"""
    frames = extract_frames(video_path, num_frames)
    score = calculate_clipiqa_plus_score(frames)
    return score

# Extract frames from video
def extract_frames(video_path, num_frames=3, frame_indices=None):
    """Extract frames from video using specified indices or random ones"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        raise ValueError(f"Video has only {total_frames} frames, but {num_frames} frames requested")
    
    # Use provided frame indices or select random ones
    if frame_indices is None:
        frame_indices = random.sample(range(total_frames), num_frames)
    else:
        # Validate that all requested indices are within bounds
        if max(frame_indices) >= total_frames:
            raise ValueError(f"Frame index {max(frame_indices)} exceeds video length {total_frames}")
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frames.append(frame)
        else:
            raise ValueError(f"Failed to read frame at index {idx}")
    
    cap.release()
    return frames

# Calculate ClipIQA+ inspired score
def calculate_clipiqa_plus_score(frames):
    """Calculate ClipIQA+ inspired score for given frames"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_quality_model().to(device)
    
    # Preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    scores = []
    for frame in frames:
        # Preprocess frame
        frame_tensor = preprocess(frame).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Extract features using ResNet50
            features = model(frame_tensor)
            
            # Calculate quality metrics
            # 1. Feature magnitude (higher = more complex/rich content)
            feature_magnitude = torch.norm(features, p=2).item()
            
            # 2. Feature variance (higher = more diverse content)
            feature_variance = torch.var(features).item()
            
            # 3. Sharpness estimation using Laplacian variance
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
            
            # 4. Contrast estimation
            contrast = np.std(gray_frame)
            
            # Combine metrics with weights (inspired by ClipIQA+ approach)
            # Normalize each metric to [0, 1] range
            norm_magnitude = min(1.0, feature_magnitude / 100.0)
            norm_variance = min(1.0, feature_variance / 1000.0)
            norm_sharpness = min(1.0, laplacian_var / 1000.0)
            norm_contrast = min(1.0, contrast / 100.0)
            
            # Weighted combination (ClipIQA+ inspired weights)
            quality_score = (
                0.3 * norm_magnitude +
                0.25 * norm_variance +
                0.25 * norm_sharpness +
                0.2 * norm_contrast
            )
            
            scores.append(quality_score)
    
    # Return the average score across all frames
    return np.mean(scores)

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

def get_video_frame_count(video_path: str) -> int:
    """
    Get frame count from video using ffprobe (works for all codecs).
    Falls back to OpenCV only if ffprobe fails.
    
    Args:
        video_path: Path to video file
        
    Returns:
        int: Number of frames in the video
    """
    try:
        # Primary method: Use ffprobe (fast and reliable for all codecs)
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-count_packets',
            '-show_entries', 'stream=nb_read_packets',
            '-of', 'csv=p=0',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
        frame_count = int(result.stdout.strip())
        logger.debug(f"ffprobe: {video_path} has {frame_count} frames")
        return frame_count
    except Exception as e:
        logger.warning(f"ffprobe failed for {video_path}: {e}, falling back to OpenCV")
        # Fallback: OpenCV (slower but works for most videos)
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Cannot open video file: {video_path}")
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            logger.debug(f"OpenCV: {video_path} has {frame_count} frames")
            return frame_count
        except Exception as opencv_error:
            logger.error(f"Both ffprobe and OpenCV failed for {video_path}")
            raise Exception(f"Cannot read frame count: {str(opencv_error)}")


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
        logger.error(f"Error calculating PieAPP score on frames: {str(e)}")
        return -100

def detect_video_codec(video_path: str) -> str:
    """
    Detect the video codec of a video file using ffprobe.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        str: The codec name (e.g., 'av1', 'h264', 'hevc', etc.)
    """
    try:
        cmd = [
            'ffprobe', '-v', 'error', 
            '-select_streams', 'v:0', 
            '-show_entries', 'stream=codec_name', 
            '-of', 'csv=p=0', 
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        codec = result.stdout.strip()
        logger.info(f"Detected codec for {video_path}: {codec}")
        return codec
    except Exception as e:
        logger.error(f"Error detecting codec for {video_path}: {e}")
        return "unknown"


def convert_av1_to_h264(input_path: str, output_suffix: str = "_h264") -> str:
    """
    Convert AV1 video to H.264 using libdav1d decoder (software).
    NVENC is disabled due to device compatibility issues.

    Args:
        input_path (str): Path to the AV1 video file
        output_suffix (str): Suffix to add to the output filename
        
    Returns:
        str: Path to the converted H.264 video file
    """
    try:
        # Generate output path
        filename, extension = os.path.splitext(input_path)
        output_path = f"{filename}{output_suffix}.mp4"

        logger.info(f"Converting AV1 video to H.264: {input_path} -> {output_path}")

        # Simplified conversion: just use libdav1d + libx264 (reliable and working)
        conversion_attempts = [
            # Attempt 1: libdav1d decoder with CPU encoder (MOST RELIABLE)
            [
                "ffmpeg", "-y",
                "-c:v", "libdav1d",
                "-i", input_path,
                "-c:v", "libx264",
                "-preset", "veryfast",  # Faster preset for quicker conversions
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-an", output_path
            ],
            # Attempt 2: Same but with ultrafast preset if first fails
            [
                "ffmpeg", "-y",
                "-c:v", "libdav1d",
                "-i", input_path,
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-an", output_path
            ],
        ]
        
        last_error = None
        start_time = time.time()
        
        for attempt_num, cmd in enumerate(conversion_attempts, 1):
            try:
                logger.info(f"AV1 conversion attempt {attempt_num}/{len(conversion_attempts)}")
                logger.debug(f"Command: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=300,  # 5 minute timeout
                    check=True
                )
                
                # If we get here, the conversion succeeded
                conversion_time = time.time() - start_time
                logger.info(f"✅ AV1 conversion successful using attempt {attempt_num} in {conversion_time:.2f}s")
                break
                
            except subprocess.CalledProcessError as e:
                last_error = e
                logger.warning(f"Attempt {attempt_num} failed: {e.stderr[:200]}...")
                
                # Clean up any partial output
                if os.path.exists(output_path):
                    try:
                        os.unlink(output_path)
                    except:
                        pass
                
                # If this was the last attempt, raise the error
                if attempt_num == len(conversion_attempts):
                    logger.error("All conversion attempts failed")
                    raise e
                    
                continue
        
        conversion_time = time.time() - start_time
        
        # Verify the output file was created
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise Exception(f"Conversion failed: output file not created or empty")
        
        logger.info(f"✅ AV1 to H264 conversion completed in {conversion_time:.2f} seconds")
        logger.info(f"Output file: {output_path} ({os.path.getsize(output_path)} bytes)")
        
        return output_path
        
    except subprocess.TimeoutExpired:
        logger.error("AV1 to H264 conversion timed out after 5 minutes")
        raise Exception("Video conversion timeout")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion error: {e.stderr}")
        raise Exception(f"Video conversion failed: {e}")
    except Exception as e:
        logger.error(f"Error converting AV1 to H264: {e}")
        raise


def convert_av1_via_intermediate(input_path: str, output_path: str) -> str:
    """
    Fallback AV1 conversion using OpenCV or frame extraction.
    This method bypasses the broken FFmpeg AV1 decoder entirely.
    """
    try:
        logger.info("Attempting OpenCV-based AV1 conversion...")
        
        # Try to open with OpenCV
        import cv2
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise Exception("OpenCV cannot open AV1 file")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties: {width}x{height} @ {fps}fps, {frame_count} frames")
        
        # Create output with OpenCV writing to H.264
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise Exception("Cannot create output video writer")
        
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            frame_num += 1
            
            if frame_num % 50 == 0:
                logger.debug(f"Processed {frame_num}/{frame_count} frames")
        
        cap.release()
        out.release()
        
        # Verify output
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise Exception("OpenCV conversion failed: output file not created")
        
        logger.info(f"✅ OpenCV AV1 conversion successful ({frame_num} frames)")
        return output_path
        
    except Exception as opencv_error:
        logger.warning(f"OpenCV conversion failed: {opencv_error}")
        logger.info("Trying frame extraction method as last resort...")
        
        # Ultimate fallback: extract frames as images, then re-encode
        import tempfile
        import shutil
        
        temp_dir = None
        try:
            # Create temporary directory for frames
            temp_dir = tempfile.mkdtemp(prefix="av1_frames_")
            logger.info(f"Extracting AV1 frames to: {temp_dir}")
            
            # Extract frames - ignore errors, just get what we can
            extract_cmd = [
                "ffmpeg", "-y",
                "-vsync", "0",
                "-i", input_path,
                "-qscale:v", "1",
                f"{temp_dir}/frame_%06d.png"
            ]
            
            subprocess.run(extract_cmd, capture_output=True, timeout=300, check=False)
            
            # Count extracted frames
            frame_files = sorted([f for f in os.listdir(temp_dir) if f.startswith("frame_")])
            
            if len(frame_files) == 0:
                raise Exception("No frames could be extracted from AV1 video")
            
            logger.info(f"Extracted {len(frame_files)} frames")
            
            # Re-encode from frames
            encode_cmd = [
                "ffmpeg", "-y",
                "-framerate", "25",
                "-i", f"{temp_dir}/frame_%06d.png",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-an",
                output_path
            ]
            
            subprocess.run(encode_cmd, capture_output=True, text=True, timeout=300, check=True)
            
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise Exception("Frame-based conversion failed")
            
            logger.info("✅ Frame-based AV1 conversion successful")
            return output_path
            
        except Exception as e:
            logger.error(f"All AV1 conversion methods failed: {e}")
            raise Exception(f"Cannot convert AV1 video - FFmpeg AV1 decoder is broken on this system")
            
        finally:
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass


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

def calculate_compression_score_with_soft_threshold(
    vmaf_score: float,
    compression_rate: float,
    vmaf_threshold: float,
    compression_weight: float = COMPRESSION_WEIGHT,
    quality_weight: float = QUALITY_WEIGHT,
    soft_threshold_margin: float = SOFT_THRESHOLD_MARGIN) -> tuple[float, str]:
    """
    Calculate compression score with soft threshold and heavy penalty for poor compression.
    
    Score is capped at maximum 1.0.
    Compression rates near 1.0 (little compression) are heavily penalized.
    
    Args:
        vmaf_score: VMAF score (0-100)
        compression_rate: Size ratio (compressed/original, smaller is better)
        vmaf_threshold: Required VMAF threshold
        compression_weight: Weight for compression component (default: 0.75)
        quality_weight: Weight for quality bonus component (default: 0.25)
        soft_threshold_margin: Points below threshold before hard cutoff (default: 5.0)
    
    Returns:
        Tuple of (final_score, reason) where final_score is in range [0.0, 1.0]
    """
    
    # Validate weights
    if abs(compression_weight + quality_weight - 1.0) > 0.01:
        raise ValueError(f"Weights must sum to 1.0, got {compression_weight + quality_weight}")
    
    hard_cutoff = vmaf_threshold - soft_threshold_margin
    
    # CASE 1: Below hard cutoff - immediate failure
    if vmaf_score < hard_cutoff:
        return 0.0, f"VMAF {vmaf_score:.1f} below hard cutoff ({hard_cutoff:.1f})"
    
    # CASE 2: In soft threshold zone - gradual penalty
    if vmaf_score < vmaf_threshold:
        soft_zone_position = (vmaf_score - hard_cutoff) / soft_threshold_margin
        quality_penalty = math.exp(-3 * (1 - soft_zone_position))
        
        # Heavy penalty for poor compression in soft zone
        if compression_rate >= 0.95:
            compression_component = 0.0  # Essentially no compression
        else:
            # Exponential decay for rates near 1.0
            compression_component = (1 - compression_rate) ** 2.5 * 0.6
        
        final_score = compression_component * quality_penalty
        
        return min(1.0, final_score), f"VMAF {vmaf_score:.1f} in soft threshold zone (penalty: {quality_penalty:.2f})"
    
    # CASE 3: Above threshold - prioritize compression with quality bonus
    else:
        # Heavy penalty for compression rates near 1.0
        if compression_rate >= 0.95:
            # Almost no compression - give very low score even with good VMAF
            compression_component = 0.0
            reason_suffix = " (WARNING: minimal compression)"
        elif compression_rate >= 0.85:
            # Poor compression (< 1.18x) - exponential penalty
            compression_component = ((1 - compression_rate) / 0.15) ** 3 * 0.3
            reason_suffix = " (poor compression)"
        else:
            # Good compression - logarithmic rewards with diminishing returns
            compression_ratio = 1 / compression_rate if compression_rate > 0 else 1
            max_expected_compression = 20.0
            compression_component = math.log(1 + compression_ratio) / math.log(1 + max_expected_compression)
            reason_suffix = ""
        
        # Quality bonus for exceeding threshold
        quality_bonus = min(1.0, (vmaf_score - vmaf_threshold) / (100 - vmaf_threshold))
        
        # Weighted combination
        final_score = (compression_weight * compression_component + 
                      quality_weight * quality_bonus)
        
        # Ensure score never exceeds 1.0
        final_score = min(1.0, final_score)
        
        return final_score, f"success{reason_suffix}"




@app.post("/score_upscaling_synthetics")
async def score_upscaling_synthetics(request: UpscalingScoringRequest) -> UpscalingScoringResponse:
    logger.info("#################### 🤖 start upscaling request scoring ####################")

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
            logger.info(f"🧩 Processing pair {idx+1}/{len(request.distorted_urls)}: UID {uid} 🧩")
            
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
    logger.info("#################### 🤖 start compression request scoring ####################")
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
        # Initialize converted paths as None at the start of each iteration
        converted_ref_path = None
        converted_dist_path = None
        
        try:
            logger.info(f"🧩 Processing pair {idx+1}/{len(request.distorted_urls)}: UID {uid} 🧩")
            
            uid_start_time = time.time()  # Start time for this UID

            ref_cap = None
            dist_path = None

            ref_cap = cv2.VideoCapture(ref_path)
            
            # Check if the reference video was actually opened successfully
            if not ref_cap.isOpened():
                # Add diagnostic information
                file_exists = os.path.exists(ref_path)
                file_size = os.path.getsize(ref_path) if file_exists else 0
                
                logger.info(f"Error opening reference video file {ref_path}.")
                logger.info(f"  File exists: {file_exists}")
                logger.info(f"  File size: {file_size} bytes")
                logger.info(f"  Current working directory: {os.getcwd()}")
                logger.info(f"  Assigning score of 0.")
                
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)  # No compression achieved
                final_scores.append(-100)
                reasons.append(f"error opening reference video file: {ref_path} (exists: {file_exists}, size: {file_size})")
                continue
            
            # Only log success after confirming the file was opened
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 1. Opened reference video in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            ref_total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 2. Retrieved reference video frame count ({ref_total_frames}) in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            if ref_total_frames < 10:
                logger.info(f"Video must contain at least 10 frames. Assigning score of 0.")
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)  # No compression achieved
                final_scores.append(-100)
                reasons.append("reference video has fewer than 10 frames")
                ref_cap.release()
                continue

            # Get reference video file size for compression rate calculation
            ref_file_size = os.path.getsize(ref_path)
            logger.info(f"Reference video file size: {ref_file_size} bytes")

            if len(dist_url) < 10:
                logger.info(f"Wrong dist download URL: {dist_url}. Assigning score of 0.")
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)  # No compression achieved
                final_scores.append(0.0)
                reasons.append("Invalid download URL: the distorted video download URL must be at least 10 characters long.")
                continue

            try:
                dist_path, download_time = await download_video(dist_url, request.verbose)
            except Exception as e:
                error_msg = f"Failed to download video from {dist_url}: {str(e)}"
                logger.error(f"{error_msg}. Assigning score of 0.")
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)  # No compression achieved
                final_scores.append(0.0)
                reasons.append("failed to download video file from url")
                continue

            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 3. Downloaded distorted video in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            # Detect video codecs
            ref_codec = detect_video_codec(ref_path)
            dist_codec = detect_video_codec(dist_path)
            
            # Get distorted video file size for compression rate calculation
            dist_file_size = os.path.getsize(dist_path)
            logger.info(f"Distorted video file size: {dist_file_size} bytes")
            
            # Use ffprobe for ALL videos (more reliable than OpenCV)
            logger.info("📊 Getting distorted video frame count via ffprobe")
            try:
                dist_total_frames = get_video_frame_count(dist_path)
                logger.info(f"Distorted video has {dist_total_frames} frames (via ffprobe).")
            except Exception as e:
                logger.error(f"Failed to get frame count: {e}")
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)
                final_scores.append(0.0)
                reasons.append(f"Failed to read video metadata: {str(e)}")
                continue
            
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 4. Retrieved distorted video frame count in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            # Verify frame counts match
            if dist_total_frames != ref_total_frames:
                logger.error(
                    f"Video length mismatch for pair {idx+1}: ref({ref_total_frames}) != dist({dist_total_frames}). Assigning score of 0."
                )
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)  # No compression achieved
                final_scores.append(0.0)
                reasons.append("video length mismatch")
                if dist_path and os.path.exists(dist_path):
                    os.unlink(dist_path)
                continue
            ''' LEGACY CODE
            # Sample frames for VMAF calculation 
            # random_frames = sorted(random.sample(range(ref_total_frames), VMAF_SAMPLE_COUNT))
            # print(f"Randomly selected {VMAF_SAMPLE_COUNT} frames for VMAF score: frame list: {random_frames}")
            #print(f"♎️ 6. Selected random frames for VMAF in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")
            # LEGACY
            #ref_y4m_path = convert_mp4_to_y4m(ref_path, random_frames)
            #print("The reference video has been successfully converted to Y4M format.")
            #step_time = time.time() - uid_start_time
            #print(f"♎️ 7. Converted reference video to Y4M in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")
            '''
            # Calculate VMAF - try direct AV1 first, convert only if it fails
            try:
                logger.info(f"🎬 Attempting VMAF calculation with original codecs (ref: {ref_codec}, dist: {dist_codec})")
                vmaf_start = time.time()
                vmaf_score = calculate_vmaf_neg_segments(ref_path=ref_path, dist_path=dist_path)
                vmaf_calc_time = time.time() - vmaf_start
                logger.info(f"✅ VMAF calculation succeeded with original codecs in {vmaf_calc_time:.2f} seconds")
                logger.info(f"☣️☣️ VMAF calculation took {vmaf_calc_time:.2f} seconds.")

                step_time = time.time() - uid_start_time
                logger.info(f"♎️ 5. Completed VMAF calculation in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

                if vmaf_score is not None:
                    vmaf_scores.append(vmaf_score)
                else:
                    vmaf_score = 0.0
                    vmaf_scores.append(vmaf_score)
                logger.info(f"🎾 VMAF score is {vmaf_score}")
                
            except Exception as vmaf_error:
                logger.warning(f"⚠️ VMAF calculation failed with original codecs: {vmaf_error}")
                logger.info("🔄 Converting AV1 videos to H264 and retrying VMAF...")
                
                try:
                    # Convert AV1 videos to H264
                    if ref_codec == 'av1':
                        logger.info("🔄 Reference video is AV1, converting to H264...")
                        converted_ref_path = convert_av1_to_h264(ref_path, "_ref_264")
                        ref_path_for_processing = converted_ref_path
                    else:
                        ref_path_for_processing = ref_path
                    
                    if dist_codec == 'av1':
                        logger.info("🔄 Distorted video is AV1, converting to H264...")
                        converted_dist_path = convert_av1_to_h264(dist_path, "_dist_264")
                        dist_path_for_processing = converted_dist_path
                    else:
                        dist_path_for_processing = dist_path
                    
                    step_time = time.time() - uid_start_time
                    logger.info(f"♎️ 5.5. Completed AV1 conversions in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")
                    
                    # Retry VMAF with converted videos
                    vmaf_start = time.time()
                    vmaf_score = calculate_vmaf_neg_segments(
                        ref_path=ref_path_for_processing, 
                        dist_path=dist_path_for_processing
                    )
                    vmaf_calc_time = time.time() - vmaf_start
                    logger.info(f"✅ VMAF calculation succeeded after conversion in {vmaf_calc_time:.2f} seconds")
                    logger.info(f"☣️☣️ VMAF calculation took {vmaf_calc_time:.2f} seconds.")

                    step_time = time.time() - uid_start_time
                    logger.info(f"♎️ 6. Completed VMAF calculation (after conversion) in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

                    if vmaf_score is not None:
                        vmaf_scores.append(vmaf_score)
                    else:
                        vmaf_score = 0.0
                        vmaf_scores.append(vmaf_score)
                    logger.info(f"🎾 VMAF score is {vmaf_score}")
                    
                except Exception as conversion_error:
                    logger.error(f"❌ VMAF calculation failed even after conversion: {conversion_error}")
                    vmaf_scores.append(0.0)
                    compression_rates.append(1.0)
                    final_scores.append(0.0)
                    reasons.append(f"VMAF calculation failed: {str(conversion_error)}")
                    continue

            # Calculate compression rate
            if ref_file_size > 0 and dist_file_size < ref_file_size:
                compression_rate = dist_file_size / ref_file_size
                logger.info(f"Compression rate: {compression_rate:.4f} ({dist_file_size}/{ref_file_size})")
            else:
                compression_rate = 1.0
                logger.info("Reference file size is 0 or distorted file size is greater than reference file size, setting compression rate to 1.0")

            # Check VMAF threshold first
            if vmaf_score < vmaf_threshold - 10:
                logger.info(f"VMAF score is lower than threshold, giving zero score, current VMAF score: {vmaf_score}, vmaf threshold: {vmaf_threshold}")
                compression_rates.append(compression_rate)  # Keep the compression rate even if VMAF fails
                final_scores.append(0.0)
                reasons.append(f"VMAF score is too low, current VMAF score: {vmaf_score}")
                continue

            compression_score, reason = calculate_compression_score_with_soft_threshold(
                compression_rate=compression_rate,
                vmaf_score=vmaf_score,
                vmaf_threshold=vmaf_threshold,
                soft_threshold_margin=5
            )
            
            # Calculate final score (for compression, this is the same as compression_score)
            logger.info(f"🎯 Final compression score is {compression_score:.4f}")

            compression_rates.append(compression_rate)
            final_scores.append(compression_score)
            reasons.append("success" + reason)

            step_time = time.time() - uid_start_time
            logger.info(f"🛑 Processed one UID in {step_time:.2f} seconds.")

        except Exception as e:
            error_msg = f"Failed to process video from {dist_url}: {str(e)}"
            logger.info(f"{error_msg}. Assigning score of 0.")
            vmaf_scores.append(0.0)
            compression_rates.append(1.0)  # No compression achieved
            final_scores.append(0.0)
            reasons.append("failed to process video")

        finally:
            # Clean up resources for this pair
            if ref_cap:
                ref_cap.release()
            if dist_path and os.path.exists(dist_path):
                os.unlink(dist_path)
            
            # Clean up converted AV1 files
            if converted_ref_path and os.path.exists(converted_ref_path):
                logger.info(f"🧹 Cleaning up converted reference file: {converted_ref_path}")
                os.unlink(converted_ref_path)
            if converted_dist_path and os.path.exists(converted_dist_path):
                logger.info(f"🧹 Cleaning up converted distorted file: {converted_dist_path}")
                os.unlink(converted_dist_path)

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
          
            

  

@app.post("/score_organics_upscaling")
async def score_organics_upscaling(request: OrganicsUpscalingScoringRequest) -> OrganicsUpscalingScoringResponse:
    print("#################### 🤖 start scoring ####################")
    batch_start_time = time.time()
    vmaf_scores = []
    pieapp_scores = []
    quality_scores = []
    length_scores = []
    final_scores = []
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
        ref_clip_path = None
        dist_clip_path = None
        ref_upscaled_clip_path = None

        scale_factor = 2
        if task_type == "SD24K":
            scale_factor = 4

        print(f"scale factor: {scale_factor}")

        # === REFERENCE VIDEO VALIDATION (NOT MINER'S FAULT) ===
        try:
            # download reference video
            if len(ref_url) < 10:
                print(f"invalid reference download url: {ref_url}. skipping scoring")
                reasons.append("invalid reference url - skipped")
                vmaf_scores.append(-1)
                pieapp_scores.append(-1)
                quality_scores.append(-1)
                length_scores.append(-1)
                final_scores.append(-1)  # -1 means skip, don't penalize
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
                
                reasons.append("corrupted reference video - skipped")
                vmaf_scores.append(-1)
                pieapp_scores.append(-1)
                quality_scores.append(-1)
                length_scores.append(-1)
                final_scores.append(-1)  # -1 means skip, don't penalize
                continue
            
            ref_total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ref_fps = ref_cap.get(cv2.CAP_PROP_FPS)
            ref_width = int(ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ref_height = int(ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if ref_total_frames < 10:
                print(f"reference video too short (<10 frames). skipping scoring")
                reasons.append("reference video too short - skipped")
                vmaf_scores.append(-1)
                pieapp_scores.append(-1)
                quality_scores.append(-1)
                length_scores.append(-1)
                final_scores.append(-1)  # -1 means skip, don't penalize
                ref_cap.release()
                continue

        except Exception as e:
            print(f"system error processing reference video: {e}. skipping scoring")
            reasons.append("system error with reference - skipped")
            vmaf_scores.append(-1)
            pieapp_scores.append(-1)
            quality_scores.append(-1)
            length_scores.append(-1)
            final_scores.append(-1)  # -1 means skip, don't penalize
            if ref_cap:
                ref_cap.release()
            continue

        # === MINER OUTPUT VALIDATION (MINER'S FAULT) ===
        try:
            # check if distorted video failed to download
            if dist_path is None:
                print(f"failed to download distorted video for uid {uid}. penalizing miner.")
                reasons.append("MINER FAILURE: failed to download distorted video, invalid url")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)  # 0 = miner penalty
                ref_cap.release()
                continue

            # Check if miner's output can be opened
            dist_cap = cv2.VideoCapture(dist_path)
            if not dist_cap.isOpened():
                print(f"error opening distorted video from {dist_path}. penalizing miner.")
                reasons.append("MINER FAILURE: corrupted output video")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)  # 0 = miner penalty
                ref_cap.release()
                continue

            dist_total_frames = int(dist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            dist_width = int(dist_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            dist_height = int(dist_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"distorted video has {dist_total_frames} frames.")

            # Check frame count match
            if dist_total_frames != ref_total_frames:
                print(f"video length mismatch: ref({ref_total_frames}) != dist({dist_total_frames}). penalizing miner.")
                reasons.append("MINER FAILURE: video length mismatch")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)  # 0 = miner penalty
                ref_cap.release()
                dist_cap.release()
                continue

            # Check minimum frame count for miner output
            if dist_total_frames < 10:
                print(f"miner output too short (<10 frames). penalizing miner.")
                reasons.append("MINER FAILURE: output video too short")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)  # 0 = miner penalty
                ref_cap.release()
                dist_cap.release()
                continue

            # === RESOLUTION CHECK ===
            expected_width = ref_width * scale_factor
            expected_height = ref_height * scale_factor
            
            if dist_width != expected_width or dist_height != expected_height:
                print(f"resolution mismatch: expected {expected_width}x{expected_height}, got {dist_width}x{dist_height}. penalizing miner.")
                reasons.append("MINER FAILURE: incorrect upscaling resolution")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)  # 0 = miner penalty
                ref_cap.release()
                dist_cap.release()
                continue

        except Exception as e:
            print(f"error validating miner output: {e}. penalizing miner.")
            reasons.append(f"MINER FAILURE: {e}")
            vmaf_scores.append(0.0)
            pieapp_scores.append(0.0)
            quality_scores.append(0.0)
            length_scores.append(0.0)
            final_scores.append(0.0)  # 0 = miner penalty
            if ref_cap:
                ref_cap.release()
            if dist_cap:
                dist_cap.release()
            continue

        # === MAIN SCORING LOGIC ===
        try:
            # Calculate 0.5 second clip duration
            clip_duration = 0.25  # seconds
            
            # Calculate maximum start time to ensure we don't exceed video length
            ref_duration = ref_total_frames / ref_fps
            max_start_time = max(0, ref_duration - clip_duration)
            start_time = random.uniform(0, max_start_time)
            
            print(f"Creating 0.5-second clips starting from {start_time:.2f} seconds")
            
            # Create 0.5-second reference clip
            ref_clip_path = trim_video(ref_path, start_time, clip_duration)
            print(f"Created reference clip: {ref_clip_path}")
            
            # Create 0.5-second distorted clip
            dist_clip_path = trim_video(dist_path, start_time, clip_duration)
            print(f"Created distorted clip: {dist_clip_path}")
            
            # Upscale the reference clip
            ref_upscaled_clip_path = upscale_video(ref_clip_path, scale_factor)
            print(f"Created upscaled reference clip: {ref_upscaled_clip_path}")
            
            # Calculate ClipIQ scores
            print("Calculating ClipIQ scores...")
            ref_clipiqa_score = get_clipiqa_score(ref_clip_path, num_frames=3)
            dist_clipiqa_score = get_clipiqa_score(dist_clip_path, num_frames=3)
            ref_upscaled_clipiqa_score = get_clipiqa_score(ref_upscaled_clip_path, num_frames=3)
            
            print(f"Reference ClipIQ score: {ref_clipiqa_score:.4f}")
            print(f"Distorted ClipIQ score: {dist_clipiqa_score:.4f}")
            print(f"Upscaled reference ClipIQ score: {ref_upscaled_clipiqa_score:.4f}")
            
            # Create Y4M files for VMAF calculation
            print("Creating Y4M files for VMAF calculation...")
            # Get the number of frames in the clips for Y4M conversion
            ref_clip_cap = cv2.VideoCapture(ref_upscaled_clip_path)
            ref_clip_frames = int(ref_clip_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ref_clip_cap.release()
            
            ref_y4m_path = convert_mp4_to_y4m(ref_upscaled_clip_path, list(range(ref_clip_frames)))
            
            # Calculate VMAF score
            print("Calculating VMAF score...")
            vmaf_score = calculate_vmaf(ref_y4m_path, dist_clip_path, list(range(ref_clip_frames)))
            
            if vmaf_score is None:
                vmaf_score = 0.0
            
            print(f"VMAF score: {vmaf_score:.4f}")
            
            # Check VMAF threshold
            if vmaf_score < 50:
                print(f"VMAF score below threshold (50): {vmaf_score}. Assigning score 0.")
                vmaf_scores.append(vmaf_score)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append(f"VMAF score too low: {vmaf_score}")
                continue
            
            # Final scoring based on ClipIQ scores
            if dist_clipiqa_score > ref_clipiqa_score:
                final_score = 3
                reason = "distorted better than reference"
            elif dist_clipiqa_score >= ref_upscaled_clipiqa_score + 0.008:
                final_score = 2
                reason = "distorted between reference and upscaled+0.008"
            else:
                final_score = 1
                reason = "distorted below upscaled+0.008"
            
            print(f"Final score: {final_score} ({reason})")
            
            vmaf_scores.append(vmaf_score)
            pieapp_scores.append(0.0)  
            quality_scores.append(0.0)  
            length_scores.append(0.0)
            final_scores.append(final_score)
            reasons.append(f"{reason}")

        except Exception as e:
            print(f"error in main scoring logic: {e}. penalizing miner.")
            reasons.append(f"MINER FAILURE: scoring error - {e}")
            vmaf_scores.append(0.0)
            pieapp_scores.append(0.0)
            quality_scores.append(0.0)
            length_scores.append(0.0)
            final_scores.append(0.0)  # 0 = miner penalty

        finally:
            # Clean up resources
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
            if ref_clip_path and os.path.exists(ref_clip_path):
                os.unlink(ref_clip_path)
            if dist_clip_path and os.path.exists(dist_clip_path):
                os.unlink(dist_clip_path)
            if ref_upscaled_clip_path and os.path.exists(ref_upscaled_clip_path):
                os.unlink(ref_upscaled_clip_path)

    processed_time = time.time() - batch_start_time
    print(f"🔯🔯🔯 calculated scores: {final_scores} 🔯🔯🔯")
    print(f"completed one batch scoring within {processed_time:.2f} seconds")
    
    # Summary statistics
    successful_miners = sum(1 for score in final_scores if score >= 1)
    failed_miners = sum(1 for score in final_scores if score == 0.0)
    skipped_miners = sum(1 for score in final_scores if score == -1)
    
    print(f"📊 SCORING SUMMARY:")
    print(f"  ✅ Successful miners: {successful_miners}")
    print(f"  ❌ Failed miners: {failed_miners}")
    print(f"  ⏭️ Skipped miners: {skipped_miners}")
    
    return OrganicsUpscalingScoringResponse(
        vmaf_scores=vmaf_scores,
        pieapp_scores=pieapp_scores,
        quality_scores=quality_scores,
        length_scores=length_scores,
        final_scores=final_scores,
        reasons=reasons
    )

@app.post("/score_organics_compression")
async def score_organics_compression(request: OrganicsCompressionScoringRequest) -> OrganicsCompressionScoringResponse:
    print("#################### 🤖 start scoring ####################")
    start_time = time.time()
    vmaf_scores = []
    compression_rates = []
    final_scores = []
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

    for idx, (ref_url, dist_path, uid, vmaf_threshold) in enumerate(
        zip(request.reference_urls, distorted_video_paths, request.uids, request.vmaf_thresholds)
    ):
        print(f"🧩 processing {uid}.... downloading reference video.... 🧩")
        ref_path = None
        ref_cap = None
        dist_cap = None
        ref_y4m_path = None
        ref_clip_path = None
        dist_clip_path = None

        uid_start_time = time.time() # start time for each uid

        # === REFERENCE VIDEO VALIDATION (NOT MINER'S FAULT) ===
        try:
            # download reference video
            if len(ref_url) < 10:
                print(f"invalid reference download url: {ref_url}. skipping scoring")
                vmaf_scores.append(-1)
                compression_rates.append(-1)
                reasons.append("invalid reference url - skipped")
                final_scores.append(-1)  # -1 means skip, don't penalize
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
                compression_rates.append(-1)
                reasons.append("corrupted reference video - skipped")
                final_scores.append(-1)  # -1 means skip, don't penalize
                continue
            
            ref_total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Get reference video file size for compression rate calculation
            ref_file_size = os.path.getsize(ref_path)
            print(f"Reference video file size: {ref_file_size} bytes")

            if ref_total_frames < 10:
                print(f"reference video too short (<10 frames). skipping scoring")
                vmaf_scores.append(-1)
                compression_rates.append(-1)
                reasons.append("reference video too short - skipped")
                final_scores.append(-1)  # -1 means skip, don't penalize
                ref_cap.release()
                continue

        except Exception as e:
            print(f"system error processing reference video: {e}. skipping scoring")
            vmaf_scores.append(-1)
            compression_rates.append(-1)
            reasons.append("system error with reference - skipped")
            final_scores.append(-1)  # -1 means skip, don't penalize
            if ref_cap:
                ref_cap.release()
            continue

        # === MINER OUTPUT VALIDATION (MINER'S FAULT) ===
        try:
            # check if distorted video failed to download
            if dist_path is None:
                print(f"failed to download distorted video for uid {uid}. penalizing miner.")
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)
                reasons.append("MINER FAILURE: failed to download distorted video, invalid url")
                final_scores.append(0.0)  # 0 = miner penalty
                ref_cap.release()
                continue

            # Check if miner's output can be opened
            dist_cap = cv2.VideoCapture(dist_path)
            if not dist_cap.isOpened():
                print(f"error opening distorted video from {dist_path}. penalizing miner.")
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)
                reasons.append("MINER FAILURE: corrupted output video")
                final_scores.append(0.0)  # 0 = miner penalty
                ref_cap.release()
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

            # Calculate 0.5 second clip duration for VMAF calculation
            clip_duration = 0.5  # seconds
            
            # Get video FPS to calculate frame-based timing
            ref_fps = ref_cap.get(cv2.CAP_PROP_FPS)
            print(f"Reference video FPS: {ref_fps}")
            
            # Calculate maximum start time to ensure we don't exceed video length
            ref_duration = ref_total_frames / ref_fps
            max_start_time = max(0, ref_duration - clip_duration)
            start_time = random.uniform(0, max_start_time)
            
            print(f"Creating 0.5-second clips starting from {start_time:.2f} seconds")
            step_time = time.time() - uid_start_time
            print(f"♎️ 6. Selected random start time for video chunks in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            # Create 0.5-second reference clip
            ref_clip_path = trim_video(ref_path, start_time, clip_duration)
            print(f"Created reference clip: {ref_clip_path}")
            step_time = time.time() - uid_start_time
            print(f"♎️ 7. Created reference video clip in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            # Create 0.5-second distorted clip
            dist_clip_path = trim_video(dist_path, start_time, clip_duration)
            print(f"Created distorted clip: {dist_clip_path}")
            step_time = time.time() - uid_start_time
            print(f"♎️ 8. Created distorted video clip in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            # Get the number of frames in the clips for Y4M conversion
            ref_clip_cap = cv2.VideoCapture(ref_clip_path)
            ref_clip_frames = int(ref_clip_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ref_clip_cap.release()
            
            # Ensure we don't sample more frames than available
            sample_count = min(VMAF_SAMPLE_COUNT, ref_clip_frames)
            random_frames = sorted(random.sample(range(ref_clip_frames), sample_count))
            ref_y4m_path = convert_mp4_to_y4m(ref_clip_path, random_frames)
            print("The reference video clip has been successfully converted to Y4M format.")
            step_time = time.time() - uid_start_time
            print(f"♎️ 9. Converted reference video clip to Y4M in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            # Calculate VMAF on chunked videos
            try:
                vmaf_start = time.time()
                vmaf_score = calculate_vmaf(ref_y4m_path, dist_clip_path, random_frames)
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
            if dist_cap:
                dist_cap.release()
            if ref_path and os.path.exists(ref_path):
                os.unlink(ref_path)
            if dist_path and os.path.exists(dist_path):
                os.unlink(dist_path)
            if ref_y4m_path and os.path.exists(ref_y4m_path):
                os.unlink(ref_y4m_path)
            # Clean up chunked video files
            if ref_clip_path and os.path.exists(ref_clip_path):
                os.unlink(ref_clip_path)
            if dist_clip_path and os.path.exists(dist_clip_path):
                os.unlink(dist_clip_path)

    return OrganicsCompressionScoringResponse(
        vmaf_scores=vmaf_scores,
        compression_rates=compression_rates,
        final_scores=final_scores,
        reasons=reasons
    )

if __name__ == "__main__":
    import uvicorn
    
    host = CONFIG.score.host
    port = CONFIG.score.port
    
    uvicorn.run(app, host=host, port=port)