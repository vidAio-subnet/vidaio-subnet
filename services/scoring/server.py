import os
import cv2
import glob
import json
import time
import math
import torch
import random
import asyncio
import aiohttp
from loguru import logger
import tempfile
import subprocess
import numpy as np
from pydantic import BaseModel
from typing import Optional, List
from firerequests import FireRequests
from vidaio_subnet_core import CONFIG
from torchvision.models import resnet50
from fastapi import FastAPI, HTTPException
import torchvision.transforms as transforms
from pieapp_metric import calculate_pieapp_score
from vidaio_subnet_core.utilities.storage_client import storage_client
from vmaf_metric import calculate_vmaf, convert_mp4_to_y4m, trim_video
from services.video_scheduler.video_utils import get_trim_video_path, delete_videos_with_fileid
from scoring_function import calculate_compression_score

# Compression scoring constants
COMPRESSION_RATE_WEIGHT = 0.7  # w_c
COMPRESSION_VMAF_WEIGHT = 0.3  # w_vmaf
SOFT_THRESHOLD_MARGIN = 5.0  # Margin below VMAF threshold for soft scoring zone


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
    payload_urls: List[str]
    video_ids: List[str]
    uploaded_object_names: List[str]
    content_lengths: List[int]
    task_types: List[str]
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

        timeout = aiohttp.ClientTimeout(sock_connect=10, total=40)

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

def extract_frames_from_y4m(y4m_path):
    """
    Extract frames from Y4M file for color validation.
    This reuses the Y4M file already created for VMAF calculation,
    eliminating redundant video processing.
    
    Args:
        y4m_path: Path to Y4M file (output of convert_mp4_to_y4m)
        
    Returns:
        list: List of frames as numpy arrays (BGR format for cv2)
    """
    frames = []
    
    # Create temporary directory for extracted frames
    temp_dir = tempfile.mkdtemp()
    output_pattern = os.path.join(temp_dir, "frame_%04d.png")
    
    try:
        # Extract all frames from Y4M as PNG using FFmpeg
        # Y4M files already contain only the selected frames, so no filtering needed
        extract_cmd = [
            "ffmpeg", "-i", y4m_path,
            output_pattern,
            "-y",
            "-hide_banner", "-loglevel", "error"
        ]
        
        result = subprocess.run(
            extract_cmd, 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        
        if result.returncode != 0:
            logger.error(f"FFmpeg Y4M frame extraction failed: {result.stderr}")
            return frames
        
        # Read extracted PNG files in order
        png_files = sorted(glob.glob(os.path.join(temp_dir, "frame_*.png")))
        
        for png_file in png_files:
            frame = cv2.imread(png_file)
            if frame is not None:
                frames.append(frame)
            else:
                logger.warning(f"Failed to read extracted frame: {png_file}")
        
        logger.debug(f"Successfully extracted {len(frames)} frames from Y4M file")
        
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg Y4M frame extraction timed out")
    except Exception as e:
        logger.error(f"Error extracting frames from Y4M: {e}")
    finally:
        # Cleanup temporary files
        try:
            for png_file in glob.glob(os.path.join(temp_dir, "*.png")):
                os.remove(png_file)
            os.rmdir(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp frames: {e}")
    
    return frames

def validate_color_channels_on_frames(frames):
    """
    Validate that frames have color information (not grayscale).
    Reuses frames already extracted for VMAF calculation.
    
    Args:
        frames: List of frames (BGR numpy arrays from cv2)
        
    Returns:
        tuple: (is_valid, reason)
    """
    try:
        if not frames:
            return False, "No frames available for color validation"
        
        color_threshold = 5.0  # Same threshold as before
        
        for i, frame in enumerate(frames):
            # Convert BGR to RGB
            img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract RGB channels
            r_channel = img_array[:,:,0].astype(float)
            g_channel = img_array[:,:,1].astype(float)
            b_channel = img_array[:,:,2].astype(float)
            
            # Calculate average difference between channels
            rg_diff = np.mean(np.abs(r_channel - g_channel))
            rb_diff = np.mean(np.abs(r_channel - b_channel))
            gb_diff = np.mean(np.abs(g_channel - b_channel))
            
            # If all channels are nearly identical, it's grayscale
            if rg_diff < color_threshold and rb_diff < color_threshold and gb_diff < color_threshold:
                logger.warning(
                    f"Frame {i} appears to be grayscale: "
                    f"RG diff={rg_diff:.2f}, RB diff={rb_diff:.2f}, GB diff={gb_diff:.2f}"
                )
                return False, f"Video has no color information (grayscale). UV channels required. Frame {i} detected as grayscale."
        
        return True, "Color channels validated"
        
    except Exception as e:
        logger.error(f"Error validating color channels: {e}")
        return False, f"Error validating color: {str(e)}"

def validate_chroma_quality_on_frames(ref_frames, dist_frames, threshold=0.7):
    """
    Validate chroma (UV) quality by comparing reference and distorted frames.
    Reuses frames already extracted for VMAF calculation.
    
    Args:
        ref_frames: List of reference frames (BGR numpy arrays)
        dist_frames: List of distorted frames (BGR numpy arrays)
        threshold: Minimum acceptable chroma similarity (0.0-1.0)
        
    Returns:
        tuple: (is_valid, reason)
    """
    try:
        if len(ref_frames) != len(dist_frames):
            return False, "Frame count mismatch between reference and distorted"
        
        if not ref_frames or not dist_frames:
            return False, "No frames available for chroma validation"
        
        chroma_ratios = []
        
        for i, (ref_frame, dist_frame) in enumerate(zip(ref_frames, dist_frames)):
            # Convert BGR to YUV
            ref_yuv = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2YUV)
            dist_yuv = cv2.cvtColor(dist_frame, cv2.COLOR_BGR2YUV)
            
            # Extract U and V channels
            ref_u = ref_yuv[:, :, 1].astype(float)
            ref_v = ref_yuv[:, :, 2].astype(float)
            dist_u = dist_yuv[:, :, 1].astype(float)
            dist_v = dist_yuv[:, :, 2].astype(float)
            
            # Calculate chroma variance/energy
            ref_u_variance = np.var(ref_u)
            ref_v_variance = np.var(ref_v)
            dist_u_variance = np.var(dist_u)
            dist_v_variance = np.var(dist_v)
            
            ref_chroma_energy = ref_u_variance + ref_v_variance
            dist_chroma_energy = dist_u_variance + dist_v_variance
            
            if ref_chroma_energy > 0:
                chroma_ratio = dist_chroma_energy / ref_chroma_energy
                chroma_ratios.append(chroma_ratio)
                logger.debug(f"Frame {i} chroma ratio: {chroma_ratio:.3f}")
        
        if not chroma_ratios:
            return False, "Could not calculate chroma ratios"
        
        # Calculate average chroma ratio
        avg_chroma_ratio = np.mean(chroma_ratios)
        
        logger.info(f"Average chroma quality ratio: {avg_chroma_ratio:.3f}, Threshold: {threshold}")
        
        if avg_chroma_ratio < threshold:
            return False, f"Chroma quality too low: {avg_chroma_ratio:.3f} < {threshold} (UV channels reduced/degraded)"
        
        return True, f"Chroma quality validated: {avg_chroma_ratio:.3f}"
        
    except Exception as e:
        logger.error(f"Error validating chroma quality: {e}")
        return False, f"Error validating chroma: {str(e)}"

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
    
    logger.info(f"Sampling {frames_to_sample} consecutive frames starting from frame {start_frame}")
    
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
    """
    Get frame count using ffprobe with optimized approach.
    First tries fast metadata read, falls back to frame counting if needed.
    """
    try:
        # Method 1: Fast metadata read (similar to OpenCV speed)
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames",  # Read from metadata (FAST)
            "-of", "default=nokey=1:noprint_wrappers=1",
            video_path
        ]
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True, 
            timeout=5  # Fast timeout for metadata read
        )
        
        frame_count_str = result.stdout.strip()
        if frame_count_str and frame_count_str != "N/A":
            frame_count = int(frame_count_str)
            logger.debug(f"ffprobe (fast): {video_path} has {frame_count} frames")
            return frame_count
        
        # Method 2: Frame counting fallback (slower but accurate)
        logger.debug(f"Metadata unavailable, counting frames for {video_path}")
        cmd = [
            "ffprobe",
            "-v", "error",
            "-count_frames",  # Actually count frames (SLOWER)
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1",
            video_path
        ]
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True, 
            timeout=30  # Longer timeout for counting
        )
        frame_count = int(result.stdout.strip())
        logger.debug(f"ffprobe (counted): {video_path} has {frame_count} frames")
        return frame_count
        
    except subprocess.TimeoutExpired:
        logger.error(f"ffprobe timeout for {video_path}")
        raise Exception(f"Frame count timeout: ffprobe took too long")
    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe failed for {video_path}: {e.stderr}")
        raise Exception(f"Frame count failed: ffprobe error - {e.stderr}")
    except ValueError as e:
        logger.error(f"ffprobe returned invalid frame count for {video_path}: {e}")
        raise Exception(f"Frame count failed: invalid ffprobe output")
    except Exception as e:
        logger.error(f"Unexpected error with ffprobe for {video_path}: {e}")
        raise Exception(f"Frame count failed: {e}")

def is_valid_video(video_path):
    """
    Check if a video file is valid and can be opened using ffprobe.
    Optimized for speed - only checks basic properties without counting frames.
    
    Returns:
        bool: True if video is valid, False otherwise
    """
    try:
        # First check if file exists and has size
        if not os.path.exists(video_path):
            logger.error(f"Video file does not exist: {video_path}")
            return False
        
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            logger.error(f"Video file is empty: {video_path}")
            return False
        
        if file_size < 1024:  # Less than 1KB is suspicious
            logger.warning(f"Video file is very small ({file_size} bytes): {video_path}")
        
        # Use ffprobe to validate video stream (FAST - no frame counting)
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_type,width,height,duration",  # Basic properties only
            "-of", "csv=p=0",
            video_path
        ]
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True, 
            timeout=5  # Reduced timeout since we're not counting frames
        )
        
        # Parse output: format is "video,width,height,duration"
        output = result.stdout.strip()
        if not output:
            logger.error(f"ffprobe returned empty output for {video_path}")
            return False
        
        parts = output.split(',')
        if len(parts) < 3:
            logger.error(f"ffprobe validation failed: incomplete output for {video_path}")
            return False
        
        # Check if it's a video stream
        if parts[0] != "video":
            logger.error(f"ffprobe validation failed: not a video file: {video_path}")
            return False
        
        # Check if width and height are valid
        try:
            width = int(parts[1])
            height = int(parts[2])
            if width <= 0 or height <= 0:
                logger.error(f"Invalid video dimensions: {width}x{height} for {video_path}")
                return False
        except (ValueError, IndexError):
            logger.error(f"Could not parse video dimensions for {video_path}")
            return False
        
        # Optionally check duration if available
        if len(parts) > 3 and parts[3]:
            try:
                duration = float(parts[3])
                if duration <= 0:
                    logger.warning(f"Video has invalid duration: {duration}s for {video_path}")
            except ValueError:
                pass  # Duration not critical for validation
        
        return True
        
    except subprocess.TimeoutExpired:
        logger.error(f"ffprobe validation timeout for {video_path}")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe validation failed for {video_path}: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error validating {video_path}: {e}")
        return False
        
    except subprocess.TimeoutExpired:
        logger.error(f"ffprobe validation timeout for {video_path}")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe validation failed for {video_path}: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error validating {video_path}: {e}")
        return False

def validate_dist_encoding_settings(dist_path: str, ref_path: str, task: str):
    """
    Validate that distorted video uses specific encoding settings.
    """
    try:
        # Enhanced ffprobe to capture both stream and format encoder tags
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,profile,level,sample_aspect_ratio,pix_fmt,"
                           "width,height,r_frame_rate,color_space,color_primaries,color_transfer",
            "-show_entries", "format=tags=encoder",
            "-show_format",
            "-of", "json",
            dist_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
        info = json.loads(result.stdout)
        
        video_stream = info.get("streams", [{}])[0]
        format_info = info.get("format", {})
        format_tags = format_info.get("tags", {})
        
        codec = video_stream.get("codec_name", "")
        profile = video_stream.get("profile", "")
        level = video_stream.get("level")
        sar = video_stream.get("sample_aspect_ratio", "1:1")
        pix_fmt = video_stream.get("pix_fmt", "")
        width = video_stream.get("width", 0)
        height = video_stream.get("height", 0)
        container = format_info.get("format_name", "")
        fps_str = video_stream.get("r_frame_rate", "0/1")
        encoder_tag = ""
        
        # Colorspace properties
        dist_color_space = video_stream.get("color_space", None)
        dist_color_primaries = video_stream.get("color_primaries", None)
        dist_color_transfer = video_stream.get("color_transfer", None)
        
        # Check encoder tags
        stream_tags = video_stream.get("tags", {})
        encoder_tag = stream_tags.get("encoder", "").lower() or format_tags.get("encoder", "").lower()
        
        errors = []

        ref_fps_str = get_video_fps(ref_path, original_str=True)
        if fps_str != ref_fps_str:
            errors.append(f"FPS must be {ref_fps_str}, got {fps_str}")

        if task == "compression":
            ref_width, ref_height = get_video_dimensions(ref_path)
            if width != ref_width or height != ref_height:
                errors.append(f"Resolution must be {ref_width}x{ref_height}, got {width}x{height}")

        # REQUIRED encoding checks
        if task == "compression" and codec != "av1" or task == "upscaling" and codec != "hevc":
            errors.append(f"Codec must be AV1 for compression & hevc for upscaling, got {codec}")
        if "ivf" in container.lower():
            errors.append("Container must be MP4, got IVF (incompatible for concatenation)")
        if container not in ["mov,mp4,m4a,3gp,3g2,mj2", "mp4", "isom"]:
            errors.append(f"Container must be proper MP4, got {container}")
        if profile != "Main":
            errors.append(f"AV1 profile must be 'Main', got {profile}")
        if sar != "1:1":
            errors.append(f"Sample aspect ratio must be 1:1, got {sar}")
        if pix_fmt != "yuv420p":
            errors.append(f"Pixel format must be yuv420p, got {pix_fmt}")

        # Colorspace validation
        if ref_path and os.path.exists(ref_path):
            try:
                # Get reference colorspace
                ref_cmd = [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=color_space,color_primaries,color_transfer",
                    "-of", "json", ref_path
                ]
                ref_result = subprocess.run(ref_cmd, capture_output=True, text=True, 
                                          timeout=5, check=True)
                ref_info = json.loads(ref_result.stdout)
                ref_stream = ref_info.get("streams", [{}])[0]
                
                ref_color_space = ref_stream.get("color_space", None)
                ref_color_primaries = ref_stream.get("color_primaries", None)
                ref_color_transfer = ref_stream.get("color_transfer", None)
                
                # Colorspace - mismatch affects color accuracy
                if (ref_color_space and dist_color_space and 
                    ref_color_space != dist_color_space):
                    errors.append(f"color_space mismatch, reference: {ref_color_space}, distorted: {dist_color_space}")
                
                # Transfer characteristics - mismatch affects VMAF significantly
                if (ref_color_transfer and dist_color_transfer and 
                    ref_color_transfer != dist_color_transfer):
                    errors.append(f"color_transfer mismatch, reference: {ref_color_transfer}, distorted: {dist_color_transfer}")
                
                # Primaries - less critical but still affects color accuracy
                if (ref_color_primaries and dist_color_primaries and 
                    ref_color_primaries != dist_color_primaries):
                    errors.append(f"color_primaries mismatch, reference: {ref_color_primaries}, distorted: {dist_color_primaries}")

                logger.warning(f"  Ref: space={ref_color_space}, primaries={ref_color_primaries}, transfer={ref_color_transfer}")
                logger.warning(f"  Dist: space={dist_color_space}, primaries={dist_color_primaries}, transfer={dist_color_transfer}")
                    # NOT a hard fail - allows innovation in HDR/Dolby Vision/etc.
                
            except subprocess.CalledProcessError:
                logger.warning("Could not read reference colorspace info")
        
        # SVT-AV1 detection, only logging for now
        is_svtav1 = any(keyword in encoder_tag for keyword in [
            "libsvtav1", "svt-av1", "svtav1"
        ])
        # # AV1 Level validation (FFmpeg uses internal integer levels)
        # av1_level_valid = False
        # if level is not None:
        #     # FFmpeg level 12 ≈ AV1 Level 5.1 (4K), map roughly
        #     # This is internal FFmpeg representation, not spec levels
        #     if isinstance(level, int) and level >= 8:  # Reasonable minimum for HD+
        #         av1_level_valid = True
        #     else:
        #         errors.append(f"AV1 level {level} too low for practical use")
        # else:
        #     # Level missing is acceptable if other params valid
        #     logger.warning("AV1 level not specified in metadata")
        
        # Encoder preference validation (not blocking)
        
        if is_svtav1:
            logger.info(f"✅ SVT-AV1 encoder detected: {encoder_tag}")
        elif encoder_tag:
            other_av1_encoders = ["libaom-av1", "rav1e", "libdav1d"]
            is_known_av1 = any(other in encoder_tag for other in other_av1_encoders)
            if is_known_av1:
                logger.info(f"ℹ️ Known AV1 encoder: {encoder_tag}")
            else:
                logger.warning(f"⚠️ Unknown encoder: {encoder_tag}")
        else:
            logger.warning("⚠️ No encoder tag found")
        
        logger.debug(f"Video analysis: {width}x{height}@{fps_str}, {codec}/{profile}, "
                    f"container={container}, color_space={dist_color_space}, "
                    f"SVT-AV1={is_svtav1}")
        
        if errors:
            return False, "; ".join(errors)
        
        color_info = f"space={dist_color_space or 'default'}" if dist_color_space else ""
        encoder_status = "SVT-AV1" if is_svtav1 else "Other AV1"
        return True, f"Valid encoding ({encoder_status}, {width}x{height}, {color_info}, level {level})"
        
    except subprocess.CalledProcessError as e:
        return False, f"ffprobe failed: {e.stderr}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def get_video_dimensions(video_path):
    """
    Get video dimensions (width, height) using ffprobe.
    This avoids OpenCV's AV1 decoder warnings.
    
    Returns:
        tuple: (width, height) or (None, None) if failed
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
        width, height = map(int, result.stdout.strip().split(','))
        return width, height
    except Exception as e:
        logger.warning(f"ffprobe dimension check failed for {video_path}: {e}")
        return None, None

def get_video_fps(video_path, original_str=False):
    """
    Get video FPS using ffprobe.
    This avoids OpenCV's AV1 decoder warnings.
    
    Returns:
        float: FPS or None if failed
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
        # r_frame_rate returns format like "30000/1001" or "30/1"
        fps_str = result.stdout.strip()
        if original_str:
            return fps_str
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            return num / den
        return float(fps_str)
    except Exception as e:
        logger.warning(f"ffprobe FPS check failed for {video_path}: {e}")
        return None

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
        logger.info("No ref frames to process")
        return -100
    
    if not dist_frames:
        logger.info("No dist frames to process")
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
    
    # Get video dimensions using ffprobe (avoids AV1 warnings)
    width, height = get_video_dimensions(input_path)
    if width is None or height is None:
        raise ValueError(f"Could not get video dimensions for {input_path}")
    
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
        logger.info(f"Successfully upscaled video to {new_width}x{new_height}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Error upscaling video: {e}")
        return input_path

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
    
    if len(request.payload_urls) != len(request.distorted_urls):
        raise HTTPException(
            status_code=400, 
            detail="Number of payload URLs must match number of distorted URLs"
        )

    for idx, (ref_path, dist_url, payload_url, uid, video_id, uploaded_object_name, content_length, task_type) in enumerate(zip(
        request.reference_paths, 
        request.distorted_urls, 
        request.payload_urls,
        request.uids,
        request.video_ids,
        request.uploaded_object_names,
        request.content_lengths,
        request.task_types
    )):
        try:
            logger.info(f"🧩 Processing pair {idx+1}/{len(request.distorted_urls)}: UID {uid} 🧩")
            
            uid_start_time = time.time()  # Start time for this UID

            ref_y4m_path = None
            dist_path = None

            scale_factor = 2
            if task_type == "SD24K":
                scale_factor = 4

            logger.info(f"scale factor: {scale_factor}")

            # Validate reference video using ffprobe (avoids AV1 warnings)
            if not is_valid_video(ref_path):
                # Add diagnostic information
                file_exists = os.path.exists(ref_path)
                file_size = os.path.getsize(ref_path) if file_exists else 0
                
                logger.error(f"Error opening reference video file {ref_path}.")
                logger.info(f"  File exists: {file_exists}")
                logger.info(f"  File size: {file_size} bytes")
                logger.info(f"  Current working directory: {os.getcwd()}")
                logger.info(f"  Assigning score of 0.")
                
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(-100)
                reasons.append(f"error opening reference video file: {ref_path} (exists: {file_exists}, size: {file_size})")
                continue
            
            try:
                dist_path, download_time = await download_video(dist_url, request.verbose)
            except Exception as e:
                error_msg = f"Failed to download distorted video from {dist_url}: {str(e)}"
                logger.error(f"{error_msg}. Assigning score of 0.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append("failed to download video file from url")
                continue

            try:
                payload_path, download_time = await download_video(payload_url, request.verbose)
            except Exception as e:
                error_msg = f"Failed to download payload video from {payload_path}: {str(e)}"
                logger.error(f"{error_msg}. Assigning score of 0.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append("failed to download video file from url")
                continue

            step_time = time.time() - uid_start_time

            # Validate video using ffprobe (avoids AV1 warnings)
            if not is_valid_video(dist_path):
                logger.error(f"Error opening distorted video file from {dist_url}. Assigning score of 0.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append("error opening distorted video file")
                if dist_path and os.path.exists(dist_path):
                    os.unlink(dist_path)
                continue

            # === RESOLUTION CHECK ===
            dist_width, dist_height = get_video_dimensions(dist_path)
            logger.info(f"Distorted video resolution: {dist_width}x{dist_height}")
            ref_width, ref_height = get_video_dimensions(payload_path)
            logger.info(f"Reference video resolution: {ref_width}x{ref_height}")
            expected_width = ref_width * scale_factor
            expected_height = ref_height * scale_factor
            
            if dist_width != expected_width or dist_height != expected_height:
                logger.info(f"resolution mismatch: expected {expected_width}x{expected_height}, got {dist_width}x{dist_height}. penalizing miner.")
                reasons.append("MINER FAILURE: incorrect upscaling resolution")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)  # 0 = miner penalty
                continue
            logger.info(f"Resolution check passed: expected {expected_width}x{expected_height}, got {dist_width}x{dist_height}.")

            # Only log success after confirming the file was validated
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 1. Validated reference video and miner output in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            ref_total_frames = get_frame_count(ref_path)
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 2. Retrieved reference video frame count ({ref_total_frames}) in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            if ref_total_frames < 10:
                logger.info(f"Video must contain at least 10 frames. Assigning score of 0.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(-100)
                reasons.append("reference video has fewer than 10 frames")
                continue

            sample_size = min(PIEAPP_SAMPLE_COUNT, ref_total_frames)
            max_start_frame = ref_total_frames - sample_size
            start_frame = 0 if max_start_frame <= 0 else random.randint(0, max_start_frame)

            logger.info(f"Selected frame range for pieapp score {idx+1}: {start_frame} to {start_frame + sample_size - 1}")
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 3. Selected frame range in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            # Open video with OpenCV only when needed for frame extraction
            ref_cap = cv2.VideoCapture(ref_path)
            ref_frames = []
            ref_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(sample_size):
                ret, frame = ref_cap.read()
                if not ret:
                    break
                ref_frames.append(frame)
            ref_cap.release()
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 4. Extracted sampled frames from reference video in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            # Validate encoding settings (MUST be AV1 in proper MP4 with specific params)
            is_valid_encoding, encoding_msg = validate_dist_encoding_settings(dist_path, ref_path, task="upscaling")
            if not is_valid_encoding:
                logger.error(f"Invalid encoding settings for distorted video {dist_path}: {encoding_msg}")
                logger.info(f"  Required: AV1 codec, Main profile, yuv420p, MP4 container, 1:1 SAR")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append(f"invalid encoding settings: {encoding_msg}")
                continue
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 5. Validated encoding settings ({encoding_msg}) in {step_time:.2f} seconds.")

            random_frames = sorted(random.sample(range(ref_total_frames), VMAF_SAMPLE_COUNT))
            logger.info(f"Randomly selected {VMAF_SAMPLE_COUNT} frames for VMAF score: frame list: {random_frames}")
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 6. Selected random frames for VMAF in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            ref_y4m_path = convert_mp4_to_y4m(ref_path, random_frames)
            logger.info("The reference video has been successfully converted to Y4M format.")
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 7. Converted reference video to Y4M in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            if len(dist_url) < 10:
                logger.info(f"Wrong dist download URL: {dist_url}. Assigning score of 0.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append("Invalid download URL: the distorted video download URL must be at least 10 characters long.")
                continue

            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 8. Validated distorted video in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            # Get frame count using ffprobe (no AV1 warnings)
            dist_total_frames = get_frame_count(dist_path)

            logger.info(f"Distorted video has {dist_total_frames} frames.")
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 9. Retrieved distorted video frame count in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            if dist_total_frames != ref_total_frames:
                logger.info(
                    f"Video length mismatch for pair {idx+1}: ref({ref_total_frames}) != dist({dist_total_frames}). Assigning score of 0."
                )
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append("video length mismatch")
                if dist_path and os.path.exists(dist_path):
                    os.unlink(dist_path)
                continue

            # Calculate VMAF
            try:
                vmaf_start = time.time()
                vmaf_score = calculate_vmaf(ref_y4m_path, dist_path, random_frames, neg_model=False)
                vmaf_calc_time = time.time() - vmaf_start
                logger.info(f"☣️☣️ VMAF calculation took {vmaf_calc_time:.2f} seconds.")

                step_time = time.time() - uid_start_time
                logger.info(f"♎️ 10. Completed VMAF calculation in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

                if vmaf_score is not None:
                    vmaf_scores.append(vmaf_score)
                else:
                    vmaf_score = 0.0
                    vmaf_scores.append(vmaf_score)
                logger.info(f"🎾 VMAF score is {vmaf_score}")
            except Exception as e:
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append("failed to calculate VMAF score due to video dimension mismatch")
                logger.error(f"Error calculating VMAF score: {e}")
                continue

            if vmaf_score / 100 < VMAF_THRESHOLD:
                logger.info(f"VMAF score is too low, giving zero score, current VMAF score: {vmaf_score}")
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append(f"VMAF score is too low, current VMAF score: {vmaf_score}")
                continue

            # Open video with OpenCV only when needed for frame extraction
            dist_cap = cv2.VideoCapture(dist_path)
            
            # Extract distorted frames
            dist_frames = []
            dist_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(sample_size):
                ret, frame = dist_cap.read()
                if not ret:
                    break
                dist_frames.append(frame)
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 11. Extracted sampled frames from distorted video in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            # Calculate PieAPP score
            pieapp_score = calculate_pieapp_score_on_samples(ref_frames, dist_frames)
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 12. Calculated PieAPP score in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")
            logger.info(f"🎾 PieAPP score is {pieapp_score}")

            if pieapp_score == -100:
                logger.error(f"Uncertain error in pieapp calculation")
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(-100)
                reasons.append("Uncertain error in pieapp calculation")
                dist_cap.release()
                continue

            pieapp_scores.append(pieapp_score)
            s_q = calculate_quality_score(pieapp_score)
            logger.info(f"🏀 Quality score is {s_q}")

            dist_cap.release()

            s_l = calculate_length_score(content_length)

            s_pre = calculate_preliminary_score(s_q, s_l)

            s_f = calculate_final_score(s_pre)

            quality_scores.append(s_q)
            length_scores.append(s_l)
            final_scores.append(s_f)
            reasons.append("success")

            step_time = time.time() - uid_start_time
            logger.info(f"🛑 Processed one UID in {step_time:.2f} seconds.")

        except Exception as e:
            error_msg = f"Failed to process video from {dist_url}: {str(e)}"
            logger.error(f"{error_msg}. Assigning score of 0.")
            vmaf_scores.append(0.0)
            pieapp_scores.append(0.0)
            quality_scores.append(0.0)
            length_scores.append(0.0)
            final_scores.append(0.0)
            reasons.append("failed to process video")

        finally:
            # Clean up resources for this pair
            # ref_cap is released immediately after use for frame extraction
            if ref_y4m_path and os.path.exists(ref_y4m_path):
                os.unlink(ref_y4m_path)
            if dist_path and os.path.exists(dist_path):
                os.unlink(dist_path)
            if payload_path and os.path.exists(payload_path):
                os.unlink(payload_path)

            # Delete the uploaded object
            storage_client.delete_file(uploaded_object_name)

    for ref_path in request.reference_paths:
        if os.path.exists(ref_path):
            os.unlink(ref_path)

    # tmp_directory = "/tmp"
    # try:
    #     logger.info("🧹 Cleaning up temporary files in /tmp...")
    #     for file_path in glob.glob(os.path.join(tmp_directory, "*.mp4")):
    #         os.remove(file_path)
    #         logger.info(f"Deleted: {file_path}")
    #     for file_path in glob.glob(os.path.join(tmp_directory, "*.y4m")):
    #         os.remove(file_path)
    #         logger.info(f"Deleted: {file_path}")
    # except Exception as e:
    #     logger.error(f"⚠️ Error during cleanup: {e}")

    processed_time = time.time() - start_time
    logger.info(f"Completed batch scoring of {len(request.distorted_urls)} pairs within {processed_time:.2f} seconds")
    logger.info(f"🍉🍉🍉 Calculated final scores: {final_scores} 🍉🍉🍉")
    
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
        try:
            logger.info(f"🧩 Processing pair {idx+1}/{len(request.distorted_urls)}: UID {uid} 🧩")
            
            uid_start_time = time.time()  # Start time for this UID

            
            ref_y4m_path = None
            dist_path = None
            dist_y4m_path = None

            # Validate reference video using ffprobe (avoids AV1 warnings)
            if not is_valid_video(ref_path):
                # Add diagnostic information
                file_exists = os.path.exists(ref_path)
                file_size = os.path.getsize(ref_path) if file_exists else 0
                
                logger.error(f"Error opening reference video file {ref_path}.")
                logger.info(f"  File exists: {file_exists}")
                logger.info(f"  File size: {file_size} bytes")
                logger.info(f"  Current working directory: {os.getcwd()}")
                logger.info(f"  Assigning score of 0.")
                
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)  # No compression achieved
                final_scores.append(-100)
                reasons.append(f"error opening reference video file: {ref_path} (exists: {file_exists}, size: {file_size})")
                continue
            
            # Only log success after confirming the file was validated
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 1. Validated reference video in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            ref_total_frames = get_frame_count(ref_path)
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 2. Retrieved reference video frame count ({ref_total_frames}) in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            if ref_total_frames < 10:
                logger.error(f"Video must contain at least 10 frames. Assigning score of 0.")
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)  # No compression achieved
                final_scores.append(-100)
                reasons.append("reference video has fewer than 10 frames")
                continue

            # Get reference video file size for compression rate calculation
            ref_file_size = os.path.getsize(ref_path)
            logger.info(f"Reference video file size: {ref_file_size} bytes")

            if len(dist_url) < 10:
                logger.error(f"Wrong dist download URL: {dist_url}. Assigning score of 0.")
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
                reasons.append(error_msg)
                continue

            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 3. Downloaded distorted video in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            # Validate video using ffprobe (avoids AV1 warnings)
            if not is_valid_video(dist_path):
                logger.error(f"Error opening distorted video file from {dist_url}. Assigning score of 0.")
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)  # No compression achieved
                final_scores.append(0.0)
                reasons.append("error opening distorted video file")
                if dist_path and os.path.exists(dist_path):
                    os.unlink(dist_path)
                continue

            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 4. Validated distorted video in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            # Validate encoding settings (MUST be AV1 in proper MP4 with specific params)
            is_valid_encoding, encoding_msg = validate_dist_encoding_settings(dist_path, ref_path, task="compression")
            if not is_valid_encoding:
                logger.error(f"Invalid encoding settings for distorted video {dist_path}: {encoding_msg}")
                logger.info(f"  Required: AV1 codec, Main profile, yuv420p, MP4 container, 1:1 SAR")
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)
                final_scores.append(0.0)
                reasons.append(f"invalid encoding settings: {encoding_msg}")
                if dist_path and os.path.exists(dist_path):
                    os.unlink(dist_path)
                continue

            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 4.5. Validated encoding settings ({encoding_msg}) in {step_time:.2f} seconds.")

            dist_total_frames = get_frame_count(dist_path)

            logger.info(f"Distorted video has {dist_total_frames} frames.")
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 5. Retrieved distorted video frame count in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

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

            # Get distorted video file size for compression rate calculation
            dist_file_size = os.path.getsize(dist_path)
            logger.info(f"Distorted video file size: {dist_file_size} bytes")

            # Calculate compression rate BEFORE VMAF calculation
            if ref_file_size > 0 and dist_file_size < ref_file_size:
                compression_rate = dist_file_size / ref_file_size
                logger.info(f"Compression rate: {compression_rate:.4f} ({dist_file_size}/{ref_file_size})")
            else:
                compression_rate = 1.0
                logger.info("Reference file size is 0 or distorted file size >= reference, setting compression rate to 1.0")

            # Sample frames for VMAF calculation
            random_frames = sorted(random.sample(range(ref_total_frames), VMAF_SAMPLE_COUNT))
            logger.info(f"Randomly selected {VMAF_SAMPLE_COUNT} frames for VMAF score: frame list: {random_frames}")
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 6. Selected random frames for VMAF in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            ref_y4m_path = convert_mp4_to_y4m(ref_path, random_frames)
            logger.info("The reference video has been successfully converted to Y4M format.")
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 7. Converted reference video to Y4M in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            # Calculate VMAF and get Y4M paths for validation
            dist_y4m_path = None
            try:
                vmaf_start = time.time()
                vmaf_score, dist_y4m_path = calculate_vmaf(ref_y4m_path, dist_path, random_frames, neg_model=True, return_y4m_path=True)
                vmaf_calc_time = time.time() - vmaf_start
                logger.info(f"☣️☣️ VMAF calculation took {vmaf_calc_time:.2f} seconds.")

                step_time = time.time() - uid_start_time
                logger.info(f"♎️ 8. Completed VMAF calculation in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

                if vmaf_score is not None:
                    vmaf_scores.append(vmaf_score)
                else:
                    vmaf_score = 0.0
                    vmaf_scores.append(vmaf_score)
                logger.info(f"🎾 VMAF score is {vmaf_score} , Threshold: {vmaf_threshold}, Diff: {vmaf_score - vmaf_threshold}")
            except Exception as e:
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)  # No compression achieved
                final_scores.append(0.0)
                reasons.append("failed to calculate VMAF score due to video dimension mismatch")
                logger.error(f"Error calculating VMAF score: {e}")
                if dist_y4m_path and os.path.exists(dist_y4m_path):
                    os.unlink(dist_y4m_path)
                continue
            
            # Now reuse the Y4M files for color validation (no redundant conversion!)
            # Extract frames for color validation from Y4M files
            logger.info(f"Extracting frames from Y4M for color validation (reusing VMAF Y4M files)")
            dist_frames = extract_frames_from_y4m(dist_y4m_path)
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 8.5. Extracted {len(dist_frames)} frames from Y4M for color validation in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            # Validate color channels (grayscale check)
            color_valid, color_reason = validate_color_channels_on_frames(dist_frames)
            if not color_valid:
                logger.error(f"UID {uid}: {color_reason}")
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)
                final_scores.append(0.0)
                reasons.append(f"Color validation failed: {color_reason}")
                if dist_path and os.path.exists(dist_path):
                    os.unlink(dist_path)
                if dist_y4m_path and os.path.exists(dist_y4m_path):
                    os.unlink(dist_y4m_path)
                continue
            
            logger.info(f"✅ UID {uid}: Color channels validated - {color_reason}")
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 8.6. Validated color channels in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            # Extract reference frames from Y4M for chroma quality comparison
            ref_frames = extract_frames_from_y4m(ref_y4m_path)
            
            # Validate chroma quality (prevents partial UV reduction)
            chroma_valid, chroma_reason = validate_chroma_quality_on_frames(
                ref_frames, dist_frames, threshold=0.7
            )
            if not chroma_valid:
                logger.error(f"UID {uid}: {chroma_reason}")
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)
                final_scores.append(0.0)
                reasons.append(f"Chroma validation failed: {chroma_reason}")
                if dist_path and os.path.exists(dist_path):
                    os.unlink(dist_path)
                if dist_y4m_path and os.path.exists(dist_y4m_path):
                    os.unlink(dist_y4m_path)
                continue
            
            logger.info(f"✅ UID {uid}: Chroma quality validated - {chroma_reason}")
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 8.7. Validated chroma quality in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            # Calculate compression score using the proper formula
            # Check scoring function for details

            final_score, compression_component, quality_component, reason = calculate_compression_score(
                vmaf_score=vmaf_score,
                compression_rate=compression_rate,
                vmaf_threshold=vmaf_threshold,
                compression_weight=COMPRESSION_RATE_WEIGHT,
                quality_weight=COMPRESSION_VMAF_WEIGHT,
                soft_threshold_margin=SOFT_THRESHOLD_MARGIN
            )
           

            #  quality component: 
            logger.info(f"VMAF quality component: {quality_component:.4f}")

            # compression component
            logger.info(f"🎯 Compression score is {compression_component:.4f}")

            # final score
            logger.info(f"🎯 Final score is {final_score:.4f}")
            compression_rates.append(compression_rate)
            final_scores.append(final_score)
            reasons.append(reason)

            step_time = time.time() - uid_start_time
            logger.info(f"🛑 Processed one UID in {step_time:.2f} seconds.")

        except Exception as e:
            error_msg = f"Failed to process video from {dist_url}: {str(e)}"
            logger.error(f"{error_msg}. Assigning score of 0.")
            vmaf_scores.append(0.0)
            compression_rates.append(1.0)  # No compression achieved
            final_scores.append(0.0)
            reasons.append("failed to process video")

        finally:
            # Clean up resources for this pair
            # ref_cap is not used in this endpoint (using ffprobe instead)
            if ref_y4m_path and os.path.exists(ref_y4m_path):
                os.unlink(ref_y4m_path)
            if dist_y4m_path and os.path.exists(dist_y4m_path):
                os.unlink(dist_y4m_path)
            if dist_path and os.path.exists(dist_path):
                os.unlink(dist_path)

            # Delete the uploaded object
            storage_client.delete_file(uploaded_object_name)
            
    for ref_path in request.reference_paths:
        if os.path.exists(ref_path):
            os.unlink(ref_path)

    # tmp_directory = "/tmp"
    # try:
    #     logger.info("🧹 Cleaning up temporary files in /tmp...")
    #     for file_path in glob.glob(os.path.join(tmp_directory, "*.mp4")):
    #         os.remove(file_path)
    #         logger.info(f"Deleted: {file_path}")
    #     for file_path in glob.glob(os.path.join(tmp_directory, "*.y4m")):
    #         os.remove(file_path)
    #         logger.info(f"Deleted: {file_path}")
    # except Exception as e:
    #     logger.error(f"⚠️ Error during cleanup: {e}")

    processed_time = time.time() - start_time
    logger.info(f"Completed batch scoring of {len(request.distorted_urls)} pairs within {processed_time:.2f} seconds")
    logger.info(f"🍉🍉🍉 Calculated final scores: {final_scores} 🍉🍉🍉")
    
    return CompressionScoringResponse(
        vmaf_scores=vmaf_scores,
        compression_rates=compression_rates,
        final_scores=final_scores,
        reasons=reasons
    )

@app.post("/score_organics_upscaling")
async def score_organics_upscaling(request: OrganicsUpscalingScoringRequest) -> OrganicsUpscalingScoringResponse:
    logger.info("#################### 🤖 start scoring ####################")
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
            logger.error(f"failed to download distorted video: {dist_url}, error: {e}")
            distorted_video_paths.append(None)

    for idx, (ref_url, dist_path, uid, task_type) in enumerate(
        zip(request.reference_urls, distorted_video_paths, request.uids, request.task_types)
    ):
        logger.info(f"🧩 processing {uid}.... downloading reference video.... 🧩")
        ref_path = None
        
        ref_upscaled_y4m_path = None
        ref_clip_path = None
        dist_clip_path = None
        ref_upscaled_clip_path = None

        scale_factor = 2
        if task_type == "SD24K":
            scale_factor = 4

        logger.info(f"scale factor: {scale_factor}")

        # === REFERENCE VIDEO VALIDATION (NOT MINER'S FAULT) ===
        try:
            # download reference video
            if len(ref_url) < 10:
                logger.info(f"invalid reference download url: {ref_url}. skipping scoring")
                reasons.append("invalid reference url - skipped")
                vmaf_scores.append(-1)
                pieapp_scores.append(-1)
                quality_scores.append(-1)
                length_scores.append(-1)
                final_scores.append(-1)  # -1 means skip, don't penalize
                continue

            ref_path, download_time = await download_video(ref_url, request.verbose)
            
            # Validate reference video using ffprobe (avoids AV1 warnings)
            if not is_valid_video(ref_path):
                file_exists = os.path.exists(ref_path)
                file_size = os.path.getsize(ref_path) if file_exists else 0
                
                logger.error(f"error opening reference video from {ref_url}. skipping scoring")
                logger.info(f"  File path: {ref_path}")
                logger.info(f"  File exists: {file_exists}")
                logger.info(f"  File size: {file_size} bytes")
                
                reasons.append("corrupted reference video - skipped")
                vmaf_scores.append(-1)
                pieapp_scores.append(-1)
                quality_scores.append(-1)
                length_scores.append(-1)
                final_scores.append(-1)  # -1 means skip, don't penalize
                continue
            
            # Get video metadata using ffprobe (no AV1 warnings)
            ref_total_frames = get_frame_count(ref_path)
            ref_fps = get_video_fps(ref_path)
            ref_width, ref_height = get_video_dimensions(ref_path)
            
            if ref_fps is None or ref_width is None or ref_height is None:
                logger.error(f"failed to get reference video metadata. skipping scoring")
                reasons.append("failed to read reference video metadata - skipped")
                vmaf_scores.append(-1)
                pieapp_scores.append(-1)
                quality_scores.append(-1)
                length_scores.append(-1)
                final_scores.append(-1)
                continue

            if ref_total_frames < 10:
                logger.info(f"reference video too short (<10 frames). skipping scoring")
                reasons.append("reference video too short - skipped")
                vmaf_scores.append(-1)
                pieapp_scores.append(-1)
                quality_scores.append(-1)
                length_scores.append(-1)
                final_scores.append(-1)  # -1 means skip, don't penalize
                continue

        except Exception as e:
            logger.error(f"system error processing reference video: {e}. skipping scoring")
            reasons.append("system error with reference - skipped")
            vmaf_scores.append(-1)
            pieapp_scores.append(-1)
            quality_scores.append(-1)
            length_scores.append(-1)
            final_scores.append(-1)  # -1 means skip, don't penalize
            continue

        # === MINER OUTPUT VALIDATION (MINER'S FAULT) ===
        try:
            # check if distorted video failed to download
            if dist_path is None:
                logger.error(f"failed to download distorted video for uid {uid}. penalizing miner.")
                reasons.append("MINER FAILURE: failed to download distorted video, invalid url")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)  # 0 = miner penalty
                continue

            # Check if miner's output can be opened (using ffprobe to avoid AV1 warnings)
            if not is_valid_video(dist_path):
                logger.error(f"error opening distorted video from {dist_path}. penalizing miner.")
                reasons.append("MINER FAILURE: corrupted output video")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)  # 0 = miner penalty
                continue

            is_valid_encoding, encoding_msg = validate_dist_encoding_settings(dist_path, ref_path, task="upscaling")
            if not is_valid_encoding:
                logger.error(f"Invalid encoding settings for distorted video {dist_path}: {encoding_msg}")
                logger.info(f"  Required: AV1 codec, Main profile, yuv420p, MP4 container, 1:1 SAR")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append(f"invalid encoding settings: {encoding_msg}")
                continue
            
            # Get frame count and dimensions using ffprobe (no AV1 warnings)
            dist_total_frames = get_frame_count(dist_path)
            dist_width, dist_height = get_video_dimensions(dist_path)
            
            if dist_width is None or dist_height is None:
                logger.error(f"failed to get video dimensions for {dist_path}. penalizing miner.")
                reasons.append("MINER FAILURE: cannot read video dimensions")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                continue
                
            logger.info(f"distorted video has {dist_total_frames} frames.")

            # Check frame count match
            if dist_total_frames != ref_total_frames:
                logger.info(f"video length mismatch: ref({ref_total_frames}) != dist({dist_total_frames}). penalizing miner.")
                reasons.append("MINER FAILURE: video length mismatch")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)  # 0 = miner penalty
                continue

            # Check minimum frame count for miner output
            if dist_total_frames < 10:
                logger.info(f"miner output too short (<10 frames). penalizing miner.")
                reasons.append("MINER FAILURE: output video too short")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)  # 0 = miner penalty
                continue

            # === RESOLUTION CHECK ===
            expected_width = ref_width * scale_factor
            expected_height = ref_height * scale_factor
            
            if dist_width != expected_width or dist_height != expected_height:
                logger.info(f"resolution mismatch: expected {expected_width}x{expected_height}, got {dist_width}x{dist_height}. penalizing miner.")
                reasons.append("MINER FAILURE: incorrect upscaling resolution")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)  # 0 = miner penalty
                continue

        except Exception as e:
            logger.error(f"error validating miner output: {e}. penalizing miner.")
            reasons.append(f"MINER FAILURE: {e}")
            vmaf_scores.append(0.0)
            pieapp_scores.append(0.0)
            quality_scores.append(0.0)
            length_scores.append(0.0)
            final_scores.append(0.0)  # 0 = miner penalty
            continue

        # === MAIN SCORING LOGIC ===
        try:
            # Calculate 0.5 second clip duration
            clip_duration = 0.25  # seconds
            
            # Calculate maximum start time to ensure we don't exceed video length
            ref_duration = ref_total_frames / ref_fps
            max_start_time = max(0, ref_duration - clip_duration)
            start_time = random.uniform(0, max_start_time)
            
            logger.info(f"Creating 0.5-second clips starting from {start_time:.2f} seconds")
            
            # Create 0.5-second reference clip
            ref_clip_path = trim_video(ref_path, start_time, clip_duration)
            logger.info(f"Created reference clip: {ref_clip_path}")
            
            # Create 0.5-second distorted clip
            dist_clip_path = trim_video(dist_path, start_time, clip_duration)
            logger.info(f"Created distorted clip: {dist_clip_path}")
            
            # Upscale the reference clip
            ref_upscaled_clip_path = upscale_video(ref_clip_path, scale_factor)
            logger.info(f"Created upscaled reference clip: {ref_upscaled_clip_path}")
            
            # Calculate ClipIQ scores
            logger.info("Calculating ClipIQ scores...")
            ref_clipiqa_score = get_clipiqa_score(ref_clip_path, num_frames=3)
            dist_clipiqa_score = get_clipiqa_score(dist_clip_path, num_frames=3)
            ref_upscaled_clipiqa_score = get_clipiqa_score(ref_upscaled_clip_path, num_frames=3)
            
            logger.info(f"Reference ClipIQ score: {ref_clipiqa_score:.4f}")
            logger.info(f"Distorted ClipIQ score: {dist_clipiqa_score:.4f}")
            logger.info(f"Upscaled reference ClipIQ score: {ref_upscaled_clipiqa_score:.4f}")
            
            # Create Y4M files for VMAF calculation
            logger.info("Creating Y4M files for VMAF calculation...")
            # Get the number of frames in the clips for Y4M conversion using ffprobe
            ref_clip_frames = get_frame_count(ref_upscaled_clip_path)
            
            ref_y4m_path = convert_mp4_to_y4m(ref_upscaled_clip_path, list(range(ref_clip_frames)))
            
            # Calculate VMAF score
            logger.info("Calculating VMAF score...")
            vmaf_score = calculate_vmaf(ref_y4m_path, dist_clip_path, list(range(ref_clip_frames)), neg_model=False)
            
            if vmaf_score is None:
                vmaf_score = 0.0
            
            logger.info(f"VMAF score: {vmaf_score:.4f}")
            
            # Check VMAF threshold
            if vmaf_score < 50:
                logger.info(f"VMAF score below threshold (50): {vmaf_score}. Assigning score 0.")
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
                reason = "upscaled better than reference"
            elif dist_clipiqa_score > (ref_upscaled_clipiqa_score * 1.05):
                final_score = 2
                reason = "upscaled at least 5% better than ffmpeg upscaled"
            elif dist_clipiqa_score > (ref_upscaled_clipiqa_score):
                final_score = 1
                reason = "Better than ffmpeg upscaled"
            else:
                final_score = 0
                reason = "Worse than ffmpeg upscaled"
            
            logger.info(f"Final score: {final_score} ({reason})")
            
            vmaf_scores.append(vmaf_score)
            pieapp_scores.append(0.0)  
            quality_scores.append(0.0)  
            length_scores.append(0.0)
            final_scores.append(final_score)
            reasons.append(f"{reason}")

        except Exception as e:
            logger.error(f"error in main scoring logic: {e}. penalizing miner.")
            reasons.append(f"MINER FAILURE: scoring error - {e}")
            vmaf_scores.append(0.0)
            pieapp_scores.append(0.0)
            quality_scores.append(0.0)
            length_scores.append(0.0)
            final_scores.append(0.0)  # 0 = miner penalty

        finally:
            # Clean up resources
            
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
    logger.info(f"🍉🍉🍉 calculated scores: {final_scores} 🍉🍉🍉")
    logger.info(f"completed one batch scoring within {processed_time:.2f} seconds")
    
    # Summary statistics
    successful_miners = sum(1 for score in final_scores if score >= 1)
    failed_miners = sum(1 for score in final_scores if score == 0.0)
    skipped_miners = sum(1 for score in final_scores if score == -1)
    
    logger.info(f"📊 SCORING SUMMARY:")
    logger.info(f"  ✅ Successful miners: {successful_miners}")
    logger.info(f"  ❌ Failed miners: {failed_miners}")
    logger.info(f"  ⏭️ Skipped miners: {skipped_miners}")
    
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
    logger.info("#################### 🤖 start scoring ####################")
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
            logger.error(f"failed to download distorted video: {dist_url}, error: {e}")
            distorted_video_paths.append(None)

    for idx, (ref_url, dist_path, uid, vmaf_threshold) in enumerate(
        zip(request.reference_urls, distorted_video_paths, request.uids, request.vmaf_thresholds)
    ):
        logger.info(f"🧩 processing {uid}.... downloading reference video.... 🧩")
        ref_path = None
        
        ref_y4m_path = None
        dist_y4m_path = None
        ref_clip_path = None
        dist_clip_path = None

        uid_start_time = time.time() # start time for each uid

        # === REFERENCE VIDEO VALIDATION (NOT MINER'S FAULT) ===
        try:
            # download reference video
            if len(ref_url) < 10:
                logger.info(f"invalid reference download url: {ref_url}. skipping scoring")
                vmaf_scores.append(-1)
                compression_rates.append(-1)
                reasons.append("invalid reference url - skipped")
                final_scores.append(-1)  # -1 means skip, don't penalize
                continue

            ref_path, download_time = await download_video(ref_url, request.verbose)
            
            # Validate reference video using ffprobe (avoids AV1 warnings)
            if not is_valid_video(ref_path):
                file_exists = os.path.exists(ref_path)
                file_size = os.path.getsize(ref_path) if file_exists else 0
                
                logger.error(f"error opening reference video from {ref_url}. skipping scoring")
                logger.info(f"  File path: {ref_path}")
                logger.info(f"  File exists: {file_exists}")
                logger.info(f"  File size: {file_size} bytes")
                
                vmaf_scores.append(-1)
                compression_rates.append(-1)
                reasons.append("corrupted reference video - skipped")
                final_scores.append(-1)  # -1 means skip, don't penalize
                continue

            ref_total_frames = get_frame_count(ref_path)

            # Get reference video file size for compression rate calculation
            ref_file_size = os.path.getsize(ref_path)
            logger.info(f"Reference video file size: {ref_file_size} bytes")

            if ref_total_frames < 10:
                logger.info(f"reference video too short (<10 frames). skipping scoring")
                vmaf_scores.append(-1)
                compression_rates.append(-1)
                reasons.append("reference video too short - skipped")
                final_scores.append(-1)  # -1 means skip, don't penalize
                continue

        except Exception as e:
            logger.error(f"system error processing reference video: {e}. skipping scoring")
            vmaf_scores.append(-1)
            compression_rates.append(-1)
            reasons.append("system error with reference - skipped")
            final_scores.append(-1)  # -1 means skip, don't penalize
            continue

        # === MINER OUTPUT VALIDATION (MINER'S FAULT) ===
        try:
            # check if distorted video failed to download
            if dist_path is None:
                logger.error(f"failed to download distorted video for uid {uid}. penalizing miner.")
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)
                reasons.append("MINER FAILURE: failed to download distorted video, invalid url")
                final_scores.append(0.0)  # 0 = miner penalty
                continue

            # Check if miner's output can be opened (using ffprobe to avoid AV1 warnings)
            if not is_valid_video(dist_path):
                logger.error(f"error opening distorted video from {dist_path}. penalizing miner.")
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)
                reasons.append("MINER FAILURE: corrupted output video")
                final_scores.append(0.0)  # 0 = miner penalty
                continue

            is_valid_encoding, encoding_msg = validate_dist_encoding_settings(dist_path, ref_path, task="compression")
            if not is_valid_encoding:
                logger.error(f"Invalid encoding settings for distorted video {dist_path}: {encoding_msg}")
                logger.info(f"  Required: AV1 codec, Main profile, yuv420p, MP4 container, 1:1 SAR")
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)
                final_scores.append(0.0)
                reasons.append(f"invalid encoding settings: {encoding_msg}")
                if dist_path and os.path.exists(dist_path):
                    os.unlink(dist_path)
                continue

            dist_total_frames = get_frame_count(dist_path)
            logger.info(f"Distorted video has {dist_total_frames} frames.")
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 5. Retrieved distorted video frame count in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

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

            # Get distorted video file size for compression rate calculation
            dist_file_size = os.path.getsize(dist_path)
            logger.info(f"Distorted video file size: {dist_file_size} bytes")

            # Calculate compression rate BEFORE VMAF calculation
            if ref_file_size > 0 and dist_file_size < ref_file_size:
                compression_rate = dist_file_size / ref_file_size
                logger.info(f"Compression rate: {compression_rate:.4f} ({dist_file_size}/{ref_file_size})")
            else:
                compression_rate = 1.0
                logger.info("Reference file size is 0 or distorted file size >= reference, setting compression rate to 1.0")

            # Calculate 0.5 second clip duration for VMAF calculation
            clip_duration = 0.5  # seconds
            
            # Get video FPS to calculate frame-based timing using ffprobe (no AV1 warnings)
            ref_fps = get_video_fps(ref_path)
            logger.info(f"Reference video FPS: {ref_fps}")
            
            # Calculate maximum start time to ensure we don't exceed video length
            ref_duration = ref_total_frames / ref_fps
            max_start_time = max(0, ref_duration - clip_duration)
            start_time = random.uniform(0, max_start_time)
            
            logger.info(f"Creating 0.5-second clips starting from {start_time:.2f} seconds")
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 6. Selected random start time for video chunks in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            # Create 0.5-second reference clip
            ref_clip_path = trim_video(ref_path, start_time, clip_duration)
            logger.info(f"Created reference clip: {ref_clip_path}")
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 7. Created reference video clip in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            # Create 0.5-second distorted clip
            dist_clip_path = trim_video(dist_path, start_time, clip_duration)
            logger.info(f"Created distorted clip: {dist_clip_path}")
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 8. Created distorted video clip in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            # Get the number of frames in the clips for Y4M conversion
            
            ref_clip_frames = get_frame_count(ref_clip_path)
            logger.info(f"Reference clip has {ref_clip_frames} frames.")
            
            
            # Ensure we don't sample more frames than available
            sample_count = min(VMAF_SAMPLE_COUNT, ref_clip_frames)
            random_frames = sorted(random.sample(range(ref_clip_frames), sample_count))
            ref_y4m_path = convert_mp4_to_y4m(ref_clip_path, random_frames)
            logger.info("The reference video clip has been successfully converted to Y4M format.")
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 9. Converted reference video clip to Y4M in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            # Calculate VMAF on chunked videos and get Y4M paths for validation
            dist_y4m_path = None
            try:
                vmaf_start = time.time()
                vmaf_score, dist_y4m_path = calculate_vmaf(ref_y4m_path, dist_clip_path, random_frames, neg_model=True, return_y4m_path=True)
                vmaf_calc_time = time.time() - vmaf_start
                logger.info(f"☣️☣️ VMAF calculation took {vmaf_calc_time:.2f} seconds.")

                step_time = time.time() - uid_start_time
                logger.info(f"♎️ 10. Completed VMAF calculation in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

                if vmaf_score is not None:
                    vmaf_scores.append(vmaf_score)
                else:
                    vmaf_score = 0.0
                    vmaf_scores.append(vmaf_score)
                logger.info(f"🎾 VMAF score is {vmaf_score}")
            except Exception as e:
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)  # No compression achieved
                final_scores.append(0.0)
                reasons.append("failed to calculate VMAF score due to video dimension mismatch")
                logger.error(f"Error calculating VMAF score: {e}")
                if dist_y4m_path and os.path.exists(dist_y4m_path):
                    os.unlink(dist_y4m_path)
                continue
            
            # Now reuse the Y4M files for color validation (no redundant conversion!)
            # Extract frames for color validation from Y4M files
            logger.info(f"Extracting frames from Y4M for color validation from clip (reusing VMAF Y4M files)")
            dist_clip_frames = extract_frames_from_y4m(dist_y4m_path)
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 10.5. Extracted {len(dist_clip_frames)} frames from Y4M for color validation in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            # Validate color channels (grayscale check)
            color_valid, color_reason = validate_color_channels_on_frames(dist_clip_frames)
            if not color_valid:
                logger.error(f"UID {uid}: {color_reason}")
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)
                final_scores.append(0.0)
                reasons.append(f"Color validation failed: {color_reason}")
                if dist_path and os.path.exists(dist_path):
                    os.unlink(dist_path)
                if ref_clip_path and os.path.exists(ref_clip_path):
                    os.unlink(ref_clip_path)
                if dist_clip_path and os.path.exists(dist_clip_path):
                    os.unlink(dist_clip_path)
                if dist_y4m_path and os.path.exists(dist_y4m_path):
                    os.unlink(dist_y4m_path)
                continue
            
            logger.info(f"✅ UID {uid}: Color channels validated - {color_reason}")
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 10.6. Validated color channels in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            # Extract reference frames from Y4M for chroma quality comparison
            ref_clip_frames_data = extract_frames_from_y4m(ref_y4m_path)
            
            # Validate chroma quality (prevents partial UV reduction)
            chroma_valid, chroma_reason = validate_chroma_quality_on_frames(
                ref_clip_frames_data, dist_clip_frames, threshold=0.7
            )
            if not chroma_valid:
                logger.error(f"UID {uid}: {chroma_reason}")
                vmaf_scores.append(0.0)
                compression_rates.append(1.0)
                final_scores.append(0.0)
                reasons.append(f"Chroma validation failed: {chroma_reason}")
                if dist_path and os.path.exists(dist_path):
                    os.unlink(dist_path)
                if ref_clip_path and os.path.exists(ref_clip_path):
                    os.unlink(ref_clip_path)
                if dist_clip_path and os.path.exists(dist_clip_path):
                    os.unlink(dist_clip_path)
                if dist_y4m_path and os.path.exists(dist_y4m_path):
                    os.unlink(dist_y4m_path)
                continue
            
            logger.info(f"✅ UID {uid}: Chroma quality validated - {chroma_reason}")
            step_time = time.time() - uid_start_time
            logger.info(f"♎️ 10.7. Validated chroma quality in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            # Calculate compression score using the proper formula
            # Check scoring function for details

            final_score, compression_component, quality_component, reason = calculate_compression_score(
                vmaf_score=vmaf_score,
                compression_rate=compression_rate,
                vmaf_threshold=vmaf_threshold,
                compression_weight=COMPRESSION_RATE_WEIGHT,
                quality_weight=COMPRESSION_VMAF_WEIGHT,
                soft_threshold_margin=SOFT_THRESHOLD_MARGIN
            )


            #  quality component: 
            logger.info(f"VMAF quality component: {quality_component:.4f}")

            # compression component
            logger.info(f"🎯 Compression score is {compression_component:.4f}")

            # final score
            logger.info(f"🎯 Final score is {final_score:.4f}")
            # vmaf_scores.append(vmaf_score)
            compression_rates.append(compression_rate)
            final_scores.append(final_score)
            reasons.append(reason)

            step_time = time.time() - uid_start_time
            logger.info(f"🛑 Processed one UID in {step_time:.2f} seconds.")

        except Exception as e:
            error_msg = f"Failed to process video from {dist_url}: {str(e)}"
            logger.error(f"{error_msg}. Assigning score of 0.")
            vmaf_scores.append(0.0)
            compression_rates.append(1.0)  # No compression achieved
            final_scores.append(0.0)
            reasons.append("failed to process video")

        finally:
            # Clean up resources for this pair
            # ref_cap is not used in this endpoint (using ffprobe instead)
            # dist_cap is not used in this endpoint (using ffprobe instead)
            if ref_path and os.path.exists(ref_path):
                os.unlink(ref_path)
            if dist_path and os.path.exists(dist_path):
                os.unlink(dist_path)
            if ref_y4m_path and os.path.exists(ref_y4m_path):
                os.unlink(ref_y4m_path)
            if dist_y4m_path and os.path.exists(dist_y4m_path):
                os.unlink(dist_y4m_path)
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
