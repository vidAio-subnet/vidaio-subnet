import os
import cv2
import math
import glob
import time
import uuid
import redis
import random
import requests
import tempfile
import threading
import subprocess
import numpy as np
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
from dotenv import load_dotenv
from typing import List, Tuple, Optional
from services.video_scheduler.redis_utils import pop_pexels_video_id
from vidaio_subnet_core import CONFIG

load_dotenv()

MAX_WORKERS = 5  

# Comprehensive video transformations for content variation
# These transformations are designed to reduce video recognizability while maintaining quality
VIDEO_TRANSFORMATIONS = [
    # === COLOR SPACE TRANSFORMATIONS ===
    # Hue shifts
    "hue=h=15:s=1.1",
    "hue=h=-10:s=1.05", 
    "hue=h=25:s=0.95",
    "hue=h=-18:s=1.08",
    "hue=h=20:s=0.92",
    
    # Color balance adjustments
    "colorbalance=rs=0.1:gs=-0.05:bs=0.05:rm=0.05:gm=0.1:bm=-0.05",
    "colorbalance=rs=-0.08:gs=0.1:bs=-0.02:rm=-0.05:gm=0.05:bm=0.1",
    "colorbalance=rs=0.05:gs=-0.03:bs=0.08:rm=-0.02:gm=0.08:bm=-0.06",
    "colorbalance=rs=-0.06:gs=0.08:bs=-0.04:rm=0.07:gm=-0.03:bm=0.09",
    
    # Curves adjustments (gamma and contrast changes)
    "curves=r='0/0 0.5/0.48 1/1':g='0/0 0.5/0.52 1/1':b='0/0 0.5/0.50 1/1'",
    "curves=r='0/0 0.5/0.52 1/1':g='0/0 0.5/0.50 1/1':b='0/0 0.5/0.48 1/1'",
    "curves=r='0/0 0.5/0.49 1/1':g='0/0 0.5/0.51 1/1':b='0/0 0.5/0.52 1/1'",
    "curves=r='0/0 0.47/0.51 0.53/0.49 1/1':g='0/0 0.5/0.49 1/1'",
    "curves=r='0/0 0.47/0.51 0.53/0.49 1/1':b='0/0 0.5/0.49 1/1'",
    "curves=g='0/0 0.47/0.51 0.53/0.49 1/1':r='0/0 0.5/0.49 1/1'",
    "curves=g='0/0 0.47/0.51 0.53/0.49 1/1':b='0/0 0.5/0.49 1/1'",
    "curves=b='0/0 0.47/0.51 0.53/0.49 1/1':r='0/0 0.5/0.49 1/1'",
    "curves=b='0/0 0.47/0.51 0.53/0.49 1/1':g='0/0 0.5/0.49 1/1'",
    
    # Exposure and vibrance
    "eq=brightness=0.02:contrast=1.05:saturation=1.1",
    "eq=brightness=-0.015:contrast=1.03:saturation=0.95",
    "eq=brightness=0.01:contrast=0.98:saturation=1.08",
    "eq=brightness=-0.01:contrast=1.07:saturation=1.02",
    
    # Temperature shifts
    "colortemperature=temperature=5800:mix=0.3",
    "colortemperature=temperature=6500:mix=0.25",
    "colortemperature=temperature=5200:mix=0.35",
    "colortemperature=temperature=6200:mix=0.28",
    
    # === CONVOLUTION FILTERS ===
    # Subtle blur effects
    "gblur=sigma=0.3",
    "gblur=sigma=0.5",
    "boxblur=2:1",
    
    # Subtle sharpen effects
    "unsharp=5:5:0.3:3:3:0.3",
    "unsharp=3:3:0.5:3:3:0.5",
    
    # Edge enhancement (very subtle)
    "convolution='0 -1 0 -1 5 -1 0 -1 0':0.1",
    "convolution='-1 -1 -1 -1 9 -1 -1 -1 -1':0.08",
    "convolution='0 -1 0 -1 5 -1 0 -1 0'",      # sharpen
    "convolution='1 1 1 1 1 1 1 1 1'",          # box blur
    "convolution='1 1 1 1 2 1 1 1 1'",          # a slight twist on the above
    "convolution='1 2 1 2 4 2 1 2 1'",          # Gaussian blur
    
    # === NOISE AND TEXTURE EFFECTS ===
    # Subtle noise (film grain effect)
    "noise=alls=2:allf=t",
    "noise=alls=3:allf=t+u",
    
    # === DISTORTION EFFECTS (VERY SUBTLE) ===
    # Subtle barrel distortion
    "lenscorrection=k1=-0.001:k2=0.0001",
    "lenscorrection=k1=0.001:k2=-0.0001",
    
    # === GAMMA AND LEVELS ===
    # Gamma adjustments
    "lutyuv=y=gammaval(0.9)",
    "lutyuv=y=gammaval(1.1)",
    "lutyuv=y=gammaval(0.95)",
    
    # === COMBINATION TRANSFORMATIONS ===
    # Multiple effects combined for more variation
    "hue=h=12:s=1.02,eq=brightness=0.01:contrast=1.02:saturation=1.05",
    "colorbalance=rs=0.05:bs=-0.03:gm=0.08,eq=brightness=-0.01:contrast=1.01",
    "curves=r='0/0 0.5/0.51 1/1':g='0/0 0.5/0.49 1/1',hue=h=-8:s=1.03",
    "gblur=sigma=0.2,unsharp=3:3:0.3:3:3:0.3",
    "colortemperature=temperature=5900:mix=0.3,eq=brightness=0.005:contrast=1.02",
    "hue=h=-12:s=1.04,colorbalance=rs=-0.03:gs=0.05:bs=-0.02",
    "noise=alls=1:allf=t,eq=brightness=0.008:contrast=1.01",
    "lenscorrection=k1=0.0008,colortemperature=temperature=6100:mix=0.25",
    "lutyuv=y=gammaval(0.98),hue=h=8:s=1.01",
    "convolution='0 -1 0 -1 5 -1 0 -1 0':0.05,eq=saturation=1.03",
    
    # === ADVANCED COMBINATIONS ===
    # Triple combinations for maximum variation
    "hue=h=10:s=1.03,gblur=sigma=0.25,eq=brightness=0.005:contrast=1.015",
    "colorbalance=rs=0.04:bs=-0.02,noise=alls=1.5:allf=t,colortemperature=temperature=5700:mix=0.2",
    "curves=r='0/0 0.5/0.505 1/1':g='0/0 0.5/0.495 1/1',unsharp=4:4:0.25:4:4:0.25,hue=h=-5:s=1.02",
]

def rand_between(a, b):
    return random.uniform(a, b)

def rand_color():
    c = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)], dtype=np.float32)
    d = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)], dtype=np.float32)
    if np.linalg.norm(c - d) < 90:  # push apart if too close
        d = np.clip(c + np.sign(d - c) * 100, 0, 255)
    return c, d

def make_linear_gradient(h, w, c0, c1):
    theta = rand_between(0, 2 * np.pi)
    ux, uy = np.cos(theta), np.sin(theta)

    # coordinate grids
    y = np.linspace(0, h - 1, h, dtype=np.float32)
    x = np.linspace(0, w - 1, w, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    # random offset so the gradient "start" is anywhere
    # scale offset by span along the gradient to actually shift visibly
    span = abs(ux) * (w - 1) + abs(uy) * (h - 1)
    offset = rand_between(-0.5, 0.5) * span

    proj = X * ux + Y * uy + offset
    # normalize to 0..1 within this rect
    pmin, pmax = proj.min(), proj.max()
    t = (proj - pmin) / max(1e-6, (pmax - pmin))
    t = t[..., None]  # (h,w,1)

    grad = (1.0 - t) * c0 + t * c1  # broadcast to 3 channels
    return np.clip(grad, 0, 255).astype(np.uint8)

def _make_multiple_of_8(n: int) -> int:
    return int(np.ceil(n / 8.0) * 8)

def preprocess_video(inp, outp, min_crop=0.05, max_crop=0.1,
                     min_banner=0.08, max_banner=0.15,
                     min_scale=0.9, max_scale=1.1,
                     codec="mp4v", preset_note=True, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    cap = cv2.VideoCapture(inp)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {inp}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:  # fallback
        fps = 30.0
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Random small crop (each side independently, bounded) ---
    max_cx = int(in_w * rand_between(min_crop, max_crop))
    max_cy = int(in_h * rand_between(min_crop, max_crop))
    l = random.randint(0, max(0, max_cx))
    r = random.randint(0, max(0, max_cx))
    t_ = random.randint(0, max(0, max_cy))
    b = random.randint(0, max(0, max_cy))

    crop_x0 = l
    crop_y0 = t_
    crop_x1 = max(crop_x0 + 2, in_w - r)  # ensure >=2 px
    crop_y1 = max(crop_y0 + 2, in_h - b)

    base_w = crop_x1 - crop_x0
    base_h = crop_y1 - crop_y0

    # --- Slight rescale after crop ---
    scale = rand_between(min_scale, max_scale)
    scaled_w = max(2, int(round(base_w * scale)))
    scaled_h = max(2, int(round(base_h * scale)))

    # Force to multiple of 8
    scaled_w = _make_multiple_of_8(scaled_w)
    scaled_h = _make_multiple_of_8(scaled_h)

    # --- Random banners: one vertical side + one horizontal side ---
    v_side = random.choice(["left", "right"])
    h_side = random.choice(["top", "bottom"])
    v_ratio = rand_between(min_banner, max_banner)
    h_ratio = rand_between(min_banner, max_banner)
    v_w = max(1, int(round(scaled_w * v_ratio)))
    h_h = max(1, int(round(scaled_h * h_ratio)))

    # Force to multiple of 8
    v_w = _make_multiple_of_8(v_w)
    h_h = _make_multiple_of_8(h_h)

    out_w = scaled_w + v_w
    out_h = scaled_h + h_h

    # Force to multiple of 8 (defensive)
    out_w = _make_multiple_of_8(out_w)
    out_h = _make_multiple_of_8(out_h)

    # Pre-generate gradient banners with random color & direction
    c0_v, c1_v = rand_color()
    c0_h, c1_h = rand_color()
    vert_banner = make_linear_gradient(out_h, v_w, c0_v, c1_v)   # full height, v_w width
    horiz_banner = make_linear_gradient(h_h, out_w, c0_h, c1_h)  # h_h height, full width

    # --- Writer ---
    fourcc = cv2.VideoWriter_fourcc(*codec)
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    writer = cv2.VideoWriter(outp, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter. Try different --codec (e.g., avc1, H264, XVID)")

    # Log the transform so you can reproduce/debug
    print({
        "input": (in_w, in_h, fps),
        "crop_rect": (crop_x0, crop_y0, crop_x1, crop_y1),
        "scale": round(scale, 4),
        "scaled": (scaled_w, scaled_h),
        "vertical_banner": {"side": v_side, "width_px": v_w, "colors": (c0_v.tolist(), c1_v.tolist())},
        "horizontal_banner": {"side": h_side, "height_px": h_h, "colors": (c0_h.tolist(), c1_h.tolist())},
        "output": (out_w, out_h),
        "seed": seed
    })

    # Processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # crop
        frame = frame[crop_y0:crop_y1, crop_x0:crop_x1]

        # rescale
        frame = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

        # start with black canvas
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        # place vertical banner
        if v_side == "left":
            canvas[:, :v_w] = vert_banner
            x0 = v_w
        else:
            canvas[:, -v_w:] = vert_banner
            x0 = 0

        # place horizontal banner (must account for vertical banner already placed)
        if h_side == "top":
            canvas[:h_h, :] = horiz_banner
            y0 = h_h
        else:
            canvas[-h_h:, :] = horiz_banner
            y0 = 0

        # place frame
        canvas[y0:y0 + scaled_h, x0:x0 + scaled_w] = frame

        writer.write(canvas)

    cap.release()
    writer.release()

    return True

def get_total_transformations() -> int:
    """
    Get the total number of available video transformations.
    
    Returns:
        int: Total number of transformations available
    """
    return len(VIDEO_TRANSFORMATIONS)

def get_random_transformation() -> str:
    """
    Get a random transformation from the available transformations.
    
    Returns:
        str: Random transformation string
    """
    return random.choice(VIDEO_TRANSFORMATIONS)

def get_transformation_by_index(index: int) -> str:
    """
    Get a transformation by its index.
    
    Args:
        index (int): Index of the transformation (0-based)
        
    Returns:
        str: Transformation string, or random if index is out of range
    """
    if 0 <= index < len(VIDEO_TRANSFORMATIONS):
        return VIDEO_TRANSFORMATIONS[index]
    else:
        return get_random_transformation()

def apply_video_transformations(video_path: str, output_path: str = None, transformation_index: int = None, preserve_original: bool = False) -> str:
    """
    Apply video transformation to make videos less identifiable while maintaining quality.
    
    This function applies various video transformations including color space adjustments,
    convolution filters, noise effects, and distortion to reduce the recognizability of 
    videos while preserving visual quality for training purposes.
    
    Args:
        video_path (str): Path to the input video file
        output_path (str, optional): Path for the output video. If None, creates a new path.
        transformation_index (int, optional): Index of the transformation to apply. If None, selects randomly.
        preserve_original (bool, optional): If True, don't delete the original file after transformation.
    
    Returns:
        str: Path to the transformed video file
    """
    if not os.path.exists(video_path):
        print(f"Input video file does not exist: {video_path}")
        return video_path
    
    if output_path is None:
        # Create output path by adding "_transformed" before the extension
        video_path_obj = Path(video_path)
        output_path = str(video_path_obj.parent / f"{video_path_obj.stem}_transformed{video_path_obj.suffix}")

    if os.path.exists(output_path):
        try:
            os.remove(output_path)
            print(f"Cleaned up existing output file before transformation: {output_path}")
        except OSError as e:
            print(f"Warning: Could not remove existing output file {output_path}: {e}")

    # Slight rotation, commented for now
    # rotation_angle_deg = random.uniform(-5, 5)  # -5 to 5 degrees
    # transformations.append(f"rotate={2*math.pi*rotation_angle_deg/360}")

    # Zoom & crop to remove black background left after rotation
    # transformations.append(f"crop=iw*0.87:ih*0.87,scale=iw/0.87:ih/0.87")

    # Select transformation based on index or randomly
    if transformation_index is not None:
        selected_transformation = get_transformation_by_index(transformation_index)
    else:
        selected_transformation = get_random_transformation()
    
    # Build FFmpeg command for video transformation
    transform_cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", selected_transformation,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",  # Faster preset, slightly higher CRF for speed
        "-an", "-threads", "30",
        str(output_path),
        "-hide_banner", "-loglevel", "error"
    ]
    
    try:
        print(f"Applying video transformation: {selected_transformation}")

        start_time = time.time()
        
        subprocess.run(transform_cmd, check=True, capture_output=True, text=True)
        
        elapsed_time = time.time() - start_time
        print(f"Video transformation completed in {elapsed_time:.2f}s: {video_path} -> {output_path}")
        
        # Verify the output file was created and is not empty
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            print(f"Warning: Output file is empty or doesn't exist: {output_path}")
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                    print(f"Removed empty/invalid output file: {output_path}")
                except OSError as e:
                    print(f"Warning: Could not remove invalid output file {output_path}: {e}")
            return video_path
        
        # Remove original file to save space only if transformation was successful and preserve_original is False
        if not preserve_original and os.path.exists(video_path) and os.path.exists(output_path):
            try:
                os.remove(video_path)
                print(f"Removed original file: {video_path}")
            except OSError as e:
                print(f"Warning: Could not remove original file {video_path}: {e}")
            
        return output_path
        
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error applying video transformation: {e}")
        if e.stderr:
            print(f"FFmpeg stderr: {e.stderr}")
        
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                print(f"Cleaned up failed transformation output: {output_path}")
            except OSError as cleanup_error:
                print(f"Warning: Could not clean up failed output {output_path}: {cleanup_error}")
        
        return None
        
    except Exception as e:
        print(f"Unexpected error applying video transformation: {e}")
        
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                print(f"Cleaned up output file after unexpected error: {output_path}")
            except OSError as cleanup_error:
                print(f"Warning: Could not clean up output after error {output_path}: {cleanup_error}")
        
        return None

def download_transform_and_trim_downscale_video(
    clip_duration: int,
    use_downscale_video: bool = True,
    redis_conn: redis.Redis = None,
    output_dir: str = "videos",
    transformations_per_video: int = 0,
    enable_transformations: bool = False
) -> Optional[Tuple[List[str], List[int], List[str]]]:
    """
    Downloads a specified video with video id from Pexels, applies transformations to the original video,
    extracts multiple chunks of the specified duration with adaptive overlapping strategies, and downscales 
    each chunk to required resolution using multi-threading.

    Args:
        clip_duration (int): Desired clip duration in seconds (5, 10, or 20).
        vid (int): Pexels video ID to download.
        task_type (str): Type of task determining resolution requirements.
        output_dir (str): Directory to save the processed videos.
        transformations_per_video (int): Number of transformations to apply per video (default: 0 = no transformations).
        enable_transformations (bool): Whether to enable transformations (default: False).

    Returns:
        Optional[Tuple[List[str], List[int], List[str]]]: Lists of paths to the downscaled videos, their generated IDs,
        and paths to reference trim files, or None on failure.
    """

    print("download_transform_and_trim_downscale_video")

    DOWNSCALE_HEIGHTS = {
        "HD24K": 1080,
        "4K28K": 2160,
    }
    EXPECTED_RESOLUTIONS = {
        "SD2HD": (1920, 1080),
        "4K28K": (7680, 4320),
    }

    print_lock = threading.Lock()
    
    def safe_print(message):
        with print_lock:
            print(message)

    os.makedirs(output_dir, exist_ok=True)
    load_dotenv()
    
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        raise ValueError("PEXELS_API_KEY not found in environment variables")

    def download_video(video_url, output_path):
        """Download video from URL with progress bar"""
        print(f"\nDownloading video from Pexels...")
        response = requests.get(video_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                size = f.write(chunk)
                pbar.update(size)
                
        print(f"Video downloaded successfully to: {output_path}")
        return output_path

    def get_video_info(video_path):
        """Get video resolution and duration using FFprobe"""
        probe_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0", 
            "-show_entries", "stream=width,height,duration", "-of", "csv=p=0", str(video_path)
        ]
        
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        width, height, duration = map(float, result.stdout.strip().split(','))
        return int(width), int(height), duration

    def process_chunk_from_transformed_video(chunk_info, transformed_source_path, base_video_id, downscale_height, transform_idx=None, use_downscale_video=True):
        i, start_time_clip, end_time_clip, chunk_video_id = chunk_info
        
        if transform_idx is not None:
            final_video_id = f"{base_video_id}_{transform_idx}_{i}"
        else:
            final_video_id = f"{base_video_id}_{i}"
        
        clipped_path = Path(output_dir) / f"{final_video_id}_trim.mp4"
        downscale_path = Path(output_dir) / f"{final_video_id}_downscale.mp4"
        
        for cleanup_path in [clipped_path, downscale_path]:
            if os.path.exists(cleanup_path):
                try:
                    os.remove(cleanup_path)
                    safe_print(f"Cleaned up existing chunk file: {cleanup_path}")
                except OSError as e:
                    safe_print(f"Warning: Could not remove existing chunk file {cleanup_path}: {e}")
        
        chunk_start_time = time.time()
        safe_print(f"Processing chunk {i+1} from transformed video: {start_time_clip:.1f}s to {end_time_clip:.1f}s")
        
        actual_duration = end_time_clip - start_time_clip
        
        trim_cmd = [
            "taskset", "-c", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20",
            "ffmpeg", "-y", "-i", str(transformed_source_path), "-ss", str(start_time_clip), 
            "-t", str(actual_duration), "-c:v", "libx264", "-preset", "ultrafast",
            "-an", str(clipped_path), "-hide_banner", "-loglevel", "error"
        ]
        
        scale_cmd = [
            "taskset", "-c", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20",
            "ffmpeg", "-y", "-i", str(clipped_path), "-vf", f"scale=-1:{downscale_height}", 
            "-c:v", "libx264", "-preset", "ultrafast", "-an",
            str(downscale_path), "-hide_banner", "-loglevel", "error"
        ]
        
        try:
            print(f"Trimming chunk {i+1} to {actual_duration}s")
            subprocess.run(trim_cmd, check=True)

            if not os.path.exists(clipped_path) or os.path.getsize(clipped_path) == 0:
                safe_print(f"Warning: Trim output is empty or invalid: {clipped_path}")
                if os.path.exists(clipped_path):
                    os.remove(clipped_path)
                return None

            if use_downscale_video:
                print(f"Downscaling chunk {i+1} to {downscale_height}p")
                subprocess.run(scale_cmd, check=True)
                
                if not os.path.exists(downscale_path) or os.path.getsize(downscale_path) == 0:
                    safe_print(f"Warning: Downscale output is empty or invalid: {downscale_path}")
                    for cleanup_path in [clipped_path, downscale_path]:
                        if os.path.exists(cleanup_path):
                            os.remove(cleanup_path)
                    return None
            else: 
                downscale_path = None
            
            chunk_elapsed_time = time.time() - chunk_start_time
            safe_print(f"Time taken to process chunk {i+1}: {chunk_elapsed_time:.2f} seconds")
            
            return (i, str(downscale_path), final_video_id, str(clipped_path))
            
        except subprocess.SubprocessError as e:
            safe_print(f"Error processing chunk {i+1}: {e}")
            
            cleanup_files = []
            if os.path.exists(clipped_path):
                cleanup_files.append(clipped_path)
            if os.path.exists(downscale_path):
                cleanup_files.append(downscale_path)
            
            for cleanup_file in cleanup_files:
                try:
                    os.remove(cleanup_file)
                    safe_print(f"Cleaned up failed chunk processing file: {cleanup_file}")
                except OSError as cleanup_error:
                    safe_print(f"Warning: Could not clean up chunk file {cleanup_file}: {cleanup_error}")
            
            return None
            
        except Exception as e:
            safe_print(f"Unexpected error processing chunk {i+1}: {e}")
            
            for cleanup_path in [clipped_path, downscale_path]:
                if os.path.exists(cleanup_path):
                    try:
                        os.remove(cleanup_path)
                        safe_print(f"Cleaned up file after unexpected error: {cleanup_path}")
                    except OSError as cleanup_error:
                        safe_print(f"Warning: Could not clean up file {cleanup_path}: {cleanup_error}")
            
            return None

    def generate_chunk_timestamps(total_duration, clip_duration):
        """
        Generate chunk timestamps using randomized approach to maximize chunks per video.
        This reduces API requests by extracting many overlapping chunks from each downloaded video
        while using randomized start points to avoid predictable patterns.
        
        Strategy:
        - Uses random start points within safe zones to avoid predictable patterns
        - Maintains reasonable overlap between chunks for continuity
        - Adapts to different clip durations with appropriate randomization ranges
        - For 10s clips from 30s+ videos, uses sliding window approach for maximum extraction
        """
        chunks = []

        # For 10s clips from 30s+ videos, use sliding window approach for maximum extraction
        if clip_duration == 10 and total_duration >= 30:
            # Use sliding window with small overlap to get maximum chunks
            slide_interval = 3.5  # Slide by 3.5s to get more chunks
            max_chunks = 6
            
            # Start from the beginning with some margin
            current_start = 1.0  # Start 1s into the video
            
            chunk_index = 0
            while chunk_index < max_chunks and current_start + clip_duration <= total_duration:
                end_time = current_start + clip_duration
                
                # Add some randomness to the start time
                random_offset = random.uniform(-0.5, 0.5)
                adjusted_start = max(0, current_start + random_offset)
                adjusted_end = adjusted_start + clip_duration
                
                # Ensure we don't exceed video duration
                if adjusted_end > total_duration:
                    adjusted_end = total_duration
                    adjusted_start = adjusted_end - clip_duration
                
                chunks.append((chunk_index, adjusted_start, adjusted_end))
                
                current_start += slide_interval
                chunk_index += 1
            
            print(f"Generated {len(chunks)} sliding window chunks for {clip_duration}s duration from {total_duration:.1f}s video")
            print(f"Using sliding window approach with {slide_interval}s intervals")
            
            return chunks
        
        # For other durations, use the original randomized approach
        # Define safe zones and overlap ranges for each clip duration
        # Safe zones avoid very beginning/end of video where quality might be poor
        safe_zone_configs = {
            1: {"safe_start": 0.2, "safe_end_ratio": 0.8, "min_overlap": 0.3, "max_overlap": 0.7},
            2: {"safe_start": 0.3, "safe_end_ratio": 0.85, "min_overlap": 0.5, "max_overlap": 1.0},
            3: {"safe_start": 0.4, "safe_end_ratio": 0.9, "min_overlap": 0.8, "max_overlap": 1.5},
            4: {"safe_start": 0.5, "safe_end_ratio": 0.9, "min_overlap": 1.0, "max_overlap": 2.0},
            5: {"safe_start": 0.5, "safe_end_ratio": 0.9, "min_overlap": 1.0, "max_overlap": 2.5},
            10: {"safe_start": 0.5, "safe_end_ratio": 0.9, "min_overlap": 0.5, "max_overlap": 1.5},  # Adjusted for better extraction
            20: {"safe_start": 0.5, "safe_end_ratio": 0.9, "min_overlap": 3.0, "max_overlap": 6.0}

        }
        
        config = safe_zone_configs.get(clip_duration, {
            "safe_start": 0.5, "safe_end_ratio": 0.9, "min_overlap": 1.0, "max_overlap": 2.0
        })
        
        # Calculate safe zone boundaries
        safe_start = config["safe_start"]
        safe_end = total_duration * config["safe_end_ratio"]
        min_overlap = config["min_overlap"]
        max_overlap = config["max_overlap"]
        
        # Set reasonable limits to avoid over-extraction from one video
        max_chunks_per_video = {
            1: 20,   # Max 20 chunks for 1s clips
            2: 15,   # Max 15 chunks for 2s clips
            3: 12,   # Max 12 chunks for 3s clips
            4: 10,   # Max 10 chunks for 4s clips
            5: 8,    # Max 8 chunks for 5s clips
            10: 6,   # Max 6 chunks for 10s clips
            20: 4    # Max 4 chunks for 20s clips

        }
        
        max_chunks = max_chunks_per_video.get(clip_duration, 8)
        
        # Ensure we have enough video duration
        if total_duration < clip_duration:
            return [(0, 0, min(total_duration, clip_duration))]
        
        # Ensure safe zone is valid
        if safe_end - safe_start < clip_duration:
            # If safe zone is too small, use the entire video
            safe_start = 0
            safe_end = total_duration
        
        # Generate randomized chunks
        chunk_index = 0
        last_end = safe_start
        
        while chunk_index < max_chunks:
            # Calculate the maximum start time for this chunk
            max_start = min(safe_end - clip_duration, last_end + max_overlap)
            
            # If we can't fit another chunk, break
            if max_start < last_end - min_overlap:
                break
            
            # Generate random start time within valid range
            min_start = max(safe_start, last_end - min_overlap)
            if min_start >= max_start:
                break
                
            start_time = random.uniform(min_start, max_start)
            end_time = start_time + clip_duration
            
            # Ensure we don't exceed video duration
            if end_time > total_duration:
                end_time = total_duration
                start_time = end_time - clip_duration
                if start_time < 0:
                    start_time = 0
            
            chunks.append((chunk_index, start_time, end_time))
            
            last_end = end_time
            chunk_index += 1
        
        # Add one final chunk from the end if we have remaining duration and space
        if (chunk_index < max_chunks and 
            total_duration > clip_duration and 
            total_duration - clip_duration > last_end - min_overlap):
            
            final_start = total_duration - clip_duration
            # Only add if it doesn't overlap too much with the last chunk
            if not chunks or final_start - chunks[-1][1] >= min_overlap:
                chunks.append((chunk_index, final_start, total_duration))
        
        print(f"Generated {len(chunks)} randomized chunks for {clip_duration}s duration from {total_duration:.1f}s video")
        print(f"Randomization details: safe_zone={safe_start:.1f}s-{safe_end:.1f}s, overlap_range={min_overlap:.1f}s-{max_overlap:.1f}s, max_chunks={max_chunks}")
        
        return chunks

    def process_video_standard(clip_duration, output_dir, transformations_per_video, enable_transformations, use_downscale_video, redis_conn):
        """
        Process 5s or 10s clips with transformation workflow:
        1. Download video once
        2. Apply transformations to original video (if enabled)
        3. Process chunks from each transformed version in parallel
        """
        try:

            downscale_height = None
            total_duration = None
            task_type = None
            temp_path = None
            max_retry = 3

            while True:

                if max_retry < 0:
                    break

                print("poping pexels video id")
                video_id_data = pop_pexels_video_id(redis_conn)

                if video_id_data == None:
                    pass

                vid = video_id_data["vid"]
                task_type = video_id_data["task_type"]
                
                # downscale_height = DOWNSCALE_HEIGHTS.get(task_type, 540)
                expected_width, expected_height = EXPECTED_RESOLUTIONS.get(task_type, (3840, 2160))
                start_time = time.time()
                
                url = f"https://api.pexels.com/videos/videos/{vid}"
                headers = {"Authorization": api_key}
                
                response = requests.get(url, headers=headers)
                response.raise_for_status()  
                
                data = response.json()
                if "video_files" not in data:
                    raise ValueError("No video found or API error")
                    
                video_url = next(
                    (video["link"] for video in data["video_files"] 
                    if video["width"] == expected_width and video["height"] == expected_height), 
                    None
                )
                
                if video_url is None:
                    return None, None, None
                
                temp_path = Path(output_dir) / f"{vid}_original.mp4"
                
                download_video(video_url, temp_path)
                elapsed_time = time.time() - start_time
                print(f"Time taken to download video: {elapsed_time:.2f} seconds")
            
                print("\nChecking video resolution...")
                width, height, total_duration = get_video_info(temp_path)
                
                if width != expected_width or height != expected_height or total_duration < CONFIG.video_scheduler.min_video_len:
                    error_msg = (f"Video resolution mismatch. Expected {expected_width}x{expected_height}, "
                                f"but got {width}x{height} or video duration is less than {CONFIG.video_scheduler.min_video_len}s: video len: {total_duration}")
                    print(error_msg)
                    os.remove(temp_path)
                    max_retry -= 1
                    continue
                else:
                    print(f"\nVideo resolution verified: {width}x{height}")
                    print(f"Video duration: {total_duration:.2f}s")
                    break
            
            chunks_info = generate_chunk_timestamps(total_duration, clip_duration)
            print(f"Extracting {len(chunks_info)} chunks of {clip_duration}s each")
            
            base_video_ids = [uuid.uuid4() for _ in range(len(chunks_info))]
            
            preprocessed_video_path = Path(output_dir) / f"{vid}_preprocessed.mp4"

            if enable_transformations and transformations_per_video > 0:
                print(f"\n Creating {transformations_per_video} transformed versions of the original video...")
                
                transformed_video_paths = []
                failed_attempts = 0
                max_failed_attempts = transformations_per_video * 2  # Allow some retries but prevent infinite loops
                
                transform_idx = 0

                transformation_start_time = time.time()
                while len(transformed_video_paths) < transformations_per_video and failed_attempts < max_failed_attempts:
                    transformed_path = Path(output_dir) / f"{vid}_transform_{transform_idx}.mp4"
                    if os.path.exists(transformed_path):
                        try:
                            os.remove(transformed_path)
                            print(f"Cleaned up existing transformation file: {transformed_path}")
                        except OSError as e:
                            print(f"Warning: Could not clean up existing file {transformed_path}: {e}")
                    
                    transformed_result = preprocess_video(temp_path, transformed_path)

                    if transformed_result and os.path.exists(transformed_path):
                        transformed_video_paths.append(transformed_path)
                        transform_idx += 1
                        print(f"✅ Created transformed version {len(transformed_video_paths)}: {transformed_path}")
                    else:
                        failed_attempts += 1
                        print(f"❌ Failed to create transformed version {transform_idx + 1} (attempt {failed_attempts}/{max_failed_attempts})")
                        if os.path.exists(transformed_path):
                            try:
                                os.remove(transformed_path)
                                print(f"Cleaned up failed transformation file: {transformed_path}")
                            except OSError as e:
                                print(f"Warning: Could not clean up failed file {transformed_path}: {e}")

                transformation_end_time = time.time()
                print(f"Time taken to create transformed versions: {transformation_end_time - transformation_start_time:.2f} seconds")

                if failed_attempts >= max_failed_attempts:
                    print(f"⚠️ Exceeded maximum failed attempts ({max_failed_attempts}), falling back to original")
                    transformed_video_paths = [str(temp_path)]
                    transformations_per_video = 1
                elif not transformed_video_paths:
                    print("❌ No transformed versions created successfully, falling back to original")
                    transformed_video_paths = [str(temp_path)]
                    transformations_per_video = 1

                print(f"transformed_video_paths: {transformed_video_paths}")

                all_results = []

                chunks_with_ids = [(i, start, end, base_video_ids[idx]) 
                                    for idx, (i, start, end) in enumerate(chunks_info)]

                for chunk in chunks_with_ids:
                    print(f"Processing chunk: {chunk}")
                
                downscaler = 2

                if task_type == "SD24K":
                    downscaler = 4
                
                print(f"task_type: {task_type}, downscaler: {downscaler}")

                for transform_idx, transformed_path in enumerate(transformed_video_paths):
                    print(f"\n📹 Processing chunks from transformed version {transform_idx + 1}...")
                    
                    _, v_height, _ = get_video_info(transformed_path)

                    downscale_height = v_height/downscaler

                    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        future_to_chunk = {
                            executor.submit(
                                process_chunk_from_transformed_video, 
                                chunk, 
                                transformed_path, 
                                base_video_ids[idx], 
                                downscale_height,
                                transform_idx,
                                use_downscale_video
                            ): chunk 
                            for idx, chunk in enumerate(chunks_with_ids)
                        }
                        
                        for future in concurrent.futures.as_completed(future_to_chunk):
                            result = future.result()
                            if result:
                                all_results.append(result)
                    
                    if os.path.exists(transformed_path):
                        os.remove(transformed_path)
                        print(f"🧹 Cleaned up transformed video: {transformed_path}")
                
                results = all_results
                
            else:
                print("\n📹 Processing chunks from original video (no transformations)...")
                
                chunks_with_ids = [(i, start, end, base_video_ids[idx]) 
                                  for idx, (i, start, end) in enumerate(chunks_info)]
                
                results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    future_to_chunk = {
                        executor.submit(
                            process_chunk_from_transformed_video,
                            chunk, 
                            temp_path, 
                            base_video_ids[idx],
                            downscale_height,
                            use_downscale_video
                        ): chunk 
                        for idx, chunk in enumerate(chunks_with_ids)
                    }
                    
                    for future in concurrent.futures.as_completed(future_to_chunk):
                        result = future.result()
                        if result:
                            results.append(result)
            
            results.sort(key=lambda x: x[0])
            
            downscale_paths = [result[1] for result in results]
            final_video_ids = [result[2] for result in results]
            reference_trim_paths = [result[3] for result in results]
            
            print("\nCleaning up original downloaded file...")
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"Deleted original video: {temp_path}")
            if os.path.exists(preprocessed_video_path):
                os.remove(preprocessed_video_path)
                print(f"Deleted preprocessed video: {preprocessed_video_path}")

            print(f"\nDone! Successfully processed {len(results)} chunks.")
            print(f"Downscaled videos: {len(downscale_paths)}")
            print(f"Reference trims: {len(reference_trim_paths)}")
            
            return downscale_paths, final_video_ids, reference_trim_paths, task_type
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading video: {e}")
            return None, None, None, None
        except Exception as e:
            print(f"Error: {str(e)}")
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    print(f"Cleaned up downloaded file after error: {temp_path}")
                except:
                    pass
            return None, None, None, None

    try:
        return process_video_standard(clip_duration, output_dir, transformations_per_video, enable_transformations, use_downscale_video, redis_conn)

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None, None, None, None


def get_trim_video_path(file_id: int, dir_path: str = "videos") -> str:
    """Returns the path of the clipped trim video based on the file ID."""
    return str(Path(dir_path) / f"{file_id}_trim.mp4")


def delete_videos_with_fileid(file_id: int, dir_path: str = "videos") -> None:
    """Deletes all video files associated with the given file ID."""

    files_to_delete = [
        Path(dir_path) / f"{file_id}_trim.mp4",
        Path(dir_path) / f"{file_id}_downscale.mp4",
    ]

    for file_path in files_to_delete:
        try:
            if file_path.exists():
                file_path.unlink()
                print(f"Deleted: {file_path}")
            else:
                print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def cleanup_orphaned_files(output_dir: str = "videos", max_age_hours: int = 720) -> None:
    """
    Cleanup function to remove orphaned files that may have been left behind by failed operations.
    This acts as a safety net to prevent storage issues from accumulating files.
    
    Args:
        output_dir (str): Directory to clean up
        max_age_hours (int): Maximum age in hours for files to keep
    """
    if not os.path.exists(output_dir):
        print(f"Output directory does not exist: {output_dir}")
        return
        
    print(f"\nCleaning up orphaned files in {output_dir} older than {max_age_hours} hours...")
    
    patterns = [
        "*_transform_*.mp4",
        "*_trim.mp4",       
        "*_downscale.mp4",  
        "*_original.mp4",   
        "*_ytransform_*.mp4"
    ]
    
    now = time.time()
    max_age_seconds = max_age_hours * 3600
    
    total_cleaned = 0
    total_size_cleaned = 0
    
    for pattern in patterns:
        pattern_path = os.path.join(output_dir, pattern)
        try:
            for file_path in glob.glob(pattern_path):
                try:
                    # Check file age
                    file_age = now - os.path.getmtime(file_path)
                    if file_age > max_age_seconds:
                        # Get file size before deleting
                        try:
                            file_size = os.path.getsize(file_path)
                        except OSError:
                            file_size = 0
                            
                        try:
                            os.remove(file_path)
                            total_cleaned += 1
                            total_size_cleaned += file_size
                            print(f"Cleaned up orphaned file: {file_path} (Age: {file_age/3600:.1f} hours, Size: {file_size/1024/1024:.1f} MB)")
                        except OSError as e:
                            print(f"Warning: Could not remove orphaned file {file_path}: {e}")
                            
                except OSError as e:
                    print(f"Warning: Error checking file {file_path}: {e}")
                    
        except glob.error as e:
            print(f"Warning: Error processing pattern {pattern}: {e}")
            
    if total_cleaned > 0:
        print(f"\nCleanup complete! Removed {total_cleaned} orphaned files (Total: {total_size_cleaned/1024/1024:.1f} MB)")
    else:
        print("\nNo orphaned files found to clean up")

def download_trim_downscale_youtube_video(
    clip_duration: int,
    youtube_video_id: str,
    task_type: str,
    output_dir: str = "videos",
    youtube_handler=None,
    transformations_per_video: int = 0,
    enable_transformations: bool = False
) -> Optional[Tuple[List[str], List[int], List[str]]]:
    """
    Downloads a YouTube video, applies transformations to the original video,
    extracts multiple chunks of the specified duration with sliding window approach,
    and downscales each chunk to required resolution.

    Args:
        clip_duration (int): Desired clip duration in seconds (5, 10, or 20).
        youtube_video_id (str): YouTube video ID to download.
        task_type (str): Type of task determining resolution requirements.
        output_dir (str): Directory to save the processed videos.
        youtube_handler: Optional YouTubeHandler instance for downloading.
        transformations_per_video (int): Number of transformations to apply per video (default: 0 = no transformations).
        enable_transformations (bool): Whether to enable transformations (default: False).

    Returns:
        Optional[Tuple[List[str], List[int], List[str]]]: Lists of paths to the downscaled videos, their generated IDs,
        and paths to reference trim files, or None on failure.
    """
    try:
        from .youtube_requests import YouTubeHandler, RESOLUTIONS
    except ImportError:
        try:
            from youtube_requests import YouTubeHandler, RESOLUTIONS
        except ImportError:
            print("Error: Could not import YouTubeHandler and RESOLUTIONS")
            return None, None, None
    
    if youtube_handler is None:
        youtube_handler = YouTubeHandler()

    DOWNSCALE_HEIGHTS = {
        "HD24K": 1080,
        "4K28K": 2160,
    }
    
    TASK_TO_RESOLUTION = {
        "SD2HD": RESOLUTIONS.HD_1080,
        "HD24K": RESOLUTIONS.HD_2160,
        "SD24K": RESOLUTIONS.HD_2160,
        "4K28K": RESOLUTIONS.HD_4320,
    }

    print_lock = threading.Lock()
    
    def safe_print(message):
        with print_lock:
            print(message)

    downscale_height = DOWNSCALE_HEIGHTS.get(task_type, 1080)
    target_resolution = TASK_TO_RESOLUTION.get(task_type, RESOLUTIONS.HD_2160)

    os.makedirs(output_dir, exist_ok=True)

    def get_video_info(video_path):
        """Get video resolution and duration using FFprobe"""
        probe_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0", 
            "-show_entries", "stream=width,height,duration", "-of", "csv=p=0", str(video_path)
        ]
        
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        width, height, duration = map(float, result.stdout.strip().split(','))
        return int(width), int(height), duration

    def process_chunk(chunk_info, source_path):
        """Process a single video chunk using FFmpeg directly"""
        i, start_time_clip, end_time_clip, video_id = chunk_info
        
        clipped_path = Path(output_dir) / f"{video_id}_trim.mp4"
        downscale_path = Path(output_dir) / f"{video_id}_downscale.mp4"
        
        for cleanup_path in [clipped_path, downscale_path]:
            if os.path.exists(cleanup_path):
                try:
                    os.remove(cleanup_path)
                    safe_print(f"Cleaned up existing chunk file: {cleanup_path}")
                except OSError as e:
                    safe_print(f"Warning: Could not remove existing chunk file {cleanup_path}: {e}")
        
        chunk_start_time = time.time()
        safe_print(f"Processing YouTube chunk {i+1}: {start_time_clip:.1f}s to {end_time_clip:.1f}s")
        
        actual_duration = end_time_clip - start_time_clip
        
        trim_cmd = [
            "ffmpeg", "-y", "-i", str(source_path), "-ss", str(start_time_clip), 
            "-t", str(actual_duration), "-c:v", "libx264", "-preset", "fast",
            "-an", str(clipped_path), "-hide_banner", "-loglevel", "error"
        ]
        
        scale_cmd = [
            "ffmpeg", "-y", "-i", str(clipped_path), "-vf", f"scale=-1:{downscale_height}", 
            "-c:v", "libx264", "-preset", "fast", "-an",
            str(downscale_path), "-hide_banner", "-loglevel", "error"
        ]
        
        try:
            subprocess.run(trim_cmd, check=True)
            subprocess.run(scale_cmd, check=True)
            
            chunk_elapsed_time = time.time() - chunk_start_time
            safe_print(f"Time taken to process YouTube chunk {i+1}: {chunk_elapsed_time:.2f} seconds")
            
            return (i, str(downscale_path), video_id, str(clipped_path))
            
        except subprocess.SubprocessError as e:
            safe_print(f"Error processing YouTube chunk {i+1}: {e}")
            
            cleanup_files = []
            if os.path.exists(clipped_path):
                cleanup_files.append(clipped_path)
            if os.path.exists(downscale_path):
                cleanup_files.append(downscale_path)
            
            for cleanup_file in cleanup_files:
                try:
                    os.remove(cleanup_file)
                    safe_print(f"Cleaned up failed chunk processing file: {cleanup_file}")
                except OSError as cleanup_error:
                    safe_print(f"Warning: Could not clean up chunk file {cleanup_file}: {cleanup_error}")
            
            return None

    def generate_chunk_timestamps(total_duration, clip_duration):
        """
        Generate chunk timestamps using sliding window approach.
        Similar to Pexels but adapted for potentially longer YouTube videos.
        """
        chunks = []
        
        # Define slide intervals for each clip duration
        slide_intervals = {
            5: 2,     # For 5s clips, slide by 2s
            10: 4,    # For 10s clips, slide by 4s
        }
        
        slide_interval = slide_intervals.get(clip_duration, 2)
        
        # Calculate maximum number of chunks we can extract
        max_chunks_possible = int((total_duration - clip_duration) / slide_interval) + 1
        
        # Set reasonable limits for YouTube videos (can be longer)
        max_chunks_per_video = {
            5: min(30, max_chunks_possible),   # Max 30 chunks for 5s clips
            10: min(20, max_chunks_possible),  # Max 20 chunks for 10s clips
        }
        
        max_chunks = max_chunks_per_video.get(clip_duration, max_chunks_possible)
        
        # Ensure we have enough video duration
        if total_duration < clip_duration:
            return [(0, 0, min(total_duration, clip_duration))]
        
        # Generate sliding window chunks
        current_start = 0
        chunk_index = 0
        
        while (current_start + clip_duration <= total_duration and 
               chunk_index < max_chunks):
            
            end_time = current_start + clip_duration
            chunks.append((chunk_index, current_start, end_time))
            
            current_start += slide_interval
            chunk_index += 1
        
        safe_print(f"Generated {len(chunks)} chunks for {clip_duration}s duration from {total_duration:.1f}s YouTube video")
        
        return chunks

    try:
        start_time = time.time()
        
        # Download YouTube video to temporary path
        temp_path = Path(output_dir) / f"{youtube_video_id}_original.mp4"
        
        safe_print(f"Downloading YouTube video: {youtube_video_id}")
        
        # Download video using YouTubeHandler
        youtube_handler.download_video(
            youtube_video_id, 
            target_resolution, 
            output_path=str(temp_path)
        )
        
        elapsed_time = time.time() - start_time
        safe_print(f"Time taken to download YouTube video: {elapsed_time:.2f} seconds")
    
        safe_print("\nChecking YouTube video resolution...")
        width, height, total_duration = get_video_info(temp_path)
        
        safe_print(f"YouTube video resolution: {width}x{height}")
        safe_print(f"YouTube video duration: {total_duration:.2f}s")
        
        # Skip very short videos
        if total_duration < clip_duration * 2:
            safe_print(f"YouTube video too short ({total_duration:.2f}s), skipping...")
            os.remove(temp_path)
            return None, None, None
        
        chunks_info = generate_chunk_timestamps(total_duration, clip_duration)
        safe_print(f"Extracting {len(chunks_info)} chunks of {clip_duration}s each from YouTube video")
        
        base_video_ids = [uuid.uuid4() for _ in range(len(chunks_info))]
        
        if enable_transformations and transformations_per_video > 0:
            safe_print(f"\n🎨 Creating {transformations_per_video} transformed versions of YouTube video...")
            
            transformed_video_paths = []
            
            for transform_idx in range(transformations_per_video):
                transformed_path = Path(output_dir) / f"{youtube_video_id}_ytransform_{transform_idx}.mp4"
                
                transformed_result = apply_video_transformations(
                    str(temp_path), 
                    str(transformed_path), 
                    preserve_original=True
                )
                
                if transformed_result and os.path.exists(transformed_result):
                    transformed_video_paths.append(transformed_result)
                    safe_print(f"✅ Created YouTube transformed version {transform_idx + 1}: {transformed_result}")
                else:
                    safe_print(f"❌ Failed to create YouTube transformed version {transform_idx + 1}")
            
            if not transformed_video_paths:
                safe_print("❌ No YouTube transformed versions created successfully, falling back to original")
                transformed_video_paths = [str(temp_path)]
                transformations_per_video = 1
            
            all_results = []
            
            for transform_idx, transformed_path in enumerate(transformed_video_paths):
                safe_print(f"\n📹 Processing YouTube chunks from transformed version {transform_idx + 1}...")
                
                chunks_with_ids = [(i, start, end, f"{base_video_ids[idx]}_{transform_idx}") 
                                  for idx, (i, start, end) in enumerate(chunks_info)]
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    future_to_chunk = {executor.submit(process_chunk, chunk, transformed_path): chunk 
                                      for chunk in chunks_with_ids}
                    
                    for future in concurrent.futures.as_completed(future_to_chunk):
                        result = future.result()
                        if result:
                            all_results.append(result)
                
                if transform_idx > 0 and os.path.exists(transformed_path):
                    os.remove(transformed_path)
                    safe_print(f"🧹 Cleaned up YouTube transformed video: {transformed_path}")
            
            results = all_results
            
        else:
            safe_print("\n📹 Processing YouTube chunks from original video (no transformations)...")
            
            chunks_with_ids = [(i, start, end, base_video_ids[idx]) 
                              for idx, (i, start, end) in enumerate(chunks_info)]
            
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_chunk = {executor.submit(process_chunk, chunk, temp_path): chunk 
                                  for chunk in chunks_with_ids}
                
                for future in concurrent.futures.as_completed(future_to_chunk):
                    result = future.result()
                    if result:
                        results.append(result)
        
        results.sort(key=lambda x: x[0])
        
        downscale_paths = [result[1] for result in results]
        final_video_ids = [result[2] for result in results]
        reference_trim_paths = [result[3] for result in results]
        
        safe_print("\nCleaning up original downloaded YouTube file...")
        os.remove(temp_path)
        safe_print(f"Deleted: {temp_path}")

        safe_print(f"\nDone! Successfully processed {len(results)} YouTube chunks.")
        safe_print(f"Downscaled videos: {len(downscale_paths)}")
        safe_print(f"Reference trims: {len(reference_trim_paths)}")
        
        return downscale_paths, final_video_ids, reference_trim_paths
        
    except Exception as e:
        safe_print(f"Error processing YouTube video: {str(e)}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                safe_print(f"Cleaned up YouTube file after error: {temp_path}")
            except:
                pass
        return None, None, None

if __name__ == "__main__":
    
    CLIP_DURATION = 2
    vid = 2257054
    
    result = download_transform_and_trim_downscale_video(
        clip_duration=CLIP_DURATION, 
        vid=vid, 
        task_type="HD24K",
        transformations_per_video=2,
        enable_transformations=True
    )
    
    if result:
        downscale_paths, video_ids, reference_trim_paths = result
        print(f"✅ Successfully processed video:")
        print(f"   Downscaled files: {len(downscale_paths)}")
        print(f"   Video IDs: {len(video_ids)}")
        print(f"   Reference trims: {len(reference_trim_paths)}")
        
        for i, (downscale, vid_id, ref_trim) in enumerate(zip(downscale_paths, video_ids, reference_trim_paths)):
            print(f"   Chunk {i+1}:")
            print(f"     Downscaled: {downscale}")
            print(f"     Video ID: {vid_id}")
            print(f"     Reference: {ref_trim}")
    else:
        print("❌ Failed to process video")
