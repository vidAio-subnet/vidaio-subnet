"""
Video Pre-processing Module

This module handles the initial video processing stage including:
- Video validation and duration checks
- Codec detection and handling
- Lossless format conversion when necessary
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from utils.video_utils import get_video_duration, get_video_codec
from utils.encode_video import encode_lossless_video


# ============================================================================
# Constants
# ============================================================================

# Quality to VMAF score mapping
QUALITY_VMAF_MAPPING = {
    'High': 95.0,
    'Medium': 90.0,  # Updated from 93.0 to 90.0
    'Low': 85.0      # Updated from 90.0 to 85.0
}

# Lossless codecs that require re-encoding
LOSSLESS_CODECS = [
    'ffv1', 'h264_lossless', 'utvideo', 'rawvideo', 
    'prores_ks', 'dnxhd', 'cineform'
]

# Lossless file extensions
LOSSLESS_EXTENSIONS = ['.y4m', '.yuv', '.raw']


# ============================================================================
# Main Function
# ============================================================================

def pre_processing(
    video_path: str, 
    target_quality: str = 'Medium',
    codec: str = 'auto', 
    max_duration: int = 60, 
    output_dir: str = './output'
) -> Optional[Dict[str, Any]]:
    """
    Part 1: Initial video checks and lossless encoding if necessary.

    This function performs the first stage of the video processing pipeline.
    It checks the video's duration and codec to determine if it's a candidate
    for processing. If the video is in a lossless format, it's re-encoded
    into a standardized lossless format (FFV1 in an MKV container) to ensure
    compatibility with later stages. Compressed videos are passed through
    unmodified, while videos exceeding the maximum duration are rejected.

    Args:
        video_path: The full path to the input video file
        target_quality: Target quality level - 'High', 'Medium', or 'Low'
                       Gets converted to VMAF scores: High=95, Medium=93, Low=90
        max_duration: Maximum allowed video duration in seconds. Default: 3600 (1 hour)
        output_dir: Directory for final output files. Default: './output'
        codec: Target encoding codec. Default: 'auto' (auto-detect best available)
               Options: 'auto', 'av1_nvenc', 'libx264', 'libx265', 'h264_nvenc', etc.

    Returns:
        dict or None: Dictionary containing video metadata and processing info if successful, 
                     otherwise None. Keys: 'path', 'codec', 'original_codec', 'duration', 
                     'was_reencoded', 'encoding_time', 'target_vmaf', 'target_quality', 
                     'directories'
    """
    # Resolve codec selection
    target_codec = _resolve_target_codec(codec)
    
    # Validate and convert target quality
    target_quality, target_vmaf = _validate_target_quality(target_quality)
    
    print(f"🎯 Target quality: {target_quality} (VMAF: {target_vmaf})")
    print(f"🎥 Target codec: {target_codec}")
    
    # Setup directories
    _setup_output_directory(output_dir)
    print(f"📁 Output directory: {output_dir}")
    
    # Validate video duration
    duration = _validate_video_duration(video_path, max_duration)
    if duration is None:
        return None
    
    # Detect video codec
    original_codec = _detect_video_codec(video_path)
    if not original_codec:
        return None
    
    # Check if video needs lossless conversion
    is_lossless = _is_lossless_format(original_codec, video_path)
    
    if is_lossless:
        return _handle_lossless_conversion(
            video_path, original_codec, duration, target_vmaf, 
            target_quality, target_codec, output_dir
        )
    else:
        return _handle_compressed_video(
            video_path, original_codec, duration, target_vmaf, 
            target_quality, target_codec
        )


# ============================================================================
# Helper Functions
# ============================================================================

def _resolve_target_codec(codec: str) -> str:
    """Resolve auto codec selection to specific codec."""
    if codec.lower() == 'auto':
        return 'av1_nvenc'  # Default to AV1 for best quality
    else:
        print(f"🎯 Using specified codec: {codec}")
        return codec


def _validate_target_quality(target_quality: str) -> tuple[str, float]:
    """Validate target quality and convert to VMAF score."""
    if target_quality not in QUALITY_VMAF_MAPPING:
        print(f"⚠️ Invalid target quality '{target_quality}'. Using 'Medium' as fallback.")
        target_quality = 'Medium'
    
    target_vmaf = QUALITY_VMAF_MAPPING[target_quality]
    return target_quality, target_vmaf


def _setup_output_directory(output_dir: str) -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)


def _validate_video_duration(video_path: str, max_duration: int) -> Optional[float]:
    """Validate video duration against maximum allowed duration."""
    print(f"⏱️ Checking video duration...")
    duration = get_video_duration(video_path)
    
    if duration is None:
        print("❌ Could not determine video duration. Aborting.")
        return None
    
    if duration > max_duration:
        print(f"❌ Video duration {duration}s exceeds maximum allowed duration of {max_duration}s. Rejecting video.")
        return None
    
    print(f"✅ Video duration: {duration}s (within limit: {max_duration}s)")
    return duration


def _detect_video_codec(video_path: str) -> Optional[str]:
    """Detect video codec from file."""
    print(f"🎥 Checking video codec...")
    original_codec = get_video_codec(video_path)
    
    if not original_codec:
        print("❌ Could not determine video codec. Aborting.")
        return None
    
    print(f"🎥 Detected codec: {original_codec}")
    return original_codec


def _is_lossless_format(codec: str, video_path: str) -> bool:
    """Check if video is in a lossless format."""
    return (
        codec.lower() in LOSSLESS_CODECS or 
        any(video_path.lower().endswith(ext) for ext in LOSSLESS_EXTENSIONS)
    )


def _handle_lossless_conversion(
    video_path: str, 
    original_codec: str, 
    duration: float, 
    target_vmaf: float, 
    target_quality: str, 
    target_codec: str, 
    output_dir: str
) -> Optional[Dict[str, Any]]:
    """Handle conversion of lossless video to standardized format."""
    print(f"🔄 Video is in a lossless format ({original_codec}). Re-encoding with standardized lossless compression...")
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = f"{base_name}_lossless_{timestamp}.mkv"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"📁 Lossless output: {output_filename}")
    
    # Perform lossless encoding
    encode_log, encode_time = encode_lossless_video(
        input_path=video_path,
        output_path=output_path,
        logging_enabled=True
    )
    
    if encode_log:
        print(f"✅ Lossless encoding successful in {encode_time:.1f}s")
        print(f"📁 Output at: {output_path}")
        
        # Return comprehensive metadata
        return {
            'path': output_path,
            'codec': 'ffv1',
            'original_codec': original_codec,
            'duration': duration,
            'was_reencoded': True,
            'encoding_time': encode_time,
            'target_vmaf': target_vmaf,
            'target_quality': target_quality,
            'target_codec': target_codec,
            'processing_info': {
                'lossless_conversion': True,
                'original_format': original_codec,
                'standardized_format': 'ffv1',
                'container': 'mkv'
            }
        }
    else:
        print("❌ Lossless encoding failed.")
        return None


def _handle_compressed_video(
    video_path: str, 
    original_codec: str, 
    duration: float, 
    target_vmaf: float, 
    target_quality: str, 
    target_codec: str
) -> Dict[str, Any]:
    """Handle compressed video that doesn't need conversion."""
    print(f"✅ Video is already compressed with codec: {original_codec}. Proceeding with original file.")
    
    # Return metadata for compressed video
    return {
        'path': video_path,
        'codec': original_codec,
        'original_codec': original_codec,
        'duration': duration,
        'was_reencoded': False,
        'encoding_time': 0,
        'target_vmaf': target_vmaf,
        'target_quality': target_quality,
        'target_codec': target_codec,
        'processing_info': {
            'lossless_conversion': False,
            'original_format': original_codec,
            'standardized_format': original_codec,
            'container': os.path.splitext(video_path)[1][1:]  # Extension without dot
        }
    }


