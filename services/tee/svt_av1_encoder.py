"""
SVT-AV1 Video Compression

This module provides CPU-based video compression using SVT-AV1
(Scalable Video Technology for AV1). SVT-AV1 is an AV1 encoder
designed for high-quality, high-efficiency encoding.

SVT-AV1 is ideal for:
- CPU-based encoding without GPU requirements
- High compression efficiency with modern AV1 codec
- Scalable performance across multiple CPU cores
- Running inside SGX enclaves for privacy-preserving compression

Reference: https://gitlab.com/AOMediaCodec/SVT-AV1
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class CodecType(Enum):
    """Supported codec types."""
    AV1 = "av1"
    HEVC = "hevc"
    H264 = "h264"
    VP9 = "vp9"


class CodecMode(Enum):
    """Encoding modes."""
    CRF = "CRF"   # Constant Rate Factor (quality-based)
    VBR = "VBR"   # Variable Bitrate
    CBR = "CBR"   # Constant Bitrate


@dataclass
class CompressionConfig:
    """Configuration for video compression."""
    target_codec: CodecType = CodecType.AV1
    codec_mode: CodecMode = CodecMode.CRF
    
    # Quality settings
    crf: int = 30                    # CRF value (lower = higher quality)
    preset: int = 6                  # SVT-AV1 preset (0-13, lower = slower/better)
    
    # Bitrate settings (for VBR/CBR modes)
    target_bitrate_mbps: float = 5.0
    max_bitrate_mbps: Optional[float] = None
    
    # VMAF settings
    vmaf_threshold: float = 90.0     # Target VMAF score
    
    # Threading
    thread_count: int = 0            # 0 = auto-detect
    
    # Output settings
    pixel_format: str = "yuv420p"
    
    
# Mapping from codec type to ffmpeg encoder
CODEC_ENCODERS = {
    CodecType.AV1: "libsvtav1",
    CodecType.HEVC: "libx265",
    CodecType.H264: "libx264",
    CodecType.VP9: "libvpx-vp9",
}

# CRF value approximations for target VMAF scores
# These are starting points - actual results vary by content
VMAF_TO_CRF = {
    # (codec, vmaf_target) -> approximate CRF
    (CodecType.AV1, 93): 25,
    (CodecType.AV1, 90): 30,
    (CodecType.AV1, 85): 35,
    (CodecType.AV1, 80): 40,
    (CodecType.HEVC, 93): 22,
    (CodecType.HEVC, 90): 26,
    (CodecType.HEVC, 85): 30,
    (CodecType.HEVC, 80): 34,
    (CodecType.H264, 93): 20,
    (CodecType.H264, 90): 23,
    (CodecType.H264, 85): 26,
    (CodecType.H264, 80): 30,
}


class SVTAV1Encoder:
    """
    Video encoder using SVT-AV1 and other CPU-based codecs.
    
    This implementation uses ffmpeg with SVT-AV1 (libsvtav1), libx265,
    libx264, or libvpx-vp9 for CPU-based encoding that can run inside
    SGX enclaves.
    """
    
    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        """
        Initialize encoder.
        
        Args:
            ffmpeg_path: Path to ffmpeg binary
        """
        self.ffmpeg_path = ffmpeg_path
        self._verify_ffmpeg()
    
    def _verify_ffmpeg(self) -> None:
        """Verify ffmpeg is available with required encoders."""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-encoders"],
                capture_output=True,
                text=True
            )
            
            # Check for required encoders
            encoders_output = result.stdout.lower()
            required = ["libsvtav1", "libx265", "libx264"]
            
            for encoder in required:
                if encoder.lower() not in encoders_output:
                    logger.warning(f"ffmpeg missing encoder: {encoder}")
                    
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("ffmpeg not available - compression may fail")
    
    def _estimate_crf_for_vmaf(
        self,
        codec: CodecType,
        vmaf_target: float
    ) -> int:
        """
        Estimate CRF value to achieve target VMAF score.
        
        This is a heuristic - actual results depend on content.
        """
        # Find closest VMAF target in our mapping
        best_crf = 30  # default
        best_diff = float('inf')
        
        for (c, v), crf in VMAF_TO_CRF.items():
            if c == codec:
                diff = abs(v - vmaf_target)
                if diff < best_diff:
                    best_diff = diff
                    best_crf = crf
        
        return best_crf
    
    def _build_encoder_args(
        self,
        config: CompressionConfig
    ) -> List[str]:
        """Build ffmpeg encoder arguments."""
        encoder = CODEC_ENCODERS.get(config.target_codec, "libsvtav1")
        args = ["-c:v", encoder]
        
        if config.target_codec == CodecType.AV1:
            # SVT-AV1 specific settings
            args.extend(["-preset", str(config.preset)])
            
            if config.codec_mode == CodecMode.CRF:
                args.extend(["-crf", str(config.crf)])
                args.extend(["-b:v", "0"])  # Required for CRF mode
            elif config.codec_mode == CodecMode.VBR:
                bitrate_k = int(config.target_bitrate_mbps * 1000)
                args.extend(["-b:v", f"{bitrate_k}k"])
                if config.max_bitrate_mbps:
                    max_k = int(config.max_bitrate_mbps * 1000)
                    args.extend(["-maxrate", f"{max_k}k"])
            elif config.codec_mode == CodecMode.CBR:
                bitrate_k = int(config.target_bitrate_mbps * 1000)
                args.extend(["-b:v", f"{bitrate_k}k"])
                args.extend(["-maxrate", f"{bitrate_k}k"])
                args.extend(["-minrate", f"{bitrate_k}k"])
            
            # SVT-AV1 specific options
            svt_params = f"tune=0"  # PSNR tuning
            if config.thread_count > 0:
                svt_params += f":lp={config.thread_count}"
            args.extend(["-svtav1-params", svt_params])
            
        elif config.target_codec == CodecType.HEVC:
            # x265 settings
            args.extend(["-preset", "medium"])
            
            if config.codec_mode == CodecMode.CRF:
                args.extend(["-crf", str(config.crf)])
            elif config.codec_mode in (CodecMode.VBR, CodecMode.CBR):
                bitrate_k = int(config.target_bitrate_mbps * 1000)
                args.extend(["-b:v", f"{bitrate_k}k"])
            
            if config.thread_count > 0:
                args.extend(["-threads", str(config.thread_count)])
            
        elif config.target_codec == CodecType.H264:
            # x264 settings
            args.extend(["-preset", "medium"])
            
            if config.codec_mode == CodecMode.CRF:
                args.extend(["-crf", str(config.crf)])
            elif config.codec_mode in (CodecMode.VBR, CodecMode.CBR):
                bitrate_k = int(config.target_bitrate_mbps * 1000)
                args.extend(["-b:v", f"{bitrate_k}k"])
            
            if config.thread_count > 0:
                args.extend(["-threads", str(config.thread_count)])
        
        elif config.target_codec == CodecType.VP9:
            # VP9 settings
            if config.codec_mode == CodecMode.CRF:
                args.extend(["-crf", str(config.crf)])
                args.extend(["-b:v", "0"])
            elif config.codec_mode in (CodecMode.VBR, CodecMode.CBR):
                bitrate_k = int(config.target_bitrate_mbps * 1000)
                args.extend(["-b:v", f"{bitrate_k}k"])
            
            if config.thread_count > 0:
                args.extend(["-threads", str(config.thread_count)])
        
        # Common output settings
        args.extend(["-pix_fmt", config.pixel_format])
        
        return args
    
    def compress(
        self,
        input_path: str,
        output_path: str,
        config: Optional[CompressionConfig] = None,
    ) -> bool:
        """
        Compress a video.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            config: Compression configuration
            
        Returns:
            True if successful, False otherwise
        """
        config = config or CompressionConfig()
        
        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()
        
        if not input_path.exists():
            logger.error(f"Input file does not exist: {input_path}")
            return False
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting compression: {input_path} -> {output_path}")
        logger.info(f"Codec: {config.target_codec.value}, Mode: {config.codec_mode.value}")
        
        # Build ffmpeg command
        encoder_args = self._build_encoder_args(config)
        
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", str(input_path),
            *encoder_args,
            "-c:a", "aac",  # Re-encode audio to AAC
            "-b:a", "128k",
            str(output_path)
        ]
        
        logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Compression failed: {result.stderr}")
                return False
            
            if not output_path.exists():
                logger.error("Output file was not created")
                return False
            
            # Log compression ratio
            input_size = input_path.stat().st_size
            output_size = output_path.stat().st_size
            ratio = input_size / output_size if output_size > 0 else 0
            
            logger.info(f"Compression completed: {output_path}")
            logger.info(f"Size: {input_size/1024/1024:.2f}MB -> {output_size/1024/1024:.2f}MB (ratio: {ratio:.2f}x)")
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Compression timed out")
            return False
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return False
    
    def compress_with_vmaf_target(
        self,
        input_path: str,
        output_path: str,
        vmaf_threshold: float = 90.0,
        target_codec: CodecType = CodecType.AV1,
        codec_mode: CodecMode = CodecMode.CRF,
        target_bitrate_mbps: float = 5.0,
    ) -> bool:
        """
        Compress video targeting a specific VMAF score.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            vmaf_threshold: Target VMAF score (0-100)
            target_codec: Codec to use
            codec_mode: Encoding mode
            target_bitrate_mbps: Target bitrate (for VBR/CBR modes)
            
        Returns:
            True if successful, False otherwise
        """
        # Estimate CRF for target VMAF
        crf = self._estimate_crf_for_vmaf(target_codec, vmaf_threshold)
        
        config = CompressionConfig(
            target_codec=target_codec,
            codec_mode=codec_mode,
            crf=crf,
            target_bitrate_mbps=target_bitrate_mbps,
            vmaf_threshold=vmaf_threshold,
        )
        
        logger.info(f"Targeting VMAF {vmaf_threshold} with estimated CRF {crf}")
        
        return self.compress(input_path, output_path, config)


# Convenience function
def compress_video(
    input_path: str,
    output_path: str,
    vmaf_threshold: float = 90.0,
    target_codec: str = "av1",
    codec_mode: str = "CRF",
    target_bitrate_mbps: float = 5.0,
) -> bool:
    """
    Convenience function to compress a video.
    
    Args:
        input_path: Path to input video
        output_path: Path for output video
        vmaf_threshold: Target VMAF score (0-100)
        target_codec: Codec to use ("av1", "hevc", "h264", "vp9")
        codec_mode: Encoding mode ("CRF", "VBR", "CBR")
        target_bitrate_mbps: Target bitrate for VBR/CBR modes
        
    Returns:
        True if successful, False otherwise
    """
    try:
        codec = CodecType(target_codec.lower())
    except ValueError:
        logger.error(f"Invalid codec: {target_codec}")
        return False
    
    try:
        mode = CodecMode(codec_mode.upper())
    except ValueError:
        logger.error(f"Invalid codec mode: {codec_mode}")
        return False
    
    encoder = SVTAV1Encoder()
    return encoder.compress_with_vmaf_target(
        input_path,
        output_path,
        vmaf_threshold=vmaf_threshold,
        target_codec=codec,
        codec_mode=mode,
        target_bitrate_mbps=target_bitrate_mbps,
    )
