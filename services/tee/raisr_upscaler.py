"""
Intel RAISR-based Video Upscaling

This module provides CPU-based video upscaling using Intel's Video Super
Resolution Library (RAISR - Rapid and Accurate Image Super Resolution).

RAISR is designed for:
- High-quality upscaling without GPU requirements
- Efficient CPU-based processing suitable for SGX enclaves
- Real-time or near-real-time super resolution

Reference: https://github.com/OpenVisualCloud/Video-Super-Resolution-Library
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class UpscaleMode(Enum):
    """Supported upscaling modes."""
    HD_TO_4K = "HD24K"      # 1080p -> 4K (2x scale)
    SD_TO_HD = "SD2HD"      # 480p/720p -> 1080p (2x scale)
    SD_TO_4K = "SD24K"      # 480p -> 4K (4x scale, two passes)
    FOUR_K_TO_8K = "4K28K"  # 4K -> 8K (2x scale)


@dataclass
class UpscaleConfig:
    """Configuration for RAISR upscaling."""
    scale_factor: int = 2
    thread_count: int = 64
    filter_folder: str = "filters_2x/filters_lowres"
    output_codec: str = "libx264"
    output_crf: int = 21
    pixel_format: str = "yuv420p"
    

class RAISRUpscaler:
    """
    Video upscaler using Intel RAISR (Rapid and Accurate Image Super Resolution).
    
    This implementation uses ffmpeg with the RAISR filter for CPU-based
    super resolution that can run inside SGX enclaves.
    """
    
    # Default RAISR Docker image
    RAISR_DOCKER_IMAGE = "raisr/raisr-xeon:ubuntu-22.04"
    
    # Filter folders for different scale factors
    FILTER_FOLDERS = {
        2: "filters_2x/filters_lowres",
        4: "filters_4x/filters_lowres",  # For 4x, typically do 2x twice
    }
    
    def __init__(
        self,
        docker_image: Optional[str] = None,
        use_docker: bool = True,
        native_ffmpeg_path: Optional[str] = None,
    ):
        """
        Initialize RAISR upscaler.
        
        Args:
            docker_image: Docker image to use. Defaults to official RAISR image.
            use_docker: Whether to use Docker. Set False if RAISR is installed natively.
            native_ffmpeg_path: Path to ffmpeg with RAISR filter compiled in.
        """
        self.docker_image = docker_image or self.RAISR_DOCKER_IMAGE
        self.use_docker = use_docker
        self.native_ffmpeg_path = native_ffmpeg_path or "ffmpeg"
        
        # Verify RAISR is available
        if use_docker:
            self._verify_docker_available()
        else:
            self._verify_native_ffmpeg()
    
    def _verify_docker_available(self) -> None:
        """Check if Docker is available."""
        try:
            subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Docker not available - RAISR upscaling may fail")
    
    def _verify_native_ffmpeg(self) -> None:
        """Check if ffmpeg with RAISR filter is available."""
        try:
            result = subprocess.run(
                [self.native_ffmpeg_path, "-filters"],
                capture_output=True,
                text=True
            )
            if "raisr" not in result.stdout.lower():
                logger.warning("ffmpeg does not have RAISR filter - upscaling may fail")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("ffmpeg not available - RAISR upscaling may fail")
    
    def _get_config_for_mode(self, mode: UpscaleMode) -> UpscaleConfig:
        """Get upscaling configuration for the given mode."""
        if mode == UpscaleMode.SD_TO_4K:
            # 4x upscaling requires two passes
            return UpscaleConfig(
                scale_factor=4,
                thread_count=64,
                filter_folder="filters_2x/filters_lowres",
            )
        else:
            return UpscaleConfig(
                scale_factor=2,
                thread_count=64,
                filter_folder="filters_2x/filters_lowres",
            )
    
    def upscale(
        self,
        input_path: str,
        output_path: str,
        mode: UpscaleMode = UpscaleMode.HD_TO_4K,
        config: Optional[UpscaleConfig] = None,
    ) -> bool:
        """
        Upscale a video using RAISR.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            mode: Upscaling mode (determines scale factor and settings)
            config: Optional custom configuration
            
        Returns:
            True if successful, False otherwise
        """
        config = config or self._get_config_for_mode(mode)
        
        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()
        
        if not input_path.exists():
            logger.error(f"Input file does not exist: {input_path}")
            return False
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting RAISR upscaling: {input_path} -> {output_path}")
        logger.info(f"Mode: {mode.value}, Scale factor: {config.scale_factor}")
        
        if mode == UpscaleMode.SD_TO_4K:
            # 4x upscaling: do two 2x passes
            return self._upscale_4x(input_path, output_path, config)
        else:
            # Single 2x pass
            return self._upscale_2x(input_path, output_path, config)
    
    def _upscale_2x(
        self,
        input_path: Path,
        output_path: Path,
        config: UpscaleConfig
    ) -> bool:
        """Perform single 2x upscaling pass."""
        if self.use_docker:
            return self._upscale_docker(input_path, output_path, config)
        else:
            return self._upscale_native(input_path, output_path, config)
    
    def _upscale_4x(
        self,
        input_path: Path,
        output_path: Path,
        config: UpscaleConfig
    ) -> bool:
        """Perform 4x upscaling (two 2x passes)."""
        # Create temporary file for intermediate result
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            intermediate_path = Path(tmp.name)
        
        try:
            # First pass: 2x
            logger.info("4x upscaling: Pass 1 of 2 (first 2x)")
            if not self._upscale_2x(input_path, intermediate_path, config):
                return False
            
            # Second pass: 2x
            logger.info("4x upscaling: Pass 2 of 2 (second 2x)")
            if not self._upscale_2x(intermediate_path, output_path, config):
                return False
            
            return True
            
        finally:
            # Clean up intermediate file
            if intermediate_path.exists():
                intermediate_path.unlink()
    
    def _upscale_docker(
        self,
        input_path: Path,
        output_path: Path,
        config: UpscaleConfig
    ) -> bool:
        """Upscale using Docker container."""
        # Mount the parent directory containing input file
        mount_dir = input_path.parent
        input_name = input_path.name
        output_name = output_path.name
        
        # If output is in a different directory, we need to handle that
        if output_path.parent != mount_dir:
            # Copy input to output directory and work there
            work_dir = output_path.parent
            work_input = work_dir / input_name
            shutil.copy2(input_path, work_input)
            mount_dir = work_dir
            input_path = work_input
            cleanup_input = True
        else:
            cleanup_input = False
        
        cmd = [
            "docker", "run", "--rm",
            "--user", "root",
            "-v", f"{mount_dir}:/data",
            self.docker_image,
            "-y",
            "-i", f"/data/{input_path.name}",
            "-vf", f"raisr=threadcount={config.thread_count}:filterfolder={config.filter_folder}",
            "-pix_fmt", config.pixel_format,
            "-c:v", config.output_codec,
            "-crf", str(config.output_crf),
            f"/data/{output_name}"
        ]
        
        logger.debug(f"Running RAISR Docker command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                logger.error(f"RAISR Docker failed: {result.stderr}")
                return False
            
            if not output_path.exists():
                logger.error("Output file was not created")
                return False
            
            logger.info(f"RAISR upscaling completed: {output_path}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("RAISR upscaling timed out")
            return False
        except Exception as e:
            logger.error(f"RAISR upscaling failed: {e}")
            return False
        finally:
            if cleanup_input and input_path.exists():
                input_path.unlink()
    
    def _upscale_native(
        self,
        input_path: Path,
        output_path: Path,
        config: UpscaleConfig
    ) -> bool:
        """Upscale using natively installed ffmpeg with RAISR filter."""
        cmd = [
            self.native_ffmpeg_path,
            "-y",
            "-i", str(input_path),
            "-vf", f"raisr=threadcount={config.thread_count}:filterfolder={config.filter_folder}",
            "-pix_fmt", config.pixel_format,
            "-c:v", config.output_codec,
            "-crf", str(config.output_crf),
            str(output_path)
        ]
        
        logger.debug(f"Running native ffmpeg RAISR command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            if result.returncode != 0:
                logger.error(f"Native ffmpeg RAISR failed: {result.stderr}")
                return False
            
            if not output_path.exists():
                logger.error("Output file was not created")
                return False
            
            logger.info(f"RAISR upscaling completed: {output_path}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("RAISR upscaling timed out")
            return False
        except Exception as e:
            logger.error(f"RAISR upscaling failed: {e}")
            return False


# Convenience function
def upscale_video(
    input_path: str,
    output_path: str,
    mode: str = "HD24K",
    use_docker: bool = True,
) -> bool:
    """
    Convenience function to upscale a video using RAISR.
    
    Args:
        input_path: Path to input video
        output_path: Path for output video
        mode: Upscaling mode ("HD24K", "SD2HD", "SD24K", "4K28K")
        use_docker: Whether to use Docker container
        
    Returns:
        True if successful, False otherwise
    """
    try:
        upscale_mode = UpscaleMode(mode)
    except ValueError:
        logger.error(f"Invalid upscaling mode: {mode}")
        return False
    
    upscaler = RAISRUpscaler(use_docker=use_docker)
    return upscaler.upscale(input_path, output_path, upscale_mode)
