"""
TEE Video Processor

This module provides the main video processing pipeline that runs inside
the Intel SGX enclave. It handles:

1. Receiving encrypted task payloads
2. Decrypting inside the enclave
3. Downloading video via HTTPS
4. Processing (upscaling or compression)
5. Uploading encrypted results
6. Returning only status (no URLs)

All sensitive operations happen inside the enclave, ensuring miners
never see video URLs or content.
"""

import os
import time
import hashlib
import tempfile
import aiohttp
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from loguru import logger

from vidaio_subnet_core.tee.tee_crypto import (
    TEECrypto,
    EncryptedPayload,
    TaskPayload,
    StorageCredentials,
)
from .raisr_upscaler import RAISRUpscaler, UpscaleMode
from .svt_av1_encoder import SVTAV1Encoder, CodecType, CodecMode


@dataclass
class ProcessingResult:
    """Result of video processing."""
    success: bool
    processing_time_seconds: float
    error_message: Optional[str] = None
    result_checksum: str = ""
    

class TEEVideoProcessor:
    """
    Secure video processor running inside SGX enclave.
    
    This class handles the complete video processing pipeline:
    1. Decrypt task payload (URL, credentials, params)
    2. Download video securely
    3. Process video (upscale or compress)
    4. Upload result securely
    5. Return status (never URLs)
    """
    
    def __init__(
        self,
        crypto: Optional[TEECrypto] = None,
        work_dir: Optional[str] = None,
    ):
        """
        Initialize TEE video processor.
        
        Args:
            crypto: TEE crypto instance for decryption
            work_dir: Working directory for temporary files
        """
        self.crypto = crypto or TEECrypto()
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.gettempdir()) / "tee_video"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        self.upscaler = RAISRUpscaler(use_docker=True)
        self.encoder = SVTAV1Encoder()
        
        logger.info(f"TEE Video Processor initialized, work_dir: {self.work_dir}")
    
    async def process_encrypted_task(
        self,
        encrypted_payload: EncryptedPayload,
        session_key: Optional[bytes] = None,
    ) -> ProcessingResult:
        """
        Process an encrypted video task.
        
        This is the main entry point. The encrypted payload contains
        all sensitive data (URLs, credentials) that will be decrypted
        only inside this enclave.
        
        Args:
            encrypted_payload: Encrypted task data from validator
            session_key: Session key for decryption (or looked up by ID)
            
        Returns:
            ProcessingResult with status and checksum (never URLs)
        """
        start_time = time.time()
        
        try:
            # Step 1: Decrypt payload
            logger.info("Decrypting task payload inside enclave...")
            try:
                task = self.crypto.decrypt_payload(encrypted_payload, session_key)
            except Exception as e:
                logger.error(f"Failed to decrypt payload: {e}")
                return ProcessingResult(
                    success=False,
                    processing_time_seconds=time.time() - start_time,
                    error_message=f"Decryption failed: {str(e)}"
                )
            
            logger.info(f"Task decrypted: type={task.task_type}")
            
            # Step 2: Process the task
            if task.task_type == "upscaling":
                result = await self._process_upscaling(task)
            elif task.task_type == "compression":
                result = await self._process_compression(task)
            else:
                return ProcessingResult(
                    success=False,
                    processing_time_seconds=time.time() - start_time,
                    error_message=f"Unknown task type: {task.task_type}"
                )
            
            result.processing_time_seconds = time.time() - start_time
            return result
            
        except Exception as e:
            logger.error(f"Task processing failed: {e}")
            return ProcessingResult(
                success=False,
                processing_time_seconds=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _process_upscaling(self, task: TaskPayload) -> ProcessingResult:
        """Process an upscaling task."""
        # Create unique temp files
        input_path = self.work_dir / f"input_{os.urandom(8).hex()}.mp4"
        output_path = self.work_dir / f"output_{os.urandom(8).hex()}.mp4"
        
        try:
            # Download video
            logger.info("Downloading video inside enclave...")
            if not await self._download_video(task.reference_video_url, input_path):
                return ProcessingResult(
                    success=False,
                    processing_time_seconds=0,
                    error_message="Failed to download video"
                )
            
            # Get upscaling mode
            mode_str = task.task_params.get("task_type", "HD24K")
            try:
                mode = UpscaleMode(mode_str)
            except ValueError:
                mode = UpscaleMode.HD_TO_4K
            
            # Upscale video
            logger.info(f"Upscaling video with mode {mode.value}...")
            success = self.upscaler.upscale(
                str(input_path),
                str(output_path),
                mode=mode
            )
            
            if not success or not output_path.exists():
                return ProcessingResult(
                    success=False,
                    processing_time_seconds=0,
                    error_message="Upscaling failed"
                )
            
            # Compute checksum
            checksum = TEECrypto.compute_file_checksum(str(output_path))
            
            # Upload result
            logger.info("Uploading result inside enclave...")
            upload_success = await self._upload_video(
                output_path,
                task.storage_credentials,
                task.result_object_key
            )
            
            if not upload_success:
                return ProcessingResult(
                    success=False,
                    processing_time_seconds=0,
                    error_message="Failed to upload result"
                )
            
            return ProcessingResult(
                success=True,
                processing_time_seconds=0,  # Will be set by caller
                result_checksum=checksum
            )
            
        finally:
            # Clean up temp files
            self._cleanup_files([input_path, output_path])
    
    async def _process_compression(self, task: TaskPayload) -> ProcessingResult:
        """Process a compression task."""
        # Create unique temp files
        input_path = self.work_dir / f"input_{os.urandom(8).hex()}.mp4"
        output_path = self.work_dir / f"output_{os.urandom(8).hex()}.mp4"
        
        try:
            # Download video
            logger.info("Downloading video inside enclave...")
            if not await self._download_video(task.reference_video_url, input_path):
                return ProcessingResult(
                    success=False,
                    processing_time_seconds=0,
                    error_message="Failed to download video"
                )
            
            # Get compression params
            vmaf_threshold = task.task_params.get("vmaf_threshold", 90.0)
            target_codec = task.task_params.get("target_codec", "av1")
            codec_mode = task.task_params.get("codec_mode", "CRF")
            target_bitrate = task.task_params.get("target_bitrate", 5.0)
            
            # Parse codec type
            try:
                codec = CodecType(target_codec.lower())
            except ValueError:
                codec = CodecType.AV1
            
            try:
                mode = CodecMode(codec_mode.upper())
            except ValueError:
                mode = CodecMode.CRF
            
            # Compress video
            logger.info(f"Compressing video with {codec.value} @ VMAF {vmaf_threshold}...")
            success = self.encoder.compress_with_vmaf_target(
                str(input_path),
                str(output_path),
                vmaf_threshold=vmaf_threshold,
                target_codec=codec,
                codec_mode=mode,
                target_bitrate_mbps=target_bitrate
            )
            
            if not success or not output_path.exists():
                return ProcessingResult(
                    success=False,
                    processing_time_seconds=0,
                    error_message="Compression failed"
                )
            
            # Compute checksum
            checksum = TEECrypto.compute_file_checksum(str(output_path))
            
            # Upload result
            logger.info("Uploading result inside enclave...")
            upload_success = await self._upload_video(
                output_path,
                task.storage_credentials,
                task.result_object_key
            )
            
            if not upload_success:
                return ProcessingResult(
                    success=False,
                    processing_time_seconds=0,
                    error_message="Failed to upload result"
                )
            
            return ProcessingResult(
                success=True,
                processing_time_seconds=0,  # Will be set by caller
                result_checksum=checksum
            )
            
        finally:
            # Clean up temp files
            self._cleanup_files([input_path, output_path])
    
    async def _download_video(self, url: str, output_path: Path) -> bool:
        """
        Download video from URL.
        
        This happens inside the enclave - the URL is only known here.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Download failed with status {response.status}")
                        return False
                    
                    with open(output_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(1024 * 1024):
                            f.write(chunk)
            
            logger.info(f"Downloaded video: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    async def _upload_video(
        self,
        file_path: Path,
        credentials: StorageCredentials,
        object_key: str
    ) -> bool:
        """
        Upload video to storage.
        
        This happens inside the enclave - credentials are only known here.
        Uses S3-compatible API (MinIO).
        """
        try:
            # Use boto3 for S3-compatible upload
            import boto3
            from botocore.config import Config
            
            s3_client = boto3.client(
                's3',
                endpoint_url=credentials.endpoint_url,
                aws_access_key_id=credentials.access_key,
                aws_secret_access_key=credentials.secret_key,
                region_name=credentials.region or 'us-east-1',
                config=Config(signature_version='s3v4')
            )
            
            # Upload file
            s3_client.upload_file(
                str(file_path),
                credentials.bucket,
                object_key
            )
            
            logger.info(f"Uploaded video to {credentials.bucket}/{object_key}")
            return True
            
        except ImportError:
            logger.error("boto3 not available - trying aiohttp upload")
            return await self._upload_video_http(file_path, credentials, object_key)
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            return False
    
    async def _upload_video_http(
        self,
        file_path: Path,
        credentials: StorageCredentials,
        object_key: str
    ) -> bool:
        """
        Fallback HTTP upload for S3-compatible storage.
        """
        try:
            # Construct presigned URL or use multipart upload
            # This is a simplified implementation
            url = f"{credentials.endpoint_url}/{credentials.bucket}/{object_key}"
            
            async with aiohttp.ClientSession() as session:
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                async with session.put(
                    url,
                    data=data,
                    headers={
                        "Content-Type": "video/mp4",
                    }
                ) as response:
                    if response.status in (200, 201):
                        logger.info(f"HTTP upload successful: {object_key}")
                        return True
                    else:
                        logger.error(f"HTTP upload failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"HTTP upload failed: {e}")
            return False
    
    def _cleanup_files(self, paths: list) -> None:
        """Clean up temporary files."""
        for path in paths:
            try:
                if path.exists():
                    path.unlink()
                    logger.debug(f"Cleaned up: {path}")
            except Exception as e:
                logger.warning(f"Failed to clean up {path}: {e}")
