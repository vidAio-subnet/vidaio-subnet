import uuid
import json
import asyncio
from pathlib import Path
import aiohttp
from fastapi import HTTPException
from loguru import logger
from vidaio_subnet_core import CONFIG
import os
import time
from typing import Optional
from dataclasses import dataclass


@dataclass
class HTTPClientPool:
    """
    Connection pool manager for HTTP requests.

    Reduces per-request latency by reusing TCP connections
    and SSL sessions across multiple downloads/uploads.
    """
    connector: Optional[aiohttp.TCPConnector] = None
    session: Optional[aiohttp.ClientSession] = None
    _lock: asyncio.Lock = None

    def __post_init__(self):
        if self._lock is None:
            self._lock = asyncio.Lock()

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create a shared ClientSession with connection pooling."""
        async with self._lock:
            if self.session is None or self.session.closed:
                self.connector = aiohttp.TCPConnector(
                    limit=20,                    # Max total connections
                    limit_per_host=10,           # Max connections per host
                    enable_cleanup_closed=True,  # Clean up closed connections
                    force_close=False,           # Keep connections alive
                    ttl_dns_cache=300,           # DNS cache TTL (5 min)
                    use_dns_cache=True,          # Enable DNS caching
                )
                timeout = aiohttp.ClientTimeout(
                    total=300,                   # Total timeout 5 min
                    connect=30,                  # Connection timeout 30s
                    sock_read=60                 # Socket read timeout 60s
                )
                self.session = aiohttp.ClientSession(
                    connector=self.connector,
                    timeout=timeout,
                    headers={"User-Agent": "SN85-Miner/1.0"}
                )
            return self.session

    async def close(self):
        """Close the session and connector."""
        async with self._lock:
            if self.session and not self.session.closed:
                await self.session.close()
                self.session = None
            if self.connector:
                await self.connector.close()
                self.connector = None


# Global connection pool instance
_http_pool: Optional[HTTPClientPool] = None


def get_http_pool() -> HTTPClientPool:
    """Get the global HTTP connection pool."""
    global _http_pool
    if _http_pool is None:
        _http_pool = HTTPClientPool()
    return _http_pool


async def download_video(video_url: str, session: Optional[aiohttp.ClientSession] = None) -> Path:
    """
    Downloads a video from the given URL with retries and redirect handling.

    Uses connection pooling when session is provided (recommended for batch
    operations). Falls back to creating a new session for single downloads.

    Args:
        video_url (str): The URL of the video to download.
        session (Optional[aiohttp.ClientSession]): Reusable session for connection pooling.
            If None, uses the global connection pool (recommended).

    Returns:
        Path: The local path of the downloaded video.

    Raises:
        HTTPException: If the download fails.
    """
    try:
        video_dir = Path(__file__).parent.parent / "upscaling" / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        # Generate a unique filename
        filename = f"{uuid.uuid4()}.mp4"
        output_path = video_dir / filename

        logger.info(f"Downloading video from {video_url} to {output_path}")
        start_time = time.time()

        # Use provided session or get one from the pool
        if session is None:
            pool = get_http_pool()
            session = await pool.get_session()

        async with session.get(video_url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download video. HTTP status: {response.status}")

                # Write the content to the temp file in chunks
                with open(output_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(3 * 1024 * 1024):  # 3 MB chunks
                        f.write(chunk)
        elapsed_time = time.time() - start_time
        logger.info(f"Chunk download time: {elapsed_time:.2f} seconds")

        # Verify the file was successfully downloaded
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise Exception(f"Download failed or file is empty: {output_path}")

        logger.info(f"File successfully downloaded to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to download video from {video_url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading video: {str(e)}")


async def video_upscaler(payload_url: str, task_type: str, session: Optional[aiohttp.ClientSession] = None) -> str | None:
    """
    Sends a video file path to the upscaling service and retrieves the processed video path.

    Uses connection pooling when session is provided for lower latency on
    repeated internal service calls.

    Args:
        payload_url (str): The url of the video to be upscaled.
        task_type (str): The type of upscaling task.
        session (Optional[aiohttp.ClientSession]): Reusable session for connection pooling.
            If None, uses the global connection pool.

    Returns:
        str | None: The path of the upscaled video or None if an error occurs.
    """
    url = f"http://{CONFIG.video_upscaler.host}:{CONFIG.video_upscaler.port}/upscale-video"
    headers = {"Content-Type": "application/json"}
    data = {
        "payload_url": payload_url,
        "task_type": task_type,
    }

    # Use provided session or get one from the pool
    if session is None:
        pool = get_http_pool()
        session = await pool.get_session()

    async with session.post(url, headers=headers, data=json.dumps(data)) as response:
        if response.status == 200:
            result = await response.json()
            uploaded_video_url = result.get("uploaded_video_url")
            if uploaded_video_url is None:
                logger.info("🩸 Received None response from video upscaler 🩸")
                return None
            logger.info("✈️ Received response from video upscaler correctly ✈️")
            return uploaded_video_url
        logger.error(f"Upscaling service error: {response.status}")
        return None

async def video_compressor(payload_url: str, vmaf_threshold: float, target_codec: str = "av1",
                          codec_mode: str = "CRF", target_bitrate: float = 10.0,
                          session: Optional[aiohttp.ClientSession] = None) -> str | None:
    """
    Sends a video file path to the compression service and retrieves the processed video path.

    Uses connection pooling when session is provided for lower latency on
    repeated internal service calls.

    Args:
        payload_url (str): The URL of the video to be compressed.
        vmaf_threshold (float): The VMAF threshold for quality control.
        target_codec (str): The target codec for compression (default: "av1").
        codec_mode (str): Codec mode - CBR, VBR, or CRF (default: "CRF").
        target_bitrate (float): Target bitrate in Mbps (default: 10.0).
        session (Optional[aiohttp.ClientSession]): Reusable session for connection pooling.
            If None, uses the global connection pool.

    Returns:
        str | None: The URL of the compressed video or None if an error occurs.
    """
    url = f"http://{CONFIG.video_compressor.host}:{CONFIG.video_compressor.port}/compress-video"
    headers = {"Content-Type": "application/json"}
    data = {
        "payload_url": payload_url,
        "vmaf_threshold": vmaf_threshold,
        "target_codec": target_codec,
        "codec_mode": codec_mode,
        "target_bitrate": target_bitrate,
    }
    logger.info(f"🎬 Sending compression request: VMAF={vmaf_threshold}, Codec={target_codec}, Mode={codec_mode}, Bitrate={target_bitrate} Mbps")

    # Use provided session or get one from the pool
    if session is None:
        pool = get_http_pool()
        session = await pool.get_session()

    async with session.post(url, headers=headers, data=json.dumps(data)) as response:
        if response.status == 200:
            result = await response.json()
            uploaded_video_url = result.get("uploaded_video_url")
            if uploaded_video_url is None:
                logger.info("🩸 Received None response from video compressor 🩸")
                return None
            logger.info("✈️ Received response from video compressor correctly ✈️")
            return uploaded_video_url
        logger.error(f"Compression service error: {response.status}")
        return None