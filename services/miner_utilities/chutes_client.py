import asyncio

import aiohttp
from loguru import logger
from vidaio_subnet_core import CONFIG


CHUTES_BASE_URL = "https://{slug}.chutes.ai"


async def _call_chute(slug: str, payload: dict) -> dict | None:
    base_url = CHUTES_BASE_URL.format(slug=slug)
    url = f"{base_url}/process"
    config = CONFIG.chutes

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=config.request_timeout),
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success"):
                        return result
                    logger.error(f"Chute returned error: {result.get('error')}")
                    return None
                logger.error(f"Chute HTTP error: {response.status}")
                return None
    except asyncio.TimeoutError:
        logger.error(f"Chute request timed out after {config.request_timeout}s")
        return None
    except Exception as e:
        logger.error(f"Chute request failed: {e}")
        return None


async def call_chute_upscaling(
    video_url: str,
    task_type: str,
    upload_url: str,
) -> str | None:
    config = CONFIG.chutes
    result = await _call_chute(
        slug=config.upscaling_slug,
        payload={
            "video_url": video_url,
            "task_type": task_type,
            "upload_url": upload_url,
        },
    )
    if result:
        return result.get("output_url")
    return None


async def call_chute_compression(
    video_url: str,
    vmaf_threshold: float,
    target_codec: str,
    codec_mode: str,
    target_bitrate: float,
    upload_url: str,
) -> str | None:
    config = CONFIG.chutes
    result = await _call_chute(
        slug=config.compression_slug,
        payload={
            "video_url": video_url,
            "vmaf_threshold": vmaf_threshold,
            "target_codec": target_codec,
            "codec_mode": codec_mode,
            "target_bitrate": target_bitrate,
            "upload_url": upload_url,
        },
    )
    if result:
        return result.get("output_url")
    return None
