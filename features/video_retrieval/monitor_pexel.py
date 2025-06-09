import asyncio
import os
import yaml
from typing import List, Dict, Optional
import httpx
from loguru import logger
import random
import requests
import time
import yaml
import shutil
import json
from pathlib import Path
from modules.config import config

script_dir = os.path.dirname(os.path.abspath(__file__))

def get_pexels_all_vids(
    min_len: int,
    max_len: int,
    width: int = None,
    height: int = None,
    task_type: str = None,
    delay: float = 1.0
) -> List[int]:
    RESOLUTIONS = {
        "HD24K": (3840, 2160),
        "SD2HD": (1920, 1080),
        "SD24K": (3840, 2160),
        "4K28K": (7680, 4320),
        "HD28K": (7680, 4320),
    }

    width, height = (width, height) if width and height else RESOLUTIONS.get(task_type, (3840, 2160))
    api_key = config["PEXELS_API_KEY"]
    if not api_key:
        logger.error("Missing Pexels API Key")
        return []

    headers = {"Authorization": api_key}
    query_list = ["nature"]  # add more keywords if needed
    per_page = 80

    logger.info(f"🔎 Fetching videos for {task_type}: {width}x{height}, {min_len}-{max_len}s")

    cache_file = Path(f"cache/pexels_{task_type}.json")
    cache_data = {}
    if cache_file.exists():
        with open(cache_file, "r") as f:
            cache_data = json.load(f)

    valid_video_ids = cache_data.get("video_ids", [])
    last_page = cache_data.get("last_page", 1)

    MAX_RETRIES = 3
    

    page = last_page
    while True:
        params = {
            "query": query_list[0],
            "per_page": per_page,
            "page": page,
            "size": "large",
        }

        if task_type == "SD2HD":
            del params["size"]

        retries = 0
        while retries < MAX_RETRIES:
            try:
                response = requests.get("https://api.pexels.com/videos/search", headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                break  # success → exit retry loop
            except requests.exceptions.RequestException as e:
                logger.error(f"[ERROR] Page {page} failed: {e}")
                retries += 1
                time.sleep(5)            
        else:
            logger.warning(f"[SKIP] Page {page} skipped after {MAX_RETRIES} retries")
        if not data.get("videos"):
            logger.info(f"[END] No more videos on page {page}")
            break

        matched = 0
        for video in data["videos"]:
            if min_len <= video["duration"] <= max_len and video["width"] == width and video["height"] == height:
                if video["id"] in valid_video_ids:
                    matched += 1
                    continue
                logger.info(f"[New Video] {video["id"]}")
                valid_video_ids.append(video["id"])

        logger.info(f"[Page {page}] Matched {matched} / {len(data['videos'])}")

        # Update and save cache
        with open(cache_file, "w") as f:
            json.dump({
                "video_ids": valid_video_ids,
                "last_page": page + 1
            }, f, indent=2)

        if len(data["videos"]) < per_page:
            break  # Last page
        page += 1
        time.sleep(delay)
            

    logger.info(f"✅ Total matching videos: {len(valid_video_ids)}")
    return valid_video_ids

def save_video_cache(data, task_type):
    Path("cache").mkdir(exist_ok=True)
    with open(f"cache/pexels_{task_type}.json", "w") as f:
        json.dump(data, f)

def load_video_cache(task_type):
    cache_path = Path(f"cache/pexels_{task_type}.json")
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)
    return None

def main():
    """Main function to manage video processing and synthetic queue handling."""
    
    logger.info("Starting worker...")

    video_min_len = 10
    video_max_len = 40
    
    for task_type in ["HD24K", "SD2HD", "SD24K", "4K28K", "HD28K"]:
        logger.info(f"Fetching all pexels videos for task type: {task_type}")
        get_pexels_all_vids(
            min_len=video_min_len,
            max_len=video_max_len,
            task_type=task_type
        )
        time.sleep(5)   

if __name__ == "__main__":
    main()