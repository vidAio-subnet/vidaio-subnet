from typing import List, Dict, Tuple, Optional
from moviepy.editor import VideoFileClip
import imagehash
import os
import time
from threading import Thread
from search.modules.search_config import search_config
from search.modules.hash_engine import video_to_phashes
from loguru import logger
from pymongo import MongoClient
import cv2
import asyncio
import sys
import random
from pathlib import Path
import requests
from tqdm import tqdm
import aiohttp
import json
import yaml
import aiohttp
import asyncio
from typing import List, Tuple
import numpy as np
from datetime import datetime

def download_video(
    vid: int,
    api_key : str,
    task_type : str,
) -> Optional[str]:
    if not api_key:
        raise ValueError("PEXELS_DOWNLOAD_API_KEY not found in environment variables")
    
    # API endpoint
    url = f"https://api.pexels.com/videos/videos/{vid}"
    headers = {"Authorization": api_key}
    
    video_id = f'{task_type}_{vid}'
    origin_path = Path(search_config['VIDEO_DIR']) / f"{video_id}_original.mp4"
    if origin_path.exists():
        logger.info(f"⚠️ Already existing: {origin_path}, deleted")
        return origin_path
    temp_path = Path(search_config['VIDEO_DIR']) / f"{video_id}_original_tmp.mp4"
    
    start_time = time.time()

    EXPECTED_RESOLUTIONS = {
        "SD2HD": (1920, 1080),
        "4K28K": (7680, 4320),
    }
    expected_width, expected_height = EXPECTED_RESOLUTIONS.get(task_type, (3840, 2160))
    logger.info(f"expected_width, expected_height: {expected_width, expected_height}")
    # Get video details
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise exception for bad status codes
    data = response.json()
    if "video_files" not in data:
        raise ValueError("No video found or API error")
        
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise exception for bad status codes
    data = response.json()
    if "video_files" not in data:
        raise ValueError("No video found or API error")
        
    # Get the highest resolution video file
    video_url = next(
        (video["link"] for video in data["video_files"] 
        if video["width"] == expected_width and video["height"] == expected_height), 
        None
    )
    
    if video_url is None:
        logger.info(f"video url is None: id: {vid}")
        return None
    
    # Download video with progress bar
    logger.info(f"\nDownloading video ID: {vid}")
    response = requests.get(video_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(temp_path, 'wb') as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=1024):
            size = f.write(chunk)
            pbar.update(size)
            
    logger.info(f"\nVideo downloaded successfully to: {temp_path}")
    elapsed_time = time.time() - start_time
    logger.info(f"Time taken to download video: {elapsed_time:.2f} seconds")

    logger.info("\nChecking video resolution...")
    video_clip = VideoFileClip(str(temp_path))
    actual_width, actual_height = video_clip.size
    
    if actual_width != expected_width or actual_height != expected_height:
        video_clip.close()
        error_msg = (f"Video resolution mismatch. Expected {expected_width}x{expected_height}, "
                    f"but got {actual_width}x{actual_height}")
        logger.info(error_msg)
        os.remove(temp_path)
        return None
    video_clip.close()
    os.rename(temp_path, origin_path)
    return origin_path

async def get_pexels_all_vids_async(
    min_len: int,
    max_len: int,
    cur_video_ids: List[int],
    all_new_video_ids: List[int],
    api_keys: List[str],
    last_page: int = 0,
    width: int = None,
    height: int = None,
    task_type: str = None,
    query: str = None,
    per_page: int = 80,
    delay: float = 1.0
) -> Tuple[List[int], List[int], int]:

    RESOLUTIONS = {
        "HD24K": (3840, 2160),
        "SD2HD": (1920, 1080),
        "SD24K": (3840, 2160),
        "4K28K": (7680, 4320),
        "HD28K": (7680, 4320),
    }

    width, height = (width, height) if width and height else RESOLUTIONS.get(task_type, (3840, 2160))

    logger.info(f"🔎 Fetching video ids for task_type: {task_type}, query: {query}, {width}x{height}, {min_len}-{max_len}s")

    new_video_ids = []
    matched_video_ids = []
    MAX_RETRIES = 3
    page = last_page
    api_key_index = 0

    async with aiohttp.ClientSession() as session:
        while True:
            api_key = api_keys[api_key_index]
            api_key_index = (api_key_index + 1) % len(api_keys)

            headers = {"Authorization": api_key}
            params = {
                "query": query,
                "per_page": per_page,
                "page": page,
                "size": "large",
            }

            if task_type == "SD2HD":
                params.pop("size", None)

            retries = 0
            data = {}
            while retries < MAX_RETRIES:
                try:
                    async with session.get("https://api.pexels.com/videos/search", headers=headers, params=params) as response:
                        if response.status != 200:
                            raise aiohttp.ClientError(f"HTTP {response.status}")
                        data = await response.json()
                        break  # success
                except Exception as e:
                    logger.info(f"Retries:{retries} [ERROR] Page {page} failed: {e}")
                    retries += 1
                    await asyncio.sleep(15 * retries)
            else:
                logger.info(f"[SKIP] Page {page} skipped after {MAX_RETRIES} retries")

            if not data.get("videos"):
                logger.info(f"[END] No more videos on page {page}")
                break

            matched = 0
            skipped = 0
            new_count = 0
            for video in data["videos"]:
                if video["id"] in cur_video_ids or video["id"] in all_new_video_ids:
                    matched += 1
                    matched_video_ids.append(video["id"])
                if min_len <= video["duration"] <= max_len and video["width"] == width and video["height"] == height:
                    new_video_ids.append(video["id"])
                    new_count += 1
                else:
                    skipped += 1

            if len(data["videos"]) < per_page:
                break  # Last page
            page += 1
            await asyncio.sleep(delay)

    logger.info(f"✅ task_type: {task_type}, query: {query}, Total new videos: {len(new_video_ids)}, matched videos: {len(matched_video_ids)}")
    return new_video_ids, matched_video_ids, page
    
class VideoDbUpdateService:
    def __init__(self):
        self.video_dir = search_config['VIDEO_DIR']
        self.test_video_dir = search_config['TEST_VIDEO_DIR']
        self.client = MongoClient(search_config['MONGO_URI'])
        self.db_name = search_config['DB_NAME']
        self.collection_name = search_config['COLLECTION_NAME']
        self.collection = self.client[self.db_name][self.collection_name]
        self.video_ids = []
        for doc in self.collection.find():
            try:
                self.video_ids.append(int(doc['filename'].split('_')[1]))
            except Exception as e:
                logger.info(f"Invalid filename format: {doc['filename']}")
                continue
        logger.info(f"✅ Loaded {len(self.video_ids)} videos from database")

    async def insert_video_to_db(self, origin_path: str):
        try:
            filename = os.path.basename(origin_path)
            cap = cv2.VideoCapture(origin_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            hashes, *_ = video_to_phashes(origin_path)

            video_info = {
                'filename': filename,
                'fps': fps,
                'width': width,
                'height': height,
                'frame_count': frame_count,
                'hashes': [str(h) for h in hashes],
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }

            self.collection.insert_one(video_info)
            logger.info(f"✅ Inserted video {filename} to db")
            
            cap.release()
        except Exception as e:
            logger.info(f"Failed to insert video to db: {str(e)}")
            return False
        return True

    async def fetch_video_ids(self, task_type: str, query_chunks: List[List[str]], api_keys_chunks: List[List[str]]):
        all_new_video_ids = []
        all_matched_video_ids = []
        async def process_query_chunk(chunk, api_keys_chunk):
            chunk_new_videos = []
            chunk_matched_videos = []
            chunk_pages = {}
            logger.info(f"task_type: {task_type}, processing query chunk: {chunk}")
            
            for query in chunk:
                new_video_ids, matched_video_ids, page = await get_pexels_all_vids_async(
                    cur_video_ids = self.video_ids,
                    all_new_video_ids = all_new_video_ids,
                    last_page = 0,
                    min_len = search_config['MIN_VIDEO_LEN'],
                    max_len = search_config['MAX_VIDEO_LEN'],
                    task_type = task_type,
                    query = query,
                    api_keys = api_keys_chunk
                )
                chunk_new_videos.extend(new_video_ids)
                chunk_matched_videos.extend(matched_video_ids)
                chunk_pages[f"{task_type}_{query}"] = page
            
            return chunk_new_videos, chunk_matched_videos, chunk_pages
        
        # Process chunks in parallel
        tasks = [process_query_chunk(chunk, api_keys_chunk) for chunk, api_keys_chunk in zip(query_chunks, api_keys_chunks)]
        results = await asyncio.gather(*tasks)
            
        # Combine results
        for chunk_new_videos, chunk_matched_videos, chunk_pages in results:
            all_new_video_ids.extend(chunk_new_videos)
            all_matched_video_ids.extend(chunk_matched_videos)
        all_new_video_ids = list(dict.fromkeys(all_new_video_ids))           
        all_matched_video_ids = list(dict.fromkeys(all_matched_video_ids))
        return all_new_video_ids, all_matched_video_ids

    async def download_and_insert_videos(self, task_type: str, video_chunks: List[List[int]], api_keys_chunks: List[List[str]]):
        def process_video_chunk(chunk, api_keys_chunk):
            chunk_results = []
            api_key_index = 0
            for video_id in chunk:
                try:
                    logger.info(f"Downloading video {video_id}")
                    origin_path = None
                    for retry in range(3):
                        try:
                            origin_path = download_video(video_id, api_keys_chunk[api_key_index], task_type)
                            api_key_index = (api_key_index + 1) % len(api_keys_chunk)
                            if origin_path is not None:
                                break
                        except Exception as e:
                            logger.info(f"❌ Retry {retry} Failed to download video {video_id}: {str(e)}")
                            time.sleep(15 * (retry + 1))
                    
                    if origin_path is None:
                        logger.info(f"❌ Failed to download video {video_id}: No origin path returned")
                        continue
                        
                    chunk_results.append((video_id, origin_path))
                except Exception as e:
                    logger.info(f"❌ Failed to download video {video_id}: {str(e)}")
                    continue
            return chunk_results

        threads = []
        for chunk, api_keys_chunk in zip(video_chunks, api_keys_chunks):
            thread = Thread(target=lambda: process_video_chunk(chunk, api_keys_chunk))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        count = 0
        for thread in threads:
            results = thread.result if hasattr(thread, 'result') else []
            for video_id, origin_path in results:
                if await self.insert_video_to_db(origin_path):
                    self.video_ids.append(video_id)
                    count += 1
                else:
                    if origin_path and os.path.exists(origin_path):
                        os.remove(origin_path)
                    logger.info(f"❌ Failed to insert video {video_id} to db")
        logger.info(f"✅ Inserted {count} videos to db")

    async def clean_up_videos(self, all_valid_video_ids: List[int]):
        missing_video_ids = [video_id for video_id in self.video_ids if video_id not in all_valid_video_ids]

        if missing_video_ids:
            logger.info(f"Found {len(missing_video_ids)} video IDs that are not in all_valid_video_ids")

            collection = self.client[self.db_name][self.collection_name]
            datetime_threshold = datetime.now() - timedelta(days=7)

            # Only fetch documents older than 7 days
            old_docs_cursor = collection.find({"created_at": {"$lt": datetime_threshold}})
            ids_to_delete = []

            for doc in old_docs_cursor:
                try:
                    video_id = int(doc['filename'].split('_')[1])
                except (IndexError, ValueError):
                    logger.info(f"Invalid filename format: {doc['filename']}")
                    continue

                if video_id not in all_valid_video_ids:
                    ids_to_delete.append(doc['_id'])

            if ids_to_delete:
                collection.delete_many({"_id": {"$in": ids_to_delete}})
                logger.info(f"✅ Deleted {len(ids_to_delete)} videos from db")
        else:
            logger.info("All video IDs are present in all_valid_video_ids")
    
    async def check_and_download(self):
        yaml_file_path = "services/video_scheduler/pexels_categories.yaml"
        query_list = ["nature"]
        with open(yaml_file_path, "r") as file:
            yaml_data = yaml.safe_load(file)
            query_list = yaml_data.get("pexels_categories", [])
        logger.info(f"query_list: {query_list}")
        
        api_keys = search_config["PEXELS_API_KEYS"]
        if not api_keys:
            logger.info("Missing Pexels API Key")
            return []
        
        all_valid_video_ids = []

        for task_type in ["HD24K", "SD2HD", "SD24K", "4K28K", "HD28K"]:
            query_chunks = np.array_split(query_list, 5)
            api_keys_chunks = np.array_split(api_keys, 5)

            new_video_ids, matched_video_ids = await self.fetch_video_ids(task_type, query_chunks, api_keys_chunks)             
            logger.info(f"Task type: {task_type} - ✅ Fetched {len(new_video_ids)} new videos")

            all_valid_video_ids.extend(new_video_ids)
            all_valid_video_ids.extend(matched_video_ids)

            yaml_file_path = f"{search_config['VIDEO_DIR']}/../pexels_video_ids_{task_type}.yaml"
            with open(yaml_file_path, "w") as file:
                yaml.dump({"new_video_ids": new_video_ids}, file)

            logger.info(f"✅ Saved {len(new_video_ids)} video ids to {yaml_file_path}")

            #video_chunks = np.array_split(new_video_ids, 5)
            #await self.download_and_insert_videos(task_type, video_chunks, api_keys_chunks)
        
        # all_valid_video_ids = list(dict.fromkeys(all_valid_video_ids))
        # logger.info(f"✅ Fetched {len(all_valid_video_ids)} valid video ids")
        # await self.clean_up_videos(all_valid_video_ids)

if __name__ == "__main__":
    try:
        service = VideoDbUpdateService()
        while True:
            asyncio.run(service.check_and_download())
            time.sleep(search_config['PEXELS_CHECK_INTERVAL'])
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down gracefully...")
        sys.exit(0)
