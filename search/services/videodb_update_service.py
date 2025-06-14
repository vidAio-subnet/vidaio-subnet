from typing import List, Dict, Tuple, Optional
from moviepy.editor import VideoFileClip
import imagehash
import os
import time
from threading import Thread
from search.modules.config import config
from search.modules.hash_engine import video_to_phashes
from loguru import logger
from search.services.video_search_service import put_video_info
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
    
    try:
        video_id = f'{task_type}_{vid}'
        origin_path = Path(config['video_dir']) / f"{video_id}_original.mp4"
        if origin_path.exists():
            print(f"⚠️ Already existing: {origin_path}, deleted")
            os.remove(origin_path)
        
        start_time = time.time()

        EXPECTED_RESOLUTIONS = {
            "SD2HD": (1920, 1080),
            "4K28K": (7680, 4320),
        }
        expected_width, expected_height = EXPECTED_RESOLUTIONS.get(task_type, (3840, 2160))
        print(f"expected_width, expected_height: {expected_width, expected_height}")
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
            print(f"video url is None: id: {vid}")
            return None
        
        # Download video with progress bar
        print(f"\nDownloading video ID: {vid}")
        response = requests.get(video_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(origin_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                size = f.write(chunk)
                pbar.update(size)
                
        print(f"\nVideo downloaded successfully to: {origin_path}")
        elapsed_time = time.time() - start_time
        print(f"Time taken to download video: {elapsed_time:.2f} seconds")
    

        print("\nChecking video resolution...")
        video_clip = VideoFileClip(str(origin_path))
        actual_width, actual_height = video_clip.size
        
        if actual_width != expected_width or actual_height != expected_height:
            video_clip.close()
            error_msg = (f"Video resolution mismatch. Expected {expected_width}x{expected_height}, "
                        f"but got {actual_width}x{actual_height}")
            print(error_msg)
            return None
        video_clip.close()
        return origin_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading video: {e}")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def get_pexels_all_vids(
    min_len: int,
    max_len: int,
    cur_video_ids: List[int],
    last_page : int = 0,
    width: int = None,
    height: int = None,
    task_type: str = None,
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
    api_key = config["PEXELS_API_KEY"]
    if not api_key:
        logger.error("Missing Pexels API Key")
        return []

    headers = {"Authorization": api_key}
    query_list = ["nature"]  # add more keywords if needed
    per_page = 80

    logger.info(f"🔎 Fetching videos for {task_type}: {width}x{height}, {min_len}-{max_len}s")

    new_video_ids = []
    matched_video_ids = []
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
                logger.error(f"Retries:{retries} [ERROR] Page {page} failed: {e}")
                retries += 1
                time.sleep(15 * retries)            
        else:
            logger.warning(f"[SKIP] Page {page} skipped after {MAX_RETRIES} retries")
        if not data.get("videos"):
            logger.info(f"[END] No more videos on page {page}")
            break

        matched = 0
        skipped = 0
        new_count = 0
        for video in data["videos"]:
            if video["id"] in cur_video_ids:
                matched += 1
                matched_video_ids.append(video["id"])
            if min_len <= video["duration"] <= max_len and video["width"] == width and video["height"] == height:
                # logger.info(f"[New Video] {video["id"]}")
                new_video_ids.append(video["id"])
                new_count += 1
            else:
                # logger.info(f"[Skip] {video["id"]} as condition not met : {video["duration"]} {video["width"]} {video["height"]}")
                skipped += 1

        logger.info(f"[Page {page}] Matched {matched} Skipped {skipped} New {new_count} Total {len(data['videos'])}")

        if len(data["videos"]) < per_page:
            break  # Last page
        page += 1
        time.sleep(delay)
            
    logger.info(f"✅ Total matching new {task_type} videos: {len(new_video_ids)}")
    return new_video_ids, matched_video_ids, page

async def put_video_info_to_search_service(video_info: dict):
    url = f"http://{config['search_service_host']}:{config['search_service_port']}/put_video_info"
    headers = {"Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, headers=headers, data=json.dumps(video_info), timeout=30) as response:
                print(f"video_search_service calling put_video_info {video_info['filename']}")
                if response.status == 200:
                    return {
                        "success": True
                    }
                logger.error(f"Video info service error: {response.status}")
                return {
                    "success": False
                }
        except asyncio.TimeoutError:
            logger.error("Timeout while putting video info")
            return {
                "success": False,
                "error": "Request timed out"
            }
        except Exception as e:
            logger.error(f"Error putting video info: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

class VideoDbUpdateService:
    def __init__(self):
        self.video_dir = config['video_dir']
        self.test_video_dir = config['test_video_dir']
        self.client = MongoClient(config['mongo_uri'])
        self.db_name = config['db_name']
        self.collection_name = config['collection']
        self.collection = self.client[self.db_name][self.collection_name]
        self.video_ids = []
        self.last_pages = {}
        for doc in self.collection.find():
            self.video_ids.append(int(doc['filename'].split('_')[1]))
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
                'hashes': [str(h) for h in hashes]
            }

            try:
                await put_video_info_to_search_service(video_info)
            except Exception as e:
                logger.error(f"Failed to put video info to search service: {str(e)}")

            self.collection.insert_one(video_info)
            logger.info(f"✅ Inserted video {filename} to db")
            
            cap.release()
        except Exception as e:
            logger.error(f"Failed to insert video to db: {str(e)}")
            return False
        return True

    async def check_and_download(self):
        all_new_video_ids = []
        all_matched_video_ids = []

        for task_type in ["HD24K", "SD2HD", "SD24K", "4K28K", "HD28K"]:
            logger.info(f"Fetching all pexels videos for task type: {task_type}")

            # last_page = self.last_pages.get(task_type, 0)
            last_page = 0
            new_video_ids, matched_video_ids, page = get_pexels_all_vids(
                cur_video_ids = self.video_ids,
                last_page = last_page,
                min_len = config['MIN_VIDEO_LEN'],
                max_len = config['MAX_VIDEO_LEN'],
                task_type = task_type
            )
            all_new_video_ids.extend(new_video_ids)
            all_matched_video_ids.extend(matched_video_ids)
            self.last_pages[task_type] = page

        logger.info(f"✅ Fetched {len(all_new_video_ids)} new videos and {len(all_matched_video_ids)} matched videos")
        unmatched_video_ids = [vid for vid in self.video_ids if vid not in all_matched_video_ids]
        logger.info(f"Found {len(unmatched_video_ids)} unmatched videos that need to be removed from database")

        for video_id in unmatched_video_ids:
            try:
                logger.info(f"Removing video {video_id} from database")
                self.collection.delete_one({'filename': {'$regex': f'{video_id}'}})
                os.system(f"rm -f {self.video_dir}/*{video_id}*")
                os.system(f"rm -f {self.test_video_dir}/*{video_id}*")
            except Exception as e:
                logger.error(f"Failed to remove video {video_id} from database: {str(e)}")
                continue
        
        download_api_keys = config['PEXELS_DOWNLOAD_API_KEY'].split(',')
        last_api_key_index = 0
        
        for video_id in all_new_video_ids:
            try:
                logger.info(f"Downloading video {video_id}")
                for retry in range(3):
                    try:
                        origin_path = download_video(video_id, download_api_keys[last_api_key_index], task_type)
                        if origin_path is not None:
                            break
                    except Exception as e:
                        logger.error(f"❌ Retry {retry} Failed to download video {video_id}: {str(e)}")
                        time.sleep(60 * retry)
                    last_api_key_index = (last_api_key_index + 1) % len(download_api_keys)
                    
                if origin_path is None:
                    logger.error(f"❌ Failed to download video {video_id}: No origin path returned")
                    continue
                    
                logger.info(f"✅ Successfully downloaded video {video_id}")
                if await self.insert_video_to_db(origin_path):
                    self.video_ids.append(video_id)
                    logger.info(f"✅ Video {video_id} inserted to db")
                else:
                    logger.error(f"❌ Failed to insert video {video_id} to db")
            except Exception as e:
                logger.error(f"❌ Failed to download video {video_id}: {str(e)}")
                continue

if __name__ == "__main__":
    try:
        service = VideoDbUpdateService()
        while True:
            asyncio.run(service.check_and_download())
            time.sleep(config['PEXELS_CHECK_INTERVAL'])
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down gracefully...")
        sys.exit(0)
