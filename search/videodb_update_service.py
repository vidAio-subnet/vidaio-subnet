from typing import List, Dict, Tuple
from moviepy.editor import VideoFileClip
import imagehash
import os
import time
from threading import Thread
from search.modules.config import config
from search.modules.hash_engine import video_to_phashes, match_query_in_video
from search.services.scoring.vmaf_metric import vmaf_metric
from loguru import logger

def download_ideo(
    vid: int,
    task_type : str,
) -> Optional[str]:
    api_keys = [
        "WD9OjZcFWztgC6no5AQksf5ixmdrW1EfLqpiYqxOWozhLX2DKzCYVkcO",
        "IVVpGOKxM8vkqJQGziB7PN9FX8WhxWAFlusWe9UvMyPgPpMcAkNR2YNK",
        "nFfgK7tYZqdcBKPOQ3xhg1MNsofTa9KoWHfRtqFgtPT9TOMfg6uqV07R"
    ]
    api_key = random.choice(api_keys)
    # api_key = os.getenv("PEXELS_API_KEY")
    
    if not api_key:
        raise ValueError("PEXELS_API_KEY not found in environment variables")
    
    # API endpoint
    url = f"https://api.pexels.com/videos/videos/{vid}"
    headers = {"Authorization": api_key}
    
    try:
        video_id = f'{task_type}_{vid}'
        origin_path = Path(config['video_dir']) / f"{video_id}_original.mp4"
        if origin_path.exists():
            print(f"✅ Already downloaded: {origin_path}")
            return str(origin_path)
        
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
    
        # Trim video
        print("\nTrimming video...")
        start_time = time.time()

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
) -> List[int], int:
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
                if video["id"] in cur_video_ids:
                    matched += 1
                    continue
                logger.info(f"[New Video] {video["id"]}")
                new_video_ids.append(video["id"])

        logger.info(f"[Page {page}] Matched {matched} / {len(data['videos'])}")

        if len(data["videos"]) < per_page:
            break  # Last page
        page += 1
        time.sleep(delay)
            
    logger.info(f"✅ Total matching new {task_type} videos: {len(new_video_ids)}")
    return new_video_ids, page

class VideoDbUpdateService:
    def __init__(self, log = None):
        if log:
            self.log = log
        else:
            self.log = logging.getLogger(__name__)
        self.video_dir = config['video_dir']
        self.reload_video_db()

        self.client = MongoClient(config['mongo_uri'])
        self.db_name = config['db_name']
        self.collection_name = config['collection']
        self.collection = self.client[self.db_name][self.collection_name]
        self.video_ids = []
        self.last_pages = {}
        for doc in self.collection.find():
            self.video_ids.append(doc['filename'].split('_')[1])
    
    def insert_video_to_db(self, origin_path: str):
        try:
            filename = os.path.basename(origin_path)
            cap = cv2.VideoCapture(origin_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            hashes = video_to_phashes(origin_path)
            self.collection.insert_one({
                'filename': filename,
                'fps': fps,
                'width': width,
                'height': height,
                'frame_count': frame_count,
                'hashes': [str(h) for h in hashes]
            })
            cap.release()

        except Exception as e:
            logger.error(f"Failed to insert video to db: {str(e)}")
            return False
        return True

    def check_and_download(self):
        for task_type in ["HD24K", "SD2HD", "SD24K", "4K28K", "HD28K"]:
            logger.info(f"Fetching all pexels videos for task type: {task_type}")
            last_page = self.last_pages.get(task_type, 0)
            new_video_ids, page = get_pexels_all_vids(
                cur_video_ids = self.video_ids,
                last_page = last_page,
                min_len = config['MIN_VIDEO_LEN'],
                max_len = config['MAX_VIDEO_LEN'],
                task_type = task_type
            )
            for video_id in new_video_ids:
                try:
                    logger.info(f"Downloading video {video_id}")
                    origin_path = download_ideo(video_id, task_type)
                    if origin_path is None:
                        logger.error(f"❌ Failed to download video {video_id}: No origin path returned")
                        continue
                    if self.insert_video_to_db(origin_path):
                        logger.info(f"✅ Video {video_id} inserted to db")
                    else:
                        logger.error(f"❌ Failed to insert video {video_id} to db")
                except Exception as e:
                    logger.error(f"Failed to download video {video_id}: {str(e)}")
                    continue
            self.last_pages[task_type] = page
            self.video_ids.extend(new_video_ids)

            time.sleep(5)

if __name__ == "__main__":
    try:
        service = VideoDbUpdateService()
        while True:
            service.check_and_download()
            time.sleep(config['PEXELS_CHECK_INTERVAL'])
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down gracefully...")
        sys.exit(0)
