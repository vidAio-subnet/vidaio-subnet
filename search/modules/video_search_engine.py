from typing import List, Dict, Tuple
from moviepy.editor import VideoFileClip
import imagehash
import os
import time
from threading import Thread
from search.modules.config import config
from search.modules.hash_engine import video_to_phashes
from services.scoring.vmaf_metric import vmaf_metric
from loguru import logger
from pymongo import MongoClient
import numpy as np
import random
import subprocess
import xml.etree.ElementTree as ET
from multiprocessing import Pool
import logging
from search.hashmatcher import cmatcher

def get_ramfs_path():
    candidates = ['/dev/shm', '/run', '/tmp']
    for path in candidates:
        if os.path.ismount(path) and os.statvfs(path).f_bsize > 0:
            return path
    return '/tmp'

def convert_mp4_to_y4m_downscale(input_path, random_frames, downscale_factor=1):
    if not input_path.lower().endswith(".mp4"):
        raise ValueError("Input file must be an MP4 file.")

    # Change extension to .y4m and keep it in the same directory
    output_path = os.path.splitext(input_path)[0] + ".y4m"

    try:
        select_expr = "+".join([f"eq(n\\,{f})" for f in random_frames])
        
        if downscale_factor >= 2:

            scale_width = f"iw/{downscale_factor}"
            scale_height = f"ih/{downscale_factor}"

            subprocess.run([
                "ffmpeg",
                "-i", input_path,
                "-vf", f"select='{select_expr}',scale={scale_width}:{scale_height}",
                "-pix_fmt", "yuv420p",
                "-vsync", "vfr",
                output_path,
                "-y"
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        else:
            subprocess.run([
                "ffmpeg",
                "-i", input_path,
                "-vf", f"select='{select_expr}'",
                "-pix_fmt", "yuv420p",
                "-vsync", "vfr",
                output_path,
                "-y"
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        return output_path

    except Exception as e:
        print(f"Error in convert_mp4_to_y4m_downscale: {e}")
        raise

def vmaf_metric_skip(ref_path, ref_skip, dist_path, dist_skip, frame_cnt, output_file="vmaf_output.xml"):
    command = [
        "vmaf",  
        "-r", ref_path,
        "--frame_skip_ref", f"{ref_skip}",
        "-d", dist_path,
        "--frame_skip_dist", f"{dist_skip}",
        "--frame_cnt", f"{frame_cnt}",
        "-out-fmt", "xml",
        "-o", output_file  
    ]
    
    # command_str = " ".join(command)
    # print(f"Running VMAF command: {command_str}")

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Error calculating VMAF: {result.stderr.strip()}")
        
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Expected output file '{output_file}' not found.")
        
        tree = ET.parse(output_file)
        root = tree.getroot()
        
        vmaf_metric = root.find(".//metric[@name='vmaf']")
        if vmaf_metric is None:
            raise ValueError("VMAF metric not found in the output.")
        
        vmaf_harmonic_mean = float(vmaf_metric.attrib['harmonic_mean'])

        os.remove(output_file)
        return vmaf_harmonic_mean
    
    except Exception as e:
        print(f"Error in calculate_vmaf: {e}")
        raise

class VideoSearchEngine:
    def __init__(self, log = None):
        if log:
            self.log = log
        else:
            self.log = logging.getLogger(__name__)
        self.video_dir = config['video_dir']
        self.hash_search_processors = config['hash_search_processors']
        self.matcher = cmatcher.HashMatcher(config['hash_search_v2_coarse_unit'], config['hash_search_v2_coarse_interval'])
        self.reload_video_db()

    def get_video_info(self, index:int):
        return self.video_db[index]
        
    def reload_video_db(self):
        client = MongoClient(config['mongo_uri'])
        db_name = config['db_name']
        collection_name = config['collection']

        self.video_db = []

        self.log.info(f"2️ Reloading video database from {collection_name}...")
        try:
            collection = client[db_name][collection_name]
            doc_count = collection.count_documents({})
            self.log.info(f"2️ Found {doc_count} documents in collection")
            start_time = time.time()
            for doc in collection.find():
                if 'hashes' in doc and doc['hashes']:
                    self.video_db.append({
                        '_id': doc['_id'],
                        'filename': doc['filename'],
                        'fps': doc['fps'],
                        'width': doc['width'],
                        'height': doc['height'],
                        'frame_count' : doc['frame_count'],
                    })
                    self.matcher.add_dataset(doc['hashes'])
            duration = time.time() - start_time
            self.log.info(f"2️ ✅ Loaded {len(self.video_db)} videos in {duration:.2f} seconds")
        except Exception as e:
            self.log.error(f"2️ ❌ Error initializing database: {e}")
        finally:
            client.close()

    def put_video_info(self, video_info: dict):
        self.log.info(f"✅ Video search engine received video info {video_info['filename']}")
        self.video_db.append({
            'filename': video_info['filename'],
            'fps': video_info['fps'],
            'width': video_info['width'],
            'height': video_info['height'],
            'frame_count': video_info['frame_count']
        })
        self.matcher.add_dataset(video_info['hashes'])

    def _search_hash(self, query_path : str):
        query_hashes, fps, width, height, frame_count = video_to_phashes(os.path.join(self.video_dir, query_path), 16)

        query_hashes_str = [str(h) for h in query_hashes]
        self.matcher.set_query(query_hashes_str, int(fps))

        best_dataset_idx, best_start = self.matcher.match() 
        return best_dataset_idx, best_start, frame_count

    def _trim_and_validate(self, query_path : str, query_scale : int, query_frame_count : int, best_video : dict, best_start : int):
        try:
            start_time_clip = best_start / best_video['fps']
            actual_duration = query_frame_count / best_video['fps']
            query_filename = os.path.splitext(os.path.basename(query_path))[0]
            vmaf_output_path = os.path.join(get_ramfs_path(), f"{query_filename}_vmaf.xml")
            ref_path = os.path.join(self.video_dir, best_video['filename'])
            # VMAF based fine tuning

            start_time = time.time()
            start_idx = best_start - 1
            if start_idx < 0:
                start_idx = 0
            end_idx = start_idx + 3
            if end_idx > best_video['frame_count'] - 1:
                end_idx = best_video['frame_count'] - 1

            query_y4m_path = convert_mp4_to_y4m_downscale(query_path, list(range(0, 2)))
            clip_y4m_path = convert_mp4_to_y4m_downscale(ref_path, list(range(start_idx, end_idx+1)), query_scale)

            max_vmaf_score = -1
            max_vmaf_idx = -1
            for i in range(start_idx, end_idx+1):
                vmaf_score = vmaf_metric_skip(ref_path = query_y4m_path, ref_skip = 0, dist_path = clip_y4m_path, dist_skip = i - start_idx, frame_cnt = 2, output_file = vmaf_output_path)
                # self.log.info(f"2️ ✅ VMAF score: {vmaf_score}, idx: {i}")
                if max_vmaf_score < vmaf_score:
                    max_vmaf_score = vmaf_score
                    max_vmaf_idx = i

            self.log.info(f"2️ ✅ VMAF based fine tuning duration: {time.time() - start_time:.2f} seconds")

            # Final clip
            clipped_path = os.path.join(get_ramfs_path(), f"{query_filename}_clipped_{max_vmaf_idx}_{query_frame_count}.mp4")
            trim_cmd = [
                "taskset", "-c", "0,1,2,3,4,5",
                "ffmpeg", "-y", "-i", str(ref_path), "-ss", str(start_time_clip), 
                "-t", str(actual_duration), "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac", str(clipped_path), "-hide_banner", "-loglevel", "error"
            ]

            start_time = time.time()
            subprocess.run(trim_cmd, check=True)

            random_frames = sorted(random.sample(range(query_frame_count), 3))
            clip_y4m_path = convert_mp4_to_y4m_downscale(clipped_path, random_frames, query_scale)
            query_y4m_path = convert_mp4_to_y4m_downscale(query_path, random_frames)
            
            vmaf_score = vmaf_metric(query_y4m_path, clip_y4m_path, vmaf_output_path)

            os.remove(clip_y4m_path)
            os.remove(query_y4m_path)
            os.remove(vmaf_output_path)

            elapsed_time = time.time() - start_time
            if vmaf_score < 75:
                self.log.info(f"2️ ❌ VMAF score is too low: {vmaf_score}")
                os.remove(clipped_path)
                return None

            self.log.info(f"2️ ✅ Valid final clip generated in {elapsed_time:.2f} seconds, vmaf_score: {vmaf_score}, max_vmaf_idx: {max_vmaf_idx}")
            return clipped_path, max_vmaf_idx
        except subprocess.SubprocessError as e:
            self.log.error(f"2️ ❌ Error trimming video: {e}")
            return None, None
        return None, None
    
    def search_and_clip(self, query_path : str, query_scale : int = 2):
        start_time = time.time()
        best_dataset_idx, best_start, query_frame_count = self._search_hash(query_path)
        duration = time.time() - start_time
        # self.log.info(f"2️ _search_hash execution duration: {duration:.2f} seconds, best_start: {best_start}")
        if best_start < 0:
            return None

        best_video = self.video_db[best_dataset_idx]
        start_time = time.time()
        clip_path, max_vmaf_idx = self._trim_and_validate(query_path, query_scale, query_frame_count, best_video, best_start)
        duration = time.time() - start_time
        # self.log.info(f"2️ _trim_and_validate execution duration: {duration:.2f} seconds, max_vmaf_idx: {max_vmaf_idx}")

        return clip_path, max_vmaf_idx


def test_multiple_files(file_count : int = -1):
    # search_and_clip
    test_video_dir = "/root/vidaio/test_videos"
    search_engine = VideoSearchEngine(logger)

    count = 0
    total_duration = 0
    max_duration = 0
    min_duration = float('inf')
    success_count = 0

    for file in os.listdir(test_video_dir):
        if (file_count > 0 and count >= file_count):
            break
        
        if file.endswith(".mp4") and "downscale" in file:
            query_path = os.path.join(test_video_dir, file)
            if file.startswith("SD24K"):
                query_scale = 4
            else:
                query_scale = 2
            
            start_time = time.time()
            clip_path, max_vmaf_idx = search_engine.search_and_clip(query_path, query_scale)
            duration = time.time() - start_time

            start_frame = int(query_path.split('downscale_')[1].split('_')[0])

            if clip_path:
                if start_frame == max_vmaf_idx:
                    logger.info(f"✅ {os.path.basename(query_path)}")                    
                elif abs(start_frame - max_vmaf_idx) == 1:
                    logger.info(f"⚠️ {os.path.basename(query_path)}, start_frame: {start_frame}, max_vmaf_idx: {max_vmaf_idx}")
                else:
                    logger.error(f"❌ {os.path.basename(query_path)}, start_frame: {start_frame}, max_vmaf_idx: {max_vmaf_idx}")

                success_count += 1
                os.remove(clip_path)                
            else:
                logger.error(f"❌ {os.path.basename(query_path)}")

            if duration > max_duration:
                max_duration = duration
            if duration < min_duration:
                min_duration = duration
            total_duration += duration
            count += 1

    logger.info(f"VideoSearchEngine processed {count} videos with {success_count} successful matches")
    logger.info(f"VideoSearchEngine took {total_duration/count:.2f} seconds on average for {count} videos")
    logger.info(f"VideoSearchEngine took {max_duration:.2f} seconds for the longest video")
    logger.info(f"VideoSearchEngine took {min_duration:.2f} seconds for the shortest video")

def test_single_file(query_path : str, query_scale : int = 2):
    start_time = time.time()
    search_engine = VideoSearchEngine(logger)
    duration = time.time() - start_time
    logger.info(f"VideoSearchEngine initialization took {duration:.2f} seconds")

    start_time = time.time()
    clip_path, max_vmaf_idx = search_engine.search_and_clip(query_path, query_scale)
    duration = time.time() - start_time
    logger.info(f"VideoSearchEngine search and clip took {duration:.2f} seconds")
    if clip_path:
        os.remove(clip_path)

if __name__ == "__main__":
    test_single_file("/root/vidaio/test_videos/SD2HD_32269855_downscale_632_10.mp4", 2)
    test_multiple_files(10)
