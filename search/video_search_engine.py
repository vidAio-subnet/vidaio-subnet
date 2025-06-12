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

def get_ramfs_path():
    candidates = ['/dev/shm', '/run', '/tmp']
    for path in candidates:
        if os.path.ismount(path) and os.statvfs(path).f_bsize > 0:
            return path
    return '/tmp'

def find_query_in_chunk(q_len, query_bits_coarse, query_bits_fine, dataset_bits_chunk, coarse_interval : int = 10):
    datset_count = len(dataset_bits_chunk)
    best_overall_score = float('inf')
    best_overall_start = -1
    best_overall_idx = -1

    for idx, dataset_bits in enumerate(dataset_bits_chunk):
        best_start, best_score = match_query_in_video_np(q_len, query_bits_coarse, query_bits_fine, dataset_bits, coarse_interval)
        
        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall_start = best_start
            best_overall_idx = idx
            
    return best_overall_start, best_overall_score, best_overall_idx

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
        "--frame_skip_ref", ref_skip,
        "-d", dist_path,
        "--frame_skip_dist", dist_skip,
        "--frame_cnt", frame_cnt,
        "-out-fmt", "xml",
        "-o", output_file  
    ]
    
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
        self.reload_video_db()

    def get_video_info(self, index:int):
        return self.video_db[index]
        
    def reload_video_db(self):
        client = MongoClient(config['mongo_uri'])
        db_name = config['db_name']
        collection_name = config['collection']

        self.video_db = []
        self.hash_bits_chunks = []
    
        self.log.info(f"2️ Reloading video database from {collection_name}...")
        try:
            collection = client[db_name][collection_name]
            doc_count = collection.count_documents({})
            self.log.info(f"2️ Found {doc_count} documents in collection")
            self.chunk_size = doc_count // self.hash_search_processors

            start_time = time.time()
            hash_bits_arr = []

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
                    hashes = [imagehash.hex_to_hash(h) for h in doc['hashes']]
                    hash_bits = np.array([np.unpackbits(np.array(h.hash, dtype=np.uint8)) for h in hashes])
                    hash_bits_arr.append(hash_bits)
                    if len(hash_bits_arr) >= self.chunk_size:
                        self.hash_bits_chunks.append(hash_bits_arr)
                        hash_bits_arr = []
            if hash_bits_arr:
                self.hash_bits_chunks.append(hash_bits_arr)
                
            duration = time.time() - start_time
            self.log.info(f"2️✅ Loaded {len(self.video_db)} videos in {duration:.2f} seconds")
        except Exception as e:
            self.log.error(f"2️❌ Error initializing database: {e}")
        finally:
            client.close()

    def _search_hash(self, query_path : str, query_scale : int = 2):
        query_hashes, fps, width, height, frame_count = video_to_phashes(os.path.join(self.video_dir, query_path), 16)

        query_bits = np.array([np.unpackbits(np.array(h.hash, dtype=np.uint8)) for h in query_hashes])
        query_bits_coarse = query_bits[:int(fps*2)]
        query_bits_fine = query_bits[:int(fps*5)]
        q_len = len(query_bits)

        # Create a pool of workers for parallel processing
        with Pool(processes=self.hash_search_processors) as pool:
            # Prepare arguments for each worker
            search_args = []
            for i in range(self.hash_search_processors):
                search_args.append((
                    q_len,
                    query_bits_coarse,
                    query_bits_fine,
                    self.hash_bits_chunks[i],
                    int(fps/2)
                ))
            
            # Process chunks in parallel
            results = pool.starmap(find_query_in_chunk, search_args)
            
            # Find best match across all chunks
            best_score = float('inf')
            best_start = -1
            best_video_idx = -1
            
            for chunk_idx, (start, score, idx) in enumerate(results):
                if score < best_score:
                    best_score = score
                    best_start = start
                    best_video_idx = chunk_idx * self.chunk_size  + idx
            
            return best_start, best_score, best_video_idx

    def _trim_and_validate(self, query_path : str, query_scale : int, query_frame_count : int, best_video : dict, best_start : int):
        try:
            start_time_clip = best_start / best_video['fps']
            actual_duration = query_frame_count / best_video['fps']
            query_filename = os.path.splitext(os.path.basename(query_path))[0]
            vmaf_output_path = os.path.join(get_ramfs_path(), f"{query_filename}_vmaf.xml")
            # VMAF based fine tuning

            start_time = time.time()
            start_idx = best_start - 1
            if start_idx < 0:
                start_idx = 0
            end_idx = start_idx + 3
            if end_idx > query_frame_count - 1:
                end_idx = query_frame_count - 1

            query_y4m_path = convert_mp4_to_y4m_downscale(query_path, list(range(0, 2)))
            clip_y4m_path = convert_mp4_to_y4m_downscale(clip_path, list(range(start_idx, end_idx+1)), query_scale)

            max_vmaf_score = -1
            max_vmaf_idx = -1
            for i in range(start_idx, end_idx+1):
                vmaf_score = vmaf_metric_skip(query_y4m_path, 0, clip_y4m_path, i, 2, vmaf_output_path)
                if max_vmaf_score < vmaf_score:
                    max_vmaf_score = vmaf_score
                    max_vmaf_idx = i

            self.log.info(f"2️✅ VMAF based fine tuning duration: {time.time() - start_time:.2f} seconds")
            print(f"2️✅ Max VMAF score: {max_vmaf_score}, Max VMAF idx: {max_vmaf_idx}")

            # Final clip
            clip_path = os.path.join(get_ramfs_path(), f"{query_filename}_clipped_{best_start}_{query_frame_count}.mp4")
            trim_cmd = [
                "taskset", "-c", "0,1,2,3,4,5",
                "ffmpeg", "-y", "-i", str(source_path), "-ss", str(start_time_clip), 
                "-t", str(actual_duration), "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac", str(clipped_path), "-hide_banner", "-loglevel", "error"
            ]

            start_time = time.time()
            subprocess.run(trim_cmd, check=True)

            random_frames = sorted(random.sample(range(query_frame_count), 3))
            clip_y4m_path = convert_mp4_to_y4m_downscale(clip_path, random_frames, query_scale)
            query_y4m_path = convert_mp4_to_y4m_downscale(query_path, random_frames)

            
            vmaf_score = vmaf_metric(query_y4m_path, clip_y4m_path, vmaf_output_path)
            self.log.info(f"2️✅ query_path: {query_path}, ref_video: {best_video['filename']}, VMAF score: {vmaf_score}")

            os.remove(clip_y4m_path)
            os.remove(query_y4m_path)
            os.remove(vmaf_output_path)

            elapsed_time = time.time() - start_time
            self.log.info(f"2️✅ Final clip generated and validated in {elapsed_time:.2f} seconds")

            if vmaf_score < 75:
                self.log.info(f"2️❌ VMAF score is too low: {vmaf_score}")
                os.remove(clip_path)
                return None

            return clip_path
        except subprocess.SubprocessError as e:
            self.log.error(f"2️❌ Error trimming video: {e}")
            return None
        return None
    
    def search_and_clip(self, query_path : str, query_scale : int = 2):
        best_start, best_score, best_video_idx = self._search_hash(query_path, query_scale)
        if best_score < 0:
            return None

        best_video = self.video_db[best_video_idx]
        return self._trim_and_validate(query_path, query_scale, best_video, best_start)


if __name__ == "__main__":
    # Create a VideoSearchEngine instance
    start_time = time.time()
    search_engine = VideoSearchEngine(logger)
    duration = time.time() - start_time
    logger.info(f"VideoSearchEngine initialization took {duration:.2f} seconds")

    start_time = time.time()
    search_engine.search_and_clip("/root/vidaio/test_videos/SD24K_30643798_downscale_126_5.mp4", 4)
    duration = time.time() - start_time
    logger.info(f"VideoSearchEngine search and clip took {duration:.2f} seconds")

    # # _search_hash
    # test_video_dir = "/root/vidaio/test_videos"
    # count = 0
    # total_duration = 0
    # max_duration = 0
    # min_duration = float('inf')
    # success_count = 0
    
    # for file in os.listdir(test_video_dir):
    #     if file.endswith(".mp4") and "downscale" in file:
    #         query_path = os.path.join(test_video_dir, file)
    #         start_frame = int(query_path.split('downscale_')[1].split('_')[0])
    #         if file.startswith("SD24K"):
    #             query_scale = 4
    #         else:
    #             query_scale = 2

    #         start_time = time.time()
    #         best_start, best_score, best_video_idx = search_engine._search_hash(query_path, query_scale)
    #         duration = time.time() - start_time
    #         best_video = search_engine.get_video_info(best_video_idx)

    #         if best_video['filename'] in query_path and best_start == start_frame:
    #             success_count += 1
    #         else:
    #             if not best_video['filename'] in query_path:
    #                 logger.error(f"❌ VideoSearchEngine search hash failed for {query_path} with best_video: {best_video['filename']} and start_frame: {start_frame}")
    #             elif abs(best_start - start_frame) == 1:
    #                 logger.warning(f"⚠️ VideoSearchEngine search hash failed for {query_path} with best_start: {best_start} and start_frame: {start_frame}")
    #             else:
    #                 logger.error(f"❌ VideoSearchEngine search hash failed for {query_path} with best_start: {best_start} and start_frame: {start_frame}")

    #         if duration > max_duration:
    #             max_duration = duration
    #         if duration < min_duration:
    #             min_duration = duration
    #         total_duration += duration
    #         count += 1

    # logger.info(f"VideoSearchEngine search hash processed {count} videos with {success_count} successful matches")
    # logger.info(f"VideoSearchEngine search hash took {total_duration/count:.2f} seconds on average for {count} videos")
    # logger.info(f"VideoSearchEngine search hash took {max_duration:.2f} seconds for the longest video")
    # logger.info(f"VideoSearchEngine search hash took {min_duration:.2f} seconds for the shortest video")

    # # search_and_clip
    # count = 0
    # total_duration = 0
    # max_duration = 0
    # min_duration = float('inf')
    # success_count = 0

    # for file in os.listdir(test_video_dir):
    #     if file.endswith(".mp4") and "downscale" in file:
    #         query_path = os.path.join(test_video_dir, file)
    #         if file.startswith("SD24K"):
    #             query_scale = 4
    #         else:
    #             query_scale = 2
            
    #         start_time = time.time()
    #         clip_path = search_engine.search_and_clip(query_path, query_scale)
    #         duration = time.time() - start_time

    #         if clip_path:
    #             success_count += 1
    #             logger.info(f"✅ VideoSearchEngine search and clip took {duration:.2f} seconds for {query_path}")
    #             os.remove(clip_path)                
    #         else:
    #             logger.error(f"❌ VideoSearchEngine search and clip failed for {query_path}")

    #         if duration > max_duration:
    #             max_duration = duration
    #         if duration < min_duration:
    #             min_duration = duration
    #         total_duration += duration
    #         count += 1

    # logger.info(f"VideoSearchEngine search and clip processed {count} videos with {success_count} successful matches")
    # logger.info(f"VideoSearchEngine search and clip took {total_duration/count:.2f} seconds on average for {count} videos")
    # logger.info(f"VideoSearchEngine search and clip took {max_duration:.2f} seconds for the longest video")
    # logger.info(f"VideoSearchEngine search and clip took {min_duration:.2f} seconds for the shortest video")



    
