from features.modules.search import VideoHasher, HashDB
from features.modules.config import config
from typing import List, Dict, Tuple
from features.modules.utils import clip_video, clip_video_precise, get_frames, find_query_start_frame
from features.modules.vmaf_metric import evaluate_vmaf
from moviepy.editor import VideoFileClip
from multiprocessing import Pool, cpu_count
import imagehash
import os
from collections import defaultdict
import time
import math
from features.hashmatcher import cmatcher
from threading import Thread

class VideoSearchEngine:
    def __init__(self, log = None):
        self.db = HashDB(config["mongo_uri"], config["db_name"], config["collection"])
        self.hasher = VideoHasher(fine_interval=config["fine_interval"])
        self.coarse_interval = config["coarse_interval"]
        self.fine_interval = config["fine_interval"]
        self.db_hash_sets = self.db.get_all_video_hashes()
        self.vmaf_score_threshold = config['vmaf_score_threshold']
        self.fps_set = defaultdict(float)
        self.log = log
        self.reload_hash_db()
        self._start_change_listener()

    def reload_hash_db(self):
        docs = self.db.get_all_video_hashes()
        self.db_hash_sets = docs
        self.fps_set = defaultdict(float)

        for doc in docs:
            try:
                fps, _ = self.db.get_fps_duration(doc['id'])
                self.fps_set[doc['id']] = fps
            except Exception as e:
                if self.log:
                    self.log.warning(f"⚠️ Could not load FPS for {doc['id']}: {e}")
        if self.log:
            self.log.info(f"🔴🔄 Reloaded hash DB: {len(self.db_hash_sets)} entries")

    def _start_change_listener(self):
        def watch_changes():
            self.log.info("📡 Starting MongoDB change stream...")
            try:
                with self.db.collection.watch(
                    [{'$match': {'operationType': {'$in': ['insert', 'update', 'replace', 'delete']}}}],
                    full_document='updateLookup'
                ) as stream:
                    for change in stream:
                        op_type = change["operationType"]
                        doc_id = change["documentKey"]["_id"]

                        if op_type in ["insert", "update", "replace"]:
                            full_doc = change.get("fullDocument")
                            if full_doc and "hashes" in full_doc:
                                self._update_in_memory(full_doc)
                        elif op_type == "delete":
                            self._delete_from_memory(doc_id)

            except Exception as e:
                if self.log:
                    self.log.error(f"❌ MongoDB Change Stream Error: {e}")

        Thread(target=watch_changes, daemon=True).start()

    def _update_in_memory(self, full_doc):
        # Update or insert
        existing = next((v for v in self.db_hash_sets if v["_id"] == full_doc["_id"]), None)
        if existing:
            self.db_hash_sets.remove(existing)
        self.db_hash_sets.append(full_doc)
        self.log.info(f"updated info: {full_doc}")

        try:
            fps, _ = self.db.get_fps_duration(full_doc["id"])
            self.fps_set[full_doc["id"]] = fps
            if self.log:
                self.log.info(f"🔴✅ Updated in-memory entry for {full_doc['id']}")
        except Exception as e:
            if self.log:
                self.log.warning(f"🔴⚠️ Could not update FPS for {full_doc['id']}: {e}")

    def _delete_from_memory(self, doc_id):
        # Remove from db_hash_sets and fps_set
        before_count = len(self.db_hash_sets)
        self.db_hash_sets = [doc for doc in self.db_hash_sets if doc["_id"] != doc_id]
        removed_count = before_count - len(self.db_hash_sets)
        if removed_count > 0:
            if self.log:
                self.log.info(f"🔴🗑️ Deleted in-memory entry for _id: {doc_id}")
        self.fps_set = {k: v for k, v in self.fps_set.items() if k != doc_id}

    def coarse_search(self, query_hashes: Dict, query_scale: int = 2, top_n: int = 3) -> List[Dict]:
        results = []        
        for db_video in self.db_hash_sets:
            if abs(self.fps_set[db_video['id']] - float(query_hashes['fps'])) > 0.1:
                continue
            source_hashes = [(h["frame"], h["hash"]) for h in db_video["hashes"]]
            query_hashes1 = [(h[0], str(h[1])) for h in query_hashes["hash"]]

            position, score = cmatcher.find_coarse_match(
                source_hashes,
                query_hashes1,
                config["tolerance"] * 5
            )

            if position is not None:
                result_scale = 2
                if "SD24K" in db_video["id"]:
                    result_scale = 4
                if query_scale == result_scale:
                    results.append({
                        "id": db_video["id"],
                        "feature": "origin",
                        "start_frame": position,
                        "scale" : result_scale,
                        "hashes": db_video["hashes"],
                        "score": score
                    })

        # Sort by score (ascending = best match first)
        ranked_results = sorted(results, key=lambda x: x["score"])
        return ranked_results[:top_n]
    
    def fine_search(self, query_hashes: Dict, query_scale: int, coarse_result: List[Dict], top_n: int = 3) -> List[Dict]:
        results = []

        for coarse in coarse_result:
            source_hashes = [(h["frame"], h["hash"]) for h in coarse["hashes"]]
            query_hashes1 = [(h[0], str(h[1])) for h in query_hashes["hash"]]
            position, score = cmatcher.find_subsequence_position(
                source_hashes,
                query_hashes1,
                self.fine_interval,
                config["tolerance"]
            )

            if position is not None:
                if query_scale == coarse['scale']:
                    results.append({
                        "id": coarse["id"],
                        "start_frame": position,
                        "scale" : query_scale,
                        "frames": len(query_hashes['hash']),
                        "fps": query_hashes['fps'],
                        "score": score
                    })

        # Sort by score (ascending = best match first)
        ranked_results = sorted(results, key=lambda x: x["score"])
        return ranked_results[:top_n]

    def get_vmaf(self, query_path: str, origin_path: str, origin_video: VideoFileClip, query_scale: int, result: Dict, start_num):
        upscale_path = f'{result['id']}_upscaled.mp4'
        fps = origin_video.fps
        start = start_num / fps
        duration = result['frames'] / fps

        upscaled_video = origin_video.subclip(start, start + duration)
        self.log.info(f'🔴 search_engine: upscale_path: {upscale_path}')
        upscaled_video.write_videofile(str(upscale_path), codec='libx264', verbose=False, logger=None)
        total_frames = get_frames(upscale_path)

        self.log.info(f"🔴 total_frames: {total_frames}")
        if total_frames != result['frames']:
            upscaled_video.close()
            os.remove(upscale_path)
            self.log.info(f"🔴 re clipping, mismatch count: {total_frames - result['frames']}")
            clip_video_precise(origin_path, upscale_path, start_num, result['frames'])
            upscaled_video = VideoFileClip(upscale_path)

        _, height = upscaled_video.size
        downscaled_height = height //  query_scale
        downscale = upscaled_video.resize(height=downscaled_height)

        downscaled_path = f'/dev/shm/{result['id']}_tmp.mp4'
        downscale.write_videofile(str(downscaled_path), codec='libx264', verbose=False, logger=None)
        vmaf_score = evaluate_vmaf(query_path, downscaled_path)
        if vmaf_score:
            self.log.info(f'🔴 search_engine: vmaf score: {vmaf_score}')
        
        os.remove(downscaled_path)
        upscaled_video.close()
        downscale.close()

        return vmaf_score, upscale_path

    def get_upscaled(self, query_path: str, query_scale: int, result: Dict) -> Tuple:
        '''
        Stage 1: calc vmaf for result.
        If vmaf score is threshold, brute-force search in source video. (check all frames.)
        '''
        result_path = f'{config['default_video_dir']}/{result['id']}_original.mp4'
        downscale_result_path = f'{config['default_video_dir']}/{result['id']}.mp4'
        
        upscale_path = f'{result['id']}_upscaled.mp4'
        

        origin_video = VideoFileClip(result_path, audio=False)
        fps = origin_video.fps
        duration = result['frames'] / fps
        start_time = time.time()
        if duration >= 4 and result['start_frame'] >= 125:
            start = result['start_frame'] / fps
            self.log.info(f'🔴 search_engine: long video: {duration}s, {result['start_frame']} frames')
            vmaf_score = cmatcher.compute_vmaf(query_path, result_path, result['id'], start, duration, query_scale)
        else:
            vmaf_score, upscale_path = self.get_vmaf(query_path=query_path, origin_path=result_path, origin_video=origin_video, query_scale=query_scale, result=result, start_num=result['start_frame'])
        
        elapsed_time = time.time() - start_time
        self.log.info(f'🔴 search_engine: get_vmaf time: {elapsed_time}')
        if vmaf_score == None or vmaf_score < self.vmaf_score_threshold:
            start_time = time.time()
            if not os.path.exists(downscale_result_path):
                _, height = origin_video.size
                downscaled_video = origin_video.resize(height=height //  query_scale)
                self.log.info(f'🔴 search_engine: downscale_result_path: {downscale_result_path}')
                downscaled_video.write_videofile(str(downscale_result_path), codec='libx264', verbose=False, logger=None)
                
            start_num, dist = cmatcher.find_query_start_frame(query_path=query_path, original_path=downscale_result_path)
            elapsed_time = time.time() - start_time
            self.log.info(f'🔴 search_engine: find_query_start_frame time: {elapsed_time}, bias: {start_num - result['start_frame']}, dist: {dist}')
            if start_num != result['start_frame']:
                if start_num == -1:
                    origin_video.close()
                    return None
                os.remove(upscale_path)
                if duration >= 4 and result['start_frame'] >= 125:
                    start = start_num / fps
                    vmaf_score = cmatcher.compute_vmaf(query_path, result_path, result['id'], start, duration, query_scale)
                else:
                    vmaf_score, upscale_path = self.get_vmaf(query_path=query_path, origin_path=result_path, origin_video=origin_video, query_scale=query_scale, result=result, start_num=start_num)
                
            
        origin_video.close()
        if vmaf_score != None and vmaf_score >= self.vmaf_score_threshold:
            return (upscale_path, vmaf_score, result['id'])
        return None
    
    def search(self, query_path: str, query_scale: int = 2, top_n: int = 3) -> List[Tuple]:
        query_hash = self.hasher.get_hashes(query_path)

        start_time = time.time()
        coarse_result = self.coarse_search(query_hashes=query_hash, query_scale=query_scale, top_n=3)
        elapsed_time1 = time.time() - start_time

        start_time = time.time()
        fine_results = self.fine_search(query_hashes=query_hash, query_scale=query_scale, coarse_result=coarse_result, top_n=1)
        elapsed_time2 = time.time() - start_time

        self.log.info(f"🔴 search_engine: fine_results: {fine_results}, delay time: {elapsed_time1} + {elapsed_time2} sec")

        final_result = []
        for idx, result in enumerate(fine_results, 1):
            res = self.get_upscaled(query_path=query_path, query_scale=query_scale, result=result)
            if (res != None):
                final_result.append(res)

        # Sort by VMAF score in descending order (highest score first)
        if len(final_result) > 0:
            final_result = sorted(final_result, key=lambda x: x[1], reverse=True)
            return final_result[:top_n]

        return []

    def searchHash(self, query: Dict, top_n: int = 3) -> List[Tuple]:
        query_hash = [(h["frame"], imagehash.hex_to_hash(h["hash"])) for h in query['hashes']]
        query1 = {'fps': query['fps'], 'hash': query_hash}
       
        start_time = time.time()
        coarse_result = self.coarse_search(query_hashes=query1, query_scale=query['scale'], top_n=3)
        elapsed_time1 = time.time() - start_time

        start_time = time.time()
        fine_results = self.fine_search(query_hashes=query1, query_scale=query['scale'], coarse_result=coarse_result, top_n=1)
        elapsed_time2 = time.time() - start_time

        self.log.info(f"🔴 search_engine: fine_results: {fine_results}, delay time: {elapsed_time1} + {elapsed_time2} sec")

        final_result = []
        for idx, result in enumerate(fine_results, 1):
            query_path = f'{config['default_video_dir']}/{query['id']}_downscale-{query['scale']}x_FPS-{query['fps']}_Frames-{query['frames']}.mp4'
            res = self.get_upscaled(query_path=query_path, query_scale=query['scale'], result=result)
            if (res != None):
                final_result.append(res)           

        # Sort by VMAF score in descending order (highest score first)
        if len(final_result) > 0:
            final_result = sorted(final_result, key=lambda x: x[1], reverse=True)
            return final_result[:top_n]

        return []
