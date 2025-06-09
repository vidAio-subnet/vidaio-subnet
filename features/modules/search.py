from pymongo import MongoClient
from typing import List, Tuple, Optional, Dict
import imagehash
from PIL import Image
import time
import cv2
from features.modules.config import config
from features.modules.utils import get_fps, get_frames

class HashDB:
    def __init__(self, mongo_uri: str, db_name: str, collection_name: str):
        client = MongoClient(mongo_uri)
        self.collection = client[db_name][collection_name]

    def get_all_video_hashes(self, feature="original") -> List[Dict]:
        # docs = self.collection.find({"feature": feature, "hashes": {"$exists": True}})
        docs = list(self.collection.find({"feature": feature, "hashes": {"$exists": True}}))

        return docs
       
    def get_fps_duration(self, id: str) -> Tuple[float, int]:
        try:
            doc = self.collection.find_one({"feature": "downscale", "id": id})
            if doc is None:
                raise ValueError(f"No document found for id: {id}")

            fps = float(doc['fps'])
            frames = doc['frames']
            return (fps, frames)
        except Exception as e:
            print(f"[WARN] get_fps_duration failed for ID {id}: {e}")
            video_path = f"{config['default_video_dir']}/{id}_origin.mp4"
            fps = get_fps(video_path)
            frames = get_frames(video_path)
            print(f"[INFO] Fallback to raw video: {video_path}")
            return (fps, frames)


class VideoHasher:
    def __init__(self, fine_interval: int = 10):
        self.fine_interval = fine_interval

    def get_hashes(self, video_path: str) -> Dict:
        """
        Extract perceptual hashes from a video at given intervals.

        Args:
            video_path (str): Path to the video file.

        Returns:
            List[Tuple[int, imagehash.ImageHash]]: A list of (frame index, hash) tuples.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        hashes = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.fine_interval == 0:
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_hash = imagehash.phash(pil_image)
                hashes.append((frame_idx, frame_hash))
            frame_idx += 1

        cap.release()
        return {"hash" : hashes, "fps": fps}

class VideoMatcher:
    def __init__(self, tolerance: int = 5):
        self.tolerance = tolerance

    def find_subsequence_position(
        self,
        source_hashes: List[Tuple[int, imagehash.ImageHash]],
        query_hashes: List[Tuple[int, imagehash.ImageHash]],
        interval: int = 1,
        tolerance: int = 5
    ) -> Tuple[Optional[int], float]:
        """
        Match query_hashes in source_hashes by comparing both at regular intervals.

        Args:
            source_hashes: [(frame_index, hash)]
            query_hashes: [(frame_index, hash)]
            interval: step size to skip frames in both source and query.

        Returns:
            (start_frame, total_diff_score) or (None, inf)
        """
        best_match = None
        min_diff_total = float('inf')

        # Apply interval skip to query
        reduced_query = query_hashes[::interval]
        q_len = len(reduced_query)

        for i in range(0, len(source_hashes) - q_len * interval + 1, interval):
            try:
                # Extract matching source subsequence at same intervals
                reduced_source = [imagehash.hex_to_hash(source_hashes[i + j * interval]['hash']) for j in range(q_len)]
                diff_total = sum(
                    s_hash - q_hash for s_hash, (_, q_hash) in zip(reduced_source, reduced_query)
                )
            except IndexError:
                break

            if diff_total < min_diff_total and diff_total <= tolerance * q_len:
                min_diff_total = diff_total
                best_match = source_hashes[i]['frame']

        return best_match, min_diff_total
    
    def find_coarse_subsequence_position(
        self,
        source_hashes: List[Tuple[int, imagehash.ImageHash]],
        query_hashes: List[Tuple[int, imagehash.ImageHash]],
        tolerance: int = 5
    ) -> Tuple[Optional[int], float]:
        """
        Match query_hashes in source_hashes by comparing both at regular intervals.

        Args:
            source_hashes: [(frame_index, hash)]
            query_hashes: [(frame_index, hash)]

        Returns:
            (start_frame, total_diff_score) or (None, inf)
        """
        best_match = None
        min_diff_total = float('inf')

        # Apply interval skip to query
        q_real_len = len(query_hashes)
        interval1 = max(q_real_len // 5, 10)
        reduced_query = query_hashes[::interval1]
        q_len = len(reduced_query)

        for i in range(0, len(source_hashes) - q_len * interval1 + 1, interval1):
            try:
                # Extract matching source subsequence at same intervals
                reduced_source = [imagehash.hex_to_hash(source_hashes[i + j * interval1]['hash']) for j in range(q_len)]
                diff_total = sum(
                    s_hash - q_hash for s_hash, (_, q_hash) in zip(reduced_source, reduced_query)
                )
            except IndexError:
                break

            if diff_total < min_diff_total and diff_total <= tolerance * q_len:
                min_diff_total = diff_total
                best_match = source_hashes[i]['frame']

        return best_match, min_diff_total
    
if __name__ == "__main__":
    full_video = "full_video.mp4"
    query_video = "query_clip.mp4"

    hasher = VideoHasher(fine_interval=10)
    matcher = VideoMatcher(tolerance=5)

    print("Hashing full video...")
    full_hashes = hasher.get_hashes(full_video)

    print("Hashing query video...")
    query_hashes = hasher.get_hashes(query_video)

    print("Matching...")
    match_start, score = matcher.find_subsequence_position(full_hashes, query_hashes)

    if match_start is not None:
        print(f"✅ Match found at frame {match_start}, score: {score}")
    else:
        print("❌ No match found.")
