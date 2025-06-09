from pymongo import MongoClient
from modules.search import VideoHasher
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

class VideoHashDBHandler:
    def __init__(self, mongo_uri: str, db_name: str, collection_name: str, fine_interval: int):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.hasher = VideoHasher(fine_interval=fine_interval)
        self.loaded_docs = []  # Cache loaded video docs

    def load_videos(self, feature: str = "original"):
        self.loaded_docs = list(self.collection.find({"feature": feature}))
        print(f"🔃 Loaded {len(self.loaded_docs)} documents.")

    def _build_video_path(self, video_doc, dir_path, feature):
        if feature == "original":
            return os.path.join(dir_path, f"{video_doc['id']}.mp4")
        elif feature == "downscale":
            return os.path.join(
                dir_path,
                f"{video_doc['id']}_downscale-{video_doc['scale']}x_FPS-{video_doc['fps']}_Frames-{video_doc['frames']}.mp4"
            )
        else:
            raise ValueError(f"Unknown feature type: {feature}")

    def _process_video(self, video_doc, dir_path, feature):
        # Skip if already hashed
        if "hashes" in video_doc and video_doc["hashes"]:
            return f"🔁 Skipped (already hashed): {video_doc['id']}"

        video_path = self._build_video_path(video_doc, dir_path, feature)
        if not os.path.exists(video_path):
            return f"⚠️ Missing file: {video_path}"

        raw_hashes = self.hasher.get_hashes(video_path)
        hashes = [{"frame": f, "hash": str(h)} for f, h in raw_hashes['hash']]
        video_doc["hashes"] = hashes  # Save locally

        self.collection.update_one(
            {"id": video_doc["id"], "feature": feature},
            {"$set": {"hashes": hashes}},
            upsert=True
        )
        return f"✅ Hashed {video_doc['id']} ({len(hashes)} frames)"
    
    def store_hashes_for_feature_parallel(self, dir_path: str, feature: str = "original", max_workers: int = 4):
        # videos = list(self.collection.find({"feature": feature}))
        if not self.loaded_docs:
            raise RuntimeError("Call `load_videos()` before processing.")
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_video, doc, dir_path, feature): doc['id']
                for doc in self.loaded_docs
            }
            # futures = {
            #     executor.submit(self._process_video, video, dir_path, feature): video['id']
            #     for video in videos
            # }
            for future in as_completed(futures):
                msg, updated_doc = future.result()
                print(msg)
                results.append(updated_doc)
            # for future in as_completed(futures):
            #     print(future.result())
        self.loaded_docs = results
    
    def store_hashes_for_feature(self, dir_path: str, feature: str = "original"):
        # videos = self.collection.find({"feature": feature})
        if not self.loaded_docs:
            print("⚠️ No video documents loaded. Run load_videos() first.")
            return

        for video_doc in self.loaded_docs:
            # Skip if hashes already exist
            if "hashes" in video_doc and video_doc["hashes"]:
                print(f"🔁 Hashes already exist for {video_doc['id']} {feature}. Skipping.")
                continue
            video_path = self._build_video_path(video_doc, dir_path, feature)
            if not os.path.exists(video_path):
                print(f"⚠️ Skipping missing file: {video_path}")
                continue
            
            raw_hashes = self.hasher.get_hashes(video_path)
            hashes = [{"frame": f, "hash": str(h)} for f, h in raw_hashes['hash']]

            self.collection.update_one(
                {"id": video_doc["id"], "feature": feature},
                {"$set": {"hashes": hashes}},
                upsert=True
            )
            print(f"✅ Stored {len(hashes)} hashes for {video_doc['id']} {feature}")
