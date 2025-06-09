from pymongo import MongoClient
from pathlib import Path
from datetime import datetime
import re
from typing import Optional, Dict

class VideoMetadataUploader:
    def __init__(self, mongo_uri: str, db_name: str, collection_name: str, video_dir: str):
        self.client = MongoClient(mongo_uri)
        self.collection = self.client[db_name][collection_name]
        self.video_dir = Path(video_dir)

    def parse_metadata(self, filename: str) -> Optional[Dict]:
        base = filename.rsplit(".", 1)[0]
        parts = base.split("_")

        if len(parts) < 2:
            return None  # Invalid filename format

        video_id = f"{parts[0]}_{parts[1]}"

        if "trim" in filename:
            return {
                "id": video_id,
                "scale": 1,
                "fps": "0",
                "frames": 0,
                "feature": "trim"
            }
        elif "original" in filename:
            return {
                "id": video_id,
                "scale": 1,
                "fps": "0",
                "frames": 0,
                "feature": "original"
            }
        else:
            try:
                scale = int(re.search(r"downscale-(\d+)x", parts[2]).group(1))
                fps = re.search(r"FPS-([\d.]+)", parts[3]).group(1)
                frames = int(re.search(r"Frames-(\d+)", parts[4]).group(1))
                return {
                    "id": video_id,
                    "scale": scale,
                    "fps": fps,
                    "frames": frames,
                    "feature": "downscale"
                }
            except Exception:
                return None

    def upload_metadata(self):
        files = [f.name for f in self.video_dir.iterdir() if f.is_file() and any(tag in f.name.lower() for tag in ["downscale", "trim", "original"])]
        
        for filename in files:
            metadata = self.parse_metadata(filename)
            if not metadata:
                print(f"⚠️ Skipping invalid filename: {filename}")
                continue

            existing = self.collection.find_one(metadata)
            if not existing:
                metadata.update({
                    "vmaf_score": 0,
                    "final_score": 0,
                    "benchmark_status": 0,
                    "benchmarkStamp": datetime.fromtimestamp(0)
                })
                self.collection.insert_one(metadata)
                print(f"✅ Inserted: {metadata['id']} ({metadata['feature']})")
            else:
                print(f"🔁 Already exists: {metadata['id']} ({metadata['feature']})")
