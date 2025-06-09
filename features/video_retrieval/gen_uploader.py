import argparse
from modules.metadata_uploader import VideoMetadataUploader
from modules.config import config

def run_uploader():
    uploader = VideoMetadataUploader(
        mongo_uri=config["mongo_uri"],
        db_name=config["db_name"],
        collection_name=config["collection"],
        video_dir=config["default_video_dir"]
    )
    uploader.upload_metadata()

if __name__ == "__main__":
    run_uploader()
