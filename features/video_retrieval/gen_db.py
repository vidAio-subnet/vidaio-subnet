from modules.db import VideoHashDBHandler
from modules.config import config
import argparse

def run_db(feature: str="original"):
    handler = VideoHashDBHandler(
        mongo_uri=config["mongo_uri"],
        db_name=config["db_name"],
        collection_name=config["collection"],
        fine_interval=config["fine_interval"]
    )

    handler.store_hashes_for_feature(config["default_video_dir"], feature)
    # handler.store_hashes_for_feature_parallel(dir_path=args.dir, feature=args.feature, max_workers=config["max_workers"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", default="original", help="Feature type: original or downscale")
    args = parser.parse_args()
    run_db(args.feature)
    
