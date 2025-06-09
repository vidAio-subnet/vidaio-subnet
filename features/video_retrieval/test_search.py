import argparse
from modules.config import config
from modules.search_engine import VideoSearchEngine
from modules.vmaf_metric import evaluate_vmaf
import time
import os
import os, math, time, logging

log_filename = "evaluation_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode='w'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Find which video contains the query video clip")
    parser.add_argument("--query_path", required=True, help="Path to query video clip")
    args = parser.parse_args()

    start_time = time.time()
    engine = VideoSearchEngine(log)
    elapsed_time = time.time() - start_time
    print(f"loading time: {elapsed_time}")

    start_time = time.time()
    result = engine.search(query_path=args.query_path, query_scale=2, top_n=1)
    elapsed_time = time.time() - start_time
    print(f"==========Search Result==========\n time = {elapsed_time}\n")

    start_time = time.time()
    result = engine.search(query_path=args.query_path, query_scale=2, top_n=1)
    elapsed_time = time.time() - start_time
    print(f"==========Search Result==========\n time = {elapsed_time}\n")
    
    for _, res in enumerate(result, 1):
        print(f"  📼 Upscaled Path: {res[0]}")
        print(f"  📊 VMAF Score: {res[1]}")
        trim_path = f'{config['default_video_dir']}/{res[2]}_trim.mp4'
        vmaf_score = evaluate_vmaf(res[0], trim_path)
        print(f"  📊 real vmaf_score: {vmaf_score}")
if __name__ == "__main__":
    main()
