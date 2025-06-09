import time
import datetime
import schedule
import sys
import os

# Add 'video-retrieval' to Python path so imports work
# sys.path.append(os.path.join(os.path.dirname(__file__), 'video-retrieval'))

from monitor_pexel import main as monitor_pexel_main
from download import run_download
from gen_uploader import run_uploader
from gen_db import run_db

# Delay between steps, in seconds
STEP_DELAY = 10

def pipeline():
    print(f"\n--- Pipeline Start: {datetime.datetime.now()} ---\n")

    pipeline_steps = [
        (monitor_pexel_main, "monitor_pexel", {}),
        (run_download, "download1", {"task_type": "HD24K", "json_path": "cache/pexels_HD24K.json"}),
        (run_download, "download2", {"task_type": "4K28K", "json_path": "cache/pexels_4K28K.json"}),
        (run_download, "download3", {"task_type": "HD28K", "json_path": "cache/pexels_HD28K.json"}),
        (run_download, "download4", {"task_type": "SD2HD", "json_path": "cache/pexels_SD2HD.json"}),
        (run_download, "download5", {"task_type": "SD24K", "json_path": "cache/pexels_SD24K.json"}),
        (run_uploader, "gen_uploader", {}),
        (run_db, "gen_db", {"feature": "HD24K"}),
        (run_db, "gen_db", {"feature": "4K28K"}),
        (run_db, "gen_db", {"feature": "HD28K"}),
        (run_db, "gen_db", {"feature": "SD2HD"}),
        (run_db, "gen_db", {"feature": "SD24K"})
    ]

    for func, name, kwargs in pipeline_steps:
        try:
            print(f"[{datetime.datetime.now()}] Running {name} with args {kwargs}...")
            func(**kwargs)
            print(f"[{datetime.datetime.now()}] {name} SUCCESS\n")
        except Exception as e:
            print(f"[{datetime.datetime.now()}] {name} FAILED with exception: {e}\n")
            print(f"Pipeline interrupted. Next run will be in 1 hour.\n")
            return
        time.sleep(STEP_DELAY)

    print(f"Pipeline finished successfully at {datetime.datetime.now()}.\n")

is_running = False

def safe_pipeline():
    global is_running
    if is_running:
        print("Previous pipeline still running. Skipping this cycle.")
        return
    is_running = True
    try:
        pipeline()
    finally:
        is_running = False

def main_loop():
    # schedule.every(1).hours.do(pipeline)
    schedule.every(1).hours.do(safe_pipeline)

    print("Scheduler started. Pipeline will run every 1 hour.\n")
    safe_pipeline()  # Run immediately at launch (optional)
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main_loop()
