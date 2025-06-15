import imagehash
from PIL import Image
import cv2
import time
from search.hashmatcher import cmatcher
import numpy as np
import os
from numba import njit
from search.modules.search_config import search_config


def video_to_phashes(video_path, hash_size=16, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    hashes = []
    frame_count = 0
        
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frames and frame_count >= max_frames):
            break
             
        # Convert to grayscale for faster processing
        frame = cv2.resize(frame, (hash_size * 4, hash_size * 4))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = Image.fromarray(gray)
        ph = imagehash.phash(image, hash_size=hash_size)
        hashes.append(ph)
        frame_count += 1
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return hashes, fps, width, height, frame_count

def test_file_cpp_matcher(query_path, video_path, iterations=10):
    # Generate hashes for both videos
    start_time = time.time()
    video_hashes, fps, width, height, frame_count = video_to_phashes(video_path, hash_size=16)
    query_hashes, *_ = video_to_phashes(query_path, hash_size=16)
    end_time = time.time()

    coarse_unit = 4
    coarse_interval = 3
    
    query_hashes_str = [str(h) for h in query_hashes]
    dataset_hashes_str = [str(h) for h in video_hashes]

    matcher = cmatcher.HashMatcher(coarse_unit, coarse_interval)
    matcher.set_query(query_hashes_str, int(fps))
    matcher.add_dataset(dataset_hashes_str)

    start_time = time.time()
    total_time = 0
    for _ in range(iterations):
        start_time = time.time()
        best_dataset_idx, best_start = matcher.match()
        end_time = time.time()
        total_time += (end_time - start_time)
    avg_time = total_time / iterations
    print(f"{os.path.basename(query_path)} : qlen: {len(query_hashes_str)}, dlen: {len(dataset_hashes_str)}, avg_time: {avg_time:.4f} seconds")

    # Extract the start frame from the query path
    start_frame = int(query_path.split('downscale_')[1].split('_')[0])
    if best_start != -1:
        if start_frame == best_start:
            print(f"✅ Match successful, {best_dataset_idx}, {best_start}")
        else:
            if abs(start_frame - best_start) == 1:
                print(f"⚠️ Match failed, start_frame: {start_frame}, best_start: {best_start}, dataset_idx: {best_dataset_idx}")
            else:
                print(f"❌ Match failed, start_frame: {start_frame}, best_start: {best_start}, dataset_idx: {best_dataset_idx}")
    else:
        print("❌ No match found")

if __name__ == "__main__":
    """
    Main function to demonstrate video hash matching functionality.
    """
    query_path = f"{search_config['TEST_VIDEO_DIR']}/SD24K_6235636_downscale_551_10.mp4"
    video_path = f"{search_config['VIDEO_DIR']}/SD24K_6235636_original.mp4"
    test_file_cpp_matcher(query_path, video_path, iterations=10)

    test_video_dir = search_config['TEST_VIDEO_DIR']
    video_dir = search_config['VIDEO_DIR']
    for file in os.listdir(test_video_dir):
        if file.endswith(".mp4") and "downscale" in file:
            query_path = os.path.join(test_video_dir, file)
            original_filename = file.split('downscale_')[0] + 'original.mp4'
            video_path = os.path.join(video_dir, original_filename)
            if os.path.exists(video_path):
                test_file_cpp_matcher(query_path, video_path, iterations=1)


