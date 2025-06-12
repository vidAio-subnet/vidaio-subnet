import imagehash
from PIL import Image
import cv2
import time
from search.hashmatcher import cmatcher
import numpy as np
import os

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

def match_query_in_video_old(query_hashes, dataset_hashes):
    q_len = len(query_hashes)
    best_score = float("inf")
    best_start = -1

    for i in range(len(dataset_hashes) - q_len + 1):
        score = 0
        for j in range(q_len):
            score += query_hashes[j] - dataset_hashes[i + j]
        if score < best_score:
            best_score = score
            best_start = i    
    return best_start, best_score

def match_query_in_video_np(q_len, query_bits_coarse, query_bits_fine, dataset_bits, coarse_interval : int = 10):
    d_len = len(dataset_bits)

    clip_len = len(query_bits_coarse)

    scores = [float('inf')] * 3  # Array with 3 items, each has integer inf values
    positions = [-1] * 3  # Array with 3 items, each has integer -1 values
    worst_index = 0
    
    for i in range(0, d_len - q_len + 1, coarse_interval):
        total_score = 0

        # Vectorized Hamming distance computation per frame
        diffs = query_bits_coarse ^ dataset_bits[i:i+clip_len]
        frame_scores = diffs.sum(axis=1)
        total_score = frame_scores.sum()
        
        if total_score < scores[worst_index]:
            scores[worst_index] = total_score
            positions[worst_index] = i
            worst_index = np.argmax(scores)

    
    # Sort positions array directly
    positions.sort()

    # Fine search around the best positions found in coarse search
    best_score = float('inf')
    best_start = -1
    
    # Define search window around each position
    prev_pos = -1
    clip_len = len(query_bits_fine)
    for pos in positions:
        if pos == -1:
            continue
        
        # Calculate search boundaries
        start_idx = max(0, pos - coarse_interval)
        end_idx = min(d_len - q_len + 1, pos + coarse_interval)
        if prev_pos != -1 and pos - prev_pos == coarse_interval:
            start_idx = pos
        # Fine search in the window
        print(f"start_idx: {start_idx}, end_idx: {end_idx}")
        for i in range(start_idx, end_idx):
            # Vectorized Hamming distance computation for clipped length
            diffs = query_bits_fine ^ dataset_bits[i:i+clip_len]
            frame_scores = diffs.sum(axis=1)
            total_score = frame_scores.sum()
            if total_score < best_score:
                best_score = total_score
                best_start = i
        prev_pos = pos
    return best_start, best_score

def test_file(query_path, video_path, iterations=10):
    # Generate hashes for both videos
    start_time = time.time()
    video_hashes, fps, width, height, frame_count = video_to_phashes(video_path, hash_size=16)
    query_hashes, *_ = video_to_phashes(query_path, hash_size=16)
    end_time = time.time()
    #print(f"Time taken to generate hashes: {end_time - start_time:.2f} seconds")
    #print(f"FPS: {fps}, Width: {width}, Height: {height}, Frame count: {frame_count}, Duration: {frame_count/fps : .2f} seconds")
    #print("Type of video_hashes:", type(video_hashes))
    #print("Type of first element in video_hashes:", type(video_hashes[0]))

    query_bits = np.array([np.unpackbits(np.array(h.hash, dtype=np.uint8)) for h in query_hashes])
    dataset_bits = np.array([np.unpackbits(np.array(h.hash, dtype=np.uint8)) for h in video_hashes])

    query_bits_coarse = query_bits[:int(fps*2)]
    query_bits_fine = query_bits[:int(fps*5)]
    q_len = len(query_bits)

    start_time = time.time()
    total_time = 0
    for _ in range(iterations):
        start_time = time.time()
        best_start, best_score = match_query_in_video_np(q_len, query_bits_coarse, query_bits_fine, dataset_bits, coarse_interval=int(fps/2))
        end_time = time.time()
        total_time += (end_time - start_time)
    avg_time = total_time / iterations
    print(f"{os.path.basename(query_path)} : q_len: {q_len}, d_len: {len(dataset_bits)}, avg_time: {avg_time:.4f} seconds")

    # Extract the start frame from the query path
    start_frame = int(query_path.split('downscale_')[1].split('_')[0])
    if best_start != -1:
        if start_frame == best_start:
            print(f"✅ Match successful, score: {best_score}")
        else:
            if abs(start_frame - best_start) == 1:
                print(f"⚠️ Match failed, start_frame: {start_frame}, best_start: {best_start}, score: {best_score}")
            else:
                print(f"❌ Match failed, start_frame: {start_frame}, best_start: {best_start}, score: {best_score}")
    else:
        print("❌ No match found")

    # str_video_hashes = [str(hash) for hash in video_hashes]
    # str_query_hashes = [str(hash) for hash in query_hashes]
    # start_time = time.time()
    # best_start, best_score = cmatcher.match_query_in_video(str_query_hashes, str_video_hashes)
    # end_time = time.time()
    # print(f"Time taken to find best match: {end_time - start_time:.2f} seconds")
    # print(f"Best match found at frame {best_start}")
    # print(f"Match score: {best_score}")

if __name__ == "__main__":
    """
    Main function to demonstrate video hash matching functionality.
    """
    query_path = "/root/vidaio/test_videos/SD24K_6235636_downscale_551_10.mp4"
    video_path = "/root/vidaio/video_src_files/SD24K_6235636_original.mp4"
    test_file(query_path, video_path, iterations=10)

    test_video_dir = "/root/vidaio/test_videos"
    video_dir = "/root/vidaio/video_src_files"
    for file in os.listdir(test_video_dir):
        if file.endswith(".mp4") and "downscale" in file:
            query_path = os.path.join(test_video_dir, file)
            original_filename = file.split('downscale_')[0] + 'original.mp4'
            video_path = os.path.join(video_dir, original_filename)
            if os.path.exists(video_path):
                test_file(query_path, video_path, iterations=1)


