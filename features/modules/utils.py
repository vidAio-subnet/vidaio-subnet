from moviepy.editor import VideoFileClip
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
import subprocess

def clip_video(input_path, output_path, start_frame, total_frame):
    cap = cv2.VideoCapture(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    end_frame = start_frame + total_frame

    # Set starting point
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Define the codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"search_engine: Clipping from frame {start_frame} to {end_frame} (FPS: {fps})")

    current_frame = start_frame
    while cap.isOpened() and current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1

    cap.release()
    out.release()
    # print(f"search_engine: Clipped video saved to {output_path}")

def get_frames(input_path) -> int:
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def get_fps(input_path) -> float:
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def clip_video_precise(input_path, output_path, start_frame, total_frame):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    start_time = start_frame / fps
    duration = total_frame / fps

    command = [
        "ffmpeg",
        "-y",
        "-ss", str(start_time),
        "-i", input_path,
        "-t", str(duration),
        "-c:v", "libx264",  # Re-encode video
        "-preset", "fast",
        "-crf", "16",       # Visually lossless
        "-c:a", "copy",     # Copy audio
        output_path
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,   # Suppress standard output
        stderr=subprocess.DEVNULL    # Suppress error/log output
    )


def to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

def to_gray_resized(frame, size=(64, 64)):
    resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

def compute_similarity(f1, f2):
    g1 = to_gray(f1)
    g2 = to_gray(f2)
    diff = np.abs(g1.astype(np.int16) - g2.astype(np.int16))
    mae = np.mean(diff)
    return mae  # lower error = better match, so return negative for max()

def find_query_start_frame1(original_path, query_path):
    original_clip = VideoFileClip(original_path)
    query_clip = VideoFileClip(query_path)
    _, origin_height = original_clip.size
    _, query_height = query_clip.size
    print(f"search_engine: query height: {query_height}")
    if origin_height != query_height:
        return -1, -1

    original_frames = list(original_clip.iter_frames(fps=original_clip.fps))
    max_frames = int(query_clip.fps * query_clip.duration)
    query_frames = list(query_clip.iter_frames(fps=query_clip.fps))[:max_frames]
    query_len = len(query_frames)
    print(f"search_engine: query_len: {query_len}")

    min_dist = float('inf')
    best_id = -1

    for i in range(len(original_frames) - query_len + 1):
        dist = compute_similarity(original_frames[i], query_frames[0])
        dist += compute_similarity(original_frames[i + query_len // 2], query_frames[query_len // 2])
        dist += compute_similarity(original_frames[i + query_len - 1], query_frames[-1])
        if dist < min_dist:
            min_dist = dist
            best_id = i
    original_clip.close()
    query_clip.close()
    return best_id, min_dist

def find_query_start_frame(original_path, query_path):
    original_clip = VideoFileClip(original_path)
    query_clip = VideoFileClip(query_path)
    origin_width, origin_height = original_clip.size
    query_width, query_height = query_clip.size

    if origin_height != query_height or origin_width != query_width:
        return -1, -1

    # Precompute grayscale reference frames
    downscale_original = original_clip.resize(height=query_height // 4)
    downscale_query = query_clip.resize(height=query_height // 4)

    query_frames = list(downscale_query.iter_frames(fps=downscale_query.fps))
    query_len = len(query_frames)

    q_start = to_gray(query_frames[0])
    q_mid = to_gray(query_frames[query_len // 2])
    q_end = to_gray(query_frames[-1])

    min_dist = float('inf')
    best_id = -1

    original_frames = list(downscale_original.iter_frames(fps=downscale_original.fps))
    for i in range(len(original_frames) - query_len + 1):
        # Grab 3 sample frames from the original at positions: start, mid, end
        try:
            f_start = to_gray(original_frames[i])
            f_mid = to_gray(original_frames[i + query_len // 2])
            f_end = to_gray(original_frames[i + query_len - 1])
        except:
            break

        # Compute simple similarity (MAE)
        dist = np.mean(np.abs(f_start.astype(np.int16) - q_start.astype(np.int16)))
        dist += np.mean(np.abs(f_mid.astype(np.int16) - q_mid.astype(np.int16)))
        dist += np.mean(np.abs(f_end.astype(np.int16) - q_end.astype(np.int16)))

        if dist < min_dist:
            min_dist = dist
            best_id = i

    original_clip.close()
    query_clip.close()
    return best_id, min_dist
