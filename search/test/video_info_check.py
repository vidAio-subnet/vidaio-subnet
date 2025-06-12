from moviepy.editor import VideoFileClip
import cv2
import asyncio
import time
import subprocess

videodb_path = "/root/vidaio/videodb"
db_original_file = "HD24K_854678_original.mp4"
db_trimmed_file = "HD24K_854678_trim.mp4"
db_downscaled_file = "HD24K_854678_downscale-2x_FPS-29.97_Frames-30.mp4"

search_engine_start_frame = 199

ffmpeg_clipped_path = f"ffmpeg_{db_trimmed_file}"
ffmpeg_downscaled_path = f"ffmpeg_{db_downscaled_file}"

async def main():
    print("running main")
    cap = cv2.VideoCapture(f"{videodb_path}/{db_original_file}")
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{videodb_path}/{db_original_file} fps: {original_fps}, width: {original_width}, height: {original_height}, total_frames: {original_total_frames}")
    cap.release()

    cap = cv2.VideoCapture(f"{videodb_path}/{db_trimmed_file}")
    trimmed_fps = cap.get(cv2.CAP_PROP_FPS)
    trimmed_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    trimmed_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    trimmed_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{videodb_path}/{db_trimmed_file} fps: {trimmed_fps}, width: {trimmed_width}, height: {trimmed_height}, total_frames: {trimmed_total_frames}")
    cap.release()

    start_time_clip = search_engine_start_frame / original_fps
    actual_duration = trimmed_total_frames / original_fps
    
    trim_cmd = [
        "taskset", "-c", "0,1,2,3,4,5",
        "ffmpeg", "-y", "-i", str(f"{videodb_path}/{db_original_file}"), "-ss", str(start_time_clip), 
        "-t", str(actual_duration), "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac", str(f"{ffmpeg_clipped_path}"), "-hide_banner", "-loglevel", "error"
    ]
    
    scale_cmd = [
        "taskset", "-c", "0,1,2,3,4,5",
        "ffmpeg", "-y", "-i", str(ffmpeg_clipped_path), "-vf", f"scale=-1:{1080}", 
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", 
        str(f"{ffmpeg_downscaled_path}"), "-hide_banner", "-loglevel", "error"
    ]
    
    try:
        chunk_start_time = time.time()
        subprocess.run(trim_cmd, check=True)
        subprocess.run(scale_cmd, check=True)
        
        chunk_elapsed_time = time.time() - chunk_start_time
        print(f"Time taken to process chunk: {chunk_elapsed_time:.2f} seconds")
        
        cap = cv2.VideoCapture(ffmpeg_clipped_path)
        ffmpeg_clipped_fps = cap.get(cv2.CAP_PROP_FPS)
        ffmpeg_clipped_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ffmpeg_clipped_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ffmpeg_clipped_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"{ffmpeg_clipped_path} fps: {ffmpeg_clipped_fps}, width: {ffmpeg_clipped_width}, height: {ffmpeg_clipped_height}, total_frames: {ffmpeg_clipped_total_frames}")
        cap.release()

        cap = cv2.VideoCapture(ffmpeg_downscaled_path)
        ffmpeg_downscaled_fps = cap.get(cv2.CAP_PROP_FPS)
        ffmpeg_downscaled_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ffmpeg_downscaled_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ffmpeg_downscaled_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"{ffmpeg_downscaled_path} fps: {ffmpeg_downscaled_fps}, width: {ffmpeg_downscaled_width}, height: {ffmpeg_downscaled_height}, total_frames: {ffmpeg_downscaled_total_frames}")
        cap.release()

                
    except subprocess.SubprocessError as e:
        print(f"Error processing chunk: {e}")


if __name__ == "__main__":
    asyncio.run(main())










