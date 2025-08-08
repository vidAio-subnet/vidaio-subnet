import cv2
import numpy as np
import subprocess

def get_video_duration(video_path):
    """Get the duration of a video in seconds"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file to get duration: {video_path}")
        return 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps is None or fps == 0 or frame_count is None:
        cap.release()
        print(f"Warning: Could not get valid FPS ({fps}) or frame count ({frame_count}) for {video_path}")
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', video_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as e:
            print(f"Error getting duration with ffprobe fallback for {video_path}: {e}")
            return 0.0
    else:
        duration = frame_count / fps
        cap.release()
        return duration

def get_video_codec(video_path):
    """Get the codec of a video stream using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return result.stdout.strip()
    except FileNotFoundError:
        print("Error: ffprobe not found. Please ensure FFmpeg is installed and in your PATH.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error getting codec for {video_path}: {e.stderr}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while getting video codec: {e}")
        return None
