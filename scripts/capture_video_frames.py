import cv2
import os
import argparse
import numpy as np
from pathlib import Path

def capture_frames(video_path, output_dir, num_frames=3):
    """
    Captures the first num_frames frames from a video file at the video's native resolution.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save the captured frames
        num_frames (int): Number of frames to capture from the start
    
    Returns:
        list: Paths to the saved frames
    """
    # Check if video exists
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video filename without extension
    video_name = Path(video_path).stem
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        raise Exception(f"Error opening video file: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total Frames: {total_frames}")
    
    # Check if we have enough frames
    if total_frames < num_frames:
        print(f"Warning: Video has only {total_frames} frames, which is less than requested {num_frames} frames.")
        num_frames = total_frames
    
    # Capture the first num_frames frames
    saved_frames = []
    for i in range(num_frames):
        ret, frame = cap.read()
        
        if not ret:
            print(f"Error reading frame {i}")
            break
        
        # Save frame with its index, preserving original resolution
        frame_path = os.path.join(output_dir, f"{video_name}_frame_{i:03d}.png")
        cv2.imwrite(frame_path, frame)
        saved_frames.append(frame_path)
        
        print(f"Saved frame {i+1}/{num_frames}: {frame_path}")
    
    # Release the video capture object
    cap.release()
    
    return saved_frames

def main():
    parser = argparse.ArgumentParser(description="Capture frames from a video at original resolution")
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--output", default="frames", help="Directory to save the frames")
    parser.add_argument("--frames", type=int, default=3, help="Number of frames to capture from the start")
    
    args = parser.parse_args()
    
    try:
        saved_frames = capture_frames(args.video, args.output, args.frames)
        print(f"Successfully captured {len(saved_frames)} frames from the video")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
