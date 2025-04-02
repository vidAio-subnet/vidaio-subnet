import os
import cv2
import numpy as np
import torch
import pyiqa

def calculate_pieapp_score(reference_video_path, processed_video_path):
    """
    Calculate PieAPP score between a reference video and a processed video using pyiqa on GPU.
    
    Args:
        reference_video_path (str): Path to the reference video file.
        processed_video_path (str): Path to the processed video file.
    
    Returns:
        float: Average PieAPP score across all frames of the videos.
    """
    if not os.path.exists(reference_video_path):
        raise FileNotFoundError(f"Reference video not found at {reference_video_path}")
    if not os.path.exists(processed_video_path):
        raise FileNotFoundError(f"Processed video not found at {processed_video_path}")
    
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Open the reference and processed videos
    ref_cap = cv2.VideoCapture(reference_video_path)
    proc_cap = cv2.VideoCapture(processed_video_path)
    
    if not ref_cap.isOpened():
        raise ValueError(f"Unable to open reference video: {reference_video_path}")
    if not proc_cap.isOpened():
        raise ValueError(f"Unable to open processed video: {processed_video_path}")
    
    # Initialize PieAPP metric from pyiqa and move it to the GPU
    pieapp_metric = pyiqa.create_metric('PieAPP').to(device)
    
    scores = []
    while True:
        # Read frames from both videos
        ret_ref, ref_frame = ref_cap.read()
        ret_proc, proc_frame = proc_cap.read()
        
        # If either video ends, stop processing
        if not ret_ref or not ret_proc:
            break
        
        # Ensure both frames are of the same shape
        if ref_frame.shape != proc_frame.shape:
            raise ValueError("Reference and processed video frames have different dimensions")
        
        # Convert frames to RGB format (if needed by the metric) and normalize to [0, 1]
        ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        proc_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Convert frames to PyTorch tensors and move them to the GPU
        ref_tensor = pyiqa.utils.img2tensor(ref_frame, bgr2rgb=False).unsqueeze(0).to(device)
        proc_tensor = pyiqa.utils.img2tensor(proc_frame, bgr2rgb=False).unsqueeze(0).to(device)
        
        # Calculate PieAPP score for the current frame
        score = pieapp_metric(ref_tensor, proc_tensor).item()
        scores.append(score)
    
    # Release video captures
    ref_cap.release()
    proc_cap.release()
    
    # Return the average PieAPP score across all frames
    return np.mean(scores) if scores else None