import os
import cv2
import numpy as np
import torch
import pyiqa

def calculate_pieapp_score(ref_cap, proc_cap):
    """
    Calculate PieAPP score between a reference video and a processed video using pyiqa on GPU.
    
    Returns:
        float: Average PieAPP score across all frames of the videos.
    """
    
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
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
    
    # Return the average PieAPP score across all frames
    return np.mean(scores) if scores else None