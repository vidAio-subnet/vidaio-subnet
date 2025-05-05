import os
import cv2
import numpy as np
import torch
import pyiqa
from tqdm import tqdm
import time

def calculate_pieapp_score(ref_cap, proc_cap, frame_interval=1):
    """
    Calculate PIE-APP score between reference and processed videos without extracting frames to disk.
    
    Args:
        frame_interval (int): Process one frame every `frame_interval` frames.
        
    Returns:
        float: Average PIE-APP score.
    """    
    if not ref_cap.isOpened():
        raise ValueError("Could not open reference video")
    if not proc_cap.isOpened():
        raise ValueError("Could not open processed video")
    
    # Get video info
    ref_frame_count = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    proc_frame_count = int(proc_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine how many frames to process
    total_frames = min(ref_frame_count, proc_frame_count)
    frames_to_process = total_frames // frame_interval

    print(f"Reference video: {ref_frame_count} frames")
    print(f"Processed video: {proc_frame_count} frames")
    print(f"Processing {frames_to_process} frames (every {frame_interval} frames)")
    
    # Initialize PIE-APP metric
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    pieapp_metric = pyiqa.create_metric('pieapp', device=device)
    
    scores = []
    frame_idx = 0

    with tqdm(total=frames_to_process, desc="Calculating PIE-APP") as pbar:
        while frame_idx < total_frames:
            # Read frames
            ref_ret, ref_frame = ref_cap.read()
            proc_ret, proc_frame = proc_cap.read()
            
            if not ref_ret or not proc_ret:
                break
            
            # Process every Nth frame
            if frame_idx % frame_interval == 0:
                # Convert BGR to RGB
                ref_frame_rgb = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
                proc_frame_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
                
                # Convert numpy arrays to tensors
                ref_tensor = torch.from_numpy(ref_frame_rgb).permute(2, 0, 1).float() / 255.0
                proc_tensor = torch.from_numpy(proc_frame_rgb).permute(2, 0, 1).float() / 255.0
                
                # Add batch dimension
                ref_tensor = ref_tensor.unsqueeze(0).to(device)
                proc_tensor = proc_tensor.unsqueeze(0).to(device)
                
                # Calculate PIE-APP score
                with torch.no_grad():
                    score = pieapp_metric(proc_tensor, ref_tensor)
                    score_value = score.item()  # Convert tensor to scalar first
                    if score_value < 0:
                        score_value = abs(score_value)  # Use Python's abs() on scalar
                scores.append(score_value)
                
                pbar.update(1)
            
            frame_idx += 1

    # Release resources
    ref_cap.release()
    proc_cap.release()

    # Return average score
    avg_score = np.mean(scores) if scores else 5.0
    
    return min(avg_score, 2.0) 
