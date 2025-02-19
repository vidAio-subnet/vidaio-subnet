import cv2
import pyiqa
import torch
import numpy as np

# Initialize the LPIPS metric using pyiqa
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
iqa_metric = pyiqa.create_metric('lpips', device=device)  # LPIPS metric instance

def calculate_lpips(ref_frame: np.ndarray, dist_frame: np.ndarray) -> float:
    """
    Calculate LPIPS score between two frames (numpy arrays).
    Args:
        ref_frame (np.ndarray): Reference frame (H, W, 3) in BGR format.
        dist_frame (np.ndarray): Distorted frame (H, W, 3) in BGR format.
    Returns:
        float: LPIPS score (lower is better).
    """
    # Convert BGR to RGB
    print("calculationg lpips")
    
    ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
    dist_frame = cv2.cvtColor(dist_frame, cv2.COLOR_BGR2RGB)

    # Normalize pixel values to [0, 1]
    ref_frame = ref_frame / 255.0
    dist_frame = dist_frame / 255.0

    # Convert numpy arrays to PyTorch tensors and add batch dimension
    ref_tensor = torch.tensor(ref_frame).permute(2, 0, 1).unsqueeze(0).float().to(device)  # (1, 3, H, W)
    dist_tensor = torch.tensor(dist_frame).permute(2, 0, 1).unsqueeze(0).float().to(device)  # (1, 3, H, W)

    # Compute LPIPS score
    lpips_score = iqa_metric(ref_tensor, dist_tensor)
    print(f"Calculated LPIPS score is : {lpips_score.item()}")
    return lpips_score.item()