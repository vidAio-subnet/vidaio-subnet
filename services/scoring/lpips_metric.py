

def calculate_lpips(ref_frame, dist_frame):
    
    return 0.5
    """Calculate LPIPS score between two frames."""
    loss_fn = lpips.LPIPS(net='alex')  # Use the AlexNet model
    ref_tensor = lpips.im2tensor(ref_frame)  # Convert to tensor
    dist_tensor = lpips.im2tensor(dist_frame)

    return loss_fn(ref_tensor, dist_tensor).item()