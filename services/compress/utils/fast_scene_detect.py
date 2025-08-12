"""
Fast scene detection utilities for video preprocessing.

This module provides efficient scene detection methods to determine
whether full PySceneDetect processing is needed or if a video can
be processed as a single scene.
"""

import cv2
import numpy as np
from typing import Tuple

def fast_scene_detection_check(video_path: str, duration: float, 
                             sample_frames: int = 10, threshold: float = 0.3,
                             logging_enabled: bool = True) -> Tuple[bool, int]:
    """
    Perform fast scene detection using frame sampling and histogram comparison.
    
    Args:
        video_path (str): Path to video file
        duration (float): Video duration in seconds
        sample_frames (int): Number of frames to sample
        threshold (float): Histogram difference threshold for scene detection
        logging_enabled (bool): Whether to log progress
        
    Returns:
        tuple: (has_multiple_scenes, estimated_scene_count)
    """
    if logging_enabled:
        print(f"      🔍 Sampling {sample_frames} frames for scene analysis...")
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            if logging_enabled:
                print("      ⚠️ Could not open video for sampling")
            return False, 1
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= sample_frames:
            cap.release()
            return False, 1
        
        # Sample frames evenly across the video
        frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
        histograms = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert to HSV and calculate histogram
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
                histograms.append(hist.flatten())
        
        cap.release()
        
        if len(histograms) < 2:
            return False, 1
        
        # Calculate histogram differences
        scene_changes = 0
        for i in range(1, len(histograms)):
            # Normalize histograms
            hist1 = histograms[i-1] / (np.sum(histograms[i-1]) + 1e-7)
            hist2 = histograms[i] / (np.sum(histograms[i]) + 1e-7)
            
            # Calculate correlation coefficient
            correlation = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)
            
            # If correlation is low, likely a scene change
            if correlation < (1.0 - threshold):
                scene_changes += 1
        
        has_multiple_scenes = scene_changes > 0
        estimated_scenes = scene_changes + 1
        
        if logging_enabled:
            print(f"      📊 Detected {scene_changes} potential scene changes")
            print(f"      📊 Estimated scenes: {estimated_scenes}")
        
        return has_multiple_scenes, estimated_scenes
        
    except Exception as e:
        if logging_enabled:
            print(f"      ❌ Fast scene detection failed: {e}")
        return False, 1

def motion_based_scene_check(video_path: str, duration: float,
                           sample_duration: float = 2.0, threshold: float = 0.4,
                           logging_enabled: bool = True) -> Tuple[bool, int]:
    """
    Detect scene changes based on motion analysis.
    
    Args:
        video_path (str): Path to video file
        duration (float): Video duration in seconds
        sample_duration (float): Duration of samples to analyze
        threshold (float): Motion change threshold
        logging_enabled (bool): Whether to log progress
        
    Returns:
        tuple: (has_multiple_scenes, estimated_scene_count)
    """
    if logging_enabled:
        print(f"      🏃 Analyzing motion patterns...")
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, 1
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or total_frames <= 0:
            cap.release()
            return False, 1
        
        # Sample multiple segments
        num_samples = max(3, int(duration / sample_duration))
        segment_length = total_frames // num_samples
        
        motion_scores = []
        
        for segment in range(num_samples):
            start_frame = segment * segment_length
            end_frame = min(start_frame + int(fps * sample_duration), total_frames - 1)
            
            if end_frame <= start_frame + 1:
                continue
            
            # Calculate motion for this segment
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, prev_frame = cap.read()
            
            if not ret:
                continue
            
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            segment_motion = 0
            frame_count = 0
            
            for frame_idx in range(start_frame + 1, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate optical flow magnitude
                flow = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, 
                    np.array([[100, 100]], dtype=np.float32), 
                    None, 
                    winSize=(15, 15),
                    maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )[0]
                
                if flow is not None and len(flow) > 0:
                    motion = np.mean(np.sqrt(flow[:, 0]**2 + flow[:, 1]**2))
                    segment_motion += motion
                    frame_count += 1
                
                prev_gray = gray
            
            if frame_count > 0:
                avg_motion = segment_motion / frame_count
                motion_scores.append(avg_motion)
        
        cap.release()
        
        if len(motion_scores) < 2:
            return False, 1
        
        # Detect significant changes in motion patterns
        scene_changes = 0
        mean_motion = np.mean(motion_scores)
        
        for i in range(1, len(motion_scores)):
            motion_change = abs(motion_scores[i] - motion_scores[i-1]) / (mean_motion + 1e-7)
            if motion_change > threshold:
                scene_changes += 1
        
        has_multiple_scenes = scene_changes > 0
        estimated_scenes = scene_changes + 1
        
        if logging_enabled:
            print(f"      📊 Motion-based scene changes: {scene_changes}")
        
        return has_multiple_scenes, estimated_scenes
        
    except Exception as e:
        if logging_enabled:
            print(f"      ❌ Motion-based scene detection failed: {e}")
        return False, 1

def adaptive_scene_detection_check(video_path: str, duration: float, 
                                 logging_enabled: bool = True) -> Tuple[bool, int, float]:
    """
    Perform adaptive scene detection using multiple methods.
    
    This function combines histogram-based and motion-based scene detection
    to provide a reliable estimate of scene count without full PySceneDetect.
    
    Args:
        video_path (str): Path to input video file
        duration (float): Video duration in seconds
        logging_enabled (bool): Whether to enable detailed logging
        
    Returns:
        tuple: (has_multiple_scenes, estimated_scene_count, analysis_time)
    """
    import time
    start_time = time.time()
    
    if logging_enabled:
        print(f"      🔍 Running adaptive scene detection analysis...")
    
    # Quick validation
    if duration < 5.0:  # Very short videos are likely single scene
        analysis_time = time.time() - start_time
        if logging_enabled:
            print(f"      📊 Video too short ({duration:.1f}s) - treating as single scene")
        return False, 1, analysis_time
    
    # Method 1: Histogram-based detection
    try:
        hist_multiple, hist_scenes = fast_scene_detection_check(
            video_path, duration, sample_frames=8, threshold=0.25, logging_enabled=logging_enabled
        )
    except Exception as e:
        if logging_enabled:
            print(f"      ⚠️ Histogram detection failed: {e}")
        hist_multiple, hist_scenes = False, 1
    
    # Method 2: Motion-based detection  
    try:
        motion_multiple, motion_scenes = motion_based_scene_check(
            video_path, duration, sample_duration=3.0, threshold=0.3, logging_enabled=logging_enabled
        )
    except Exception as e:
        if logging_enabled:
            print(f"      ⚠️ Motion detection failed: {e}")
        motion_multiple, motion_scenes = False, 1
    
    # Combine results using conservative approach
    if hist_multiple or motion_multiple:
        has_multiple_scenes = True
        # Use the higher estimate but cap it reasonably
        estimated_scenes = max(hist_scenes, motion_scenes)
        estimated_scenes = min(estimated_scenes, max(3, int(duration / 10)))  # Cap based on duration
    else:
        has_multiple_scenes = False
        estimated_scenes = 1
    
    analysis_time = time.time() - start_time
    
    if logging_enabled:
        print(f"      📊 Combined analysis results:")
        print(f"         Histogram method: {hist_scenes} scenes")
        print(f"         Motion method: {motion_scenes} scenes")
        print(f"         Final estimate: {estimated_scenes} scenes")
        print(f"         Analysis time: {analysis_time:.2f}s")
    
    return has_multiple_scenes, estimated_scenes, analysis_time