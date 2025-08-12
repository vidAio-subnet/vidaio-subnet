import os
import json
import numpy as np
import cv2
import subprocess

def analyze_video_fast(video_path, max_frames=150,logging_enabled=True,include_quality_metrics=False):
    """
    Enhanced video analysis with ONLY required features for efficient processing.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum frames to analyze
        logging_enabled: Whether to log progress
    """
    if not os.path.exists(video_path):
        if logging_enabled:
            print(f"❌ Video file not found: {video_path}")
        return None
    
    features = {}
    try:
        # Step 1: Get basic video properties using ffprobe
        basic_features = extract_basic_video_properties(video_path, logging_enabled)
        if not basic_features:
            return None
        
        features.update(basic_features)
        
        # Step 2: Extract comprehensive motion and complexity metrics using OpenCV
        advanced_features = extract_comprehensive_video_metrics(video_path, max_frames, logging_enabled)
        features.update(advanced_features)

        # Step 3: Add quality metrics for preprocessing if requested
        if include_quality_metrics:
            if logging_enabled:
                print("   🔍 Analyzing video quality for preprocessing recommendations...")
            quality_metrics = analyze_video_quality_metrics(video_path, num_frames=20)
            if quality_metrics:
                # Add quality metrics with prefix to avoid conflicts
                for key, value in quality_metrics.items():
                    features[f'quality_{key}'] = value
        
        if logging_enabled:
            print(f"✅ Video analysis completed:")
        
        return features
        
    except Exception as e:
        if logging_enabled:
            print(f"❌ Video analysis failed: {e}")
        return None

def extract_basic_video_properties(video_path, logging_enabled=True):
    """Extract basic video properties using ffprobe."""
    try:
        ffprobe_cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]
        
        result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            if logging_enabled:
                print(f"❌ ffprobe failed: {result.stderr}")
            return None
        
        probe_data = json.loads(result.stdout)
        
        video_stream = None
        for stream in probe_data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            if logging_enabled:
                print("❌ No video stream found")
            return None
        
        features = {}
        
        features['metrics_resolution_width'] = int(video_stream.get('width', 0))
        features['metrics_resolution_height'] = int(video_stream.get('height', 0))
        
        frame_rate_str = video_stream.get('r_frame_rate', '0/1')
        if '/' in frame_rate_str:
            num, den = frame_rate_str.split('/')
            features['metrics_frame_rate'] = float(num) / float(den) if float(den) != 0 else 0
        else:
            features['metrics_frame_rate'] = float(frame_rate_str)
        
        features['metrics_bit_depth'] = int(video_stream.get('bits_per_raw_sample', 8))
        
        features['input_codec'] = video_stream.get('codec_name', 'unknown')
        
        format_info = probe_data.get('format', {})
        bitrate_bps = 0
        
        if video_stream.get('bit_rate'):
            bitrate_bps = int(video_stream['bit_rate'])
        elif format_info.get('bit_rate'):
            bitrate_bps = int(format_info['bit_rate'])
        else:
            file_size_bytes = int(format_info.get('size', 0))
            duration_seconds = float(format_info.get('duration', 0))
            if file_size_bytes > 0 and duration_seconds > 0:
                bitrate_bps = int((file_size_bytes * 8) / duration_seconds)
        
        features['input_bitrate_kbps'] = bitrate_bps / 1000 if bitrate_bps > 0 else 0
        
        features['metrics_resolution'] = f"({features['metrics_resolution_width']}, {features['metrics_resolution_height']})"
        
        if (features['metrics_resolution_width'] > 0 and 
            features['metrics_resolution_height'] > 0 and 
            features['metrics_frame_rate'] > 0 and 
            features['input_bitrate_kbps'] > 0):
            
            pixels_per_second = (features['metrics_resolution_width'] * 
                               features['metrics_resolution_height'] * 
                               features['metrics_frame_rate'])
            features['bits_per_pixel'] = (features['input_bitrate_kbps'] * 1000) / pixels_per_second
        else:
            features['bits_per_pixel'] = 0
        
        return features
        
    except Exception as e:
        if logging_enabled:
            print(f"❌ Basic video property extraction failed: {e}")
        return None

def extract_comprehensive_video_metrics(video_path, max_frames=150, logging_enabled=True, use_middle_section=True):
    """
    Extract ALL required video metrics using OpenCV analysis.
    
    Args:
        use_middle_section (bool): If True, sample from middle 60% of video
    """
    features = {
        'metrics_avg_motion': 0,
        'metrics_avg_edge_density': 0,
        'metrics_avg_texture': 0,
        'metrics_avg_color_complexity': 0,
        'metrics_avg_motion_variance': 0,
        'metrics_avg_grain_noise': 0
    }
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            if logging_enabled:
                print(f"❌ Cannot open video: {video_path}")
            return features
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if use_middle_section and total_frames > max_frames * 2:
            start_frame = int(total_frames * 0.2)  # Skip first 20%
            end_frame = int(total_frames * 0.8)    # Skip last 20%
            effective_total = end_frame - start_frame
        else:
            start_frame = min(int(fps * 1.0), total_frames // 10)  # Skip first second or 10%
            end_frame = total_frames
            effective_total = end_frame - start_frame
        
        if effective_total > max_frames:
            frame_interval = effective_total // max_frames
        else:
            frame_interval = 1
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        edge_density_values = []
        texture_values = []
        color_complexity_values = []
        spatial_info_values = []
        temporal_info_values = []
        grain_noise_values = []
        motion_values = []
        
        prev_frame = None
        frame_count = 0
        processed_frames = 0
        current_frame_pos = start_frame
        
        if logging_enabled:
            print(f"   📊 Analyzing {min(max_frames, effective_total)} frames from middle section...")
            print(f"   📍 Sampling range: frame {start_frame} to {end_frame} ({effective_total} frames)")
        
        while processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval != 0:
                frame_count += 1
                current_frame_pos += 1
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            edge_density = compute_edge_density(gray)
            edge_density_values.append(edge_density)
            
            texture_complexity = compute_texture_complexity(gray)
            texture_values.append(texture_complexity)
            
            color_complexity = compute_color_complexity(frame)
            color_complexity_values.append(color_complexity)
            
            spatial_info = compute_spatial_information(gray)
            spatial_info_values.append(spatial_info)
            
            grain_noise = compute_grain_noise_level(gray)
            grain_noise_values.append(grain_noise)
            
            if prev_frame is not None:
                motion = compute_motion_metric(prev_frame, gray)
                motion_values.append(motion)
                temporal_info = compute_temporal_information(prev_frame, gray)
                temporal_info_values.append(temporal_info)
            
            prev_frame = gray.copy()
            processed_frames += 1
            frame_count += 1
            current_frame_pos += 1
        
        cap.release()
        
        if edge_density_values:
            features['metrics_avg_edge_density'] = np.mean(edge_density_values)
        
        if texture_values:
            features['metrics_avg_texture'] = np.mean(texture_values)
        
        if color_complexity_values:
            features['metrics_avg_color_complexity'] = np.mean(color_complexity_values)
        
        if spatial_info_values:
            features['metrics_avg_spatial_information'] = np.mean(spatial_info_values)
        
        if grain_noise_values:
            features['metrics_avg_grain_noise'] = np.mean(grain_noise_values)
        
        if motion_values:
            features['metrics_avg_motion'] = np.mean(motion_values)
            features['metrics_avg_motion_variance'] = np.var(motion_values)
        
        if temporal_info_values:
            features['metrics_avg_temporal_information'] = np.mean(temporal_info_values)
        

        if logging_enabled:
            print(f"   ✅ Processed {processed_frames} frames")
        return features
        
    except Exception as e:
        if logging_enabled:
            print(f"⚠️ Comprehensive video analysis failed: {e}")
        
        return {
            'metrics_avg_motion': 0.1,
            'metrics_avg_edge_density': 0.05,      
            'metrics_avg_texture': 4.0,            
            'metrics_avg_color_complexity': 3.0,   
            'metrics_avg_motion_variance': 1.0,    
            'metrics_avg_grain_noise': 5.0         
        }

def compute_edge_density(gray_frame, threshold1=100, threshold2=200):
    """Compute edge density using Canny edge detection."""
    edges = cv2.Canny(gray_frame, threshold1, threshold2)
    edge_pixels = np.count_nonzero(edges)
    total_pixels = edges.size
    return edge_pixels / total_pixels

def compute_texture_complexity(gray_frame):
    """Compute texture complexity using histogram entropy."""
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256]).ravel()
    hist_norm = hist / hist.sum()  # normalize histogram
    entropy = -np.sum([p * np.log2(p) for p in hist_norm if p > 0])
    return entropy

def compute_color_complexity(frame):
    """Compute color complexity as average entropy across color channels."""
    channels = cv2.split(frame)
    entropies = []
    for channel in channels:
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).ravel()
        hist_sum = np.sum(hist)
        if hist_sum > 0:
            hist_norm = hist / hist_sum
            entropy = -np.sum([p * np.log2(p) for p in hist_norm if p > 0])
        else:
            entropy = 0
        entropies.append(entropy)
    return np.mean(entropies)

def compute_spatial_information(gray_frame):
    """Compute spatial information using Sobel gradients."""
    sobelx = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    return np.std(gradient_magnitude)

def compute_temporal_information(prev_gray, curr_gray):
    """Compute temporal information as standard deviation of frame difference."""
    diff = cv2.absdiff(prev_gray, curr_gray)
    return np.std(diff)

def compute_motion_metric(prev_gray, curr_gray):
    """Compute motion metric as normalized frame difference."""
    diff = cv2.absdiff(prev_gray, curr_gray)
    motion_value = np.mean(diff) / 255.0  # Normalize to 0-1
    return motion_value

def compute_grain_noise_level(gray_frame, kernel_size=3):
    """Compute grain/noise level using high-pass filtering."""
    blurred = cv2.GaussianBlur(gray_frame, (kernel_size, kernel_size), 0)
    noise = gray_frame.astype(np.float32) - blurred.astype(np.float32)
    return np.std(noise)

def analyze_video_quality_metrics(video_path, num_frames=30):
    """
    Analyze video to determine which preprocessing filters would be beneficial.
    Returns a dictionary of quality metrics and recommendations.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    metrics = {
        'noise_level': 0,
        'sharpness': 0,
        'contrast': 0,
        'brightness': 0,
        'color_saturation': 0,
        'motion_blur': 0,
        'compression_artifacts': 0,
        'text_content': 0,
        'edge_density': 0,
        'temporal_consistency': 0
    }
    
    frames = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames)
    
    while frame_count < num_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * step)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1
    
    cap.release()
    
    if not frames:
        return None
    
    # Analyze each metric
    metrics.update({
        'noise_level': _analyze_noise_level(frames),
        'sharpness': _analyze_sharpness(frames),
        'contrast': _analyze_contrast(frames),
        'brightness': _analyze_brightness(frames),
        'color_saturation': _analyze_color_saturation(frames),
        'motion_blur': _analyze_motion_blur(frames),
        'compression_artifacts': _analyze_compression_artifacts(frames),
        'text_content': _analyze_text_content(frames),
        'edge_density': _analyze_edge_density(frames),
        'temporal_consistency': _analyze_temporal_consistency(frames)
    })
    
    return metrics

def _analyze_noise_level(frames):
    """Estimate noise level in video frames."""
    noise_scores = []
    
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_var = laplacian.var()
        
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        high_freq_mask = np.zeros_like(magnitude)
        high_freq_mask[center_h-h//8:center_h+h//8, center_w-w//8:center_w+w//8] = 1
        high_freq_energy = np.sum(magnitude * (1 - high_freq_mask)) / np.sum(magnitude)
        
        noise_scores.append(min(noise_var / 1000, high_freq_energy * 2))
    
    return np.mean(noise_scores)

def _analyze_sharpness(frames):
    """Analyze image sharpness using multiple methods."""
    sharpness_scores = []
    
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = np.mean(sobelx**2 + sobely**2)
        
        combined_sharpness = (laplacian_var / 2000 + tenengrad / 50000) / 2
        sharpness_scores.append(min(combined_sharpness, 1.0))
    
    return np.mean(sharpness_scores)

def _analyze_contrast(frames):
    """Analyze contrast using RMS contrast and histogram analysis."""
    contrast_scores = []
    
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        rms_contrast = np.std(gray) / np.mean(gray) if np.mean(gray) > 0 else 0
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / hist.sum()
        
        pixels_in_range = []
        for i in range(0, 256, 32):
            pixels_in_range.append(np.sum(hist_norm[i:i+32]))
        
        hist_spread = np.std(pixels_in_range)
        
        combined_contrast = (rms_contrast / 100 + hist_spread * 3) / 2
        contrast_scores.append(min(combined_contrast, 1.0))
    
    return np.mean(contrast_scores)

def _analyze_brightness(frames):
    """Analyze brightness distribution."""
    brightness_scores = []
    
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray) / 255.0
        brightness_scores.append(mean_brightness)
    
    return np.mean(brightness_scores)

def _analyze_color_saturation(frames):
    """Analyze color saturation levels."""
    saturation_scores = []
    
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        mean_saturation = np.mean(saturation) / 255.0
        saturation_scores.append(mean_saturation)
    
    return np.mean(saturation_scores)

def _analyze_motion_blur(frames):
    """Detect motion blur using edge analysis."""
    if len(frames) < 2:
        return 0
    
    blur_scores = []
    
    for i in range(len(frames) - 1):
        gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        
        edges1 = cv2.Canny(gray1, 50, 150)
        edges2 = cv2.Canny(gray2, 50, 150)
        

        edge_diff = np.sum(np.abs(edges1.astype(float) - edges2.astype(float)))
        frame_size = gray1.shape[0] * gray1.shape[1]
        blur_score = edge_diff / frame_size / 255.0
        
        blur_scores.append(blur_score)
    
    return np.mean(blur_scores)

def _analyze_compression_artifacts(frames):
    """Detect compression artifacts using blockiness detection."""
    artifact_scores = []
    
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        block_diffs = []
        h, w = gray.shape
        
        for y in range(0, h - 8, 8):
            for x in range(0, w - 8, 8):
                block = gray[y:y+8, x:x+8]
                
                if x + 8 < w:
                    right_block = gray[y:y+8, x+8:x+16] if x+16 <= w else None
                    if right_block is not None:
                        edge_diff = np.mean(np.abs(block[:, -1].astype(float) - right_block[:, 0].astype(float)))
                        block_diffs.append(edge_diff)
        
        if block_diffs:
            artifact_score = np.mean(block_diffs) / 255.0
            artifact_scores.append(artifact_score)
    
    return np.mean(artifact_scores) if artifact_scores else 0

def _analyze_text_content(frames):
    """Detect text/high-contrast content."""
    text_scores = []
    
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contrast_regions = np.sum(binary > 0) / (binary.shape[0] * binary.shape[1])
        
        text_score = (edge_density * 2 + abs(contrast_regions - 0.5) * 2)
        text_scores.append(min(text_score, 1.0))
    
    return np.mean(text_scores)

def _analyze_edge_density(frames):
    """Analyze edge density for detail preservation."""
    edge_scores = []
    
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        edge_scores.append(edge_density)
    
    return np.mean(edge_scores)

def _analyze_temporal_consistency(frames):
    """Analyze temporal consistency between frames."""
    if len(frames) < 3:
        return 1.0
    
    consistency_scores = []
    
    for i in range(1, len(frames) - 1):
        prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
        
        diff1 = np.mean(np.abs(curr_gray.astype(float) - prev_gray.astype(float)))
        diff2 = np.mean(np.abs(next_gray.astype(float) - curr_gray.astype(float)))
        
        consistency = 1.0 - min(abs(diff1 - diff2) / 255.0, 1.0)
        consistency_scores.append(consistency)
    
    return np.mean(consistency_scores)