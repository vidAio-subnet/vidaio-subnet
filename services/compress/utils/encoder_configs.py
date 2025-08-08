
ENCODER_SETTINGS = {
    "AV1_Optimized": {
        "codec": "libsvtav1", "preset": "8", "crf": 30, "keyint": 50,
        # Add SVT-AV1 specific AQ if needed, e.g., --enable-variance-boost
    },
    "libsvtav1": {
        "codec": "libsvtav1", "preset": "8", "crf": 30, "keyint": 50,
        # SVT-AV1 encoder settings
    },
    "av1_nvenc": {
        "codec": "av1_nvenc", "preset": "p6", "cq": 30, "keyint": 50, 'pix_fmt': 'yuv420p' # Default AQ settings for NVENC AV1
    },
    "av1_rust": {
        "codec": "librav1e", "preset": "4", "crf": 35, "keyint": 50,
        # rav1e has AQ settings too, check its documentation
    },
    "av1_fallback": {
        "codec": "libaom-av1", "preset": "medium", "crf": 30, "b:v": "0", "cpu-used": 4, "keyint": 50,
        "aq-mode": 1 # Default AQ mode for libaom
    },
    "h264": {
        "codec": "libx264", "preset": "medium", "crf": 23, "keyint": 50,
        "aq-mode": 1, "aq-strength": 1.0 # Default AQ for x264
    },
    "h266_vvc": {
        "codec": "libvvenc", "preset": "p4", "crf": 28, "keyint": 50,
        # Check libvvenc documentation for AQ parameters
    },
    "hevc": {
        "codec": "libx265", "preset": "medium", "crf": 28, "keyint": 50,
        "aq-mode": 2, "aq-strength": 1.0 # Default AQ for x265
    },
    "hevc_nvenc": {
        "codec": "hevc_nvenc", "preset": "p4", "rc": "constqp", "cq": 22, "keyint": 50,
        "spatial-aq": 1, "temporal-aq": 0 # Default AQ settings for NVENC HEVC
    },
    "h264_nvenc": {
        "codec": "h264_nvenc", "preset": "p4", "rc": "constqp", "cq": 22, "keyint": 50,
        "spatial-aq": 1, "temporal-aq": 0 # Default AQ settings for NVENC H264
    },
    "h264_videotoolbox": {
        "codec": "h264_videotoolbox", 
        "profile:v": "main", 
        "crf": 23, 
        "keyint": 50,
        'pix_fmt': 'yuv420p' # Often required for hardware encoders
    },
    "hevc_videotoolbox": {
        "codec": "hevc_videotoolbox",
        "profile:v": "main", 
        "crf": 28, 
        "keyint": 50,
        'pix_fmt': 'yuv420p'
    },
    "ffv1": {
        "codec": "ffv1",
        "level": 3,           
        "coder": 1,           
        "context": 1,         
        "g": 1,               
        "slices": 4,          
        "slicecrc": 1,        
        "pix_fmt": "yuv420p", 
    },
}

SCENE_SPECIFIC_PARAMS = {
    'AV1_NVENC': {
        'Screen Content / Text': {'preset': 'p7', 'spatial-aq': 1, 'temporal-aq': 0, 'keyint': 250},
        'Faces / People': {'preset': 'p6', 'spatial-aq': 1, 'temporal-aq': 1, 'keyint': 100},
        'Animation / Cartoon / Rendered Graphics': {'preset': 'p5', 'spatial-aq': 1, 'temporal-aq': 0, 'keyint': 150},
        'Gaming Content': {'preset': 'p5', 'spatial-aq': 1, 'temporal-aq': 0, 'keyint': 75},
        'other': {'keyint': 100},
        'unclear': {'keyint': 100},
    },
    'HEVC_NVENC': {
        'Screen Content / Text': {'preset': 'p7', 'spatial-aq': 1, 'temporal-aq': 0, 'keyint': 250},
        'Faces / People': {'preset': 'p6', 'spatial-aq': 1, 'temporal-aq': 1, 'keyint': 100},
        'Animation / Cartoon / Rendered Graphics': {'preset': 'p5', 'spatial-aq': 1, 'temporal-aq': 0, 'keyint': 150},
        'Gaming Content': {'preset': 'p5', 'spatial-aq': 1, 'temporal-aq': 0, 'keyint': 75},
        'other': {'keyint': 100},
        'unclear': {'keyint': 100},
    },
    'H264_NVENC': {
        'Screen Content / Text': {'preset': 'p7', 'spatial-aq': 1, 'temporal-aq': 0, 'keyint': 250},
        'Faces / People': {'preset': 'p6', 'spatial-aq': 1, 'temporal-aq': 1, 'keyint': 100},
        'Animation / Cartoon / Rendered Graphics': {'preset': 'p5', 'spatial-aq': 1, 'temporal-aq': 0, 'keyint': 150},
        'Gaming Content': {'preset': 'p5', 'spatial-aq': 1, 'temporal-aq': 0, 'keyint': 75},
        'other': {'keyint': 100},
        'unclear': {'keyint': 100},
    },
    'libaom-av1': {
        'Screen Content / Text': {'cpu-used': 3, 'tune': 'ssim', 'aq-mode': 2, 'keyint': 250},
        'Animation / Cartoon / Rendered Graphics': {'cpu-used': 5, 'tune': 'animation', 'aq-mode': 1, 'keyint': 150},
        'Faces / People': {'cpu-used': 4, 'tune': 'psnr', 'aq-mode': 1, 'keyint': 100},
        'Gaming Content': {'cpu-used': 6, 'tune': 'fastdecode', 'aq-mode': 1, 'keyint': 75},
        'other': {'cpu-used': 4, 'tune': 'film', 'aq-mode': 1, 'keyint': 100},
        'unclear': {'cpu-used': 4, 'tune': 'film', 'aq-mode': 1, 'keyint': 100},
    },
    'libx264': {
        'Screen Content / Text': {'preset': 'slow', 'tune': 'film', 'aq-mode': 1, 'aq-strength': 1.2, 'keyint': 250},
        'Animation / Cartoon / Rendered Graphics': {'preset': 'medium', 'tune': 'animation', 'aq-mode': 1, 'aq-strength': 1.0, 'keyint': 150},
        'Faces / People': {'preset': 'medium', 'tune': 'film', 'aq-mode': 1, 'aq-strength': 0.8, 'keyint': 100},
        'Gaming Content': {'preset': 'fast', 'tune': 'fastdecode', 'aq-mode': 1, 'aq-strength': 1.0, 'keyint': 75},
        'other': {'preset': 'medium', 'tune': 'film', 'aq-mode': 1, 'aq-strength': 1.0, 'keyint': 100},
        'unclear': {'preset': 'medium', 'tune': 'film', 'aq-mode': 1, 'aq-strength': 1.0, 'keyint': 100},
    },
     'libx265': {
        'Screen Content / Text': {'preset': 'medium', 'tune': 'grain', 'aq-mode': 2, 'aq-strength': 1.1, 'keyint': 250},
        'Animation / Cartoon / Rendered Graphics': {'preset': 'medium', 'tune': 'animation', 'aq-mode': 2, 'aq-strength': 1.0, 'keyint': 150},
        'Faces / People': {'preset': 'medium', 'tune': 'none', 'aq-mode': 2, 'aq-strength': 0.9, 'keyint': 100},
        'Gaming Content': {'preset': 'fast', 'tune': 'fastdecode', 'aq-mode': 2, 'aq-strength': 1.0, 'keyint': 75},
        'other': {'preset': 'medium', 'tune': 'none', 'aq-mode': 2, 'aq-strength': 1.0, 'keyint': 100},
        'unclear': {'preset': 'medium', 'tune': 'none', 'aq-mode': 2, 'aq-strength': 1.0, 'keyint': 100},
    },
}

MODEL_CQ_REFERENCE_CODEC = "libsvtav1" 

QUALITY_MAPPING_ANCHORS = {
    "libaom-av1": { 
        "model_ref_cq_range": [10, 63], 
        "target_param_type": "crf",
        "target_param_range": [10, 63], 
        "anchor_points": [ 
            [20, 22],
            [30, 32],
            [40, 42],
            [50, 52]
        ]
    },
    "H264": { 
        "model_ref_cq_range": [10, 63],
        "target_param_type": "crf",
        "target_param_range": [0, 51], 
        "anchor_points": [ 
            [20, 18],
            [30, 23],
            [40, 28],
            [50, 33]
        ]
    },

    "HEVC_NVENC": { 
        "model_ref_cq_range": [10, 63],
        "target_param_type": "cq",
        "target_param_range": [10, 51], 
        "anchor_points": [ 
            [20, 19], 
            [30, 25],
            [40, 31],
            [50, 37]
        ]
    },
    "libx265": { 
        "model_ref_cq_range": [10, 63],
        "target_param_type": "crf",
        "target_param_range": [0, 51], 
        "anchor_points": [ 
            [20, 20],
            [30, 26],
            [40, 32],
            [50, 38]
        ]
    },
}

CODEC_CQ_LIMITS = {
    'av1_nvenc': {
        'max_cq': 45,          
        'recommended_max': 40,  
        'quality_range': (20, 40), 
        'description': 'NVIDIA AV1 encoder'
    },
    'libsvtav1': {
        'max_cq': 50,
        'recommended_max': 45,
        'quality_range': (25, 45),
        'description': 'SVT-AV1 encoder'
    },
    'libaom-av1': {
        'max_cq': 50,
        'recommended_max': 45,
        'quality_range': (25, 45),
        'description': 'AOM AV1 encoder'
    },
    
    'h264_nvenc': {
        'max_cq': 32,
        'recommended_max': 28,
        'quality_range': (18, 28),
        'description': 'NVIDIA H.264 encoder'
    },
    'libx264': {
        'max_cq': 32,
        'recommended_max': 28,
        'quality_range': (18, 28),
        'description': 'x264 H.264 encoder'
    },
    
    'hevc_nvenc': {
        'max_cq': 35,
        'recommended_max': 30,
        'quality_range': (20, 30),
        'description': 'NVIDIA HEVC encoder'
    },
    'libx265': {
        'max_cq': 35,
        'recommended_max': 30,
        'quality_range': (20, 30),
        'description': 'x265 HEVC encoder'
    },
    
    'libvpx-vp9': {
        'max_cq': 45,
        'recommended_max': 40,
        'quality_range': (25, 40),
        'description': 'VP9 encoder'
    },
    
    'default': {
        'max_cq': 40,
        'recommended_max': 35,
        'quality_range': (20, 35),
        'description': 'Default codec limits'
    }
}