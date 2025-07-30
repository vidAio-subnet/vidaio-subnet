
# Configurable parameters for all encoders
# Base settings, including default AQ where applicable
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
        "profile:v": "main", # Example: main profile, can be high, baseline etc.
        # VideoToolbox might use -q:v for quality or rely on bitrate.
        # FFmpeg's wrapper might accept -crf for some videotoolbox encoders, but it's not universal.
        # For CQ-like behavior, you might need to experiment with -q:v or a specific -b:v.
        # Using a placeholder CRF here, assuming ffmpeg handles it or it gets overridden by mapping.
        "crf": 23, # Placeholder, check ffmpeg docs for h264_videotoolbox quality params
        "keyint": 50,
        'pix_fmt': 'yuv420p' # Often required for hardware encoders
    },
    "hevc_videotoolbox": {
        "codec": "hevc_videotoolbox",
        "profile:v": "main", # Example
        # Similar to h264_videotoolbox, quality parameters might vary.
        "crf": 28, # Placeholder
        "keyint": 50,
        'pix_fmt': 'yuv420p'
    },
    # FFV1 lossless codec
    "ffv1": {
        "codec": "ffv1",
        "level": 3,           # FFV1 level 3 is most common
        "coder": 1,           # Range coder (better compression)
        "context": 1,         # Large context
        "g": 1,               # GOP size 1 for lossless
        "slices": 4,          # Number of slices for parallel processing
        "slicecrc": 1,        # Enable slice CRC for error detection
        "pix_fmt": "yuv420p", # Pixel format
    },
}

# Scene-Specific Parameter Overrides (including AQ and keyint)
# These are examples and need tuning based on content and codec specifics.
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
    # Add/update other codecs as needed
}



# --- START: New Configuration for Quality Parameter Mapping ---

# Define which codec's CQ scale your VMAF prediction model was trained on.
# Example: If your model predicts CQ values suitable for libsvtav1.
MODEL_CQ_REFERENCE_CODEC = "libsvtav1" # IMPORTANT: User must set this correctly

# Define anchor points for mapping the model's reference CQ (from MODEL_CQ_REFERENCE_CODEC)
# to other codecs' quality parameters.
# This requires empirical tuning and knowledge of codec quality scales.
# 'model_ref_cq_range': The typical input CQ range from your VMAF model.
# 'target_param_type': 'cq' or 'crf' for the target codec.
# 'target_param_range': The typical/effective output range for the target codec's parameter.
# 'anchor_points': A list of [model_ref_cq_value, target_codec_param_value] pairs.
#                  These points define the mapping relationship.
QUALITY_MAPPING_ANCHORS = {
    # Example: If MODEL_CQ_REFERENCE_CODEC is "libsvtav1"
    "libaom-av1": { # Mapping libsvtav1 CQ to libaom-av1 CRF
        "model_ref_cq_range": [10, 63], # Expected input range from your AV1 model
        "target_param_type": "crf",
        "target_param_range": [10, 63], # Typical CRF range for libaom-av1
        "anchor_points": [ # [model_libsvtav1_cq, libaom_av1_crf]
            [20, 22], # Lower CQ (better quality) maps to lower CRF
            [30, 32], # Medium CQ maps to medium CRF
            [40, 42], # Higher CQ (lower quality) maps to higher CRF
            [50, 52]
        ]
    },
    "H264": { # Mapping libsvtav1 CQ to libx264 CRF
        "model_ref_cq_range": [10, 63],
        "target_param_type": "crf",
        "target_param_range": [0, 51],  # Typical CRF range for libx264
        "anchor_points": [ # [model_libsvtav1_cq, libx264_crf]
            [20, 18],
            [30, 23],
            [40, 28],
            [50, 33]
        ]
    },

    "HEVC_NVENC": { # Mapping libsvtav1 CQ to HEVC_NVENC CQ
        "model_ref_cq_range": [10, 63],
        "target_param_type": "cq",
        "target_param_range": [10, 51], # Typical CQ range for NVENC HEVC
        "anchor_points": [ # [model_libsvtav1_cq, hevc_nvenc_cq]
            [20, 19], # Might be slightly different scale or perception
            [30, 25],
            [40, 31],
            [50, 37]
        ]
    },
    "libx265": { # Mapping libsvtav1 CQ to libx265 CRF
        "model_ref_cq_range": [10, 63],
        "target_param_type": "crf",
        "target_param_range": [0, 51], # Typical CRF range for libx265
        "anchor_points": [ # [model_libsvtav1_cq, libx265_crf]
            [20, 20],
            [30, 26],
            [40, 32],
            [50, 38]
        ]
    },
    # Add other codecs as needed.
    # If a codec is the same as MODEL_CQ_REFERENCE_CODEC, it will use the CQ directly.
}

# =============================================================================
# CODEC-SPECIFIC CQ LIMITS AND QUALITY MAPPINGS
# =============================================================================

# Define maximum recommended CQ values for different codecs to maintain reasonable quality
CODEC_CQ_LIMITS = {
    # AV1 Codecs
    'av1_nvenc': {
        'max_cq': 45,           # Maximum CQ for reasonable quality
        'recommended_max': 40,   # Conservative maximum for good quality
        'quality_range': (20, 40),  # Typical quality range
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
    
    # H.264 Codecs
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
    
    # H.265/HEVC Codecs
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
    
    # VP9 Codec
    'libvpx-vp9': {
        'max_cq': 45,
        'recommended_max': 40,
        'quality_range': (25, 40),
        'description': 'VP9 encoder'
    },
    
    # Default fallback for unknown codecs
    'default': {
        'max_cq': 40,
        'recommended_max': 35,
        'quality_range': (20, 35),
        'description': 'Default codec limits'
    }
}