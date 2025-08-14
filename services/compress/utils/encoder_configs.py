
# Configurable parameters for all encoders
# Base settings, including default AQ where applicable
ENCODER_SETTINGS = {
    
    "libsvtav1": {  # Changed from "AV1_Optimized"
        "codec": "libsvtav1", "preset": "8", "crf": 30, "keyint": 50,
        # SVT-AV1 specific settings
    },
    "av1_nvenc": {
        "codec": "av1_nvenc", "preset": "p6", "cq": 30, "keyint": 50, 'pix_fmt': 'yuv420p'
    },
    "libvpx_vp9": {  # Changed from "vp9"
        "codec": "libvpx-vp9", "deadline": "good", "cpu-used": 2, "crf": 32, "keyint": 50,
        "aq-mode": 1, "arnr-maxframes": 7, "arnr-strength": 4, "auto-alt-ref": 1,
        "tune": "psnr", "row-mt": 1
    },
    "libx264": {  # Changed from "h264"
        "codec": "libx264", "preset": "medium", "crf": 23, "keyint": 50,
        "aq-mode": 1, "aq-strength": 1.0
    },
    "libvvenc": {  # Changed from "h266_vvc"
        "codec": "libvvenc", "preset": "medium", "crf": 28, "keyint": 50,
        # VVC/H.266 encoder
    },
    "libx265": {  # Changed from "hevc"
        "codec": "libx265", "preset": "medium", "crf": 28, "keyint": 50,
        "aq-mode": 2, "aq-strength": 1.0
    },
    "hevc_nvenc": {
        "codec": "hevc_nvenc", "preset": "p4", "rc": "constqp", "cq": 22, "keyint": 50,
        "spatial-aq": 1, "temporal-aq": 0
    },
    "h264_nvenc": {
        "codec": "h264_nvenc", "preset": "p4", "rc": "constqp", "cq": 22, "keyint": 50,
        "spatial-aq": 1, "temporal-aq": 0
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

# Scene-Specific Parameter Overrides (including AQ and keyint)
# These are examples and need tuning based on content and codec specifics.
SCENE_SPECIFIC_PARAMS = {
    'av1_nvenc': {  # Changed from 'AV1_NVENC'
        'Screen Content / Text': {'preset': 'p7', 'spatial-aq': 1, 'temporal-aq': 0, 'keyint': 250},
        'Faces / People': {'preset': 'p6', 'spatial-aq': 1, 'temporal-aq': 1, 'keyint': 100},
        'Animation / Cartoon / Rendered Graphics': {'preset': 'p5', 'spatial-aq': 1, 'temporal-aq': 0, 'keyint': 150},
        'Gaming Content': {'preset': 'p5', 'spatial-aq': 1, 'temporal-aq': 0, 'keyint': 75},
        'other': {'keyint': 100},
        'unclear': {'keyint': 100},
    },
    'hevc_nvenc': {  # Changed from 'HEVC_NVENC'
        'Screen Content / Text': {'preset': 'p7', 'spatial-aq': 1, 'temporal-aq': 0, 'keyint': 250},
        'Faces / People': {'preset': 'p6', 'spatial-aq': 1, 'temporal-aq': 1, 'keyint': 100},
        'Animation / Cartoon / Rendered Graphics': {'preset': 'p5', 'spatial-aq': 1, 'temporal-aq': 0, 'keyint': 150},
        'Gaming Content': {'preset': 'p5', 'spatial-aq': 1, 'temporal-aq': 0, 'keyint': 75},
        'other': {'keyint': 100},
        'unclear': {'keyint': 100},
    },
    'h264_nvenc': {  # Changed from 'H264_NVENC'
        'Screen Content / Text': {'preset': 'p7', 'spatial-aq': 1, 'temporal-aq': 0, 'keyint': 250},
        'Faces / People': {'preset': 'p6', 'spatial-aq': 1, 'temporal-aq': 1, 'keyint': 100},
        'Animation / Cartoon / Rendered Graphics': {'preset': 'p5', 'spatial-aq': 1, 'temporal-aq': 0, 'keyint': 150},
        'Gaming Content': {'preset': 'p5', 'spatial-aq': 1, 'temporal-aq': 0, 'keyint': 75},
        'other': {'keyint': 100},
        'unclear': {'keyint': 100},
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
        'Faces / People': {'preset': 'medium', 'aq-mode': 2, 'aq-strength': 0.9, 'keyint': 100},
        'Gaming Content': {'preset': 'fast', 'tune': 'fastdecode', 'aq-mode': 2, 'aq-strength': 1.0, 'keyint': 75},
        'other': {'preset': 'medium', 'aq-mode': 2, 'aq-strength': 1.0, 'keyint': 100},
        'unclear': {'preset': 'medium', 'aq-mode': 2, 'aq-strength': 1.0, 'keyint': 100},
    },
    'libvpx_vp9': {
        'Screen Content / Text': {'deadline': 'best', 'cpu-used': 1, 'aq-mode': 2, 'keyint': 250},
        'Faces / People': {'deadline': 'good', 'cpu-used': 2, 'aq-mode': 1, 'keyint': 100},
        'Animation / Cartoon / Rendered Graphics': {'deadline': 'good', 'cpu-used': 3, 'aq-mode': 1, 'keyint': 150},
        'Gaming Content': {'deadline': 'realtime', 'cpu-used': 5, 'aq-mode': 1, 'keyint': 75},
        'other': {'deadline': 'good', 'cpu-used': 2, 'aq-mode': 1, 'keyint': 100},
        'unclear': {'deadline': 'good', 'cpu-used': 2, 'aq-mode': 1, 'keyint': 100},
    },
    'libsvtav1': {
        'Screen Content / Text': {'preset': '6', 'tune': 0, 'keyint': 250},  # tune 0 = subjective quality
        'Faces / People': {'preset': '8', 'tune': 1, 'keyint': 100},        # tune 1 = objective quality
        'Animation / Cartoon / Rendered Graphics': {'preset': '7', 'tune': 0, 'keyint': 150},
        'Gaming Content': {'preset': '9', 'tune': 2, 'keyint': 75},         # tune 2 = fast decode
        'other': {'preset': '8', 'tune': 1, 'keyint': 100},
        'unclear': {'preset': '8', 'tune': 1, 'keyint': 100},
    },
    'libvvenc': {
        'Screen Content / Text': {'preset': 'slow', 'keyint': 250},
        'Faces / People': {'preset': 'medium', 'keyint': 100},
        'Animation / Cartoon / Rendered Graphics': {'preset': 'medium', 'keyint': 150},
        'Gaming Content': {'preset': 'fast', 'keyint': 75},
        'other': {'preset': 'medium', 'keyint': 100},
        'unclear': {'preset': 'medium', 'keyint': 100},
    },
}



# --- START: New Configuration for Quality Parameter Mapping ---

# Define which codec's CQ scale your VMAF prediction model was trained on.
# Example: If your model predicts CQ values suitable for AV1_NVENC.
MODEL_CQ_REFERENCE_CODEC = "av1_nvenc" # IMPORTANT: User must set this correctly

# Define anchor points for mapping the model's reference CQ (from MODEL_CQ_REFERENCE_CODEC)
# to other codecs' quality parameters.
# This requires empirical tuning and knowledge of codec quality scales.
# 'model_ref_cq_range': The typical input CQ range from your VMAF model.
# 'target_param_type': 'cq' or 'crf' for the target codec.
# 'target_param_range': The typical/effective output range for the target codec's parameter.
# 'anchor_points': A list of [model_ref_cq_value, target_codec_param_value] pairs.
#                  These points define the mapping relationship.
QUALITY_MAPPING_ANCHORS = {
    #   Changed "libaom-av1" to match ENCODER_SETTINGS
    #   Changed "H264" to "libx264" to match ENCODER_SETTINGS
    "libx264": { # Mapping av1_nvenc CQ to libx264 CRF
        "model_ref_cq_range": [10, 63],
        "target_param_type": "crf",
        "target_param_range": [0, 51],  # Typical CRF range for libx264
        "anchor_points": [ # [model_av1_nvenc_cq, libx264_crf]
            [20, 16], #   H.264 is less efficient, needs lower CRF
            [30, 21], #   More realistic mapping
            [40, 26], #   Conservative mapping for quality
            [50, 31]  #   Upper range adjustment
        ]
    },
    #   Changed "HEVC_NVENC" to "hevc_nvenc" to match ENCODER_SETTINGS
    "hevc_nvenc": { # Mapping av1_nvenc CQ to hevc_nvenc CQ
        "model_ref_cq_range": [10, 63],
        "target_param_type": "cq",
        "target_param_range": [10, 51], # Typical CQ range for NVENC HEVC
        "anchor_points": [ # [model_av1_nvenc_cq, hevc_nvenc_cq]
            [20, 18], #  HEVC NVENC slightly different scale
            [30, 23], #  More conservative mapping
            [40, 28], #  Quality preservation
            [50, 33]  #  Upper range adjustment
        ]
    },
    "libx265": { # Mapping av1_nvenc CQ to libx265 CRF
        "model_ref_cq_range": [10, 63],
        "target_param_type": "crf",
        "target_param_range": [0, 51], # Typical CRF range for libx265
        "anchor_points": [ # [model_av1_nvenc_cq, libx265_crf]
            [20, 18], #  x265 is more efficient than H.264
            [30, 23], #  Similar to H.264 but slightly better
            [40, 28], #  Conservative quality mapping
            [50, 33]  #  Upper range adjustment
        ]
    },
    #   Missing codec mappings
    "h264_nvenc": { # Mapping av1_nvenc CQ to h264_nvenc CQ
        "model_ref_cq_range": [10, 63],
        "target_param_type": "cq",
        "target_param_range": [10, 51], # Typical CQ range for NVENC H.264
        "anchor_points": [ # [model_av1_nvenc_cq, h264_nvenc_cq]
            [20, 17], # NVENC H.264 needs lower CQ for similar quality
            [30, 22],
            [40, 27],
            [50, 32]
        ]
    },
    "libvpx_vp9": { # Mapping av1_nvenc CQ to libvpx-vp9 CRF
        "model_ref_cq_range": [10, 63],
        "target_param_type": "crf",
        "target_param_range": [0, 63], # VP9 CRF range
        "anchor_points": [ # [model_av1_nvenc_cq, vp9_crf]
            [20, 22], # VP9 similar efficiency to AV1
            [30, 32],
            [40, 42],
            [50, 52]
        ]
    },
    "libsvtav1": { # Mapping av1_nvenc CQ to libsvtav1 CRF
        "model_ref_cq_range": [10, 63],
        "target_param_type": "crf",
        "target_param_range": [0, 63], # SVT-AV1 CRF range
        "anchor_points": [ # [model_av1_nvenc_cq, svt_av1_crf]
            [20, 21], # SVT-AV1 slightly different from NVENC
            [30, 31],
            [40, 41],
            [50, 51]
        ]
    },
    "libvvenc": { # Mapping av1_nvenc CQ to libvvenc CRF
        "model_ref_cq_range": [10, 63],
        "target_param_type": "crf",
        "target_param_range": [0, 51], # H.266/VVC CRF range
        "anchor_points": [ # [model_av1_nvenc_cq, vvc_crf]
            [20, 20], # VVC should be more efficient than AV1
            [30, 25], # Better compression than AV1
            [40, 30],
            [50, 35]
        ]
    },
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
    'libsvtav1': {  #   Standardized name
        'max_cq': 50,
        'recommended_max': 45,
        'quality_range': (25, 45),
        'description': 'SVT-AV1 encoder'
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
    'libvpx_vp9': {  #   Changed from 'libvpx-vp9'
        'max_cq': 45,
        'recommended_max': 40,
        'quality_range': (25, 40),
        'description': 'VP9 encoder'
    },
    
    #   H.266/VVC Codec
    'libvvenc': {
        'max_cq': 45,
        'recommended_max': 40,
        'quality_range': (20, 40),
        'description': 'VVenC H.266 encoder'
    },
    
    #   Lossless codec
    'ffv1': {
        'max_cq': 0,     # Lossless doesn't use quality parameters
        'recommended_max': 0,
        'quality_range': (0, 0),
        'description': 'FFV1 lossless encoder'
    },
    
    # Default fallback for unknown codecs
    'default': {
        'max_cq': 40,
        'recommended_max': 35,
        'quality_range': (20, 35),
        'description': 'Default codec limits'
    }
}