'''
This script performs video encoding using various codecs, potentially optimized
for scene content type.
'''
import ffmpeg
import re
import time
import traceback
import numpy as np
from .encoder_configs import ENCODER_SETTINGS, SCENE_SPECIFIC_PARAMS, MODEL_CQ_REFERENCE_CODEC, QUALITY_MAPPING_ANCHORS

def cleanup_quality_params(settings, keep_param=None):
    """
    Remove conflicting quality parameters (CRF, CQ, QP).

    Args:
        settings (dict): Current settings dictionary
        keep_param (str, optional): Which parameter to keep ('crf', 'cq', or 'qp')
                                   If None, removes all quality parameters

    Returns:
        dict: Updated settings with cleaned quality parameters
    """
    quality_params = ['crf', 'cq', 'qp']

    if keep_param:
        # Remove all except the one we want to keep
        for param in quality_params:
            if param != keep_param:
                settings.pop(param, None)
    else:
        # Remove all quality parameters
        for param in quality_params:
            settings.pop(param, None)

    return settings


def apply_rate_mapping(codec, rate, current_settings, logging_enabled=True):
    """
    Apply rate (CQ/CRF) mapping to current settings based on codec type.

    Args:
        codec (str): Video codec name
        rate (int/float): Model-predicted CQ value
        current_settings (dict): Current encoder settings
        logging_enabled (bool): Enable logging

    Returns:
        dict: Updated settings with rate parameter applied
    """
    model_predicted_ref_cq = int(rate)

    if codec == MODEL_CQ_REFERENCE_CODEC:
        # Use the predicted CQ directly for the reference codec
        if 'cq' in current_settings:
            current_settings['cq'] = model_predicted_ref_cq
            cleanup_quality_params(current_settings, keep_param='cq')
            if logging_enabled:
                print(f"Applying model CQ directly for {MODEL_CQ_REFERENCE_CODEC}: {current_settings['cq']}")
        elif 'crf' in current_settings:
            current_settings['crf'] = model_predicted_ref_cq
            cleanup_quality_params(current_settings, keep_param='crf')
            if logging_enabled:
                print(f"Applying model CQ as CRF for {MODEL_CQ_REFERENCE_CODEC}: {current_settings['crf']}")
        else:
            if logging_enabled:
                print(f"Warning: Neither 'cq' nor 'crf' in base settings for {MODEL_CQ_REFERENCE_CODEC}. Using CRF fallback.")
            current_settings['crf'] = model_predicted_ref_cq

    elif codec in QUALITY_MAPPING_ANCHORS:
        # Use quality mapping for non-reference codecs
        mapping_config = QUALITY_MAPPING_ANCHORS[codec]

        model_anchor_cqs = [p[0] for p in mapping_config['anchor_points']]
        target_anchor_params = [p[1] for p in mapping_config['anchor_points']]

        # Interpolate with clamping
        clamped_model_cq = np.clip(model_predicted_ref_cq,
                                 mapping_config['model_ref_cq_range'][0],
                                 mapping_config['model_ref_cq_range'][1])
        if logging_enabled and clamped_model_cq != model_predicted_ref_cq:
            print(f"Clamped model_predicted_ref_cq from {model_predicted_ref_cq} to {clamped_model_cq}")

        mapped_param_float = np.interp(clamped_model_cq, model_anchor_cqs, target_anchor_params)

        # Clamp to target parameter range
        min_target_param, max_target_param = mapping_config['target_param_range']
        mapped_param_clamped = np.clip(mapped_param_float, min_target_param, max_target_param)
        mapped_param_int = int(round(mapped_param_clamped))

        target_param_type = mapping_config['target_param_type']
        if target_param_type == 'cq':
            current_settings['cq'] = mapped_param_int
            cleanup_quality_params(current_settings, keep_param='cq')
            if logging_enabled:
                print(f"Applying mapped CQ for {codec} from model ref CQ {model_predicted_ref_cq}: {current_settings['cq']}")
        elif target_param_type == 'crf':
            current_settings['crf'] = mapped_param_int
            cleanup_quality_params(current_settings, keep_param='crf')
            if logging_enabled:
                print(f"Applying mapped CRF for {codec} from model ref CQ {model_predicted_ref_cq}: {current_settings['crf']}")
        else:
            if logging_enabled:
                print(f"Warning: Unknown target_param_type '{target_param_type}' for {codec}. Using fallback.")
            if 'cq' in current_settings:
                current_settings['cq'] = model_predicted_ref_cq
            elif 'crf' in current_settings:
                current_settings['crf'] = model_predicted_ref_cq

    else:
        # Fallback for codecs without mapping
        if logging_enabled:
            print(f"Warning: No quality mapping found for {codec}. Applying model ref CQ {model_predicted_ref_cq} directly.")
        if 'cq' in current_settings:
            current_settings['cq'] = model_predicted_ref_cq
            cleanup_quality_params(current_settings, keep_param='cq')
        elif 'crf' in current_settings:
            current_settings['crf'] = model_predicted_ref_cq
            cleanup_quality_params(current_settings, keep_param='crf')
        else:
            current_settings['crf'] = model_predicted_ref_cq
        if logging_enabled:
            print(f"Applied direct rate {model_predicted_ref_cq} to {codec}.")

    return current_settings


def get_contrast_optimized_params(scene_type, contrast_value, codec):
    """
    Get contrast-optimized encoding parameters for a specific scene type and codec.
    
    Args:
        scene_type (str): The classified scene type
        contrast_value (float): The calculated perceptual contrast (0.0-1.0)
        codec (str): The codec being used
        
    Returns:
        dict: Additional encoding parameters optimized for contrast
    """
    params = {}
    
    if contrast_value > 0.7:
        contrast_category = "high"
    elif contrast_value < 0.3:
        contrast_category = "low"
    else:
        contrast_category = "medium"
    
    # --- NVENC-based codecs (AV1, HEVC, H264) ---
    if codec in ["av1_nvenc", "hevc_nvenc", "h264_nvenc"]:      
        # Set spatial AQ strength based on contrast
        if contrast_category == "high":
            params['spatial-aq'] = 1
            params['aq-strength'] = 8  # Higher strength for high contrast (NVENC 1-15)
        elif contrast_category == "low":
            params['spatial-aq'] = 1
            params['aq-strength'] = 4  # Lower strength for low contrast
        else:
            params['spatial-aq'] = 1
            params['aq-strength'] = 6  # Medium strength
        
        # Adjust temporal AQ based on scene type and contrast
        if scene_type == 'Faces / People' and contrast_category == "high":
            params['temporal-aq'] = 1  # Enable temporal AQ for high contrast faces
        else:
            params['temporal-aq'] = 0  # Default
    
    # --- x264 ---
    elif codec == "libx264":  # ✅ FIXED: Use standardized codec name
        # AQ mode and strength adjustments
        if contrast_category == "high":
            params['aq-mode'] = 2  # Variance AQ with auto-variance
            params['aq-strength'] = 1.2  # Higher strength for high contrast
        elif contrast_category == "low":
            params['aq-mode'] = 1  # Standard AQ
            params['aq-strength'] = 0.8  # Lower strength for low contrast
        else:
            params['aq-mode'] = 1
            params['aq-strength'] = 1.0  # Default
    
    # --- x265 ---
    elif codec == "libx265":  # ✅ FIXED: Use standardized codec name
        # AQ mode and strength adjustments
        if contrast_category == "high":
            params['aq-mode'] = 3  # Auto-variance AQ with more aggressive bias
            params['aq-strength'] = 1.2  # Higher strength
        elif contrast_category == "low":
            params['aq-mode'] = 2  # Standard AQ
            params['aq-strength'] = 0.8  # Lower strength
        else:
            params['aq-mode'] = 2
            params['aq-strength'] = 1.0  # Default
    
    # --- SVT-AV1 ---
    elif codec == "libsvtav1":  # ✅ FIXED: Use standardized codec name
        # SVT-AV1 doesn't have enable-hdr or aq-mode options
        # Use CRF adjustments and available parameters instead
        if contrast_category == "high":
            # For high contrast, we can rely on the encoder's default behavior
            # or use dolbyvision if HDR content is detected
            pass  # SVT-AV1 handles high contrast well by default
        elif contrast_category == "low":
            # For low contrast, we might want a slightly lower CRF for better quality
            pass  # This would be handled at the CRF level, not here
        else:
            pass  # Default SVT-AV1 behavior is generally good
    
    # --- VP9 ---
    elif codec == "libvpx_vp9":  # ✅ FIXED: Use standardized codec name (underscore)
        # VP9 adaptive quantization and quality parameters
        if contrast_category == "high":
            params['aq-mode'] = 3  # Complexity-based AQ for high contrast
            params['arnr-maxframes'] = 7  # More noise reduction frames
            params['arnr-strength'] = 5  # Higher noise reduction strength
        elif contrast_category == "low":
            params['aq-mode'] = 2  # Variance-based AQ for low contrast  
            params['arnr-maxframes'] = 5  # Standard noise reduction
            params['arnr-strength'] = 3  # Moderate noise reduction
        else:
            params['aq-mode'] = 1  # Variance-based AQ (default)
            params['arnr-maxframes'] = 7  # Standard noise reduction
            params['arnr-strength'] = 4  # Standard noise reduction strength
        
        # VP9-specific quality settings
        params['tune'] = 'psnr'  # Optimize for quality
        params['auto-alt-ref'] = 1  # Enable alternate reference frames
    
    # --- Scene-specific contrast adjustments ---
    # For text content, adjust parameters further based on contrast
    if scene_type == 'Screen Content / Text':
        if contrast_category == "high":
            params['sharpness'] = 0  # Preserve sharpness
            if "_nvenc" in codec:  # ✅ FIXED: lowercase nvenc
                params['rc-lookahead'] = 20  # More lookahead for complex text
        elif contrast_category == "low":
            params['sharpness'] = 1  # Some sharpening for low contrast text
            if "_nvenc" in codec:  # ✅ FIXED: lowercase nvenc
                params['rc-lookahead'] = 8  # Less lookahead needed
    
    # For faces, special handling based on contrast
    elif scene_type == 'Faces / People':
        if contrast_category == "high":
            if "_nvenc" in codec:  # ✅ FIXED: lowercase nvenc
                params['rc-lookahead'] = 20  # More lookahead for dramatic lighting
        else:
            if "_nvenc" in codec:  # ✅ FIXED: lowercase nvenc
                params['rc-lookahead'] = 15  # Standard lookahead
    
    return params

def encode_video(input_path, output_path, codec, rate=None, preset=None, scene_type=None, contrast_value=None, codec_mode=None, target_bitrate=None, logging_enabled=True):
    """
    Encodes a video using specified codec settings, optimized for scene type and contrast.
    Audio is copied without re-encoding for efficiency.

    Note: encoding decisions should be made BEFORE calling this function.

    Args:
        input_path (str): Path to input video file
        output_path (str): Path for output video file
        codec (str): Video codec to use (e.g., 'av1_nvenc', 'libx264')
        rate (int/float, optional): Quality parameter (CQ value from model)
        preset (str, optional): Encoder preset override
        scene_type (str, optional): Scene classification for optimization
        contrast_value (float, optional): Perceptual contrast (0.0-1.0)
        codec_mode (str, optional): Encoding mode - 'CRF', 'CBR', or 'VBR'
        target_bitrate (float, optional): Target bitrate in Mbps (for CBR/VBR modes)
        logging_enabled (bool): Enable detailed logging

    Returns:
        tuple: (encoding_results_log, encoding_time) or (None, None) on failure
    """
    
    if logging_enabled:
        print(f"Encoding video using codec: {codec}, scene: {scene_type}, model_predicted_rate: {rate}")
        if contrast_value is not None:
            print(f"Using contrast value: {contrast_value:.2f}")

    # Get base encoder settings
    base_settings = ENCODER_SETTINGS.get(codec)
    if not base_settings:
        print(f"Error: Codec settings for '{codec}' not found in ENCODER_SETTINGS.")
        return None, None

    # --- Parameter Prioritization ---
    # 1. Start with base settings
    current_settings = base_settings.copy()

    # 2. Apply scene-specific overrides if scene_type is provided
    if scene_type:
        codec_scene_params = SCENE_SPECIFIC_PARAMS.get(codec, {})
        scene_params = codec_scene_params.get(scene_type, codec_scene_params.get('other', {}))
        if scene_params:
            if logging_enabled:
                print(f"Applying scene-specific params for '{scene_type}': {scene_params}")
            current_settings.update(scene_params)

    # 3. Apply contrast-specific overrides if contrast_value is provided
    if contrast_value is not None:
        contrast_params = get_contrast_optimized_params(scene_type, contrast_value, codec)
        if contrast_params:
            if logging_enabled:
                print(f"Applying contrast-specific params: {contrast_params}")
            current_settings.update(contrast_params)

    # 4. Apply codec_mode and target_bitrate if provided (BEFORE rate mapping)
    # This determines whether we should apply CQ/CRF or use bitrate control
    skip_rate_application = False
    if codec_mode and target_bitrate:
        if logging_enabled:
            print(f"Applying codec_mode='{codec_mode}' with target_bitrate={target_bitrate} Mbps")

        if codec_mode.upper() == 'CBR':
            # Constant Bitrate Mode - fixed bitrate throughout
            skip_rate_application = True  # Don't apply CQ/CRF in CBR mode
            bitrate_kbps = int(target_bitrate * 1000)  # Convert Mbps to kbps
            bitrate_value = f"{bitrate_kbps}k"

            current_settings['bitrate'] = bitrate_value
            current_settings['maxrate'] = bitrate_value
            current_settings['bufsize'] = f"{bitrate_kbps * 2}k"  # Buffer = 2x bitrate

            # Remove quality parameters (incompatible with CBR)
            cleanup_quality_params(current_settings)

            # Set rate control mode for NVENC/QSV
            if codec.endswith('_nvenc'):
                current_settings['rc'] = 'cbr'
                if logging_enabled:
                    print(f"Set NVENC rate control to CBR")
            elif codec.endswith('_qsv'):
                current_settings['rc'] = 'cbr'

            if logging_enabled:
                print(f"CBR mode: bitrate={bitrate_value}, maxrate={bitrate_value}, bufsize={bitrate_kbps * 2}k")

        elif codec_mode.upper() == 'VBR':
            # Variable Bitrate Mode - allows bitrate to vary but caps at target
            # We will still apply CQ/CRF for quality, but add maxrate constraint
            bitrate_kbps = int(target_bitrate * 1000)
            bitrate_value = f"{bitrate_kbps}k"

            current_settings['maxrate'] = bitrate_value
            current_settings['bufsize'] = f"{bitrate_kbps * 2}k"

            # Set rate control mode for NVENC/QSV
            if codec.endswith('_nvenc') or codec.endswith('_qsv'):
                current_settings['rc'] = 'vbr'
                if logging_enabled:
                    print(f"Set rate control to VBR")

            if logging_enabled:
                print(f"VBR mode: maxrate={bitrate_value}, bufsize={bitrate_kbps * 2}k, will apply CQ/CRF for quality")

        elif codec_mode.upper() == 'CRF':
            # CRF mode is the default - will apply rate parameter below
            if logging_enabled:
                print(f"CRF mode: Will apply quality-based encoding (rate={rate})")

        else:
            if logging_enabled:
                print(f"Warning: Unknown codec_mode '{codec_mode}', defaulting to CRF behavior")

    # 5. Apply rate (CQ) parameter with codec-specific handling (skip if CBR mode)
    if rate is not None and not skip_rate_application:
        current_settings = apply_rate_mapping(codec, rate, current_settings, logging_enabled)

    # 6. Apply preset override if provided
    if preset is not None:
        current_settings['preset'] = preset
        if logging_enabled:
            print(f"Applying preset override: {preset}")

    # 7. Handle CRF usage - disable constqp when CRF is used
    try:
        uses_crf = 'crf' in current_settings
        has_maxrate = 'maxrate' in current_settings

        if uses_crf:
            # Remove constqp explicitly when CRF is used
            if str(current_settings.get('rc', '')).lower() == 'constqp':
                if logging_enabled:
                    print("Disabling 'constqp' because CRF is in use")
                del current_settings['rc']

            # For NVENC/QSV with CRF + maxrate, use VBR (only if rc not already set by codec_mode)
            if (codec.endswith('_nvenc') or codec.endswith('_qsv')) and has_maxrate and 'rc' not in current_settings:
                current_settings['rc'] = 'vbr'
                if logging_enabled:
                    print(f"Using 'vbr' with CRF due to specified maxrate")
    except Exception:
        pass

    # 8. CQ policy for NVENC/QSV - use VBR with CQ (only if rc not already set by codec_mode)
    try:
        has_cq = 'cq' in current_settings and isinstance(current_settings.get('cq'), (int, float))
        has_maxrate = 'maxrate' in current_settings
        rc_already_set = 'rc' in current_settings

        if (codec.endswith('_nvenc') or codec.endswith('_qsv')) and has_cq and not rc_already_set:
            # Standard CQ encoding with VBR (fallback when codec_mode not specified)
            current_settings['rc'] = 'vbr'
            cleanup_quality_params(current_settings, keep_param='cq')
            if logging_enabled:
                maxrate_info = " and maxrate" if has_maxrate else ""
                print(f"Using VBR with CQ{maxrate_info} for {codec}")
    except Exception:
        pass

    # --- Build FFmpeg arguments ---
    output_args = {}
    
    # Map internal setting names to FFmpeg argument names
    key_map = {
        'keyint': 'g',
        'bitrate': 'b:v',
        'codec': 'vcodec'
    }

    # For NVENC in constqp mode, translate 'cq' to 'qp'
    try:
        if codec.endswith('_nvenc') and str(current_settings.get('rc', '')).lower() == 'constqp':
            if 'cq' in current_settings and 'qp' not in current_settings:
                current_settings['qp'] = int(current_settings.pop('cq'))
                if logging_enabled:
                    print("Translating NVENC constqp: using -qp instead of -cq")
    except Exception:
        pass

    # Add all current settings to output_args
    for key, value in current_settings.items():
        ffmpeg_key = key_map.get(key, key)
        output_args[ffmpeg_key] = value

    # Ensure vcodec is set
    if 'vcodec' not in output_args and 'codec' in current_settings:
        output_args['vcodec'] = current_settings['codec']

    # Remove internal 'codec' key if different from 'vcodec'
    if 'codec' in output_args and output_args.get('codec') != output_args.get('vcodec'):
        del output_args['codec']

    # --- Audio handling ---
    has_audio = check_audio_stream(input_path)
    
    if has_audio:
        output_args['acodec'] = 'copy'
        output_args['map'] = '0'  # Map all streams
    else:
        output_args['map'] = '0:v:0'  # Only video stream
    
    if logging_enabled:
        print(f"Audio detected: {has_audio}")
        print("Final FFmpeg output args:", output_args)

    # --- Execute FFmpeg ---
    try:
        start_time = time.time()
        
        input_stream = ffmpeg.input(input_path)
        output_stream = ffmpeg.output(input_stream, output_path, **output_args)
        
        result = output_stream.run(
            overwrite_output=True,
            capture_stdout=True,
            capture_stderr=True,
            quiet=(not logging_enabled)
        )
        
        end_time = time.time()
        encoding_time_calculated = round(end_time - start_time, 2)
        stderr = result[1].decode("utf-8") if result[1] else ""

        # Extract final encoding log line for debugging
        encoding_results_log = None
        if logging_enabled:
            for line in stderr.splitlines():
                if "time=" in line:
                    encoding_results_log = line

        if logging_enabled:
            print(f"Successfully encoded using {codec} scene '{scene_type}': {output_path}")
        return encoding_results_log, encoding_time_calculated

    except ffmpeg.Error as e:
        err_text = e.stderr.decode('utf8') if getattr(e, 'stderr', None) else str(e)
        if logging_enabled:
            print(f"Error encoding with {codec} scene '{scene_type}': {e}")
            print(f"FFmpeg stderr: {err_text}")
            print("Trying with libsvtav1 encoder...")

        try:
            # switch codec from gpu to cpu
            output_args["vcodec"] = "libsvtav1"

            # remove nvenc-specific options that might cause fallback errors
            for key in ["spatial-aq", "temporal-aq", "aq-strength", "rc-lookahead", "preset"]:
                output_args.pop(key, None)

            start_time = time.time()
            input_stream = ffmpeg.input(input_path)
            output_stream = ffmpeg.output(input_stream, output_path, **output_args)

            result = output_stream.run(
                overwrite_output=True,
                capture_stdout=True,
                capture_stderr=True,
                quiet=(not logging_enabled)
            )

            end_time = time.time()
            encoding_time_calculated = round(end_time - start_time, 2)
            stderr = result[1].decode("utf-8") if result[1] else ""

            encoding_results_log = None
            if logging_enabled:
                for line in stderr.splitlines():
                    if "time=" in line:
                        encoding_results_log = line

            if logging_enabled:
                print(f"Successfully encoded using libsvtav1 fallback for scene '{scene_type}': {output_path}")
            return encoding_results_log, encoding_time_calculated

        except Exception as fallback_error:
            if logging_enabled:
                print(f"Fallback encoding with libsvtav1 also failed: {fallback_error}")
            return None, None
            
def encode_lossless_video(input_path, output_path, logging_enabled=True):
    """
    Encode video using FFV1 lossless codec with optimized settings.
    This function is specifically for lossless encoding and doesn't use the main
    encode_video function to avoid codec mapping issues.
    """
    import subprocess
    import time
    
    if logging_enabled:
        print(f"🎥 Encoding lossless video with FFV1 codec...")
    
    # FFV1 lossless encoding command
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', 'ffv1',         
        '-level', '3',          
        '-coder', '1',          
        '-context', '1',        
        '-g', '1',              
        '-slices', '4',         
        '-slicecrc', '1',       
        '-pix_fmt', 'yuv420p',  
        '-c:a', 'copy',         
        '-y',                   
        output_path
    ]
    
    try:
        start_time = time.time()
        
        if logging_enabled:
            print(f"🔧 FFmpeg command: {' '.join(cmd)}")
        
        # Run FFmpeg command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        end_time = time.time()
        encoding_time = round(end_time - start_time, 2)
        
        if logging_enabled:
            print(f"✅ FFV1 lossless encoding completed in {encoding_time:.1f}s")
        
        return "FFV1 lossless encoding successful", encoding_time
        
    except subprocess.CalledProcessError as e:
        if logging_enabled:
            print(f"❌ FFmpeg error: {e}")
            print(f"   stdout: {e.stdout}")
            print(f"   stderr: {e.stderr}")
        return None, None
        
    except Exception as e:
        if logging_enabled:
            print(f"❌ Unexpected error during lossless encoding: {e}")
        return None, None

def check_audio_stream(input_path):
    """Check if the input video has an audio stream."""
    try:
        import subprocess
        import json
        
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_streams', input_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        # Check if any stream is audio
        has_audio = any(stream.get('codec_type') == 'audio' for stream in data.get('streams', []))
        return has_audio
    except Exception as e:
        # If we can't detect, assume no audio to be safe
        return False
