import os
import json
import torch
import time
import subprocess
import sys
import traceback

from utils.processing_utils import (
    should_skip_encoding,
    encode_scene_with_size_check,
    analyze_input_compression,
    # classify_scene_from_path 
)
from utils.encode_video import encode_video
from utils.classify_scene import load_scene_classifier_model, CombinedModel

def get_cq_from_lookup_table(scene_type, config):
    """
    Gets a CQ value from a lookup table in the config based on the scene type.
    """
    # Default CQ values if not found in config
    default_cq_map = {
        'animation': 28,
        'low-action': 26,
        'medium-action': 24,
        'high-action': 22,
        'default': 25
    }
    
    # Get the lookup table from the config, or use the default
    cq_lookup_table = config.get('video_processing', {}).get('basic_cq_lookup', default_cq_map)
    
    # Return the CQ for the scene type, or the default value
    return cq_lookup_table.get(scene_type, cq_lookup_table.get('default', 25))


def load_encoding_resources(config, logging_enabled=True):
    """Load AI models and other resources needed for BASIC encoding."""
    
    model_paths = config.get('model_paths', {})
    scene_model_path = model_paths.get('scene_classifier_model', "services/compress/models/scene_classifier_model.pth")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print("💪💪💪💪💪")
    # Load scene classifier
    model_state_dict, available_metrics, class_mapping = load_scene_classifier_model(scene_model_path, device, logging_enabled)
    scene_classifier_model = CombinedModel(num_classes=len(class_mapping), metrics_dim=len(available_metrics))
    scene_classifier_model.load_state_dict(model_state_dict)
    scene_classifier_model.to(device)
    scene_classifier_model.eval()
    
    return {
        "scene_classifier_model": scene_classifier_model,
        "available_metrics": available_metrics,
        "device": device,
        "class_mapping": class_mapping
    }

def ai_encoding(scene_metadata, config, resources, target_vmaf=None, logging_enabled=True):
    """
    Part 3: BASIC AI-powered analysis and encoding for a single scene.
    This version uses a scene classifier and a CQ lookup table.
    """
    logging_enabled=True
    if not scene_metadata:
        return None, {
            'scene_number': 0,
            'encoding_success': False,
            'error_reason': 'Scene metadata is None or empty',
            'processing_time_seconds': 0.0,
            'encoded_path': None,
            'path': None
        }
    
    def safe_float(value, default=0.0):
        if value is None: return default
        try:
            result = float(value)
            return result if not (result != result) else default
        except (TypeError, ValueError):
            return default
    
    def safe_positive_float(value, default=1.0):
        result = safe_float(value, default)
        return max(result, 0.1)
    
    scene_path = scene_metadata.get('path')
    scene_number = int(safe_float(scene_metadata.get('scene_number', 1), 1))
    start_time = safe_float(scene_metadata.get('start_time'), 0.0)
    end_time = safe_float(scene_metadata.get('end_time'), 0.0)
    scene_duration = safe_positive_float(scene_metadata.get('duration'), 1.0)
    
    if end_time <= start_time or scene_duration <= 0:
        if end_time > start_time:
            scene_duration = end_time - start_time
        else:
            try:
                cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', scene_path]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
                if result.returncode == 0 and result.stdout.strip():
                    scene_duration = safe_positive_float(result.stdout.strip(), 1.0)
                    end_time = start_time + scene_duration
                else:
                    scene_duration = 1.0
            except Exception:
                scene_duration = 1.0

    original_video_metadata = scene_metadata.get('original_video_metadata', {})
    
    if logging_enabled:
        print(f"\n🎬 Processing Scene {scene_number} (Basic Mode)")
        print(f"   📁 File: {os.path.basename(scene_path) if scene_path else 'None'}")
        print(f"   ⏱️ Timing: {start_time:.1f}s - {end_time:.1f}s (duration: {scene_duration:.1f}s)")

    if not scene_path or not os.path.exists(scene_path) or os.path.getsize(scene_path) == 0:
        return None, {
            'scene_number': scene_number,
            'encoding_success': False,
            'error_reason': 'Scene file is missing, empty, or inaccessible',
            'processing_time_seconds': 0.0,
            'encoded_path': None,
            'path': scene_path
        }

    processing_start_time = time.time()
    
    temp_dir = config.get('directories', {}).get('temp_dir', './videos/temp_scenes')
    
    # BASIC MODE: Use a fixed target VMAF for reference if needed, but it doesn't drive CQ selection
    target_vmaf = safe_float(target_vmaf or original_video_metadata.get('target_vmaf') or 
                           config.get('video_processing', {}).get('target_vmaf', 93.0), 93.0)

    # Codec selection logic from enhanced version
    original_codec = original_video_metadata.get('original_codec', 'unknown')
    target_codec_from_part1 = original_video_metadata.get('target_codec', 'auto')
    current_codec = original_video_metadata.get('codec', original_codec)
    config_codec = config.get('video_processing', {}).get('codec', 'auto')
    
    if target_codec_from_part1 and target_codec_from_part1 != 'auto':
        codec = target_codec_from_part1
    elif config_codec != 'auto':
        codec = config_codec
    else:
        codec_upgrade_map = {'h264': 'av1_nvenc', 'hevc': 'av1_nvenc', 'vp9': 'av1_nvenc', 'av1': 'av1_nvenc'}
        codec = codec_upgrade_map.get(current_codec.lower(), 'av1_nvenc')

    if logging_enabled:
        print(f"   🎥 Selected Codec: {codec}")

    # BASIC ANALYSIS: Classify scene to select CQ from lookup table
    if logging_enabled:
        print(f"   🤖 Running basic scene classification...")

    scene_type = 'default'
    confidence_score = 0.0
    try:        
        classification_result = classify_scene_from_path(
            scene_path=scene_path,
            temp_dir=temp_dir,
            scene_classifier_model=resources['scene_classifier_model'],
            available_metrics=resources['available_metrics'],
            device=resources['device'],
            class_mapping=resources['class_mapping'],
            logging_enabled=logging_enabled
        )
        scene_type = classification_result.get('scene_type', 'default')
        confidence_score = classification_result.get('confidence_score', 0.0)

        if logging_enabled:
            print(f"   🎭 Scene classified as: '{scene_type}' (Confidence: {confidence_score:.2f})")

    except Exception as e:
        if logging_enabled:
            print(f"   ❌ Scene classification failed: {e}")
            traceback.print_exc()
        # Continue with default scene type
    
    # Get CQ from lookup table
    base_cq = get_cq_from_lookup_table(scene_type, config)
    if logging_enabled:
        print(f"   🎚️ Base CQ from lookup table for '{scene_type}': {base_cq}")

    # Apply conservative adjustment from config
    conservative_cq_adjustment = safe_float(config.get('video_processing', {}).get('conservative_cq_adjustment', 2), 2)
    final_cq = min(base_cq + conservative_cq_adjustment, 51.0)
    if logging_enabled:
        print(f"   🔧 Applied conservative adjustment: +{conservative_cq_adjustment} -> Final CQ: {final_cq}")

    # Create a placeholder scene_data object
    scene_data = {
        'path': scene_path,
        'scene_number': scene_number,
        'start_time': start_time,
        'end_time': end_time,
        'duration': scene_duration,
        'original_video_metadata': original_video_metadata,
        'scene_type': scene_type,
        'confidence_score': confidence_score,
        'optimal_cq': base_cq,
        'adjusted_cq': final_cq,
        'final_adjusted_cq': final_cq,
        'codec_selection_process': {'final_selected_codec': codec},
        'model_training_data': {} # Placeholder
    }

    # ENCODING
    output_scene_path = os.path.join(
        temp_dir,
        f"encoded_scene_{scene_number:03d}_{start_time:.0f}s-{end_time:.0f}s_{codec.lower()}.mp4"
    )
    os.makedirs(temp_dir, exist_ok=True)

    size_increase_protection = config.get('video_processing', {}).get('size_increase_protection', True)
    max_encoding_retries = int(safe_float(config.get('video_processing', {}).get('max_encoding_retries', 2), 2))
    
    encoding_start_time = time.time()
    encoded_path = None
    encoding_time = 0

    if size_increase_protection:
        if logging_enabled:
            print(f"   🛡️ Size increase protection enabled")
        try:
            encoded_path, encoding_time = encode_scene_with_size_check(
                scene_path=scene_path,
                output_path=output_scene_path,
                codec=codec,
                adjusted_cq=final_cq,
                content_type=scene_type,
                contrast_value=0.5, # Default contrast for basic mode
                max_retries=max_encoding_retries,
                logging_enabled=logging_enabled
            )
        except Exception as e:
            if logging_enabled:
                print(f"   ❌ Size-protected encoding failed: {e}")
    else:
        if logging_enabled:
            print(f"   ⚡ Standard encoding (size protection disabled)")
        try:
            _, encoding_time = encode_video(
                input_path=scene_path,
                output_path=output_scene_path,
                codec=codec,
                rate=final_cq,
                scene_type=scene_type,
                contrast_value=0.5, # Default contrast
                logging_enabled=logging_enabled
            )
            encoded_path = output_scene_path
        except Exception as e:
            if logging_enabled:
                print(f"   ❌ Standard encoding failed: {e}")

    # Final update of scene_data
    encoding_success = bool(encoded_path and os.path.exists(encoded_path) and os.path.getsize(encoded_path) > 0)
    scene_data['encoding_success'] = encoding_success
    scene_data['encoded_path'] = encoded_path if encoding_success else None
    scene_data['encoding_time'] = safe_float(encoding_time, 0)
    scene_data['processing_time_seconds'] = time.time() - processing_start_time

    if encoding_success:
        if logging_enabled:
            print(f"   ✅ Scene {scene_number} encoded successfully.")
        return encoded_path, scene_data
    else:
        if logging_enabled:
            print(f"   ❌ Encoding failed for scene {scene_number}.")
        scene_data['error_reason'] = 'Encoding process failed or produced an empty file'
        return None, scene_data

if __name__ == '__main__':
    print("🧪 --- Part 3 AI Encoding (Basic - Lookup Table) Testing ---")
    
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        print("✅ Configuration loaded successfully")
    except FileNotFoundError:
        print("⚠️ Config file not found, using default configuration")
        config = {}

    test_scene_path = './videos/ducks_take_off_1080p50_full.mp4'
    
    if os.path.exists(test_scene_path):
        test_scene_metadata = {
            'path': test_scene_path,
            'scene_number': 1,
            'start_time': 0.0,
            'end_time': 10.0,
            'duration': 10.0,
            'original_video_metadata': {
                'path': 'original_video.mp4',
                'codec': 'h264',
                'target_codec': 'av1_nvenc',
            }
        }
        
        print(f"📊 Testing with basic scene metadata:")
        print(f"   Scene: {os.path.basename(test_scene_path)}")
        
        try:
            resources = load_encoding_resources(config, logging_enabled=True)
            encoded_path, scene_data = part3_ai_encoding(
                scene_metadata=test_scene_metadata,
                config=config,
                resources=resources,
                logging_enabled=True
            )
            
            if encoded_path and scene_data.get('encoding_success', False):
                print(f"✅ Part 3 basic test completed successfully!")
                print(f"   📁 Encoded: {os.path.basename(encoded_path)}")
                print(f"   🎭 Scene type: {scene_data.get('scene_type', 'unknown')}")
                print(f"   🎚️ Final CQ: {scene_data.get('final_adjusted_cq', 'N/A')}")
            else:
                print(f"❌ Part 3 basic test failed: {scene_data.get('error_reason', 'Unknown error')}")
        except Exception as e:
            print(f"❌ Part 3 basic test failed with exception: {e}")
            traceback.print_exc()
    else:
        print(f"❌ Test scene not found: {test_scene_path}")
        print("   Ensure a test video exists at the specified path.")
    
    print(f"\n🎉 Part 3 (basic) testing completed!")
