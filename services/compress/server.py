"""
Video Compression Service API Server

This module provides a FastAPI server for video compression services.
It handles video upload, compression, and storage operations.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger

from video_preprocessor import pre_processing
from scene_detector import scene_detection
from encoder import ai_encoding, load_encoding_resources
from vmaf_calculator import scene_vmaf_calculation
from validator_merger import validation_and_merging
from vidaio_subnet_core.utilities import storage_client, download_video
from vidaio_subnet_core import CONFIG


# ============================================================================
# FastAPI Application Setup
# ============================================================================

app = FastAPI(title="Video Compression Service", version="1.0.0")


# ============================================================================
# Data Models
# ============================================================================

class CompressPayload(BaseModel):
    """Payload for video compression requests."""
    payload_url: str
    vmaf_threshold: float
    target_quality: str = 'Medium'  # High, Medium, Low
    max_duration: int = 3600  # Maximum allowed video duration in seconds
    output_dir: str = './output'  # Output directory for final files


class TestCompressPayload(BaseModel):
    """Payload for test compression requests."""
    video_path: str


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/compress-video")
async def compress_video(video: CompressPayload):
    """
    Compress a video from a URL payload.
    
    Args:
        video: Compression request payload
        
    Returns:
        dict: Compression results with uploaded video URL
    """
    print(f"video url: {video.payload_url}")
    print(f"vmaf threshold: {video.vmaf_threshold}")
    
    # Download video from URL
    input_path = await download_video(video.payload_url)
    input_file = Path(input_path)
    vmaf_threshold = video.vmaf_threshold

    # Map VMAF threshold to target quality
    if vmaf_threshold == 85:
        target_quality = 'Low'
    elif vmaf_threshold == 90:
        target_quality = 'Medium'
    elif vmaf_threshold == 95:
        target_quality = 'High'
    else:
        raise HTTPException(status_code=400, detail="Invalid VMAF threshold.")

    # Validate input file
    if not input_file.is_file():
        raise HTTPException(status_code=400, detail="Input video file does not exist.")

    # Create output directory
    output_dir = Path(video.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Perform video compression
    try:
        compressed_video_path = video_compressor(
            input_file=str(input_file),
            target_quality=target_quality,
            max_duration=video.max_duration,
            output_dir=str(output_dir)
        )
        print(f"compressed_video_path: {compressed_video_path}")

        if compressed_video_path and Path(compressed_video_path).exists():
            # Upload compressed video to storage
            try:
                compressed_video_name = os.path.basename(compressed_video_path)
                object_name: str = compressed_video_name
                
                # Upload file
                await storage_client.upload_file(object_name, compressed_video_path)
                print(f"object_name: {object_name}")
                print("Video uploaded successfully.")
                
                # Clean up local file
                if os.path.exists(compressed_video_path):
                    os.remove(compressed_video_path)
                    print(f"{compressed_video_path} has been deleted.")
                else:
                    print(f"{compressed_video_path} does not exist.")
                
                # Get sharing link
                sharing_link: Optional[str] = await storage_client.get_presigned_url(object_name)
                print(f"sharing_link: {sharing_link}")
                
                if not sharing_link:
                    print("Upload failed")
                    return {"uploaded_video_url": None}
                
                return {
                    "uploaded_video_url": sharing_link,
                    "status": "success",
                    "compressed_video_path": str(compressed_video_path)
                }
            except Exception as upload_error:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to upload compressed video: {str(upload_error)}"
                )
        else:
            raise HTTPException(status_code=500, detail="Video compression failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video compression error: {str(e)}")


@app.post("/test-compress")
async def test_compress_video(test_payload: TestCompressPayload):
    """
    Test endpoint for video compression using local video path.
    
    Args:
        test_payload: Test compression request payload
        
    Returns:
        dict: Test compression results
    """
    video_path = Path(test_payload.video_path)
    
    # Validate input file
    if not video_path.is_file():
        raise HTTPException(
            status_code=400, 
            detail=f"Video file does not exist: {video_path}"
        )
    
    try:
        # Perform test compression
        compressed_video_path = test_video_compression(str(video_path))
        
        if compressed_video_path and Path(compressed_video_path).exists():
            return {
                "status": "success",
                "message": "Video compression test completed successfully",
                "input_path": str(video_path),
                "output_path": compressed_video_path,
                "output_size_mb": round(
                    Path(compressed_video_path).stat().st_size / (1024 * 1024), 2
                )
            }
        else:
            raise HTTPException(status_code=500, detail="Video compression test failed")
            
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Video compression test error: {str(e)}"
        )


# ============================================================================
# Core Video Compression Functions
# ============================================================================

def video_compressor(
    input_file: str, 
    target_quality: str = 'Medium', 
    max_duration: int = 3600, 
    output_dir: str = './output'
) -> Optional[str]:
    """
    Main video compression pipeline orchestrator.
    
    Args:
        input_file: Path to input video file
        target_quality: Target quality level ('High', 'Medium', 'Low')
        max_duration: Maximum allowed video duration in seconds
        output_dir: Output directory for final files
        
    Returns:
        str: Path to compressed video file, or None if failed
    """
    # Record pipeline start time
    pipeline_start_time = time.time()
    
    # Get current directory and setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = _load_configuration(current_dir)
    config['directories']['output_dir'] = str(output_dir_path)
    if 'video_processing' not in config:
        config['video_processing'] = {}
    config['video_processing']['target_quality'] = target_quality

    # Create temp directory
    temp_dir = Path(config['directories']['temp_dir'])
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Display pipeline information
    _display_pipeline_info(input_file, target_quality, max_duration, output_dir)

    # PART 1: Pre-processing
    part1_result = _execute_preprocessing(input_file, target_quality, max_duration, output_dir_path)
    if not part1_result:
        print("❌ Part 1 failed. Pipeline terminated.")
        return False
    
    part1_time = time.time() - pipeline_start_time
    _display_preprocessing_results(part1_result, part1_time)

    # PART 2: Scene Detection
    part2_start_time = time.time()
    scenes_metadata = scene_detection(part1_result)
    if not scenes_metadata:
        print("❌ Part 2 failed. Pipeline terminated.")
        return False
    
    part2_time = time.time() - part2_start_time
    _display_scene_detection_results(scenes_metadata, part2_time)

    # PART 3: AI Encoding
    part3_result = _execute_ai_encoding(scenes_metadata, config, target_quality)
    if not part3_result:
        print("❌ Part 3 failed completely. Pipeline terminated.")
        return False
    
    part3_time = part3_result['processing_time']
    encoded_scenes_data = part3_result['encoded_scenes_data']
    successful_encodings = part3_result['successful_encodings']

    # PART 4: Validation and Merging
    part4_result = _execute_validation_and_merging(
        part1_result, encoded_scenes_data, config
    )
    if not part4_result:
        print("❌ Part 4 failed. Could not create final video.")
        return False
    
    part4_time = part4_result['processing_time']
    final_video_path = part4_result['final_video_path']
    final_vmaf = part4_result['final_vmaf']
    comprehensive_report = part4_result['comprehensive_report']
    
    _display_validation_results(part4_result)

    # Pipeline completion summary
    total_pipeline_time = time.time() - pipeline_start_time
    _display_pipeline_summary(
        input_file, final_video_path, part1_result, scenes_metadata,
        successful_encodings, output_dir, pipeline_start_time,
        part1_time, part2_time, part3_time, part4_time, total_pipeline_time,
        final_vmaf, comprehensive_report
    )
    
    return final_video_path


def test_video_compression(video_path: str) -> Optional[str]:
    """
    Test function for video compression using default parameters.
    
    Args:
        video_path: Path to input video file
        
    Returns:
        str: Path to compressed video file, or None if failed
    """
    print(f"\n🧪 === Testing Video Compression ===")
    print(f"   📁 Input: {video_path}")
    print(f"   🎯 Using default test parameters")
    
    # Default test parameters
    test_params = {
        'target_quality': 'Medium',
        'max_duration': 3600,
        'output_dir': './test_output'
    }
    
    try:
        # Validate input file
        input_path = Path(video_path)
        if not input_path.is_file():
            print(f"❌ Input file does not exist: {video_path}")
            return None
        # Create test output directory
        test_output_dir = Path(test_params['output_dir'])
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Perform compression
        result = video_compressor(
            input_file=str(input_path),
            target_quality=test_params['target_quality'],
            max_duration=test_params['max_duration'],
            output_dir=str(test_output_dir)
        )
        
        if result and Path(result).exists():
            print(f"\n✅ Test completed successfully!")
            print(f"   📁 Compressed video: {result}")
            return result
        else:
            print(f"\n❌ Test failed - no output file generated")
            return None
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        return None


# ============================================================================
# Helper Functions
# ============================================================================

def _load_configuration(current_dir: str) -> dict:
    """Load configuration from config.json or use defaults."""
    try:
        config_path = os.path.join(current_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("✅ Configuration loaded successfully")
        return config
    except FileNotFoundError:
        print("⚠️ Config file not found, using default configuration")
        return _get_default_config()


def _get_default_config() -> dict:
    """Get default configuration when config.json is not available."""
    return {
        'directories': {
            'temp_dir': './videos/temp_scenes',
            'output_dir': './output'
        },
        'video_processing': {
            'SHORT_VIDEO_THRESHOLD': 20,
            'target_vmaf': 93.0,
            'codec': 'auto',
            'size_increase_protection': True,
            'conservative_cq_adjustment': 2,
            'max_output_size_ratio': 1.15,
            'max_encoding_retries': 2,
            'basic_cq_lookup_by_quality': {
                'High': {
                    'animation': 22,
                    'low-action': 20,
                    'medium-action': 18,
                    'high-action': 16,
                    'default': 19
                },
                'Medium': {
                    'animation': 25,
                    'low-action': 23,
                    'medium-action': 21,
                    'high-action': 19,
                    'default': 22
                },
                'Low': {
                    'animation': 28,
                    'low-action': 26,
                    'medium-action': 24,
                    'high-action': 22,
                    'default': 25
                }
            },
        },
        'scene_detection': {
            'enable_time_based_fallback': True,
            'time_based_scene_duration': 90
        },
        'vmaf_calculation': {
            'calculate_full_video_vmaf': True,
            'vmaf_use_sampling': True,
            'vmaf_num_clips': 3,
            'vmaf_clip_duration': 2
        },
        'output_settings': {
            'save_individual_scene_reports': True,
            'save_comprehensive_report': True
        },
        'model_paths': {
            'scene_classifier_model': 'services/compress/models/scene_classifier_model.pth'
        }
    }


def _display_pipeline_info(input_file: str, target_quality: str, max_duration: int, output_dir: str):
    """Display pipeline initialization information."""
    print(f"\n🎬 === AI Video Compression Pipeline ===")
    print(f"   📁 Input: {Path(input_file).name}")
    print(f"   🎯 Target Quality: {target_quality}")
    print(f"   ⏱️ Max Duration: {max_duration}s")
    print(f"   📁 Output Dir: {output_dir}")
    print(f"   🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def _execute_preprocessing(input_file: str, target_quality: str, max_duration: int, output_dir_path: Path) -> Optional[dict]:
    """Execute Part 1: Pre-processing."""
    print(f"\n🔧 === Part 1: Pre-processing ===")
    part1_start_time = time.time()
    
    part1_result = pre_processing(
        video_path=input_file,
        target_quality=target_quality,
        max_duration=max_duration,
        output_dir=output_dir_path
    )
    
    print("part1_result", part1_result)
    return part1_result


def _display_preprocessing_results(part1_result: dict, part1_time: float):
    """Display Part 1 results."""
    print(f"\n✅ Part 1 completed in {part1_time:.1f}s:")
    print(f"   📁 Video: {os.path.basename(part1_result['path'])}")
    print(f"   🎥 Codec: {part1_result['codec']} (original: {part1_result['original_codec']})")
    print(f"   ⏱️ Duration: {part1_result['duration']:.1f}s")
    print(f"   🔄 Reencoded: {part1_result['was_reencoded']}")
    print(f"   🎯 Target VMAF: {part1_result['target_vmaf']} ({part1_result['target_quality']})")
    
    if part1_result['was_reencoded']:
        print(f"   🔄 Lossless conversion: {part1_result['processing_info']['original_format']} → {part1_result['processing_info']['standardized_format']}")
        print(f"   ⏱️ Encoding time: {part1_result['encoding_time']:.1f}s")


def _display_scene_detection_results(scenes_metadata: list, part2_time: float):
    """Display Part 2 results."""
    print(f"\n✅ Part 2 completed in {part2_time:.1f}s: {len(scenes_metadata)} scenes detected")
    
    # Display scene information
    total_scene_size = 0
    for scene in scenes_metadata:
        scene_size = scene.get('file_size_mb', 0)
        total_scene_size += scene_size
        print(f"   Scene {scene['scene_number']}: {scene['start_time']:.1f}s - {scene['end_time']:.1f}s "
              f"(duration: {scene['duration']:.1f}s)")
        if scene_size > 0:
            print(f"      📁 File: {os.path.basename(scene['path'])} ({scene_size:.1f} MB)")
        else:
            print(f"      📁 File: {os.path.basename(scene['path'])}")
    
    if total_scene_size > 0:
        print(f"   📊 Total scene files: {total_scene_size:.1f} MB")


def _execute_ai_encoding(scenes_metadata: list, config: dict, target_quality: str) -> Optional[dict]:
    """Execute Part 3: AI Encoding."""
    print(f"\n🧠 === Part 3: AI Encoding ===")
    part3_start_time = time.time()
    
    print(f"   🔧 Loading AI models and resources...")
    print(f"   📋 Using quality-based CQ lookup tables for {target_quality} quality")
    print(f"   🎯 Target Quality Level: {target_quality}")
    
    # Display CQ ranges for selected quality level
    quality_info = {
        'High': {'vmaf': 95, 'cq_range': '16-22'},
        'Medium': {'vmaf': 93, 'cq_range': '19-25'},
        'Low': {'vmaf': 90, 'cq_range': '22-28'}
    }
    
    if target_quality in quality_info:
        info = quality_info[target_quality]
        print(f"   🎚️ CQ Range for {target_quality}: {info['cq_range']} (Target VMAF: {info['vmaf']})")
    
    print(f"   🔧 Loading AI models and resources...")
    
    try:
        resources = load_encoding_resources(config, logging_enabled=True)
        print(f"   ✅ AI resources loaded successfully")
        print(f"   🧠 Mode: Scene classification + CQ lookup table")
    except Exception as e:
        print(f"   ❌ Failed to load GGG AI resources: {e}")
        return None
    
    # Process each scene individually
    encoded_scenes_data = []
    successful_encodings = 0
    failed_encodings = 0
    total_input_size = 0
    total_output_size = 0
    
    print(f"\n   📊 Processing {len(scenes_metadata)} scenes with AI approach...")
    
    for i, scene_metadata in enumerate(scenes_metadata):
        scene_result = _process_single_scene(
            scene_metadata, i, len(scenes_metadata), config, resources, target_quality
        )
        
        if scene_result['success']:
            successful_encodings += 1
            total_input_size += scene_result['input_size_mb']
            total_output_size += scene_result['output_size_mb']
        else:
            failed_encodings += 1
        
        encoded_scenes_data.append(scene_result['scene_data'])
    
    part3_time = time.time() - part3_start_time
    
    # Display Part 3 summary
    _display_ai_encoding_summary(
        successful_encodings, failed_encodings, len(scenes_metadata),
        part3_time, target_quality, total_input_size, total_output_size
    )
    
    if successful_encodings == 0:
        print("❌ Part 3 failed completely. No scenes were encoded. Pipeline terminated.")
        return None
    
    return {
        'encoded_scenes_data': encoded_scenes_data,
        'successful_encodings': successful_encodings,
        'failed_encodings': failed_encodings,
        'processing_time': part3_time,
        'total_input_size': total_input_size,
        'total_output_size': total_output_size
    }


def _process_single_scene(scene_metadata: dict, scene_index: int, total_scenes: int, 
                         config: dict, resources: dict, target_quality: str) -> dict:
    """Process a single scene for AI encoding."""
    scene_number = scene_metadata['scene_number']
    scene_path = scene_metadata['path']
    scene_duration = scene_metadata['duration']
    
    print(f"\n   🎬 Scene {scene_number}/{total_scenes}: {os.path.basename(scene_path)}")
    print(f"      ⏱️ Duration: {scene_duration:.1f}s")
    
    indicative_vmaf = scene_metadata['original_video_metadata'].get('target_vmaf')
    print(f"      🎯 Target Quality: {target_quality}" + (f" (VMAF≈{indicative_vmaf})" if indicative_vmaf else ""))
    print(f"      🧠 Method: scene classification + CQ lookup")
    
    scene_start_time = time.time()
    
    try:
        encoded_path, scene_data = ai_encoding(
            scene_metadata=scene_metadata,
            config=config,
            resources=resources,
            target_vmaf=None,
            target_quality_level=target_quality,
            logging_enabled=True
        )
        
        scene_processing_time = time.time() - scene_start_time
        
        if encoded_path and scene_data.get('encoding_success', False):
            size_mb = scene_data.get('encoded_file_size_mb', 0)
            input_size_mb = scene_data.get('input_size_mb', 0)
            compression = scene_data.get('compression_ratio', 0)
            
            print(f"      ✅ Scene {scene_number} encoded successfully")
            print(f"         📁 Output: {os.path.basename(encoded_path)}")
            print(f"         📊 Size: {input_size_mb:.1f} MB → {size_mb:.1f} MB ({compression:+.1f}% compression)")
            print(f"         🎭 Scene type: {scene_data.get('scene_type', 'unknown')}")
            print(f"         🎯 Quality: {scene_data.get('target_quality_level', target_quality)}")
            print(f"         🎚️ CQ used: {scene_data.get('base_cq_for_quality', 'N/A')} → {scene_data.get('final_adjusted_cq', 'unknown')} (after adjustment)")
            print(f"         📋 Method: Quality-based lookup table CQ selection")
            print(f"         ⏱️ Processing: {scene_processing_time:.1f}s")
            
            # Update scene metadata
            scene_metadata['encoded_path'] = encoded_path
            scene_metadata['encoding_data'] = scene_data
            
            return {
                'success': True,
                'scene_data': scene_data,
                'input_size_mb': input_size_mb,
                'output_size_mb': size_mb
            }
        else:
            error_reason = scene_data.get('error_reason', 'Unknown error')
            print(f"      ❌ Scene {scene_number} encoding failed: {error_reason}")
            print(f"         ⏱️ Processing: {scene_processing_time:.1f}s")
            
            scene_metadata['encoded_path'] = None
            scene_metadata['encoding_data'] = scene_data
            
            return {
                'success': False,
                'scene_data': scene_data,
                'input_size_mb': 0,
                'output_size_mb': 0
            }
            
    except Exception as e:
        scene_processing_time = time.time() - scene_start_time
        print(f"      ❌ Scene {scene_number} processing failed with exception: {e}")
        print(f"         ⏱️ Processing: {scene_processing_time:.1f}s")
        
        error_scene_data = {
            'scene_number': scene_number,
            'encoding_success': False,
            'error_reason': f'Exception: {str(e)}',
            'processing_time_seconds': scene_processing_time,
            'encoded_path': None,
            'original_video_metadata': scene_metadata['original_video_metadata']
        }
        
        scene_metadata['encoded_path'] = None
        scene_metadata['encoding_data'] = error_scene_data
        
        return {
            'success': False,
            'scene_data': error_scene_data,
            'input_size_mb': 0,
            'output_size_mb': 0
        }


def _display_ai_encoding_summary(successful_encodings: int, failed_encodings: int, total_scenes: int,
                                part3_time: float, target_quality: str, total_input_size: float, total_output_size: float):
    """Display Part 3 summary."""
    print(f"\n   📊 Part 3 Processing Summary:")
    print(f"      ✅ Successful encodings: {successful_encodings}")
    print(f"      ❌ Failed encodings: {failed_encodings}")
    print(f"      📈 Success rate: {successful_encodings/total_scenes*100:.1f}%")
    print(f"      ⏱️ Total processing time: {part3_time:.1f}s")
    print(f"      🎯 Quality Level: {target_quality}")
    print(f"      🧠 AI Method: Scene classification + quality-based CQ lookup")
    
    if total_input_size > 0 and total_output_size > 0:
        overall_compression = (1 - total_output_size / total_input_size) * 100
        print(f"      🗜️ Overall compression: {overall_compression:+.1f}%")
        print(f"      📊 Total size: {total_input_size:.1f} MB → {total_output_size:.1f} MB")
    
    print(f"✅ Part 3 completed with {successful_encodings} successful encodings")


def _execute_validation_and_merging(part1_result: dict, encoded_scenes_data: list, config: dict) -> Optional[dict]:
    """Execute Part 4: Validation and Merging."""
    part4_start_time = time.time()
    
    try:
        final_video_path, final_vmaf, comprehensive_report = validation_and_merging(
            original_video_path=part1_result['path'],
            encoded_scenes_data=encoded_scenes_data,
            config=config,
            logging_enabled=True
        )
        
        part4_time = time.time() - part4_start_time
        
        if final_video_path and os.path.exists(final_video_path):
            return {
                'final_video_path': final_video_path,
                'final_vmaf': final_vmaf,
                'comprehensive_report': comprehensive_report,
                'processing_time': part4_time
            }
        else:
            print("❌ Part 4 failed. Could not create final video.")
            return None
            
    except Exception as e:
        print(f"❌ Part 4 failed with exception: {e}")
        return None


def _display_validation_results(part4_result: dict):
    """Display Part 4 results."""
    final_video_path = part4_result['final_video_path']
    final_vmaf = part4_result['final_vmaf']
    comprehensive_report = part4_result['comprehensive_report']
    part4_time = part4_result['processing_time']
    
    print(f"✅ Part 4 completed successfully in {part4_time:.1f}s!")
    print(f"   📁 Final video: {os.path.basename(final_video_path)}")
    
    if final_vmaf:
        print(f"   🎯 Final VMAF: {final_vmaf:.2f}")
    
    if comprehensive_report:
        compression_info = comprehensive_report.get('compression_metrics', {})
        final_compression = compression_info.get('overall_compression_ratio_percent', 0)
        final_size = compression_info.get('final_file_size_mb', 0)
        
        print(f"   🗜️ Overall compression: {final_compression:+.1f}%")
        print(f"   📊 Final file size: {final_size:.1f} MB")


def _display_pipeline_summary(input_file: str, final_video_path: str, part1_result: dict,
                            scenes_metadata: list, successful_encodings: int, output_dir: str,
                            pipeline_start_time: float, part1_time: float, part2_time: float,
                            part3_time: float, part4_time: float, total_pipeline_time: float,
                            final_vmaf: Optional[float], comprehensive_report: Optional[dict]):
    """Display complete pipeline summary."""
    print(f"\n🎉 === Pipeline Completed Successfully ===")
    print(f"   📁 Input video: {os.path.basename(input_file)}")
    print(f"   📁 Final video: {os.path.basename(final_video_path)}")
    print(f"   🎯 Target quality: {part1_result['target_quality']} (VMAF: {part1_result['target_vmaf']})")
    print(f"   📊 Scenes processed: {len(scenes_metadata)} total, {successful_encodings} successful")
    print(f"   📁 Output directory: {output_dir}")
    print(f"   🕐 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Performance breakdown
    print(f"\n   ⏱️ Performance Breakdown:")
    print(f"      Part 1 (Pre-processing): {part1_time:.1f}s")
    print(f"      Part 2 (Scene Detection): {part2_time:.1f}s")
    print(f"      Part 3 (AI Encoding): {part3_time:.1f}s")
    print(f"      Part 4 (Validation & Merging): {part4_time:.1f}s")  
    print(f"      Total Pipeline Time: {total_pipeline_time:.1f}s")
    
    # Final file size comparison
    input_file_path = Path(input_file)
    final_video_path_obj = Path(final_video_path)
    if input_file_path.exists() and final_video_path_obj.exists():
        input_size = input_file_path.stat().st_size / (1024 * 1024)
        output_size = final_video_path_obj.stat().st_size / (1024 * 1024)
        final_compression = (1 - output_size / input_size) * 100
        
        print(f"\n   📊 Final Size Comparison:")
        print(f"      Input: {input_size:.1f} MB")
        print(f"      Output: {output_size:.1f} MB")
        print(f"      Compression: {final_compression:+.1f}%")
        
        if final_compression > 0:
            print(f"      💾 Space saved: {input_size - output_size:.1f} MB")
    
    # Quality achievement summary
    if final_vmaf and comprehensive_report:
        quality_info = comprehensive_report.get('quality_metrics', {})
        scenes_meeting_target = quality_info.get('scenes_meeting_target', 0)
        avg_scene_vmaf = quality_info.get('average_scene_vmaf', 0)
        
        print(f"\n   🎯 Quality Achievement:")
        print(f"      Final VMAF: {final_vmaf:.2f}")
        print(f"      Average Scene VMAF: {avg_scene_vmaf:.2f}")
        print(f"      Scenes meeting target: {scenes_meeting_target}/{len(scenes_metadata)}")
        
        if 'prediction_accuracy_stats' in comprehensive_report.get('scene_analysis', {}):
            pred_stats = comprehensive_report['scene_analysis']['prediction_accuracy_stats']
            avg_error = pred_stats.get('average_prediction_error')
            if avg_error:
                print(f"      AI prediction accuracy: ±{avg_error:.1f} VMAF points")
    
    # Report file locations
    if comprehensive_report:
        print(f"\n   📄 Reports Generated:")
        print(f"      📁 Output directory: {output_dir}")
        print(f"      📊 Comprehensive report: comprehensive_processing_report_*.json")
        print(f"      📄 Individual scene reports: scene_reports/scene_*_report.json")
    
    print(f"\n   🎉 Pipeline completed successfully!")
    print(f"   🚀 Ready for playback: {final_video_path}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting video compressor server")
    logger.info(f"Video compressor server running on http://{CONFIG.video_compressor.host}:{CONFIG.video_compressor.port}")

    uvicorn.run(app, host=CONFIG.video_compressor.host, port=CONFIG.video_compressor.port)

    # result = test_video_compression('test1.mp4')
    # print(result)

    #python services/compress/server.py