from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os
from pathlib import Path
from loguru import logger
import sys
import json
import argparse
import time
from datetime import datetime

from video_preprocessor import pre_processing
from scene_detector import scene_detection
from encoder import ai_encoding, load_encoding_resources  
from vmaf_calculator import scene_vmaf_calculation  
from validator_merger import validation_and_merging 
from vidaio_subnet_core.global_config import CONFIG
from vidaio_subnet_core.utilities import storage_client, download_video

app = FastAPI()

class CompressPayload(BaseModel):
    payload_url: str
    vmaf_threshold: float
    target_quality: str = 'Medium'  # High, Medium, Low
    max_duration: int = 3600  # Maximum allowed video duration in seconds
    output_dir: str = './output'  # Output directory for final files

class TestCompressPayload(BaseModel):
    video_path: str

@app.post("/compress-video")
async def compress_video(video: CompressPayload):
    print(f"video url: {video.payload_url}")
    print(f"vmaf threshold: {video.vmaf_threshold}")
    input_path = await download_video(video.payload_url)
    input_file = Path(input_path)
    vmaf_threshold = video.vmaf_threshold

    if vmaf_threshold == 90:
        target_quality = 'Low'
    elif vmaf_threshold == 93:
        target_quality = 'Medium'
    elif vmaf_threshold == 95:
        target_quality = 'High'
    else:
        raise HTTPException(status_code=400, detail="Invalid VMAF threshold.")

    # Check if input file exists
    if not input_file.is_file():
        raise HTTPException(status_code=400, detail="Input video file does not exist.")

    # Create output directory if it doesn't exist
    output_dir = Path(video.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Call video_compressor with the payload parameters
    try:
        compressed_video_path = video_compressor(
            input_file=str(input_file),  # Use input_file consistently
            target_quality=target_quality,
            max_duration=video.max_duration,
            output_dir=str(output_dir)
        )
        print(f"compressed_video_path: {compressed_video_path}")

        if compressed_video_path and Path(compressed_video_path).exists():
            # Upload the compressed video to storage
            try:
                # Generate object name from the compressed video filename
                compressed_video_name = os.path.basename(compressed_video_path)
                object_name: str = compressed_video_name
                
                # Upload the compressed video file
                await storage_client.upload_file(object_name, compressed_video_path)
                print(f"object_name: {object_name}")
                print("Video uploaded successfully.")
                
                # Delete the local file since we've already uploaded it to MinIO
                # if os.path.exists(compressed_video_path):
                #     os.remove(compressed_video_path)
                #     print(f"{compressed_video_path} has been deleted.")
                # else:
                #     print(f"{compressed_video_path} does not exist.")
                
                # Get the presigned URL for sharing
                sharing_link: str | None = await storage_client.get_presigned_url(object_name)
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
                raise HTTPException(status_code=500, detail=f"Failed to upload compressed video: {str(upload_error)}")
        else:
            raise HTTPException(status_code=500, detail="Video compression failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video compression error: {str(e)}")


@app.post("/test-compress")
async def test_compress_video(test_payload: TestCompressPayload):
    """
    Test endpoint for video compression that only requires a local video path.
    Uses default parameters for testing purposes.
    """
    video_path = Path(test_payload.video_path)
    
    # Check if input file exists
    if not video_path.is_file():
        raise HTTPException(status_code=400, detail=f"Video file does not exist: {video_path}")
    
    try:
        # Call the test function
        compressed_video_path = test_video_compression(str(video_path))
        
        if compressed_video_path and Path(compressed_video_path).exists():
            return {
                "status": "success",
                "message": "Video compression test completed successfully",
                "input_path": str(video_path),
                "output_path": compressed_video_path,
                "output_size_mb": round(Path(compressed_video_path).stat().st_size / (1024 * 1024), 2)
            }
        else:
            raise HTTPException(status_code=500, detail="Video compression test failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video compression test error: {str(e)}")


def video_compressor(input_file, target_quality='Medium', max_duration=3600, output_dir='./output'):
    
    # Record pipeline start time
    pipeline_start_time = time.time()
    #python modular_pipeline_main.py -i test1.mp4 -q Medium --max_duration 3600 --output_dir ./output
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create output directory if it doesn't exist
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    try:
        config_path = os.path.join(current_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("✅ Configuration loaded successfully")
    except FileNotFoundError:
        print("⚠️ Config file not found, using default configuration")
        config = {
            'directories': {
                'temp_dir': str(Path('./videos/temp_scenes').absolute()),
                'output_dir': str(output_dir_path)
            },
            'video_processing': {
                'SHORT_VIDEO_THRESHOLD': 20,
                'target_vmaf': 93.0,
                'codec': 'auto',
                'size_increase_protection': True,
                'conservative_cq_adjustment': 2,
                'max_output_size_ratio': 1.15,
                'max_encoding_retries': 2
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
            }
        }
    
    # Update output directory from arguments
    config['directories']['output_dir'] = str(output_dir_path)
    
    # Create temp directory if it doesn't exist
    temp_dir = Path(config['directories']['temp_dir'])
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎬 === AI Video Compression Pipeline ===")
    print(f"   📁 Input: {Path(input_file).name}")
    print(f"   🎯 Target Quality: {target_quality}")
    print(f"   ⏱️ Max Duration: {max_duration}s")
    print(f"   📁 Output Dir: {output_dir}")
    print(f"   🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ✅ PART 1: Pre-processing
    print(f"\n🔧 === Part 1: Pre-processing ===")
    part1_start_time = time.time()
    
    part1_result = pre_processing(
        video_path=input_file,
        target_quality=target_quality,
        max_duration=max_duration,
        output_dir=output_dir_path
    )
    
    part1_time = time.time() - part1_start_time
    
    if not part1_result:
        print("❌ Part 1 failed. Pipeline terminated.")
        return False
    
    print(f"\n✅ Part 1 completed in {part1_time:.1f}s:")
    print(f"   📁 Video: {os.path.basename(part1_result['path'])}")
    print(f"   🎥 Codec: {part1_result['codec']} (original: {part1_result['original_codec']})")
    print(f"   ⏱️ Duration: {part1_result['duration']:.1f}s")
    print(f"   🔄 Reencoded: {part1_result['was_reencoded']}")
    print(f"   🎯 Target VMAF: {part1_result['target_vmaf']} ({part1_result['target_quality']})")
    
    if part1_result['was_reencoded']:
        print(f"   🔄 Lossless conversion: {part1_result['processing_info']['original_format']} → {part1_result['processing_info']['standardized_format']}")
        print(f"   ⏱️ Encoding time: {part1_result['encoding_time']:.1f}s")

    # ✅ PART 2: Scene Detection
    print(f"\n🎭 === Part 2: Scene Detection ===")
    part2_start_time = time.time()
    
    scenes_metadata = scene_detection(part1_result)
    
    part2_time = time.time() - part2_start_time
    
    if not scenes_metadata:
        print("❌ Part 2 failed. Pipeline terminated.")
        return False
    
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

    # ✅ PART 3: AI Encoding (Multiple Scene Processing)
    print(f"\n🤖 === Part 3: AI Encoding ===")
    part3_start_time = time.time()
    
    print(f"   🔧 Loading AI models and resources...")
    
    try:
        resources = load_encoding_resources(config, logging_enabled=True)
        print(f"   ✅ AI resources loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load AI resources: {e}")
        return False
    
    # Process each scene individually
    encoded_scenes_data = []
    successful_encodings = 0
    failed_encodings = 0
    total_input_size = 0
    total_output_size = 0
    
    print(f"\n   📊 Processing {len(scenes_metadata)} scenes individually...")
    
    for i, scene_metadata in enumerate(scenes_metadata):
        scene_number = scene_metadata['scene_number']
        scene_path = scene_metadata['path']
        scene_duration = scene_metadata['duration']
        
        print(f"\n   🎬 Scene {scene_number}/{len(scenes_metadata)}: {os.path.basename(scene_path)}")
        print(f"      ⏱️ Duration: {scene_duration:.1f}s")
        print(f"      🎯 Target VMAF: {scene_metadata['original_video_metadata']['target_vmaf']}")
        
        scene_start_time = time.time()
        
        try:
            # ✅ SINGLE SCENE PROCESSING: Call Part 3 for individual scene
            encoded_path, scene_data = ai_encoding(
                scene_metadata=scene_metadata,
                config=config,
                resources=resources,
                target_vmaf=None,  # Let it use the target from original video metadata
                logging_enabled=True
            )
            
            scene_processing_time = time.time() - scene_start_time
            
            if encoded_path and scene_data.get('encoding_success', False):
                successful_encodings += 1
                size_mb = scene_data.get('encoded_file_size_mb', 0)
                input_size_mb = scene_data.get('input_size_mb', 0)
                compression = scene_data.get('compression_ratio', 0)
                
                total_input_size += input_size_mb
                total_output_size += size_mb
                
                print(f"      ✅ Scene {scene_number} encoded successfully")
                print(f"         📁 Output: {os.path.basename(encoded_path)}")
                print(f"         📊 Size: {input_size_mb:.1f} MB → {size_mb:.1f} MB ({compression:+.1f}% compression)")
                print(f"         🎭 Scene type: {scene_data.get('scene_type', 'unknown')}")
                print(f"         🎚️ CQ used: {scene_data.get('final_adjusted_cq', 'unknown')}")
                print(f"         ⏱️ Processing: {scene_processing_time:.1f}s")
                
                # Update scene metadata with encoded path for Part 4
                scene_metadata['encoded_path'] = encoded_path
                scene_metadata['encoding_data'] = scene_data
                
            else:
                failed_encodings += 1
                error_reason = scene_data.get('error_reason', 'Unknown error')
                
                print(f"      ❌ Scene {scene_number} encoding failed: {error_reason}")
                print(f"         ⏱️ Processing: {scene_processing_time:.1f}s")
                
                # Still add to scene metadata for potential recovery
                scene_metadata['encoded_path'] = None
                scene_metadata['encoding_data'] = scene_data
            
            # Add to results regardless of success/failure
            encoded_scenes_data.append(scene_data)
            
        except Exception as e:
            failed_encodings += 1
            scene_processing_time = time.time() - scene_start_time
            
            print(f"      ❌ Scene {scene_number} processing failed with exception: {e}")
            print(f"         ⏱️ Processing: {scene_processing_time:.1f}s")
            
            # Create minimal error data
            error_scene_data = {
                'scene_number': scene_number,
                'encoding_success': False,
                'error_reason': f'Exception: {str(e)}',
                'processing_time_seconds': scene_processing_time,
                'encoded_path': None,
                'original_video_metadata': scene_metadata['original_video_metadata']
            }
            encoded_scenes_data.append(error_scene_data)
            scene_metadata['encoded_path'] = None
            scene_metadata['encoding_data'] = error_scene_data
    
    part3_time = time.time() - part3_start_time
    
    # ✅ PART 3 SUMMARY
    print(f"\n   📊 Part 3 Processing Summary:")
    print(f"      ✅ Successful encodings: {successful_encodings}")
    print(f"      ❌ Failed encodings: {failed_encodings}")
    print(f"      📈 Success rate: {successful_encodings/len(scenes_metadata)*100:.1f}%")
    print(f"      ⏱️ Total processing time: {part3_time:.1f}s")
    
    if successful_encodings == 0:
        print("❌ Part 3 failed completely. No scenes were encoded. Pipeline terminated.")
        return False
    
    if total_input_size > 0 and total_output_size > 0:
        overall_compression = (1 - total_output_size / total_input_size) * 100
        print(f"      🗜️ Overall compression: {overall_compression:+.1f}%")
        print(f"      📊 Total size: {total_input_size:.1f} MB → {total_output_size:.1f} MB")
    
    print(f"✅ Part 3 completed with {successful_encodings} successful encodings")


    # ✅ PART 4: Scene VMAF Calculation (NEW)
    print(f"\n📊 === Part 4: Scene VMAF Calculation ===")
    part4_start_time = time.time()
    
    # Calculate VMAF for individual scenes
    try:
        encoded_scenes_data_with_vmaf = scene_vmaf_calculation(
            encoded_scenes_data=encoded_scenes_data,
            config=config,
            logging_enabled=True
        )
        
        part4_time = time.time() - part4_start_time
        print(f"✅ Part 4 completed in {part4_time:.1f}s")
        
    except Exception as e:
        print(f"❌ Part 4 failed with exception: {e}")
        return False

    # ✅ PART 5: Validation and Merging (RENAMED)
    print(f"\n🔗 === Part 5: Validation and Merging ===")
    part5_start_time = time.time()
    
    try:
        final_video_path, final_vmaf, comprehensive_report = validation_and_merging(
            original_video_path=part1_result['path'],
            encoded_scenes_data_with_vmaf=encoded_scenes_data_with_vmaf,  # Now includes VMAF
            config=config,
            logging_enabled=True
        )
        
        part5_time = time.time() - part5_start_time
        
        if final_video_path and os.path.exists(final_video_path):
            print(f"✅ Part 5 completed successfully in {part5_time:.1f}s!")
            print(f"   📁 Final video: {os.path.basename(final_video_path)}")
            
            if final_vmaf:
                target_vmaf = part1_result['target_vmaf']
                vmaf_status = "✅ ACHIEVED" if final_vmaf >= target_vmaf else "❌ MISSED"
                print(f"   🎯 Final VMAF: {final_vmaf:.2f} (target: {target_vmaf:.1f}) - {vmaf_status}")
            
            if comprehensive_report:
                compression_info = comprehensive_report.get('compression_metrics', {})
                final_compression = compression_info.get('overall_compression_ratio_percent', 0)
                final_size = compression_info.get('final_file_size_mb', 0)
                
                print(f"   🗜️ Overall compression: {final_compression:+.1f}%")
                print(f"   📊 Final file size: {final_size:.1f} MB")
        else:
            print("❌ Part 5 failed. Could not create final video.")
            return False
            
    except Exception as e:
        print(f"❌ Part 5 failed with exception: {e}")
        return False

    # ✅ PIPELINE COMPLETION SUMMARY
    total_pipeline_time = time.time() - pipeline_start_time
    
    print(f"\n🎉 === Pipeline Completed Successfully ===")
    print(f"   📁 Input video: {os.path.basename(input_file)}")
    print(f"   📁 Final video: {os.path.basename(final_video_path)}")
    print(f"   🎯 Target quality: {part1_result['target_quality']} (VMAF: {part1_result['target_vmaf']})")
    print(f"   📊 Scenes processed: {len(scenes_metadata)} total, {successful_encodings} successful")
    print(f"   📁 Output directory: {output_dir}")
    print(f"   🕐 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Performance breakdown
    # Update performance breakdown
    print(f"\n   ⏱️ Performance Breakdown:")
    print(f"      Part 1 (Pre-processing): {part1_time:.1f}s")
    print(f"      Part 2 (Scene Detection): {part2_time:.1f}s")
    print(f"      Part 3 (AI Encoding): {part3_time:.1f}s")
    print(f"      Part 4 (Scene VMAF): {part4_time:.1f}s")       
    print(f"      Part 5 (Validation & Merging): {part5_time:.1f}s")  
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
    
    return final_video_path


def test_video_compression(video_path: str):
    """
    Test function for video compression that only requires a video path.
    Uses default parameters for testing purposes.
    
    Args:
        video_path (str): Path to the input video file
        
    Returns:
        str: Path to the compressed video file, or None if failed
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
        # Check if input file exists
        input_path = Path(video_path)
        if not input_path.is_file():
            print(f"❌ Input file does not exist: {video_path}")
            return None
            
        # Create test output directory if it doesn't exist
        test_output_dir = Path(test_params['output_dir'])
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Call the main video_compressor function with test parameters
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


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting video compressor server")
    logger.info(f"Video compressor server running on http://{CONFIG.video_compressor.host}:{CONFIG.video_compressor.port}")

    uvicorn.run(app, host=CONFIG.video_compressor.host, port=CONFIG.video_compressor.port)

    # result = test_video_compression('test1.mp4')
    # print(result)


    #python services/compress/server.py