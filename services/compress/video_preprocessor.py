import os
import subprocess
import json
import sys

from utils.video_utils import get_video_duration, get_video_codec
from utils.encode_video import encode_lossless_video

#TODO: Add more checks for max resolution, bitrate, etc. as needed
#TODO: Add checks for user defined parameters and override defaults in config
#TODO: Add more lossless codecs as needed and check if more containers are needed


def pre_processing(video_path, target_quality='Medium',codec='auto', max_duration=60, output_dir='./output'):
    """
    Part 1: Initial video checks and lossless encoding if necessary.

    This function performs the first stage of the video processing pipeline.
    It checks the video's duration and codec to determine if it's a candidate
    for processing. If the video is in a lossless format, it's re-encoded
    into a standardized lossless format (FFV1 in an MKV container) to ensure
    compatibility with later stages. Compressed videos are passed through
    unmodified, while videos exceeding the maximum duration are rejected.

    Args:
        video_path (str): The full path to the input video file.
        target_quality (str): Target quality level - 'High', 'Medium', or 'Low'.
                             Gets converted to VMAF scores: High=95, Medium=93, Low=90.
        max_duration (int): Maximum allowed video duration in seconds. Default: 3600 (1 hour).
        output_dir (str): Directory for final output files. Default: './output'.
        codec (str): Target encoding codec. Default: 'auto' (auto-detect best available).
                    Options: 'auto', 'av1_nvenc', 'libx264', 'libx265', 'h264_nvenc', etc.
       

    Returns:
        dict or None: Dictionary containing video metadata and processing info if successful, otherwise None.
                     Keys: 'path', 'codec', 'original_codec', 'duration', 'was_reencoded', 
                           'encoding_time', 'target_vmaf', 'target_quality', 'directories'
    """
    
    # ✅ CODEC HANDLING: Resolve auto codec selection
    if codec.lower() == 'auto':
        target_codec = 'av1_nvenc'  # Default to AV1 for best quality
        
    else:
        target_codec = codec
        print(f"🎯 Using specified codec: {target_codec}")
    
    # ✅ QUALITY MAPPING: Convert target quality to VMAF score
    quality_vmaf_mapping = {
        'High': 95.0,
        'Medium': 93.0,
        'Low': 90.0
    }
    
    # Validate and convert target quality
    if target_quality not in quality_vmaf_mapping:
        print(f"⚠️ Invalid target quality '{target_quality}'. Using 'Medium' as fallback.")
        target_quality = 'Medium'
    
    target_vmaf = quality_vmaf_mapping[target_quality]
    
    print(f"🎯 Target quality: {target_quality} (VMAF: {target_vmaf})")
    print(f"🎥 Target codec: {target_codec}")
    
    # ✅ DIRECTORY SETUP: Create output and temp directories
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # ✅ DURATION CHECK: Validate video duration
    print(f"⏱️ Checking video duration...")
    duration = get_video_duration(video_path)
    
    if duration is None:
        print("❌ Could not determine video duration. Aborting.")
        return None
    
    if duration > max_duration:
        print(f"❌ Video duration {duration}s exceeds maximum allowed duration of {max_duration}s. Rejecting video.")
        return None
    
    print(f"✅ Video duration: {duration}s (within limit: {max_duration}s)")
    
    # CODEC CHECK: Determine video codec
    print(f"🎥 Checking video codec...")
    original_codec = get_video_codec(video_path)
    
    if not original_codec:
        print("❌ Could not determine video codec. Aborting.")
        return None
    
    print(f"🎥 Detected codec: {original_codec}")
    
    # LOSSLESS CODEC HANDLING: Define lossless codecs and special cases
    lossless_codecs = ['ffv1', 'h264_lossless', 'utvideo', 'rawvideo', 'prores_ks', 'dnxhd', 'cineform']
    lossless_extensions = ['.y4m', '.yuv', '.raw']
    
    is_lossless = (
        original_codec.lower() in lossless_codecs or 
        any(video_path.lower().endswith(ext) for ext in lossless_extensions)
    )
    
    if is_lossless:
        print(f"🔄 Video is in a lossless format ({original_codec}). Re-encoding with standardized lossless compression...")
        
        # Generate output filename with timestamp to avoid conflicts
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{base_name}_lossless_{timestamp}.mkv"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"📁 Lossless output: {output_filename}")
        
        # ✅ IMPROVED: Use direct FFmpeg command for lossless encoding
        # This bypasses the main encode_video function which expects lossy codecs
        encode_log, encode_time = encode_lossless_video(
            input_path=video_path,
            output_path=output_path,
            logging_enabled=True
        )
        
        if encode_log:
            print(f"✅ Lossless encoding successful in {encode_time:.1f}s")
            print(f"📁 Output at: {output_path}")
            
            # Return comprehensive metadata
            return {
                'path': output_path,
                'codec': 'ffv1',
                'original_codec': original_codec,
                'duration': duration,
                'was_reencoded': True,
                'encoding_time': encode_time,
                'target_vmaf': target_vmaf,
                'target_quality': target_quality,
                'target_codec': target_codec,
                'processing_info': {
                    'lossless_conversion': True,
                    'original_format': original_codec,
                    'standardized_format': 'ffv1',
                    'container': 'mkv'
                }
            }
        else:
            print("❌ Lossless encoding failed.")
            return None
    else:
        print(f"✅ Video is already compressed with codec: {original_codec}. Proceeding with original file.")
        
        # Return metadata for compressed video
        return {
            'path': video_path,
            'codec': original_codec,
            'original_codec': original_codec,
            'duration': duration,
            'was_reencoded': False,
            'encoding_time': 0,
            'target_vmaf': target_vmaf,
            'target_quality': target_quality,
            'target_codec': target_codec,  # ✅ Added target codec
            'processing_info': {
                'lossless_conversion': False,
                'original_format': original_codec,
                'standardized_format': original_codec,
                'container': os.path.splitext(video_path)[1][1:]  # Extension without dot
            }
        }


if __name__ == '__main__':
    # ✅ EXAMPLE USAGE: Test with different quality levels and codecs
    
    # Create dummy video for testing
    dummy_video = "test_video.mp4"
    if not os.path.exists(dummy_video):
        print("🎬 Creating dummy video for testing...")
        subprocess.run([
            'ffmpeg', '-f', 'lavfi', '-i', 'testsrc=duration=5:size=1280x720:rate=30', 
            '-c:v', 'libx264', '-preset', 'fast', '-y', dummy_video
        ], capture_output=True)
        print(f"✅ Created test video: {dummy_video}")
    
    # Test with different quality levels and codec options
    test_cases = [
        ('High', 'auto'),
        ('Medium', 'av1_nvenc'),
        ('Low', 'libx264'),
        ('Medium', 'auto')  # Test auto-detection
    ]
    
    for quality, test_codec in test_cases:
        print(f"\n🧪 Testing with quality: {quality}, codec: {test_codec}")
        print("=" * 60)
        
        result = pre_processing(
            video_path=dummy_video,
            target_quality=quality,
            max_duration=7200,  # 2 hours
            codec=test_codec
        )
        
        if result:
            print(f"✅ Part 1 finished successfully!")
            print(f"   📁 Processed video: {result['path']}")
            print(f"   🎥 Current codec: {result['codec']} (original: {result['original_codec']})")
            print(f"   🎯 Target codec: {result['target_codec']}")
            print(f"   🔄 Was re-encoded: {result['was_reencoded']}")
            print(f"   ⏱️ Duration: {result['duration']}s")
            print(f"   🎯 Target VMAF: {result['target_vmaf']} ({result['target_quality']})")
        
            if result['processing_info']['lossless_conversion']:
                print(f"   🔄 Lossless conversion: {result['processing_info']['original_format']} → {result['processing_info']['standardized_format']}")
        else:
            print(f"❌ Part 1 failed for quality: {quality}, codec: {test_codec}")
    
    # ✅ TEST LOSSLESS VIDEO WITH CODEC PARAMETER
    print(f"\n🧪 Testing lossless video with codec parameter")
    print("=" * 60)
    
    # Create a lossless test video
    lossless_video = "test_lossless.y4m"
    if not os.path.exists(lossless_video):
        print("🎬 Creating lossless test video...")
        subprocess.run([
            'ffmpeg', '-f', 'lavfi', '-i', 'testsrc=duration=3:size=640x480:rate=30', 
            '-pix_fmt', 'yuv420p', '-y', lossless_video
        ], capture_output=True)
        print(f"✅ Created lossless test video: {lossless_video}")
    
    result = pre_processing(
        video_path=lossless_video,
        target_quality='High',
        max_duration=7200,
        codec='av1_nvenc'  # Test with specific codec
    )
    
    if result:
        print(f"✅ Lossless video processed successfully!")
        print(f"   🔄 Lossless conversion: {result['processing_info']['lossless_conversion']}")
        print(f"   📁 Output: {result['path']}")
        print(f"   🎯 Target codec for next stages: {result['target_codec']}")
        print(f"   ⏱️ Encoding time: {result['encoding_time']:.1f}s")
    
    print(f"\n🎉 All tests completed!")