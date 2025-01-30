import subprocess
import json
import sys

def get_video_info(file_path):
    """
    Get detailed video information using FFprobe.
    :param file_path: Path to the video file
    """
    try:
        # Run FFprobe to get video details in JSON format
        command = [
            "ffprobe",
            "-v", "error",  # Suppress unnecessary logs
            "-show_format",  # Show container format
            "-show_streams",  # Show codec and stream details
            "-count_frames",  # Count the number of frames
            "-print_format", "json",  # Output as JSON
            file_path
        ]
        
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Parse the JSON output
        video_info = json.loads(result.stdout)
        
        # Print details
        print("\n=== Video File Information ===")
        print(f"File Path: {file_path}")
        print("=" * 40)
        
        # Format information
        if "format" in video_info:
            format_info = video_info["format"]
            print(f"File Format: {format_info.get('format_name', 'N/A')}")
            print(f"File Size: {int(format_info.get('size', 0)) / (1024 ** 2):.2f} MB")
            print(f"Duration: {float(format_info.get('duration', 0)):.2f} seconds")
            print(f"Bitrate: {int(format_info.get('bit_rate', 0)) / 1000:.2f} kbps")
            print(f"Compressed: {'Yes' if 'bit_rate' in format_info else 'No'}")
        
        print("\n=== Stream Information ===")
        for stream in video_info.get("streams", []):
            codec_type = stream.get("codec_type", "unknown").capitalize()
            print(f"\nStream Type: {codec_type}")
            print(f"Codec: {stream.get('codec_name', 'N/A')}")
            print(f"Codec Long Name: {stream.get('codec_long_name', 'N/A')}")
            if codec_type == "Video":
                print(f"Resolution: {stream.get('width', 'N/A')}x{stream.get('height', 'N/A')}")
                print(f"Frame Rate: {eval(stream.get('r_frame_rate', '0')):.2f} fps")
                print(f"Number of Frames: {stream.get('nb_frames', 'N/A')}")
                print(f"Pixel Format: {stream.get('pix_fmt', 'N/A')}")
            elif codec_type == "Audio":
                print(f"Sample Rate: {stream.get('sample_rate', 'N/A')} Hz")
                print(f"Channels: {stream.get('channels', 'N/A')}")
                print(f"Channel Layout: {stream.get('channel_layout', 'N/A')}")
            print(f"Bitrate: {int(stream.get('bit_rate', 0)) / 1000:.2f} kbps")
        
        print("\nDetailed Info Extraction Complete.")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

# Example usage
if __name__ == "__main__":
    # Input video file path
    video_path = input("Enter the path to the video file: ").strip()
    get_video_info(video_path)
