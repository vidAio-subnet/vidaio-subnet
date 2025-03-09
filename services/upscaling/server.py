from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import subprocess
from fastapi.responses import JSONResponse
import time
import asyncio
from vidaio_subnet_core import CONFIG
import re
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class UpscaleRequest(BaseModel):
    task_file_path: str
    task_type: str
    # output_file_upscaled: Optional[str] = None
    

def get_frame_rate(input_file: Path) -> float:
    """
    Extracts the frame rate of the input video using FFmpeg.

    Args:
        input_file (Path): The path to the video file.

    Returns:
        float: The frame rate of the video.
    """
    frame_rate_command = [
        "ffmpeg",
        "-i", str(input_file),
        "-hide_banner"
    ]
    process = subprocess.run(frame_rate_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = process.stderr  # Frame rate is usually in stderr

    # Extract frame rate using regex
    match = re.search(r"(\d+(?:\.\d+)?) fps", output)
    if match:
        return float(match.group(1))
    else:
        raise HTTPException(status_code=500, detail="Unable to determine frame rate of the video.")


@app.post("/upscale-video")
def video_upscaler(request: UpscaleRequest):
    """
    Upscales a video using the video2x tool and returns the full paths of the upscaled video and the converted mp4 file.

    Args:
        request (UpscaleRequest): Input containing the path to the video to upscale.

    Returns:
        dict: A dictionary containing the full paths to the upscaled video and the mp4 file.
    """
    try:
        ppath = "/workspace/vidaio-subnet/234.mp4"
        print(f"Returning from FastAPI: {ppath}")
        return {"upscaled_video_path": ppath}



        input_file = Path(request.task_file_path)
        task_type = request.task_type

        scale_factor = 2

        if task_type == "SD24K":
            scale_factor = 4

        # Validate input file
        if not input_file.exists() or not input_file.is_file():
            raise HTTPException(status_code=400, detail="Input file does not exist or is not a valid file.")

        # Get the frame rate of the video
        frame_rate = get_frame_rate(input_file)
        print(f"Frame rate detected: {frame_rate} fps")

        # Calculate the duration to duplicate 2 frames
        stop_duration = 2 / frame_rate

        # Generate output file paths
        output_file_with_extra_frames = input_file.with_name(f"{input_file.stem}_extra_frames.mp4")
        output_file_upscaled = input_file.with_name(f"{input_file.stem}_upscaled.mp4")

        # Step 1: Duplicate the last frame two times
        print("Step 1: Duplicating the last frame two times...")
        start_time = time.time()

        duplicate_last_frame_command = [
            "ffmpeg",
            "-i", str(input_file),
            "-vf", f"tpad=stop_mode=clone:stop_duration={stop_duration}",  # Dynamically calculated duration
            "-c:v", "libx264",  # Ensure proper encoding
            "-crf", "18",  # High-quality encoding
            "-preset", "fast",  # Fast encoding preset
            str(output_file_with_extra_frames)
        ]

        duplicate_last_frame_process = subprocess.run(
            duplicate_last_frame_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        elapsed_time = time.time() - start_time
        if duplicate_last_frame_process.returncode != 0:
            print(f"Duplicating frames failed: {duplicate_last_frame_process.stderr.strip()}")
            raise HTTPException(status_code=500, detail=f"Duplicating frames failed: {duplicate_last_frame_process.stderr.strip()}")
        if not output_file_with_extra_frames.exists():
            print("MP4 video file with extra frames was not created.")
            raise HTTPException(status_code=500, detail="MP4 video file with extra frames was not created.")
        print(f"Step 1 completed in {elapsed_time:.2f} seconds. File with extra frames: {output_file_with_extra_frames}")

        # Step 2: Upscale video using video2x
        print("Step 2: Upscaling video using video2x...")
        start_time = time.time()
        video2x_command = [
            "video2x",
            "-i", str(output_file_with_extra_frames),
            "-o", str(output_file_upscaled),
            "-p", "realesrgan",  # Use Real-ESRGAN for upscaling
            "-s", scale_factor,  # Scale factor of 2 or 4
            "-c", "libx264",  # Encode with H.264
            "-e", "preset=slow",  # Slow preset for better quality
            "-e", "crf=24"  # Compression level
        ]
        video2x_process = subprocess.run(video2x_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        elapsed_time = time.time() - start_time
        if video2x_process.returncode != 0:
            print(f"Upscaling failed: {video2x_process.stderr.strip()}")
            raise HTTPException(status_code=500, detail=f"Upscaling failed: {video2x_process.stderr.strip()}")
        if not output_file_upscaled.exists():
            print("Upscaled MP4 video file was not created.")
            raise HTTPException(status_code=500, detail="Upscaled MP4 video file was not created.")
        print(f"Step 2 completed in {elapsed_time:.2f} seconds. Upscaled MP4 file: {output_file_upscaled}")

        # Cleanup intermediate files if needed
        if output_file_with_extra_frames.exists():
            output_file_with_extra_frames.unlink()
            print(f"Intermediate file {output_file_with_extra_frames} deleted.")
        
        print(f"Returning from FastAPI: {output_file_upscaled}")
        return {"upscaled_video_path": str(output_file_upscaled)}
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    
    import uvicorn
    
    host = CONFIG.video_upscaler.host
    port = CONFIG.video_upscaler.port
    
    uvicorn.run(app, host=host, port=port)