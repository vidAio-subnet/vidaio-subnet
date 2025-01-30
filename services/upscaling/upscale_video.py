from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import subprocess
import time
import asyncio

# Initialize FastAPI
app = FastAPI()

class UpscaleRequest(BaseModel):
    task_file_path: str

@app.post("/upscale-video")
async def video_upscaler(request: UpscaleRequest):
    """
    Upscales a video using the video2x tool and returns the full paths of the upscaled video and the converted mp4 file.
    
    Args:
        request (UpscaleRequest): Input containing the path to the video to upscale.
    
    Returns:
        dict: A dictionary containing the full paths to the upscaled video and the mp4 file.
    """
    try:
        input_file = Path(request.task_file_path)
        
        # Validate input file
        if not input_file.exists() or not input_file.is_file():
            raise HTTPException(status_code=400, detail="Input file does not exist or is not a valid file.")
        
        # Generate intermediate and output file paths
        intermediate_file_hevc = input_file.with_name(f"intermediate_{input_file.stem}.hevc")
        output_file_hevc = input_file.with_name(f"4k_{input_file.stem}.hevc")
        output_file_mp4 = input_file.with_name(f"4k_{input_file.stem}.mp4")

        # Step 1: Convert .mp4 (H.264) to .hevc
        print("Step 1: Converting .mp4 to .hevc...")
        start_time = time.time()
        conversion_to_hevc_command = [
            "ffmpeg",
            "-i", str(input_file),
            "-c:v", "libx265",
            "-preset", "slow",
            "-crf", "24",
            "-fps_mode", "cfr",  # Ensure constant frame rate
            "-r", "25",          # Explicitly set frame rate
            str(intermediate_file_hevc)
        ]
        conversion_to_hevc_process = subprocess.run(conversion_to_hevc_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        elapsed_time = time.time() - start_time
        if conversion_to_hevc_process.returncode != 0:
            print(f"Conversion to HEVC failed: {conversion_to_hevc_process.stderr.strip()}")
            raise HTTPException(status_code=500, detail=f"Conversion to HEVC failed: {conversion_to_hevc_process.stderr.strip()}")
        if not intermediate_file_hevc.exists():
            print("Intermediate HEVC video file was not created.")
            raise HTTPException(status_code=500, detail="Intermediate HEVC video file was not created.")
        print(f"Step 1 completed in {elapsed_time:.2f} seconds. Intermediate HEVC file: {intermediate_file_hevc}")

        # Step 2: Upscale video using video2x
        print("Step 2: Upscaling video using video2x...")
        start_time = time.time()
        video2x_command = [
            "video2x",
            "-i", str(intermediate_file_hevc),
            "-o", str(output_file_hevc),
            "-p", "realesrgan",
            "-s", "2",
            "-c", "libx265",
            "-e", "preset=slow",
            "-e", "crf=24"
        ]
        video2x_process = subprocess.run(video2x_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        elapsed_time = time.time() - start_time
        if video2x_process.returncode != 0:
            print(f"Upscaling failed: {video2x_process.stderr.strip()}")
            raise HTTPException(status_code=500, detail=f"Upscaling failed: {video2x_process.stderr.strip()}")
        if not output_file_hevc.exists():
            print("Upscaled HEVC video file was not created.")
            raise HTTPException(status_code=500, detail="Upscaled HEVC video file was not created.")
        print(f"Step 2 completed in {elapsed_time:.2f} seconds. Upscaled HEVC file: {output_file_hevc}")

        # Step 3: Convert upscaled .hevc to .mp4
        print("Step 3: Converting upscaled .hevc to .mp4...")
        start_time = time.time()
        conversion_to_mp4_command = [
            "ffmpeg",
            "-i", str(output_file_hevc),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "24",
            "-fps_mode", "cfr",  # Ensure constant frame rate
            "-r", "25",          # Explicitly set frame rate
            str(output_file_mp4)
        ]
        conversion_to_mp4_process = subprocess.run(conversion_to_mp4_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        elapsed_time = time.time() - start_time
        if conversion_to_mp4_process.returncode != 0:
            print(f"Conversion to MP4 failed: {conversion_to_mp4_process.stderr.strip()}")
            raise HTTPException(status_code=500, detail=f"Conversion to MP4 failed: {conversion_to_mp4_process.stderr.strip()}")
        if not output_file_mp4.exists():
            print("Final MP4 video file was not created.")
            raise HTTPException(status_code=500, detail="Final MP4 video file was not created.")
        print(f"Step 3 completed in {elapsed_time:.2f} seconds. Final MP4 file: {output_file_mp4}")

        # Cleanup intermediate files if needed
        if intermediate_file_hevc.exists():
            intermediate_file_hevc.unlink()
            print(f"Intermediate file {intermediate_file_hevc} deleted.")

        return {
            "upscaled_video_path": str(output_file_hevc),
            "mp4_video_path": str(output_file_mp4)
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    
    # video = UpscaleRequest
    # video.task_file_path = "/workspace/vidaio-subnet/services/upscaling/videos/6153734_hd.mp4"
    # asyncio.run(video_upscaler(video))


