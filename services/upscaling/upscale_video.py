from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import subprocess
import asyncio
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
        
        # Generate output file paths
        output_file_hevc = input_file.with_name(f"4k_{input_file.stem}.hevc")
        output_file_mp4 = input_file.with_name(f"4k_{input_file.stem}.mp4")

        # video2x command
        command = [
            "video2x",
            "-i", str(input_file),
            "-o", str(output_file_hevc),
            "-p", "realesrgan",
            "-s", "2",
            "-c", "libx265",
            "-e", "crf=24"
        ]

        # Run the command
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check for errors in the subprocess
        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Upscaling failed: {process.stderr.strip()}")

        # Verify output file
        if not output_file_hevc.exists():
            raise HTTPException(status_code=500, detail="Upscaled video file was not created.")

        # Convert .hevc to .mp4
        conversion_command = [
            "ffmpeg",
            "-i", str(output_file_hevc),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "24",
            str(output_file_mp4)
        ]
        
        # Run the conversion command
        conversion_process = subprocess.run(conversion_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check for errors in the conversion subprocess
        if conversion_process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Conversion to MP4 failed: {conversion_process.stderr.strip()}")

        # Verify mp4 output file
        if not output_file_mp4.exists():
            raise HTTPException(status_code=500, detail="MP4 video file was not created.")

        return {
            "upscaled_video_path": str(output_file_hevc),
            "mp4_video_path": str(output_file_mp4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    # video = UpscaleRequest
    # video.task_file_path = "/workspace/vidaio-subnet/services/video_scheduler/videos/3214448_5sec_hd.hevc"
    # asyncio.run(video_upscaler(video))


