


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import subprocess

app = FastAPI()

class UpscaleRequest(BaseModel):
    task_file_path: str

@app.post("/upscale-video")
async def video_upscaler(request: UpscaleRequest):
    """
    Upscales a video using the video2x tool and returns the full path of the upscaled video.
    
    Args:
        request (UpscaleRequest): Input containing the path to the video to upscale.
    
    Returns:
        dict: A dictionary containing the full path to the upscaled video.
    """
    try:
        input_file = Path(request.task_file_path)
        
        # Validate input file
        if not input_file.exists() or not input_file.is_file():
            raise HTTPException(status_code=400, detail="Input file does not exist or is not a valid file.")
        
        # Generate output file path
        output_file = input_file.with_name(f"4k_{input_file.stem}.hevc")

        # video2x command
        command = [
            "video2x",
            "-i", str(input_file),
            "-o", str(output_file),
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
        if not output_file.exists():
            raise HTTPException(status_code=500, detail="Upscaled video file was not created.")

        return {"upscaled_video_path": str(output_file)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
