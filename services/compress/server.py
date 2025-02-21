from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os
from pathlib import Path

app = FastAPI()


class VideoPath(BaseModel):
    input_path: str


@app.post("/compress-video")
def compress_video(video: VideoPath):
    input_path = Path(video.input_path)

    # Check if input file exists
    if not input_path.is_file():
        raise HTTPException(status_code=400, detail="Input video file does not exist.")

    # Generate output path
    output_path = input_path.with_name(input_path.stem + "_compressed.mp4")

    # FFmpeg compression command
    command = [
        "ffmpeg",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-crf",
        "28",
        "-preset",
        "medium",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(output_path),
    ]

    try:
        # Run FFmpeg command
        subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500, detail=f"FFmpeg error: {e.stderr.decode('utf-8')}"
        )

    # Check if the output file was created
    if not output_path.is_file():
        raise HTTPException(
            status_code=500, detail="Compression failed, output file not created."
        )

    return {"compressed_video_path": str(output_path)}


@app.post("/video-codec")
def get_video_codec(video: VideoPath):
    input_path = Path(video.input_path)

    # Check if input file exists
    if not input_path.is_file():
        raise HTTPException(status_code=400, detail="Input video file does not exist.")

    # FFprobe command to get codec information
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]

    try:
        # Run FFprobe command
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        codec = result.stdout.strip()
        if not codec:
            raise HTTPException(
                status_code=500, detail="Could not determine video codec"
            )

        return {"video_path": str(input_path), "codec": codec}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"FFprobe error: {e.stderr}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=40152)
