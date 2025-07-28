from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os
from pathlib import Path
from loguru import logger
from vidaio_subnet_core.global_config import CONFIG
from vidaio_subnet_core.utilities import storage_client, download_video

from video_preprocessor import part1_pre_processing
from scene_detector import part2_scene_detection
from ai_encoding_enhanced_miner import part3_ai_encoding, load_encoding_resources
from vmaf_calculator import part4_scene_vmaf_calculation  
from validator_merger import part5_validation_and_merging 

app = FastAPI()


class CompressPayload(BaseModel):
    payload_url: str
    vmaf_threshold: float


@app.post("/compress-video")
def compress_video(video: CompressPayload):

    input_path = download_video(video.payload_url)
    vmaf_threshold = video.vmaf_threshold

    # Check if input file exists
    if not input_path.is_file():
        raise HTTPException(status_code=400, detail="Input video file does not exist.")

    # Generate output path
    output_path = input_path.with_name(input_path.stem + "_compressed.mp4")

    

    return {"compressed_video_path": str(output_path)}

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting video compressor server")
    logger.info(f"Video compressor server running on http://{CONFIG.video_compressor.host}:{CONFIG.video_compressor.port}")

    uvicorn.run(app, host=CONFIG.video_compressor.host, port=CONFIG.video_compressor.port)