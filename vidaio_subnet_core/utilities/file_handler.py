import time
import httpx
import os
import sys
import uuid
from firerequests import FireRequests
from loguru import logger
from pathlib import Path
from rich.progress import Progress, TaskID

fire_downloader = FireRequests()

def clean_tmp_directory():
    """Clean the tmp directory if running as validator and delete only .mp4 files."""
    if (
        __name__ != "__main__"
        and os.path.basename(os.path.abspath(sys.argv[0])) == "validator.py"
    ):
        tmp_dir = Path("tmp")
        tmp_dir.mkdir(exist_ok=True)  # Create the tmp directory if it doesn't exist
        
        # Iterate over all files in the tmp directory
        for file in track(tmp_dir.iterdir(), description="Cleaning .mp4 files in tmp directory"):
            if file.suffix == ".mp4":  # Only delete .mp4 files
                os.remove(file)
                print(f"Deleted: {file}").remove(os.path.join("tmp", file))

def _generate_filename(url: str) -> str:
    """Generate a unique filename for downloaded file."""
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)  # Create the tmp directory if it doesn't exist
    return os.path.join("tmp", str(uuid.uuid4()) + ".mp4")

async def download_video(url: str) -> str:
    """
    Simple function to download a video file from a URL.
    
    Args:
        url (str): The URL of the video to download.
        
    Returns:
        str: The local file path of the downloaded video.
    """
    file_path = _generate_filename(url)
    
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("Content-Length", 0))
            
            with open(file_path, "wb") as f:
                with Progress() as progress:
                    task = progress.add_task("[cyan]Downloading...", total=total_size)
                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))
    
    print(f"Video downloaded to: {file_path}")
    return file_path
