import uuid
import requests
from pathlib import Path
from tqdm import tqdm
from typing import Optional
from requests.exceptions import RequestException
import httpx


class DownloadError(Exception):
    """Custom exception for download-related errors"""
    pass


def download_video(url: str) -> Optional[str]:
    """
    Downloads video from URL to the upscaling/videos directory with progress tracking.
    
    Args:
        url (str): Direct video URL to download
        
    Returns:
        str: Full path of the downloaded video file if successful
        
    Raises:
        DownloadError: If download fails or path issues occur
        ValueError: If URL is invalid or empty
    """
    try:
        if not url:
            raise ValueError("URL cannot be empty")

        # Construct and validate video directory path
        try:
            video_dir = Path(__file__).parent.parent / "upscaling" / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise DownloadError(f"Failed to create directory structure: {str(e)}")

        # Generate unique filename
        filename = f"{uuid.uuid4()}.mp4"
        output_path = video_dir / filename

        # Initialize download
        try:
            response = requests.get(url, stream=True, timeout=20)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
        except RequestException as e:
            raise DownloadError(f"Failed to initialize download: {str(e)}")

        # Download with progress tracking
        try:
            with open(output_path, 'wb') as f:
                with tqdm(
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {filename}",
                    bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                ) as pbar:
                    for data in response.iter_content(chunk_size=8192):
                        size = f.write(data)
                        pbar.update(size)
        except Exception as e:
            # Clean up partially downloaded file if it exists
            if output_path.exists():
                output_path.unlink()
            raise DownloadError(f"Download failed: {str(e)}")

        # Verify download
        if output_path.stat().st_size == 0:
            output_path.unlink()
            raise DownloadError("Downloaded file is empty")

        # Return the full path of the downloaded file
        print(f"File Downloading Success, Pass: {str(output_path)}")
        return str(output_path)

    except Exception as e:
        raise DownloadError(f"Download failed: {str(e)}")


async def video_upscaler(input_file_path: str):
    """
    Calls the local FastAPI endpoint to upscale the video.
    
    Args:
        input_file_path (str): The full path of the input video file.
    
    Returns:
        str: Full path of the upscaled video, if successful.
    """
    
    return input_file_path
    try:
        # Define the API URL
        url = "http://0.0.0.0:8000/upscale-video"
        
        # Define the payload
        payload = {"task_file_path": input_file_path}
        
        # Send the POST request
        response = httpx.post(url, json=payload)

        # Check the response status
        if response.status_code == 200:
            result = response.json()
            return result["upscaled_video_path"]
        else:
            print(f"Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}")
    except Exception as e:
        print(f"Exception occurred: {e}")

    return None


if __name__ == "__main__":
    video_url = "https://drive.google.com/uc?id=1QN9B54sYl7ZeI2KyZDwFmlZc_dW2ybyE&export=download"
    download_video(video_url)

