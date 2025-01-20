import uuid
import requests
from pathlib import Path
from tqdm import tqdm
from typing import Optional
from requests.exceptions import RequestException

class DownloadError(Exception):
    """Custom exception for download-related errors"""
    pass

def download_video(url: str) -> Optional[str]:
    """
    Downloads video from URL to the upscaling/videos directory with progress tracking.
    
    Args:
        url (str): Direct video URL to download
        
    Returns:
        str: Generated video filename if successful
        
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
            response = requests.get(url, stream=True, timeout=10)
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

        return filename

    except Exception as e:
        raise DownloadError(f"Download failed: {str(e)}")

if __name__ == "__main__":
    url = "https://www.pexels.com/download/video/3173312/"
    
    download_video(url)