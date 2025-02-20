import time
import httpx
import os
from rich.progress import track
import sys
import uuid
from firerequests import FireRequests
from loguru import logger
import os
import sys
from pathlib import Path
from tqdm import track

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
    return os.path.join("tmp", str(uuid.uuid4()))


async def _download(url: str) -> tuple[str, float, str]:
    """Download file using hf_transfer."""
    debug_start_time = time.time()
    try:
        filename = _generate_filename(url)
        start_time = time.time()

        await fire_downloader.download_file(
            url=url,
            filename=filename,
            max_files=10,  # Number of parallel downloads
            chunk_size=1024 * 1024,  # 1 MB chunks
            parallel_failures=3,
            max_retries=5,
            headers=None,
            show_progress=False,
        )

        download_time = time.time() - start_time
        logger.info(f"Time taken to download: {download_time:.2f} seconds")
        return filename, download_time, ""
    except Exception as e:
        return "", time.time() - debug_start_time, "Download failed: " + str(e)


clean_tmp_directory()
