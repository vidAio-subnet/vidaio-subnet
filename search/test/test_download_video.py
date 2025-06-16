from vidaio_subnet_core.utilities import storage_client, download_video
from search.modules.search_config import search_config
import asyncio

async def download_from_video_search_service(filename: str):
    url = f"http://{search_config["SEARCH_SERVICE_HOST"]}:{search_config["SEARCH_SERVICE_PORT"]}/download/{filename}?task_type=HD24K"
    file_path = await download_video(url)
    return file_path

async def main():
    file_path = await download_from_video_search_service("HD24K_854678_original.mp4")
    print(file_path)

if __name__ == "__main__":
    asyncio.run(main())