import time
import asyncio
import logging
from redis_utils import get_files_to_delete
from loguru import logger
from vidaio_subnet_core.utilities.storage_client import storage_client

async def delete_files_from_minio(files_to_delete):
    """Delete files from MinIO"""
    for file_info in files_to_delete:
        object_name = file_info.get("object_name")
        if object_name:
            try:
                await storage_client.delete_file(object_name)
                logger.info(f"Successfully deleted {object_name} from MinIO")
            except Exception as e:
                logger.error(f"Failed to delete {object_name} from MinIO: {e}")

async def run_deletion_service(check_interval=600):
    """
    Run the file deletion service that checks Redis for files to delete
    
    Args:
        check_interval: How often to check Redis for files to delete (in seconds)
    """
    logger.info("Starting file deletion service")
    
    while True:
        try:
            # Get files that are due for deletion
            files_to_delete = get_files_to_delete()
            
            if files_to_delete:
                logger.info(f"Found {len(files_to_delete)} files to delete")
                await delete_files_from_minio(files_to_delete)
            
            # Wait for the next check interval
            await asyncio.sleep(check_interval)
            
        except Exception as e:
            logger.error(f"Error in file deletion service: {e}")
            await asyncio.sleep(check_interval)

if __name__ == "__main__":
    asyncio.run(run_deletion_service())
