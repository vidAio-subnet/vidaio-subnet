import time
import os
from loguru import logger
from services.miner_utilities.redis_utils import get_local_files_to_delete

def delete_files_loop():
    """
    Continuously monitors the Redis queue for files that need to be deleted
    and removes them from the local filesystem.
    """
    logger.info("Starting file deletion service...")
    
    try:
        while True:
            try:
                # Get files that are due for deletion
                files_to_delete = get_local_files_to_delete()
                
                for file_info in files_to_delete:
                    file_path = file_info["filepath"]
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            logger.info(f"Successfully deleted file: {file_path}")
                        else:
                            logger.warning(f"File not found: {file_path}")
                    except Exception as e:
                        logger.error(f"Error deleting file {file_path}: {str(e)}")
                
                # Sleep for a short interval before checking again
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in file deletion service: {str(e)}")
                time.sleep(5)  # Sleep longer on error
    except KeyboardInterrupt:
        logger.info("File deletion service stopped by user")
        return

if __name__ == "__main__":
    delete_files_loop()
