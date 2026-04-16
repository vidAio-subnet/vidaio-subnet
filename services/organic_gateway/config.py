import os
import logging
from functools import lru_cache
from dotenv import load_dotenv
from vidaio_subnet_core import CONFIG

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("validator-api")

REDIS_CONFIG = CONFIG.redis

class Settings:
    def __init__(self):
        self.REDIS_HOST = REDIS_CONFIG.host
        self.REDIS_PORT = REDIS_CONFIG.port
        self.REDIS_DB = REDIS_CONFIG.db
        self.REDIS_PASSWORD = None
        self.REQUEST_TIMEOUT = 30
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 2
        # Make sure to include http:// or https:// in the endpoint
        protocol = "http://"
        self.REDIS_SERVICE_ENDPOINT = f"{protocol}{CONFIG.video_scheduler.host}:{CONFIG.video_scheduler.port}"
        self.DATA_RETENTION_DAYS = 3
        self.CLEANUP_INTERVAL_HOURS = 72  # hours
        self.ORGANIC_API_KEY = os.getenv("ORGANIC_API_KEY", "")

@lru_cache()
def get_settings():
    return Settings()
