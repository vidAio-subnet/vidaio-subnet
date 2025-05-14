from pydantic import BaseModel, Field
from enum import Enum
from dotenv import load_dotenv
import os

load_dotenv()

class BucketType(str, Enum):
    BACKBLAZE = "backblaze"
    HIPPIUS = "hippius"
    AMAZON_S3 = "amazon_s3"
    CLOUDFLARE = "cloudflare"

class StorageConfig(BaseModel):
    bucket_type: BucketType = Field(
        default=os.getenv("BUCKET_TYPE", "backblaze"),
        description="Type of bucket service"
    )
    endpoint: str = Field(default=os.getenv("BUCKET_COMPATIBLE_ENDPOINT", "localhost:9000"))
    access_key: str = Field(default=os.getenv("BUCKET_COMPATIBLE_ACCESS_KEY", "minioadmin"))
    secret_key: str = Field(default=os.getenv("BUCKET_COMPATIBLE_SECRET_KEY", "minioadmin"))
    secure: bool = Field(default=True)
    bucket_name: str = Field(default=os.getenv("BUCKET_NAME", "vidaio-subnet"))
