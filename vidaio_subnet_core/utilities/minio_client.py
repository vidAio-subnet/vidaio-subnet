from minio import Minio
import asyncio
import datetime
import os
from concurrent.futures import ThreadPoolExecutor
# from ..global_config import CONFIG
from vidaio_subnet_core.global_config import CONFIG


class VideoSubnetMinioClient:
    def __init__(self, endpoint, access_key, secret_key, bucket_name, secure=True, region="us-east-1"):
        self.endpoint = endpoint
        self.bucket_name = bucket_name
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            region=region,
        )
        self.executor = ThreadPoolExecutor()

    async def upload_file(self, object_name, file_path):
        func = self.client.fput_object
        args = (self.bucket_name, object_name, file_path)
        print("Attempting to upload")
        try:
            loop = asyncio.get_running_loop()  # Get the current event loop dynamically
            result = await loop.run_in_executor(self.executor, func, *args)
            print(f"The bucket_name is {self.bucket_name} and the result was {result}")
            return result
        except Exception as e:
            print(f"There was an issue with uploading: {e}")

    async def download_file(self, object_name, file_path):
        func = self.client.fget_object
        args = (self.bucket_name, object_name, file_path)
        print("Attempting to download")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    async def delete_file(self, object_name):
        func = self.client.remove_object
        args = (self.bucket_name, object_name)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    async def list_objects(self, prefix=None, recursive=True):
        func = self.client.list_objects
        args = (self.bucket_name, prefix, recursive)
        print("Listing objects in bucket...")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    async def ensure_bucket_exists(self):
        loop = asyncio.get_running_loop()
        exists = await loop.run_in_executor(self.executor, self.client.bucket_exists, self.bucket_name)
        if not exists:
            await loop.run_in_executor(self.executor, self.client.make_bucket, self.bucket_name)

    async def set_bucket_public_policy(self):
        """Sets the bucket policy to allow public read access."""
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": "s3:GetObject",
                    "Resource": f"arn:aws:s3:::{self.bucket_name}/*"
                }
            ]
        }
        
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self.executor, self.client.set_bucket_policy, self.bucket_name, policy)
            print(f"Public policy set for bucket: {self.bucket_name}")
        except Exception as e:
            print(f"Failed to set public policy for bucket {self.bucket_name}: {e}")

    async def delete_all_items(self):
        """Deletes all items in the Minio bucket."""
        print(f"Deleting all items in bucket: {self.bucket_name}")
        try:
            # List all objects in the bucket
            objects = await self.list_objects()
            if not objects:
                print("No objects to delete.")
                return

            count = 0
            # Delete each object
            for obj in objects:
                print(f"Deleting object: {obj.object_name}")
                await self.delete_file(obj.object_name)
                count += 1
            print(f"All {count} objects deleted successfully.")
        except Exception as e:
            print(f"Error deleting all items in bucket: {e}")

    async def get_presigned_url(self, object_name, expires=604800):
        expires_duration = datetime.timedelta(seconds=expires)
        func = self.client.presigned_get_object
        args = (self.bucket_name, object_name, expires_duration)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    def get_public_url(self, object_name):
        return f"https://{self.endpoint}/{self.bucket_name}/{object_name}"

    def __del__(self):
        self.executor.shutdown(wait=False)


minio_client = VideoSubnetMinioClient(
    endpoint=CONFIG.minio.endpoint,
    access_key=CONFIG.minio.access_key,
    secret_key=CONFIG.minio.secret_key,
    bucket_name=CONFIG.minio.bucket_name,
)


# async def main():
#     await minio_client.upload_file("567.mp4", "/workspace/vidaio-subnet/4k_33.mp4")
#     presigned_url = await minio_client.get_presigned_url("567.mp4")
#     print(presigned_url)

if __name__ == "__main__":
    asyncio.run(main())