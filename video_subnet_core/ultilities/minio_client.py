from minio import Minio
import asyncio
import datetime
import os
from concurrent.futures import ThreadPoolExecutor
from ..global_config import CONFIG


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
        self.loop = asyncio.get_running_loop()
        self.executor = ThreadPoolExecutor()


    async def upload_file(self, object_name, file_path):
        func = self.client.fput_object
        args = (self.bucket_name, object_name, file_path)
        print("Attempting to upload")
        try:
            result = await self.loop.run_in_executor(self.executor, func, *args)
            print(f"The bucket_name is {self.bucket_name} and the result was {result}")
            return result
        except Exception as e:
            print(f"There was an issue with uploading: {e}")


    async def download_file(self, object_name, file_path):
        func = self.client.fget_object
        args = (self.bucket_name, object_name, file_path)
        print("Attempting to download")
        return await self.loop.run_in_executor(self.executor, func, *args)


    async def delete_file(self, object_name):
        func = self.client.remove_object
        args = (self.bucket_name, object_name)
        return await self.loop.run_in_executor(self.executor, func, *args)


    async def list_objects(self, prefix=None, recursive=True):
        func = self.client.list_objects
        args = (self.bucket_name, prefix, recursive)
        print("Listing objects")
        return await self.loop.run_in_executor(self.executor, func, *args)


    async def ensure_bucket_exists(self):
        exists = await self.loop.run_in_executor(self.executor, self.client.bucket_exists, self.bucket_name)
        if not exists:
            await self.loop.run_in_executor(self.executor, self.client.make_bucket, self.bucket_name)


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
            await self.loop.run_in_executor(self.executor, self.client.set_bucket_policy, self.bucket_name, policy)
            print(f"Public policy set for bucket: {self.bucket_name}")
        except Exception as e:
            print(f"Failed to set public policy for bucket {self.bucket_name}: {e}")


    async def get_presigned_url(self, object_name, expires=604800):
        expires_duration = datetime.timedelta(seconds=expires)
        func = self.client.presigned_get_object
        args = (self.bucket_name, object_name, expires_duration)
        return await self.loop.run_in_executor(self.executor, func, *args)


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


async def main():
    await minio_client.ensure_bucket_exists()
    await minio_client.upload_file("123.md", "/Users/mac/Documents/work/video-streaming/vidaio-subnet/README.md")
    result = await minio_client.list_objects()
    print(result)
    # await minio_client.download_file
    await minio_client.set_bucket_public_policy()
    url = await minio_client.get_presigned_url("123.md")
    print(url)


if __name__ == "__main__":
    asyncio.run(main())
