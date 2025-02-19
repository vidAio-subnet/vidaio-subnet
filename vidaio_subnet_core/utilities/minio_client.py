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
        print("Listing objects")
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
            
            # Delete each object
            for obj_name in objects:
                print(f"Deleting object: {obj_name}")
                await self.delete_file(obj_name)
            print("All objects deleted successfully.")
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


async def main():
    await minio_client.ensure_bucket_exists()
    await minio_client.upload_file("345.mp4", "/root/workspace/vidaio-subnet/videos/4k_4887282_hd.mp4")
    result = await minio_client.list_objects()
    print(result)
    # await minio_client.download_file
    await minio_client.set_bucket_public_policy()
    url = await minio_client.get_presigned_url("123.md")
    print(url)


if __name__ == "__main__":
    asyncio.run(main())



# from minio import Minio
# import asyncio
# import datetime
# import os
# from concurrent.futures import ThreadPoolExecutor
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# class VideoSubnetMinioClient:
#     def __init__(self, endpoint, access_key, secret_key, bucket_name, secure=True, region="us-east-1", loop=None):
#         self.endpoint = endpoint
#         self.bucket_name = bucket_name
#         self.client = Minio(
#             endpoint,
#             access_key=access_key,
#             secret_key=secret_key,
#             secure=secure,
#             region=region,
#         )
#         # Use the provided loop or get the current running loop
#         self.loop = loop or asyncio.get_running_loop()
#         self.executor = ThreadPoolExecutor()

#     async def upload_file(self, object_name, file_path):
#         """Uploads a file to the Minio bucket."""
#         func = self.client.fput_object
#         args = (self.bucket_name, object_name, file_path)
#         print(f"Uploading file: {file_path} as {object_name}")
#         try:
#             result = await self.loop.run_in_executor(self.executor, func, *args)
#             print(f"Upload successful: {result}")
#             return result
#         except Exception as e:
#             print(f"Error during upload: {e}")
#             raise

#     async def download_file(self, object_name, file_path):
#         """Downloads a file from the Minio bucket."""
#         func = self.client.fget_object
#         args = (self.bucket_name, object_name, file_path)
#         print(f"Downloading file: {object_name} to {file_path}")
#         try:
#             await self.loop.run_in_executor(self.executor, func, *args)
#             print("Download successful.")
#         except Exception as e:
#             print(f"Error during download: {e}")
#             raise

#     async def delete_file(self, object_name):
#         """Deletes a file from the Minio bucket."""
#         func = self.client.remove_object
#         args = (self.bucket_name, object_name)
#         print(f"Deleting file: {object_name}")
#         try:
#             await self.loop.run_in_executor(self.executor, func, *args)
#             print("Delete successful.")
#         except Exception as e:
#             print(f"Error during delete: {e}")
#             raise

#     async def list_objects(self, prefix=None, recursive=True):
#         """Lists objects in the Minio bucket."""
#         func = self.client.list_objects
#         args = (self.bucket_name, prefix, recursive)
#         print("Listing objects in bucket...")
#         try:
#             objects = await self.loop.run_in_executor(self.executor, func, *args)
#             return [obj.object_name for obj in objects]
#         except Exception as e:
#             print(f"Error during listing objects: {e}")
#             raise

#     async def ensure_bucket_exists(self):
#         """Ensures the Minio bucket exists, creates it if not."""
#         print(f"Ensuring bucket exists: {self.bucket_name}")
#         try:
#             exists = await self.loop.run_in_executor(self.executor, self.client.bucket_exists, self.bucket_name)
#             if not exists:
#                 await self.loop.run_in_executor(self.executor, self.client.make_bucket, self.bucket_name)
#                 print("Bucket created.")
#             else:
#                 print("Bucket already exists.")
#         except Exception as e:
#             print(f"Error ensuring bucket exists: {e}")
#             raise

#     async def set_bucket_public_policy(self):
#         """Sets the bucket policy to allow public read access."""
#         policy = {
#             "Version": "2012-10-17",
#             "Statement": [
#                 {
#                     "Effect": "Allow",
#                     "Principal": "*",
#                     "Action": "s3:GetObject",
#                     "Resource": f"arn:aws:s3:::{self.bucket_name}/*"
#                 }
#             ]
#         }
#         print(f"Setting public policy for bucket: {self.bucket_name}")
#         try:
#             await self.loop.run_in_executor(self.executor, self.client.set_bucket_policy, self.bucket_name, policy)
#             print(f"Public policy set for bucket: {self.bucket_name}")
#         except Exception as e:
#             print(f"Failed to set public policy for bucket {self.bucket_name}: {e}")
#             raise

#     async def get_presigned_url(self, object_name, expires=604800):
#         """Generates a presigned URL for accessing a file."""
#         expires_duration = datetime.timedelta(seconds=expires)
#         func = self.client.presigned_get_object
#         args = (self.bucket_name, object_name, expires_duration)
#         print(f"Generating presigned URL for: {object_name}")
#         try:
#             url = await self.loop.run_in_executor(self.executor, func, *args)
#             print(f"Presigned URL: {url}")
#             return url
#         except Exception as e:
#             print(f"Error generating presigned URL: {e}")
#             raise

#     def get_public_url(self, object_name):
#         """Returns the public URL of an object."""
#         return f"https://{self.endpoint}/{self.bucket_name}/{object_name}"
    

#     async def delete_all_items(self):
#         """Deletes all items in the Minio bucket."""
#         print(f"Deleting all items in bucket: {self.bucket_name}")
#         try:
#             # List all objects in the bucket
#             objects = await self.list_objects()
#             if not objects:
#                 print("No objects to delete.")
#                 return
            
#             # Delete each object
#             for obj_name in objects:
#                 print(f"Deleting object: {obj_name}")
#                 await self.delete_file(obj_name)
#             print("All objects deleted successfully.")
#         except Exception as e:
#             print(f"Error deleting all items in bucket: {e}")


#     def __del__(self):
#         """Shuts down the executor."""
#         self.executor.shutdown(wait=False)

# async def main():
#     # Load environment variables
#     endpoint = os.getenv("S3_COMPATIBLE_ENDPOINT", "localhost:9000")
#     access_key = os.getenv("S3_COMPATIBLE_ACCESS_KEY", "minioadmin")
#     secret_key = os.getenv("S3_COMPATIBLE_SECRET_KEY", "minioadmin")
#     bucket_name = os.getenv("BUCKET_NAME", "vidaio-subnet")

#     # Initialize Minio client
#     minio_client = VideoSubnetMinioClient(
#         endpoint=endpoint,
#         access_key=access_key,
#         secret_key=secret_key,
#         bucket_name=bucket_name,
#     )

#     # Ensure bucket exists
#     await minio_client.ensure_bucket_exists()

#     # Upload a file
#     # await minio_client.upload_file("345.mp4", "/root/workspace/vidaio-subnet/videos/4k_4887282_hd.mp4")

#     # List objects
#     objects = await minio_client.list_objects()
#     print(f"Objects in bucket: {objects}")

#     # await minio_client.delete_all_items()
    
#     objects = await minio_client.list_objects()
#     print(f"Objects are : {objects}")

#     # Set bucket public policy
#     # await minio_client.set_bucket_public_policy()

#     # Generate a presigned URL
#     # url = await minio_client.get_presigned_url("345.mp4")
#     # print(f"Presigned URL: {url}")

# if __name__ == "__main__":
#     asyncio.run(main())