from minio import Minio
import os
import asyncio
import datetime
import boto3
from botocore.exceptions import ClientError
from botocore.client import Config
from concurrent.futures import ThreadPoolExecutor
from vidaio_subnet_core.global_config import CONFIG


class BackblazeClient:
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
            loop = asyncio.get_running_loop()  
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
        print(f"Attempting to delete file: {object_name}")
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

    def __del__(self):
        try:
            if hasattr(self, 'executor') and self.executor:
                self.executor.shutdown(wait=False)
        except (TypeError, AttributeError, RuntimeError):
            pass


class AmazonS3Client:
    def __init__(self, endpoint, access_key, secret_key, bucket_name, secure=True, region="eu-west-1"):
        print(f"[AmazonS3Client] Initializing S3 client")
        print(f"[AmazonS3Client] Endpoint: {endpoint}")
        print(f"[AmazonS3Client] Bucket: {bucket_name}")
        print(f"[AmazonS3Client] Region: {region}")
        print(f"[AmazonS3Client] Secure: {secure}")
        
        self.endpoint = endpoint
        self.bucket_name = bucket_name
        
        try:
            self.client = boto3.client(
                's3',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                endpoint_url=endpoint,
                region_name=region,
                config=Config(signature_version='s3v4')
            )
            print(f"[AmazonS3Client] Successfully created boto3 S3 client")
        except Exception as e:
            print(f"[AmazonS3Client] ERROR: Failed to initialize boto3 client: {e}")
            raise
        
        self.executor = ThreadPoolExecutor()
        print(f"[AmazonS3Client] ThreadPoolExecutor initialized")

    async def upload_file(self, object_name, file_path):
        print(f"[AmazonS3Client] ========== UPLOAD OPERATION ==========")
        print(f"[AmazonS3Client] Object name: {object_name}")
        print(f"[AmazonS3Client] File path: {file_path}")
        print(f"[AmazonS3Client] Bucket: {self.bucket_name}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"[AmazonS3Client] ERROR: File does not exist at path: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file size
        file_size = os.path.getsize(file_path)
        print(f"[AmazonS3Client] File size: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
        
        func = self.client.upload_file
        args = (file_path, self.bucket_name, object_name)
        
        try:
            print(f"[AmazonS3Client] Starting upload to S3...")
            start_time = datetime.datetime.now()
            
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(self.executor, func, *args)
            
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"[AmazonS3Client] Upload completed successfully")
            print(f"[AmazonS3Client] Duration: {duration:.2f} seconds")
            print(f"[AmazonS3Client] Upload speed: {(file_size / 1024 / 1024) / duration:.2f} MB/s")
            print(f"[AmazonS3Client] Result: {result}")
            print(f"[AmazonS3Client] =====================================")
            
            return {"etag": None, "version_id": None, "status": "success"}
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            print(f"[AmazonS3Client] ERROR: AWS ClientError during upload")
            print(f"[AmazonS3Client] Error Code: {error_code}")
            print(f"[AmazonS3Client] Error Message: {error_message}")
            print(f"[AmazonS3Client] Full response: {e.response}")
            raise
        except Exception as e:
            print(f"[AmazonS3Client] ERROR: Unexpected error during upload: {type(e).__name__}")
            print(f"[AmazonS3Client] Error details: {str(e)}")
            raise

    async def download_file(self, object_name, file_path):
        print(f"[AmazonS3Client] ========== DOWNLOAD OPERATION ==========")
        print(f"[AmazonS3Client] Object name: {object_name}")
        print(f"[AmazonS3Client] File path: {file_path}")
        print(f"[AmazonS3Client] Bucket: {self.bucket_name}")
        
        # Check if destination directory exists
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            print(f"[AmazonS3Client] Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)
        
        func = self.client.download_file
        args = (self.bucket_name, object_name, file_path)
        
        try:
            print(f"[AmazonS3Client] Starting download from S3...")
            start_time = datetime.datetime.now()
            
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(self.executor, func, *args)
            
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Get downloaded file size
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"[AmazonS3Client] Download completed successfully")
                print(f"[AmazonS3Client] Downloaded file size: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
                print(f"[AmazonS3Client] Duration: {duration:.2f} seconds")
                print(f"[AmazonS3Client] Download speed: {(file_size / 1024 / 1024) / duration:.2f} MB/s")
            else:
                print(f"[AmazonS3Client] WARNING: Download reported success but file not found at destination")
            
            print(f"[AmazonS3Client] Result: {result}")
            print(f"[AmazonS3Client] ========================================")
            return result
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            print(f"[AmazonS3Client] ERROR: AWS ClientError during download")
            print(f"[AmazonS3Client] Error Code: {error_code}")
            print(f"[AmazonS3Client] Error Message: {error_message}")
            if error_code == 'NoSuchKey':
                print(f"[AmazonS3Client] The object '{object_name}' does not exist in bucket '{self.bucket_name}'")
            raise
        except Exception as e:
            print(f"[AmazonS3Client] ERROR: Unexpected error during download: {type(e).__name__}")
            print(f"[AmazonS3Client] Error details: {str(e)}")
            raise

    async def delete_file(self, object_name):
        print(f"[AmazonS3Client] ========== DELETE OPERATION ==========")
        print(f"[AmazonS3Client] Object name: {object_name}")
        print(f"[AmazonS3Client] Bucket: {self.bucket_name}")
        
        func = self.client.delete_object
        kwargs = {
            'Bucket': self.bucket_name,
            'Key': object_name
        }
        
        try:
            print(f"[AmazonS3Client] Attempting to delete object from S3...")
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self.executor, 
                lambda: func(**kwargs)
            )
            
            print(f"[AmazonS3Client] Delete operation completed")
            print(f"[AmazonS3Client] Response: {result}")
            
            if 'DeleteMarker' in result:
                print(f"[AmazonS3Client] Delete marker created: {result['DeleteMarker']}")
            if 'VersionId' in result:
                print(f"[AmazonS3Client] Version ID: {result['VersionId']}")
            
            print(f"[AmazonS3Client] ======================================")
            return result
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            print(f"[AmazonS3Client] ERROR: AWS ClientError during deletion")
            print(f"[AmazonS3Client] Error Code: {error_code}")
            print(f"[AmazonS3Client] Error Message: {error_message}")
            raise
        except Exception as e:
            print(f"[AmazonS3Client] ERROR: Unexpected error during deletion: {type(e).__name__}")
            print(f"[AmazonS3Client] Error details: {str(e)}")
            raise

    async def list_objects(self, prefix=None, recursive=True):
        print(f"[AmazonS3Client] ========== LIST OBJECTS OPERATION ==========")
        print(f"[AmazonS3Client] Bucket: {self.bucket_name}")
        print(f"[AmazonS3Client] Prefix: {prefix if prefix else 'None (all objects)'}")
        print(f"[AmazonS3Client] Recursive: {recursive}")
        
        func = self.client.list_objects_v2
        kwargs = {
            'Bucket': self.bucket_name
        }
        if prefix:
            kwargs['Prefix'] = prefix
        
        if not recursive:
            kwargs['Delimiter'] = '/'
            print(f"[AmazonS3Client] Using delimiter '/' for non-recursive listing")
        
        loop = asyncio.get_running_loop()
        
        def list_and_convert():
            try:
                print(f"[AmazonS3Client] Executing list_objects_v2 with params: {kwargs}")
                response = func(**kwargs)
                
                print(f"[AmazonS3Client] Response received")
                print(f"[AmazonS3Client] KeyCount: {response.get('KeyCount', 0)}")
                print(f"[AmazonS3Client] IsTruncated: {response.get('IsTruncated', False)}")
                
                objects = []
                if 'Contents' in response:
                    print(f"[AmazonS3Client] Found {len(response['Contents'])} objects")
                    for idx, item in enumerate(response['Contents']):
                        class ObjectInfo:
                            def __init__(self, obj_data):
                                self.object_name = obj_data.get('Key')
                                self.size = obj_data.get('Size', 0)
                                self.last_modified = obj_data.get('LastModified')
                        
                        obj_info = ObjectInfo(item)
                        objects.append(obj_info)
                        print(f"[AmazonS3Client]   [{idx+1}] Key: {obj_info.object_name}, Size: {obj_info.size} bytes, Last Modified: {obj_info.last_modified}")
                else:
                    print(f"[AmazonS3Client] No objects found (Contents field not in response)")
                
                if 'CommonPrefixes' in response:
                    print(f"[AmazonS3Client] Found {len(response['CommonPrefixes'])} common prefixes (folders)")
                    for prefix_info in response['CommonPrefixes']:
                        print(f"[AmazonS3Client]   Prefix: {prefix_info.get('Prefix')}")
                
                print(f"[AmazonS3Client] Total objects to return: {len(objects)}")
                print(f"[AmazonS3Client] ==============================================")
                return objects
            except ClientError as e:
                error_code = e.response['Error']['Code']
                error_message = e.response['Error']['Message']
                print(f"[AmazonS3Client] ERROR: AWS ClientError during list operation")
                print(f"[AmazonS3Client] Error Code: {error_code}")
                print(f"[AmazonS3Client] Error Message: {error_message}")
                raise
            except Exception as e:
                print(f"[AmazonS3Client] ERROR: Unexpected error during list: {type(e).__name__}")
                print(f"[AmazonS3Client] Error details: {str(e)}")
                raise
        
        return await loop.run_in_executor(self.executor, list_and_convert)

    async def ensure_bucket_exists(self):
        print(f"[AmazonS3Client] ========== ENSURE BUCKET EXISTS ==========")
        print(f"[AmazonS3Client] Checking if bucket '{self.bucket_name}' exists...")
        
        loop = asyncio.get_running_loop()
        
        async def check_bucket_exists():
            try:
                print(f"[AmazonS3Client] Executing head_bucket operation...")
                await loop.run_in_executor(
                    self.executor,
                    lambda: self.client.head_bucket(Bucket=self.bucket_name)
                )
                print(f"[AmazonS3Client] Bucket '{self.bucket_name}' exists")
                return True
            except ClientError as e:
                error_code = e.response['Error']['Code']
                print(f"[AmazonS3Client] head_bucket returned error code: {error_code}")
                if error_code == '404':
                    print(f"[AmazonS3Client] Bucket '{self.bucket_name}' does not exist")
                    return False
                elif error_code == '403':
                    print(f"[AmazonS3Client] Access denied to bucket (may exist but no permission)")
                    raise
                else:
                    print(f"[AmazonS3Client] Unexpected error: {e.response['Error']['Message']}")
                    raise
        
        try:
            exists = await check_bucket_exists()
            if not exists:
                print(f"[AmazonS3Client] Creating bucket '{self.bucket_name}'...")
                print(f"[AmazonS3Client] Region: {self.client.meta.region_name}")
                
                await loop.run_in_executor(
                    self.executor,
                    lambda: self.client.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': self.client.meta.region_name}
                    )
                )
                print(f"[AmazonS3Client] Bucket '{self.bucket_name}' created successfully")
            print(f"[AmazonS3Client] ===========================================")
        except Exception as e:
            print(f"[AmazonS3Client] ERROR in ensure_bucket_exists: {type(e).__name__}")
            print(f"[AmazonS3Client] Error details: {str(e)}")
            raise

    async def set_bucket_public_policy(self):
        """Sets the bucket policy to allow public read access."""
        print(f"[AmazonS3Client] ========== SET BUCKET PUBLIC POLICY ==========")
        print(f"[AmazonS3Client] Bucket: {self.bucket_name}")
        
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
        
        print(f"[AmazonS3Client] Policy to apply:")
        print(f"[AmazonS3Client] {json.dumps(policy, indent=2)}")
        
        try:
            loop = asyncio.get_running_loop()
            import json
            policy_str = json.dumps(policy)
            
            print(f"[AmazonS3Client] Applying public read policy...")
            await loop.run_in_executor(
                self.executor, 
                lambda: self.client.put_bucket_policy(
                    Bucket=self.bucket_name,
                    Policy=policy_str
                )
            )
            print(f"[AmazonS3Client] Public policy set successfully for bucket: {self.bucket_name}")
            print(f"[AmazonS3Client] ===============================================")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            print(f"[AmazonS3Client] ERROR: Failed to set public policy")
            print(f"[AmazonS3Client] Error Code: {error_code}")
            print(f"[AmazonS3Client] Error Message: {error_message}")
            print(f"[AmazonS3Client] This may be due to insufficient permissions or bucket policy restrictions")
            raise
        except Exception as e:
            print(f"[AmazonS3Client] ERROR: Unexpected error setting bucket policy: {type(e).__name__}")
            print(f"[AmazonS3Client] Error details: {str(e)}")
            raise

    async def delete_all_items(self):
        """Deletes all items in the S3 bucket."""
        print(f"[AmazonS3Client] ========== DELETE ALL ITEMS ==========")
        print(f"[AmazonS3Client] Bucket: {self.bucket_name}")
        print(f"[AmazonS3Client] WARNING: This will delete ALL objects in the bucket!")
        
        try:
            # List all objects in the bucket
            print(f"[AmazonS3Client] Listing all objects to delete...")
            objects = await self.list_objects()
            
            if not objects:
                print(f"[AmazonS3Client] No objects to delete. Bucket is empty.")
                print(f"[AmazonS3Client] ======================================")
                return

            count = 0
            total = len(objects)
            print(f"[AmazonS3Client] Found {total} objects to delete")
            
            # Delete each object
            for idx, obj in enumerate(objects, 1):
                print(f"[AmazonS3Client] Deleting object {idx}/{total}: {obj.object_name}")
                await self.delete_file(obj.object_name)
                count += 1
            
            print(f"[AmazonS3Client] =====================================")
            print(f"[AmazonS3Client] All {count} objects deleted successfully.")
            print(f"[AmazonS3Client] ======================================")
        except Exception as e:
            print(f"[AmazonS3Client] ERROR: Error deleting all items in bucket")
            print(f"[AmazonS3Client] Error type: {type(e).__name__}")
            print(f"[AmazonS3Client] Error details: {str(e)}")
            print(f"[AmazonS3Client] Objects deleted before error: {count}")
            raise

    async def get_presigned_url(self, object_name, expires=604800):
        print(f"[AmazonS3Client] ========== GENERATE PRESIGNED URL ==========")
        print(f"[AmazonS3Client] Object name: {object_name}")
        print(f"[AmazonS3Client] Bucket: {self.bucket_name}")
        print(f"[AmazonS3Client] Expires in: {expires} seconds ({expires / 3600:.2f} hours)")
        
        loop = asyncio.get_running_loop()
        
        def generate_url():
            try:
                print(f"[AmazonS3Client] Generating presigned URL...")
                url = self.client.generate_presigned_url(
                    'get_object',
                    Params={
                        'Bucket': self.bucket_name,
                        'Key': object_name
                    },
                    ExpiresIn=expires
                )
                print(f"[AmazonS3Client] Presigned URL generated successfully")
                print(f"[AmazonS3Client] URL length: {len(url)} characters")
                print(f"[AmazonS3Client] URL (first 100 chars): {url[:100]}...")
                print(f"[AmazonS3Client] ============================================")
                return url
            except ClientError as e:
                error_code = e.response['Error']['Code']
                error_message = e.response['Error']['Message']
                print(f"[AmazonS3Client] ERROR: Failed to generate presigned URL")
                print(f"[AmazonS3Client] Error Code: {error_code}")
                print(f"[AmazonS3Client] Error Message: {error_message}")
                raise
            except Exception as e:
                print(f"[AmazonS3Client] ERROR: Unexpected error generating presigned URL: {type(e).__name__}")
                print(f"[AmazonS3Client] Error details: {str(e)}")
                raise
        
        return await loop.run_in_executor(self.executor, generate_url)

    def __del__(self):
        print(f"[AmazonS3Client] Destructor called, shutting down ThreadPoolExecutor...")
        try:
            if hasattr(self, 'executor') and self.executor:
                self.executor.shutdown(wait=False)
                print(f"[AmazonS3Client] ThreadPoolExecutor shutdown complete")
        except (TypeError, AttributeError, RuntimeError) as e:
            print(f"[AmazonS3Client] Warning during cleanup: {e}")

class CloudflareR2Client:
    def __init__(self, endpoint, access_key, secret_key, bucket_name, secure=True, region="auto"):
        self.endpoint = endpoint
        self.bucket_name = bucket_name
        
        self.client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint,
            region_name=region,
            config=Config(signature_version='s3v4')
        )
        self.executor = ThreadPoolExecutor()

    async def upload_file(self, object_name, file_path):
        func = self.client.upload_file
        args = (file_path, self.bucket_name, object_name)
        print("Attempting to upload to Cloudflare R2")
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(self.executor, func, *args)
            print(f"The bucket_name is {self.bucket_name} and the result was {result}")
            # Return similar structure to maintain consistency
            return {"etag": None, "version_id": None}
        except Exception as e:
            print(f"There was an issue with uploading to R2: {e}")

    async def download_file(self, object_name, file_path):
        func = self.client.download_file
        args = (self.bucket_name, object_name, file_path)
        print("Attempting to download from Cloudflare R2")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    async def delete_file(self, object_name):
        func = self.client.delete_object
        kwargs = {'Bucket': self.bucket_name, 'Key': object_name}
        loop = asyncio.get_running_loop()
        print(f"Attempting to delete file from R2: {object_name}")
        return await loop.run_in_executor(
            self.executor, 
            lambda: func(**kwargs)
        )

    async def list_objects(self, prefix=None, recursive=True):
        func = self.client.list_objects_v2
        kwargs = {
            'Bucket': self.bucket_name
        }
        if prefix:
            kwargs['Prefix'] = prefix
        
        if not recursive:
            kwargs['Delimiter'] = '/'
            
        print("Listing objects in R2 bucket...")
        loop = asyncio.get_running_loop()
        
        def list_and_convert():
            response = func(**kwargs)
            objects = []
            if 'Contents' in response:
                for item in response['Contents']:
                    class ObjectInfo:
                        def __init__(self, obj_data):
                            self.object_name = obj_data.get('Key')
                            self.size = obj_data.get('Size', 0)
                            self.last_modified = obj_data.get('LastModified')
                    
                    objects.append(ObjectInfo(item))
            return objects
            
        return await loop.run_in_executor(self.executor, list_and_convert)

    async def ensure_bucket_exists(self):
        loop = asyncio.get_running_loop()
        
        async def check_bucket_exists():
            try:
                await loop.run_in_executor(
                    self.executor,
                    lambda: self.client.head_bucket(Bucket=self.bucket_name)
                )
                return True
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == '404':
                    return False
                raise
        
        exists = await check_bucket_exists()
        if not exists:
            await loop.run_in_executor(
                self.executor,
                lambda: self.client.create_bucket(Bucket=self.bucket_name)
            )

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
            import json
            policy_str = json.dumps(policy)
            await loop.run_in_executor(
                self.executor, 
                lambda: self.client.put_bucket_policy(
                    Bucket=self.bucket_name,
                    Policy=policy_str
                )
            )
            print(f"Public policy set for R2 bucket: {self.bucket_name}")
        except Exception as e:
            print(f"Failed to set public policy for R2 bucket {self.bucket_name}: {e}")

    async def delete_all_items(self):
        """Deletes all items in the R2 bucket."""
        print(f"Deleting all items in R2 bucket: {self.bucket_name}")
        try:
            objects = await self.list_objects()
            if not objects:
                print("No objects to delete.")
                return

            count = 0
            for obj in objects:
                print(f"Deleting object: {obj.object_name}")
                await self.delete_file(obj.object_name)
                count += 1
            print(f"All {count} objects deleted successfully.")
        except Exception as e:
            print(f"Error deleting all items in R2 bucket: {e}")

    async def get_presigned_url(self, object_name, expires=604800):
        loop = asyncio.get_running_loop()
        
        def generate_url():
            return self.client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': object_name
                },
                ExpiresIn=expires
            )
            
        return await loop.run_in_executor(self.executor, generate_url)

    def __del__(self):
        try:
            if hasattr(self, 'executor') and self.executor:
                self.executor.shutdown(wait=False)
        except (TypeError, AttributeError, RuntimeError):
            pass


class HippiusClient:
    """
    Client for interacting with Hippius storage service.
    """
    def __init__(self, endpoint, access_key, secret_key, bucket_name, secure=True, region=None):
        """
        Initialize the Hippius client.
        
        Args:
            endpoint (str): Hippius service endpoint
            access_key (str): Access key for authentication
            secret_key (str): Secret key for authentication
            bucket_name (str): Name of the bucket to use
            secure (bool): Whether to use HTTPS (True) or HTTP (False)
            region (str): Region for the Hippius service (may not be applicable)
        """
        self.endpoint = endpoint
        self.bucket_name = bucket_name
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure
        self.region = region
        self.executor = ThreadPoolExecutor()
        
        # Placeholder for the actual client implementation
        self.client = Minio(
            endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure,
        )

    async def upload_file(self, object_name, file_path):
        """
        Upload a file to Hippius storage.
        
        Args:
            object_name (str): Name to give the object in storage
            file_path (str): Path to the file to upload
            
        Returns:
            dict: Information about the uploaded object, including etag and version_id
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor, self.client.fput_object, self.bucket_name, object_name, file_path
        )
        return {"etag": result.etag, "version_id": result.version_id}

    async def download_file(self, object_name, file_path):
        """
        Download a file from Hippius storage.
        
        Args:
            object_name (str): Name of the object in storage
            file_path (str): Path where to save the downloaded file
            
        Returns:
            dict: Information about the downloaded object
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self.client.fget_object, self.bucket_name, object_name, file_path)
        return {"status": "success"}

    async def delete_file(self, object_name):
        """
        Delete a file from Hippius storage.
        
        Args:
            object_name (str): Name of the object to delete
            
        Returns:
            dict: Information about the deletion operation
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self.client.remove_object, self.bucket_name, object_name)
        return {"status": "success"}

    async def list_objects(self, prefix=None, recursive=True):
        """
        List objects in the Hippius bucket.
        
        Args:
            prefix (str, optional): Prefix to filter objects by
            recursive (bool): Whether to list objects recursively
            
        Returns:
            list: List of objects with attributes matching other client implementations
        """
        loop = asyncio.get_event_loop()
        objects = await loop.run_in_executor(
            self.executor, lambda: list(self.client.list_objects(self.bucket_name, prefix=prefix, recursive=recursive))
        )
        return objects

    async def ensure_bucket_exists(self):
        """
        Ensure the specified bucket exists, creating it if necessary.
        """
        loop = asyncio.get_event_loop()
        exists = await loop.run_in_executor(self.executor, self.client.bucket_exists, self.bucket_name)
        if not exists:
            await loop.run_in_executor(self.executor, self.client.make_bucket, self.bucket_name)

    async def set_bucket_public_policy(self):
        """
        Sets the bucket policy to allow public read access.
        """
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": "s3:GetObject",
                    "Resource": f"arn:aws:s3:::{self.bucket_name}/*",
                }
            ],
        }
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor, self.client.set_bucket_policy, self.bucket_name, str(policy).replace("'", '"')
        )

    async def delete_all_items(self):
        """
        Deletes all items in the Hippius bucket.
        """
        objects = await self.list_objects()
        if objects:
            object_names = [obj.object_name for obj in objects]
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor, lambda: list(self.client.remove_objects(self.bucket_name, object_names))
            )

    async def get_presigned_url(self, object_name, expires=604800):
        """
        Generate a presigned URL for an object.
        
        Args:
            object_name (str): Name of the object
            expires (int): Expiration time in seconds
            
        Returns:
            str: Presigned URL
        """
        loop = asyncio.get_event_loop()
        url = await loop.run_in_executor(
            self.executor, self.client.presigned_get_object, self.bucket_name, object_name, timedelta(seconds=expires)
        )
        return url

    def __del__(self):
        try:
            if hasattr(self, "executor") and self.executor:
                self.executor.shutdown(wait=False)
        except (TypeError, AttributeError, RuntimeError):
            pass


def get_storage_client():
    bucket_type = CONFIG.storage.bucket_type
    
    if bucket_type == "backblaze":
        return BackblazeClient(
            endpoint=CONFIG.storage.endpoint,
            access_key=CONFIG.storage.access_key,
            secret_key=CONFIG.storage.secret_key,
            bucket_name=CONFIG.storage.bucket_name,
        )
    elif bucket_type == "amazon_s3":
        return AmazonS3Client(
            endpoint=CONFIG.storage.endpoint,
            access_key=CONFIG.storage.access_key,
            secret_key=CONFIG.storage.secret_key,
            bucket_name=CONFIG.storage.bucket_name,
        )
    elif bucket_type == "cloudflare":
        return CloudflareR2Client(
            endpoint=CONFIG.storage.endpoint,
            access_key=CONFIG.storage.access_key,
            secret_key=CONFIG.storage.secret_key,
            bucket_name=CONFIG.storage.bucket_name,
        )
    elif bucket_type == "hippius":
        return HippiusClient(
            endpoint=CONFIG.storage.endpoint,  # s3.hippius.com
            access_key=CONFIG.storage.access_key,  # base64 encoded subaccount seed phrase
            secret_key=CONFIG.storage.secret_key,  # the raw seed phrase from above (used locally for signing)
            bucket_name=CONFIG.storage.bucket_name,
            secure=True,
            region="decentralized",
        )
    else:
        raise ValueError(f"Unsupported bucket type: {bucket_type}")


storage_client = get_storage_client()

async def main():
    await storage_client.upload_file("normal_03.mp4", "test1.mp4")
    presigned_url = await storage_client.get_presigned_url("normal_03.mp4")
    print(presigned_url)

if __name__ == "__main__":
    asyncio.run(main())
