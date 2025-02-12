import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import os
from dotenv import load_dotenv

class R2Manager:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        self.bucket_name = os.getenv('BUCKET_NAME')
        self.access_key_id = os.getenv('R2_ACCESS_KEY_ID')
        self.secret_access_key = os.getenv('R2_SECRET_ACCESS_KEY')
        self.endpoint_url = os.getenv('R2_ENDPOINT_URL')

        # Create S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            endpoint_url=self.endpoint_url
        )

        # Check if the bucket exists, if not create it
        self._check_or_create_bucket()

    def _check_or_create_bucket(self):
        try:
            # List existing buckets
            buckets = self.s3_client.list_buckets()
            bucket_exists = any(bucket['Name'] == self.bucket_name for bucket in buckets['Buckets'])

            if not bucket_exists:
                # Create the bucket
                self.s3_client.create_bucket(Bucket=self.bucket_name)
                print(f"Bucket '{self.bucket_name}' created successfully.")
            else:
                print(f"Bucket '{self.bucket_name}' already exists.")

        except ClientError as e:
            print(f"Failed to check or create bucket: {e}")

    def upload_file(self, full_file_path):
        try:
            # Extract the file name from the full file path
            file_name = os.path.basename(full_file_path)

            # Upload the file
            self.s3_client.upload_file(full_file_path, self.bucket_name, file_name)

            # Make the file public
            self.s3_client.put_object_acl(ACL='public-read', Bucket=self.bucket_name, Key=file_name)

            # Generate the public URL
            public_url = f"https://{self.bucket_name}.{self.s3_client.meta.endpoint_url}/{file_name}"
            return file_name, public_url
            
        except FileNotFoundError:
            print(f"The file {full_file_path} was not found.")
            return None, None
        except NoCredentialsError:
            print("Credentials not available.")
            return None, None
        except ClientError as e:
            print(f"Failed to upload {full_file_path}: {e}")
            return None, None

    def delete_file(self, file_name):
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=file_name)
            print(f"{file_name} has been deleted from {self.bucket_name}.")
        except ClientError as e:
            print(f"Failed to delete {file_name}: {e}")

# Example Usage
if __name__ == "__main__":
    r2_manager = R2Manager()

    # Upload a file and get the public URL
    file_name, public_url = r2_manager.upload_file('/Users/mac/Documents/work/video-streaming/vidaio-subnet/README.md')
    if public_url:
        print(f"File '{file_name}' uploaded successfully. Public URL: {public_url}")

    # Delete a specific file
    # r2_manager.delete_file('file.txt')
