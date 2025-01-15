from minio import Minio
from ..global_config import CONFIG


class VideoSubnetMinioClient:
    def __init__(self):
        self.client = Minio(
            endpoint=CONFIG.minio.endpoint,
            access_key=CONFIG.minio.access_key,
            secret_key=CONFIG.minio.secret_key,
            secure=CONFIG.minio.secure,
        )

    def upload_file(self, bucket_name: str, file_path: str, object_name: str):
        self.client.fput_object(bucket_name, object_name, file_path)

    def download_file(self, bucket_name: str, object_name: str, file_path: str):
        self.client.fget_object(bucket_name, object_name, file_path)

    def delete_file(self, bucket_name: str, object_name: str):
        self.client.remove_object(bucket_name, object_name)
