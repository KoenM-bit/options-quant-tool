"""
MinIO S3-Compatible Object Storage Client
==========================================
Provides utilities for interacting with MinIO object storage.
Compatible with both MinIO and AWS S3.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List
from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)


class MinIOClient:
    """
    MinIO client for S3-compatible object storage operations.
    
    Usage:
        client = MinIOClient()
        client.upload_file('local/file.parquet', 'parquet/gold/file.parquet')
        client.download_file('parquet/gold/file.parquet', 'local/file.parquet')
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket: Optional[str] = None,
        secure: Optional[bool] = None
    ):
        """
        Initialize MinIO client.
        
        Args:
            endpoint: MinIO endpoint (default from env: MINIO_ENDPOINT)
            access_key: Access key (default from env: MINIO_ACCESS_KEY)
            secret_key: Secret key (default from env: MINIO_SECRET_KEY)
            bucket: Default bucket name (default from env: MINIO_BUCKET)
            secure: Use HTTPS (default from env: MINIO_SECURE)
        """
        self.endpoint = endpoint or os.getenv('MINIO_ENDPOINT', 'minio:9000')
        self.access_key = access_key or os.getenv('MINIO_ACCESS_KEY', 'admin')
        self.secret_key = secret_key or os.getenv('MINIO_SECRET_KEY', 'miniopassword123')
        self.bucket = bucket or os.getenv('MINIO_BUCKET', 'options-data')
        self.secure = secure if secure is not None else os.getenv('MINIO_SECURE', 'false').lower() == 'true'
        
        # Initialize MinIO client
        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
        
        logger.info(f"MinIO client initialized: {self.endpoint}/{self.bucket}")
    
    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist."""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info(f"✅ Created bucket: {self.bucket}")
            else:
                logger.debug(f"Bucket already exists: {self.bucket}")
        except S3Error as e:
            logger.error(f"Failed to create/check bucket: {e}")
            raise
    
    def upload_file(
        self,
        local_path: str,
        object_name: str,
        content_type: Optional[str] = None
    ) -> bool:
        """
        Upload a file to MinIO.
        
        Args:
            local_path: Local file path
            object_name: Object name in bucket (e.g., 'parquet/gold/file.parquet')
            content_type: MIME type (auto-detected if None)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Auto-detect content type
            if content_type is None:
                if local_path.endswith('.parquet'):
                    content_type = 'application/octet-stream'
                elif local_path.endswith('.json'):
                    content_type = 'application/json'
                elif local_path.endswith('.csv'):
                    content_type = 'text/csv'
            
            # Upload file
            self.client.fput_object(
                self.bucket,
                object_name,
                local_path,
                content_type=content_type
            )
            
            file_size = Path(local_path).stat().st_size
            logger.info(f"✅ Uploaded: {object_name} ({file_size:,} bytes)")
            return True
            
        except S3Error as e:
            logger.error(f"Failed to upload {local_path} to {object_name}: {e}")
            return False
    
    def download_file(
        self,
        object_name: str,
        local_path: str
    ) -> bool:
        """
        Download a file from MinIO.
        
        Args:
            object_name: Object name in bucket
            local_path: Local file path to save to
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure local directory exists
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            self.client.fget_object(
                self.bucket,
                object_name,
                local_path
            )
            
            logger.info(f"✅ Downloaded: {object_name} → {local_path}")
            return True
            
        except S3Error as e:
            logger.error(f"Failed to download {object_name}: {e}")
            return False
    
    def list_objects(
        self,
        prefix: Optional[str] = None,
        recursive: bool = True
    ) -> List[str]:
        """
        List objects in bucket.
        
        Args:
            prefix: Filter by prefix (e.g., 'parquet/gold/')
            recursive: List recursively
        
        Returns:
            List of object names
        """
        try:
            objects = self.client.list_objects(
                self.bucket,
                prefix=prefix,
                recursive=recursive
            )
            
            object_names = [obj.object_name for obj in objects]
            logger.info(f"Found {len(object_names)} objects with prefix '{prefix}'")
            return object_names
            
        except S3Error as e:
            logger.error(f"Failed to list objects: {e}")
            return []
    
    def delete_object(self, object_name: str) -> bool:
        """
        Delete an object from MinIO.
        
        Args:
            object_name: Object name to delete
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.remove_object(self.bucket, object_name)
            logger.info(f"✅ Deleted: {object_name}")
            return True
            
        except S3Error as e:
            logger.error(f"Failed to delete {object_name}: {e}")
            return False
    
    def object_exists(self, object_name: str) -> bool:
        """
        Check if an object exists.
        
        Args:
            object_name: Object name to check
        
        Returns:
            True if exists, False otherwise
        """
        try:
            self.client.stat_object(self.bucket, object_name)
            return True
        except S3Error:
            return False
    
    def get_object_url(self, object_name: str, expires: int = 3600) -> str:
        """
        Get a presigned URL for temporary access.
        
        Args:
            object_name: Object name
            expires: URL expiration time in seconds (default: 1 hour)
        
        Returns:
            Presigned URL
        """
        try:
            url = self.client.presigned_get_object(
                self.bucket,
                object_name,
                expires=expires
            )
            logger.info(f"Generated presigned URL for {object_name} (expires in {expires}s)")
            return url
        except S3Error as e:
            logger.error(f"Failed to generate URL for {object_name}: {e}")
            return ""


# Convenience function
def get_minio_client() -> MinIOClient:
    """Get a configured MinIO client instance."""
    return MinIOClient()
