#!/usr/bin/env python3
"""
Initialize MinIO bucket and folder structure.

This script ensures the MinIO bucket exists and creates the folder structure
for organized parquet storage. Run this on container startup.
"""

import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, '/opt/airflow')

from src.utils.minio_client import get_minio_client


def wait_for_minio(max_retries=30, retry_delay=2):
    """
    Wait for MinIO to be ready before initializing.
    
    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Seconds to wait between retries
    """
    print("⏳ Waiting for MinIO to be ready...")
    
    for attempt in range(max_retries):
        try:
            client = get_minio_client()
            # Try to list buckets to verify connection
            client.client.list_buckets()
            print(f"✅ MinIO is ready after {attempt + 1} attempt(s)")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"   Attempt {attempt + 1}/{max_retries} failed, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print(f"❌ MinIO not available after {max_retries} attempts: {e}")
                return False
    
    return False


def init_minio_storage():
    """
    Initialize MinIO bucket and folder structure.
    """
    print("\n" + "=" * 80)
    print("🪣 MINIO INITIALIZATION")
    print("=" * 80)
    print()
    
    # Check if MinIO is enabled
    use_minio = os.getenv('USE_MINIO', 'true').lower() == 'true'
    if not use_minio:
        print("⚠️  MinIO is disabled (USE_MINIO=false)")
        return
    
    # Wait for MinIO to be available
    if not wait_for_minio():
        print("⚠️  Skipping MinIO initialization")
        return
    
    try:
        client = get_minio_client()
        bucket_name = os.getenv('MINIO_BUCKET', 'options-data')
        
        # Check if bucket exists
        if client.client.bucket_exists(bucket_name):
            print(f"✅ Bucket '{bucket_name}' already exists")
        else:
            # Create bucket
            client.client.make_bucket(bucket_name)
            print(f"✅ Created bucket '{bucket_name}'")
        
        # Create folder structure by uploading empty .keep files
        # (MinIO doesn't have real folders, but this helps with organization)
        folders = [
            'parquet/gold/',
            'parquet/silver/',
        ]
        
        print("\n📁 Creating folder structure:")
        for folder in folders:
            try:
                # Upload a small .keep file to create the folder path
                keep_file = f"{folder}.keep"
                client.client.put_object(
                    bucket_name,
                    keep_file,
                    data=b'',
                    length=0,
                    content_type='text/plain'
                )
                print(f"   ✅ {folder}")
            except Exception as e:
                print(f"   ⚠️  Could not create {folder}: {e}")
        
        print("\n" + "=" * 80)
        print("✅ MINIO INITIALIZATION COMPLETE")
        print("=" * 80)
        print()
        print(f"📊 Bucket: {bucket_name}")
        print(f"🌐 Console: http://{os.getenv('MINIO_ENDPOINT', 'localhost:9000').split(':')[0]}:9001")
        print(f"🔑 Username: {os.getenv('MINIO_ROOT_USER', 'admin')}")
        print()
        
    except Exception as e:
        print(f"\n❌ MinIO initialization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    init_minio_storage()
