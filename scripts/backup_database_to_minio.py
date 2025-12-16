#!/usr/bin/env python3
"""
Daily Database Backup to MinIO

Creates a compressed PostgreSQL dump and uploads it to MinIO with date-based naming.
Implements a 7-day retention policy (deletes backups older than 7 days).

Usage:
    python scripts/backup_database_to_minio.py
    python scripts/backup_database_to_minio.py --date 2025-12-16
    python scripts/backup_database_to_minio.py --retention-days 14
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import subprocess
import tempfile
import logging
from datetime import datetime, timedelta, date
from typing import Optional

from src.utils.minio_client import get_minio_client
from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseBackup:
    """Handles database backup to MinIO with retention management."""
    
    def __init__(self, retention_days: int = 7):
        self.minio_client = get_minio_client()
        self.retention_days = retention_days
        self.backup_prefix = "backups/postgres/"
        
        # Ensure lifecycle policy is set
        self._ensure_lifecycle_policy()
    
    def _ensure_lifecycle_policy(self):
        """Ensure MinIO lifecycle policy is configured for automatic cleanup."""
        try:
            from minio import Minio
            from minio.lifecycleconfig import LifecycleConfig, Rule, Expiration, Filter
            
            minio = Minio(
                self.minio_client.endpoint,
                access_key=self.minio_client.access_key,
                secret_key=self.minio_client.secret_key,
                secure=False
            )
            
            # Check if lifecycle policy exists
            try:
                lifecycle = minio.get_bucket_lifecycle(self.minio_client.bucket)
                
                # Check if our rule exists
                has_backup_rule = False
                if lifecycle and lifecycle.rules:
                    for rule in lifecycle.rules:
                        if rule.rule_id == "delete_old_backups":
                            has_backup_rule = True
                            break
                
                if has_backup_rule:
                    logger.debug(f"✅ Lifecycle policy already configured ({self.retention_days} days)")
                    return
            
            except Exception as e:
                # No lifecycle policy exists, we'll create it
                pass
            
            # Create lifecycle rule
            rule = Rule(
                rule_id="delete_old_backups",
                rule_filter=Filter(prefix=self.backup_prefix),
                status="Enabled",
                expiration=Expiration(days=self.retention_days)
            )
            
            lifecycle_config = LifecycleConfig([rule])
            minio.set_bucket_lifecycle(self.minio_client.bucket, lifecycle_config)
            
            logger.info(f"✅ MinIO lifecycle policy configured: {self.retention_days} days retention")
        
        except Exception as e:
            logger.warning(f"⚠️  Could not set lifecycle policy (will use script-based cleanup): {e}")
    
    def create_backup(self, backup_date: Optional[date] = None) -> str:
        """
        Create a compressed database backup.
        
        Args:
            backup_date: Date for backup naming (default: today)
            
        Returns:
            Path to the backup file
        """
        if backup_date is None:
            backup_date = date.today()
        
        # Create backup filename with timestamp
        timestamp = datetime.now().strftime("%H%M%S")
        backup_filename = f"ahold_options_{backup_date}_{timestamp}.sql.gz"
        
        logger.info(f"📦 Creating database backup: {backup_filename}")
        
        # Create temporary file for backup
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.sql.gz', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Run pg_dump with compression
            # Format: pg_dump -h host -p port -U user -d database | gzip > backup.sql.gz
            cmd = [
                'pg_dump',
                '-h', settings.postgres_host,
                '-p', str(settings.postgres_port),
                '-U', settings.postgres_user,
                '-d', settings.postgres_db,
                '--clean',  # Include DROP commands
                '--if-exists',  # Add IF EXISTS to DROP commands
                '--no-owner',  # Don't dump ownership commands
                '--no-privileges',  # Don't dump privilege commands
            ]
            
            # Set password via environment
            env = {
                'PGPASSWORD': settings.postgres_password
            }
            
            logger.info(f"🔄 Running pg_dump for database '{settings.postgres_db}'...")
            
            # Run pg_dump and pipe to gzip
            with open(tmp_path, 'wb') as f:
                pg_dump = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env
                )
                
                gzip = subprocess.Popen(
                    ['gzip', '-c'],
                    stdin=pg_dump.stdout,
                    stdout=f,
                    stderr=subprocess.PIPE
                )
                
                pg_dump.stdout.close()
                gzip_err = gzip.communicate()[1]
                pg_dump_err = pg_dump.communicate()[1]
                
                if pg_dump.returncode != 0:
                    raise RuntimeError(f"pg_dump failed: {pg_dump_err.decode()}")
                
                if gzip.returncode != 0:
                    raise RuntimeError(f"gzip failed: {gzip_err.decode()}")
            
            # Get file size
            file_size = Path(tmp_path).stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            logger.info(f"✅ Backup created: {file_size_mb:.2f} MB")
            
            return tmp_path, backup_filename
            
        except Exception as e:
            # Cleanup temp file on error
            Path(tmp_path).unlink(missing_ok=True)
            raise
    
    def upload_backup(self, backup_path: str, backup_filename: str) -> str:
        """
        Upload backup to MinIO.
        
        Args:
            backup_path: Local path to backup file
            backup_filename: Filename for MinIO
            
        Returns:
            MinIO object path
        """
        s3_path = f"{self.backup_prefix}{backup_filename}"
        
        logger.info(f"☁️  Uploading backup to MinIO: {s3_path}")
        
        try:
            self.minio_client.upload_file(backup_path, s3_path)
            
            file_size = Path(backup_path).stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            logger.info(f"✅ Uploaded {file_size_mb:.2f} MB to s3://{self.minio_client.bucket}/{s3_path}")
            
            return s3_path
            
        finally:
            # Cleanup local backup file
            Path(backup_path).unlink(missing_ok=True)
            logger.info("🧹 Cleaned up local backup file")
    
    def cleanup_old_backups(self):
        """
        Delete backups older than retention_days from MinIO.
        """
        logger.info(f"🧹 Cleaning up backups older than {self.retention_days} days...")
        
        cutoff_date = date.today() - timedelta(days=self.retention_days)
        
        # List all backup objects
        try:
            from minio import Minio
            
            # Get MinIO client directly
            minio = Minio(
                self.minio_client.endpoint,
                access_key=self.minio_client.access_key,
                secret_key=self.minio_client.secret_key,
                secure=False
            )
            
            # List objects with prefix
            objects = minio.list_objects(
                self.minio_client.bucket,
                prefix=self.backup_prefix,
                recursive=True
            )
            
            deleted_count = 0
            total_size_mb = 0
            
            for obj in objects:
                object_name = obj.object_name
                
                # Extract date from filename: ahold_options_2025-12-16_HHMMSS.sql.gz
                try:
                    # Get filename without path
                    filename = object_name.split('/')[-1]
                    
                    # Extract date part (YYYY-MM-DD)
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        date_str = parts[2]  # Should be YYYY-MM-DD
                        backup_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        
                        if backup_date < cutoff_date:
                            # Delete old backup
                            minio.remove_object(self.minio_client.bucket, object_name)
                            
                            size_mb = obj.size / (1024 * 1024)
                            total_size_mb += size_mb
                            deleted_count += 1
                            
                            logger.info(f"🗑️  Deleted old backup: {filename} ({size_mb:.2f} MB)")
                
                except (ValueError, IndexError) as e:
                    logger.warning(f"⚠️  Could not parse date from {object_name}: {e}")
            
            if deleted_count > 0:
                logger.info(f"✅ Deleted {deleted_count} old backups (freed {total_size_mb:.2f} MB)")
            else:
                logger.info("✅ No old backups to delete")
        
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
    
    def run_backup(self, backup_date: Optional[date] = None) -> dict:
        """
        Run full backup workflow: create, upload, cleanup.
        
        Returns:
            Dict with backup information
        """
        try:
            # Create backup
            backup_path, backup_filename = self.create_backup(backup_date)
            
            # Upload to MinIO
            s3_path = self.upload_backup(backup_path, backup_filename)
            
            # Cleanup old backups
            self.cleanup_old_backups()
            
            result = {
                'status': 'success',
                'backup_filename': backup_filename,
                's3_path': s3_path,
                'date': str(backup_date or date.today())
            }
            
            logger.info("=" * 80)
            logger.info("✅ DATABASE BACKUP COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Backup: s3://{self.minio_client.bucket}/{s3_path}")
            logger.info(f"Retention: {self.retention_days} days")
            logger.info("=" * 80)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Backup database to MinIO')
    parser.add_argument(
        '--date',
        type=str,
        help='Backup date (YYYY-MM-DD, default: today)'
    )
    parser.add_argument(
        '--retention-days',
        type=int,
        default=7,
        help='Number of days to retain backups (default: 7)'
    )
    
    args = parser.parse_args()
    
    # Parse date if provided
    backup_date = None
    if args.date:
        backup_date = datetime.strptime(args.date, '%Y-%m-%d').date()
    
    # Run backup
    backup = DatabaseBackup(retention_days=args.retention_days)
    result = backup.run_backup(backup_date)
    
    return result


if __name__ == '__main__':
    main()
