#!/usr/bin/env python3
"""
Restore Database from MinIO Backup

Downloads a backup from MinIO and restores it to the PostgreSQL database.

Usage:
    # List available backups
    python scripts/restore_database_from_minio.py --list
    
    # Restore latest backup
    python scripts/restore_database_from_minio.py --latest
    
    # Restore specific backup
    python scripts/restore_database_from_minio.py --backup ahold_options_2025-12-16_120000.sql.gz
    
    # Restore with confirmation prompt
    python scripts/restore_database_from_minio.py --latest --confirm
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import subprocess
import tempfile
import logging
from datetime import datetime
from typing import List, Optional

from src.utils.minio_client import get_minio_client
from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseRestore:
    """Handles database restore from MinIO backups."""
    
    def __init__(self):
        self.minio_client = get_minio_client()
        self.backup_prefix = "backups/postgres/"
    
    def list_backups(self) -> List[dict]:
        """
        List all available backups in MinIO.
        
        Returns:
            List of backup info dicts sorted by date (newest first)
        """
        try:
            from minio import Minio
            
            # Get MinIO client
            minio = Minio(
                self.minio_client.endpoint,
                access_key=self.minio_client.access_key,
                secret_key=self.minio_client.secret_key,
                secure=False
            )
            
            # List objects
            objects = minio.list_objects(
                self.minio_client.bucket,
                prefix=self.backup_prefix,
                recursive=True
            )
            
            backups = []
            
            for obj in objects:
                filename = obj.object_name.split('/')[-1]
                
                # Parse filename: ahold_options_2025-12-16_120000.sql.gz
                try:
                    parts = filename.replace('.sql.gz', '').split('_')
                    if len(parts) >= 4:
                        date_str = parts[2]
                        time_str = parts[3]
                        
                        backup_datetime = datetime.strptime(
                            f"{date_str}_{time_str}",
                            "%Y-%m-%d_%H%M%S"
                        )
                        
                        backups.append({
                            'filename': filename,
                            'path': obj.object_name,
                            'date': backup_datetime,
                            'size_mb': obj.size / (1024 * 1024),
                            'last_modified': obj.last_modified
                        })
                
                except (ValueError, IndexError) as e:
                    logger.warning(f"Could not parse backup: {filename}")
            
            # Sort by date, newest first
            backups.sort(key=lambda x: x['date'], reverse=True)
            
            return backups
        
        except Exception as e:
            logger.error(f"Error listing backups: {e}")
            raise
    
    def download_backup(self, backup_path: str) -> str:
        """
        Download backup from MinIO to temporary file.
        
        Args:
            backup_path: Path to backup in MinIO
            
        Returns:
            Path to downloaded file
        """
        filename = backup_path.split('/')[-1]
        
        logger.info(f"📥 Downloading backup: {filename}")
        
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.sql.gz', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            from minio import Minio
            
            minio = Minio(
                self.minio_client.endpoint,
                access_key=self.minio_client.access_key,
                secret_key=self.minio_client.secret_key,
                secure=False
            )
            
            # Download file
            minio.fget_object(
                self.minio_client.bucket,
                backup_path,
                tmp_path
            )
            
            file_size = Path(tmp_path).stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            logger.info(f"✅ Downloaded {file_size_mb:.2f} MB")
            
            return tmp_path
        
        except Exception as e:
            Path(tmp_path).unlink(missing_ok=True)
            raise
    
    def restore_backup(self, backup_path: str, confirm: bool = True):
        """
        Restore database from backup.
        
        Args:
            backup_path: Local path to compressed backup file
            confirm: If True, prompt for confirmation before restore
        """
        if confirm:
            response = input(
                f"\n⚠️  WARNING: This will REPLACE the database '{settings.postgres_db}' "
                f"on {settings.postgres_host}:{settings.postgres_port}\n"
                f"Are you sure? Type 'yes' to continue: "
            )
            
            if response.lower() != 'yes':
                logger.info("❌ Restore cancelled by user")
                return
        
        logger.info(f"🔄 Restoring database from backup...")
        
        try:
            # Decompress and restore in one command
            # gunzip -c backup.sql.gz | psql
            
            gunzip = subprocess.Popen(
                ['gunzip', '-c', backup_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            psql = subprocess.Popen(
                [
                    'psql',
                    '-h', settings.postgres_host,
                    '-p', str(settings.postgres_port),
                    '-U', settings.postgres_user,
                    '-d', settings.postgres_db,
                    '-q'  # Quiet mode
                ],
                stdin=gunzip.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={'PGPASSWORD': settings.postgres_password}
            )
            
            gunzip.stdout.close()
            
            stdout, stderr = psql.communicate()
            gunzip_err = gunzip.communicate()[1]
            
            if gunzip.returncode != 0:
                raise RuntimeError(f"Decompression failed: {gunzip_err.decode()}")
            
            if psql.returncode != 0:
                raise RuntimeError(f"Database restore failed: {stderr.decode()}")
            
            logger.info("✅ Database restored successfully")
            
        except Exception as e:
            logger.error(f"❌ Restore failed: {e}")
            raise
        
        finally:
            # Cleanup temp file
            Path(backup_path).unlink(missing_ok=True)
    
    def restore_latest(self, confirm: bool = True):
        """Restore the most recent backup."""
        backups = self.list_backups()
        
        if not backups:
            raise ValueError("No backups found in MinIO")
        
        latest = backups[0]
        
        logger.info(f"📦 Latest backup: {latest['filename']}")
        logger.info(f"📅 Date: {latest['date']}")
        logger.info(f"💾 Size: {latest['size_mb']:.2f} MB")
        
        # Download and restore
        backup_path = self.download_backup(latest['path'])
        self.restore_backup(backup_path, confirm=confirm)
    
    def restore_specific(self, backup_filename: str, confirm: bool = True):
        """Restore a specific backup by filename."""
        backups = self.list_backups()
        
        # Find the backup
        backup = None
        for b in backups:
            if b['filename'] == backup_filename:
                backup = b
                break
        
        if not backup:
            raise ValueError(f"Backup not found: {backup_filename}")
        
        logger.info(f"📦 Backup: {backup['filename']}")
        logger.info(f"📅 Date: {backup['date']}")
        logger.info(f"💾 Size: {backup['size_mb']:.2f} MB")
        
        # Download and restore
        backup_path = self.download_backup(backup['path'])
        self.restore_backup(backup_path, confirm=confirm)


def main():
    parser = argparse.ArgumentParser(description='Restore database from MinIO backup')
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available backups'
    )
    parser.add_argument(
        '--latest',
        action='store_true',
        help='Restore latest backup'
    )
    parser.add_argument(
        '--backup',
        type=str,
        help='Specific backup filename to restore'
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        default=True,
        help='Prompt for confirmation before restore (default: True)'
    )
    parser.add_argument(
        '--no-confirm',
        action='store_true',
        help='Skip confirmation prompt'
    )
    
    args = parser.parse_args()
    
    restore = DatabaseRestore()
    
    if args.list:
        # List backups
        backups = restore.list_backups()
        
        if not backups:
            print("No backups found in MinIO")
            return
        
        print("\n" + "=" * 80)
        print("AVAILABLE BACKUPS")
        print("=" * 80)
        
        for i, backup in enumerate(backups, 1):
            print(f"\n{i}. {backup['filename']}")
            print(f"   Date: {backup['date']}")
            print(f"   Size: {backup['size_mb']:.2f} MB")
            print(f"   Modified: {backup['last_modified']}")
        
        print("\n" + "=" * 80)
        
    elif args.latest:
        # Restore latest
        confirm = not args.no_confirm
        restore.restore_latest(confirm=confirm)
        
    elif args.backup:
        # Restore specific backup
        confirm = not args.no_confirm
        restore.restore_specific(args.backup, confirm=confirm)
        
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
