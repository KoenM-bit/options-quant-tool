#!/usr/bin/env python3
"""
Setup MinIO Lifecycle Policy for Database Backups

Configures a bucket lifecycle policy to automatically delete backup files
older than 7 days. This provides an additional safety layer beyond the
application-level retention cleanup.

Usage:
    python scripts/setup_minio_lifecycle_policy.py
    python scripts/setup_minio_lifecycle_policy.py --retention-days 14
    python scripts/setup_minio_lifecycle_policy.py --remove
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import logging
from datetime import datetime

from src.utils.minio_client import get_minio_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MinIOLifecycleManager:
    """Manages MinIO bucket lifecycle policies."""
    
    def __init__(self):
        self.minio_client = get_minio_client()
        self.bucket = self.minio_client.bucket
        self.backup_prefix = "backups/postgres/"
    
    def setup_lifecycle_policy(self, retention_days: int = 7):
        """
        Set up lifecycle policy to delete backups after retention period.
        
        Args:
            retention_days: Number of days to retain backups
        """
        from minio import Minio
        from minio.lifecycleconfig import LifecycleConfig, Rule, Expiration, Filter
        
        logger.info(f"🔧 Setting up lifecycle policy for bucket: {self.bucket}")
        logger.info(f"📅 Retention: {retention_days} days")
        logger.info(f"📁 Prefix: {self.backup_prefix}")
        
        # Create MinIO client
        minio = Minio(
            self.minio_client.endpoint,
            access_key=self.minio_client.access_key,
            secret_key=self.minio_client.secret_key,
            secure=False
        )
        
        try:
            # Create lifecycle rule with Filter object
            rule = Rule(
                rule_id="delete_old_backups",
                rule_filter=Filter(prefix=self.backup_prefix),  # Use Filter object
                status="Enabled",
                expiration=Expiration(days=retention_days)
            )
            
            # Create lifecycle config
            lifecycle_config = LifecycleConfig([rule])
            
            # Set lifecycle policy
            minio.set_bucket_lifecycle(self.bucket, lifecycle_config)
            
            logger.info("✅ Lifecycle policy configured successfully")
            logger.info(f"🗑️  Backups older than {retention_days} days will be automatically deleted")
            
            # Verify policy
            self.get_lifecycle_policy()
            
        except Exception as e:
            logger.error(f"❌ Failed to set lifecycle policy: {e}")
            raise
    
    def get_lifecycle_policy(self):
        """Get and display current lifecycle policy."""
        from minio import Minio
        
        minio = Minio(
            self.minio_client.endpoint,
            access_key=self.minio_client.access_key,
            secret_key=self.minio_client.secret_key,
            secure=False
        )
        
        try:
            lifecycle = minio.get_bucket_lifecycle(self.bucket)
            
            logger.info("=" * 80)
            logger.info("CURRENT LIFECYCLE POLICY")
            logger.info("=" * 80)
            
            if lifecycle and lifecycle.rules:
                for rule in lifecycle.rules:
                    logger.info(f"Rule ID: {rule.rule_id}")
                    logger.info(f"Status: {rule.status}")
                    if hasattr(rule.rule_filter, 'prefix'):
                        logger.info(f"Prefix: {rule.rule_filter.prefix}")
                    else:
                        logger.info(f"Filter: {rule.rule_filter}")
                    if rule.expiration:
                        logger.info(f"Expiration: {rule.expiration.days} days")
                    logger.info("-" * 80)
            else:
                logger.info("No lifecycle rules configured")
            
            logger.info("=" * 80)
            
            return lifecycle
            
        except Exception as e:
            if "NoSuchLifecycleConfiguration" in str(e) or "not found" in str(e).lower():
                logger.info("ℹ️  No lifecycle policy currently configured")
                return None
            else:
                logger.error(f"❌ Failed to get lifecycle policy: {e}")
                raise
    
    def remove_lifecycle_policy(self):
        """Remove lifecycle policy from bucket."""
        from minio import Minio
        
        logger.info(f"🗑️  Removing lifecycle policy from bucket: {self.bucket}")
        
        minio = Minio(
            self.minio_client.endpoint,
            access_key=self.minio_client.access_key,
            secret_key=self.minio_client.secret_key,
            secure=False
        )
        
        try:
            minio.delete_bucket_lifecycle(self.bucket)
            logger.info("✅ Lifecycle policy removed")
            
        except Exception as e:
            if "NoSuchLifecycleConfiguration" in str(e):
                logger.info("ℹ️  No lifecycle policy to remove")
            else:
                logger.error(f"❌ Failed to remove lifecycle policy: {e}")
                raise


def main():
    parser = argparse.ArgumentParser(description='Setup MinIO lifecycle policy for backups')
    parser.add_argument(
        '--retention-days',
        type=int,
        default=7,
        help='Number of days to retain backups (default: 7)'
    )
    parser.add_argument(
        '--remove',
        action='store_true',
        help='Remove lifecycle policy'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show current lifecycle policy'
    )
    
    args = parser.parse_args()
    
    manager = MinIOLifecycleManager()
    
    if args.remove:
        manager.remove_lifecycle_policy()
    elif args.show:
        manager.get_lifecycle_policy()
    else:
        manager.setup_lifecycle_policy(args.retention_days)
    
    logger.info("=" * 80)
    logger.info("✅ COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
