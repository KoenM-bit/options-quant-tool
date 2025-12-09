#!/usr/bin/env python3
"""
Sync data from MinIO parquet files to ClickHouse.

This script loads parquet files from MinIO S3 storage into ClickHouse
for fast analytics queries. It's designed to run after the parquet export.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, '/opt/airflow')

from src.utils.clickhouse_client import get_clickhouse_client
from src.utils.minio_client import get_minio_client


# Table configurations: (table_name, order_by_columns, partition_by_column)
# Note: order_by and partition_by set to None to avoid nullable column issues
# ClickHouse will use default ordering. We can optimize later with NOT NULL constraints.
GOLD_TABLES = [
    ('gold_gamma_exposure_weekly', None, None),
    ('gold_gex_positioning_trends', None, None),
    ('gold_max_pain', None, None),
    ('gold_skew_analysis', None, None),
    ('gold_key_levels', None, None),
    ('gold_volatility_surface', None, None),
    ('gold_options_summary_daily', None, None),
    ('gold_volatility_term_structure', None, None),
    ('gold_open_interest_flow', None, None),
    ('gold_put_call_metrics', None, None),
]

SILVER_TABLES = [
    ('silver_options', None, None),
    ('silver_underlying_price', None, None),
]


def sync_table_to_clickhouse(
    clickhouse_client,
    minio_client,
    table_name: str,
    layer: str,
    order_by: list = None,
    partition_by: str = None
) -> dict:
    """
    Sync a table from MinIO to ClickHouse.
    
    Args:
        clickhouse_client: ClickHouse client instance
        minio_client: MinIO client instance
        table_name: Name of the table
        layer: Data layer (gold or silver)
        order_by: List of columns to order by
        partition_by: Column to partition by
        
    Returns:
        dict with sync statistics
    """
    print(f"📦 Syncing {table_name}...")
    
    try:
        # Get MinIO configuration
        minio_endpoint = os.getenv('MINIO_ENDPOINT', 'minio:9000')
        minio_bucket = os.getenv('MINIO_BUCKET', 'options-data')
        minio_access_key = os.getenv('MINIO_ACCESS_KEY', 'admin')
        minio_secret_key = os.getenv('MINIO_SECRET_KEY', 'miniopassword123')
        
        # Construct S3 URL for parquet file
        object_name = f"parquet/{layer}/{table_name}.parquet"
        s3_url = f"http://{minio_endpoint}/{minio_bucket}/{object_name}"
        
        # Check if file exists in MinIO
        if not minio_client.object_exists(object_name):
            print(f"  ⚠️  File not found in MinIO: {object_name}")
            return {'status': 'skipped', 'reason': 'file_not_found'}
        
        # Drop and recreate table in ClickHouse (full refresh)
        clickhouse_client.create_table_from_s3(
            table_name=table_name,
            s3_url=s3_url,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            order_by=order_by,
            partition_by=partition_by
        )
        
        # Get row count
        row_count = clickhouse_client.get_table_count(table_name)
        
        # Optimize table (merge parts)
        clickhouse_client.optimize_table(table_name)
        
        print(f"  ✅ Synced {row_count:,} rows to ClickHouse")
        
        return {
            'status': 'success',
            'rows': row_count,
            'table': table_name
        }
        
    except Exception as e:
        print(f"  ❌ Error syncing {table_name}: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'table': table_name
        }


def main():
    """Main function to sync all tables from MinIO to ClickHouse."""
    
    print("\n" + "=" * 80)
    print("🔄 SYNCING DATA FROM MINIO TO CLICKHOUSE")
    print("=" * 80)
    print()
    
    # Initialize clients
    print("🔌 Connecting to services...")
    clickhouse_client = get_clickhouse_client()
    minio_client = get_minio_client()
    print("  ✅ Connected to ClickHouse and MinIO")
    print()
    
    # Track statistics
    stats = {
        'success': [],
        'skipped': [],
        'error': []
    }
    
    # Sync Gold tables
    print("🥇 GOLD LAYER:")
    print("-" * 80)
    for table_name, order_by, partition_by in GOLD_TABLES:
        result = sync_table_to_clickhouse(
            clickhouse_client,
            minio_client,
            table_name,
            'gold',
            order_by,
            partition_by
        )
        stats[result['status']].append(result)
    
    print()
    
    # Sync Silver tables
    print("🥈 SILVER LAYER:")
    print("-" * 80)
    for table_name, order_by, partition_by in SILVER_TABLES:
        result = sync_table_to_clickhouse(
            clickhouse_client,
            minio_client,
            table_name,
            'silver',
            order_by,
            partition_by
        )
        stats[result['status']].append(result)
    
    print()
    print("=" * 80)
    print("✅ SYNC COMPLETE")
    print("=" * 80)
    print()
    
    # Calculate totals
    total_rows = sum(r.get('rows', 0) for r in stats['success'])
    success_count = len(stats['success'])
    skipped_count = len(stats['skipped'])
    error_count = len(stats['error'])
    
    print(f"📊 Summary:")
    print(f"  Tables synced: {success_count}/{len(GOLD_TABLES) + len(SILVER_TABLES)}")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")
    print()
    
    if stats['error']:
        print("❌ Failed tables:")
        for result in stats['error']:
            print(f"  - {result['table']}: {result['error']}")
        print()
    
    print("🔌 CLICKHOUSE CONNECTION INFO:")
    print(f"  Host: {os.getenv('CLICKHOUSE_HOST', 'clickhouse')}")
    print(f"  Port: {os.getenv('CLICKHOUSE_PORT', '8123')}")
    print(f"  Database: {os.getenv('CLICKHOUSE_DB', 'ahold_options')}")
    print(f"  User: {os.getenv('CLICKHOUSE_USER', 'default')}")
    print()
    
    print("📊 POWER BI CONNECTION:")
    print("  1. In Power BI Desktop, click 'Get Data' > 'More' > 'ClickHouse'")
    print(f"  2. Server: {os.getenv('CLICKHOUSE_HOST', 'localhost')}:{os.getenv('CLICKHOUSE_PORT', '8123')}")
    print(f"  3. Database: {os.getenv('CLICKHOUSE_DB', 'ahold_options')}")
    print("  4. Select tables to import or use DirectQuery")
    print()
    
    # Close clients
    clickhouse_client.close()
    
    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    exit(main())
