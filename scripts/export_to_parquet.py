"""
Export Gold Layer to Parquet Files for Power BI
===============================================
Exports all gold layer tables to Parquet format for fast querying in Power BI.

Parquet Benefits:
1. Columnar storage - 10-100x faster queries than row-based formats
2. Built-in compression - 50-80% smaller file sizes
3. Schema preservation - data types maintained
4. Native Power BI support - direct import without database connection
5. Portable - can be shared, backed up, versioned easily
6. No database load - queries run on local files

Power BI can connect to Parquet via:
- File connector (folder of Parquet files)
- Python/Pandas script
- Direct Parquet import (newer versions)
"""

import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/opt/airflow/dags/..')

import pandas as pd
from sqlalchemy import text

from src.utils.db import get_db_session
from src.config import settings

# Parquet export directory (mounted from host)
PARQUET_DIR = Path("/opt/airflow/data/parquet")
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

# Tables to export (all gold layer - now including 4 new layers!)
GOLD_TABLES = [
    'gold_gamma_exposure_weekly',  # WEEKLY: Accurate GEX from Friday OI data
    'gold_gex_positioning_trends',  # NEW: Time-series GEX trends and positioning shifts
    'gold_max_pain', 
    'gold_skew_analysis',
    'gold_key_levels',
    'gold_volatility_surface',
    'gold_options_summary_daily',
    'gold_volatility_term_structure',  # NEW: Term structure analysis
    'gold_open_interest_flow',  # NEW: Weekly OI flow (institutional positioning)
    'gold_put_call_metrics'  # NEW: Put/Call ratio sentiment
]

# Also export key silver tables
SILVER_TABLES = [
    'silver_options',
    'silver_underlying_price'
]


def export_table_to_parquet(
    table_name: str,
    schema: str = 'public_public',
    partition_by: str = None
) -> dict:
    """
    Export a database table to Parquet format.
    
    Args:
        table_name: Name of the table to export
        schema: Database schema (default: public_public)
        partition_by: Column to partition by (e.g., 'trade_date')
    
    Returns:
        dict with export statistics
    """
    print(f"📦 Exporting {table_name}...")
    
    with get_db_session() as session:
        # Read entire table
        query = f"SELECT * FROM {schema}.{table_name}"
        df = pd.read_sql(query, session.bind)
        
        if df.empty:
            print(f"  ⚠️  Table is empty, skipping")
            return {'rows': 0, 'size_mb': 0, 'file': None}
        
        # Export to parquet
        output_file = PARQUET_DIR / f"{table_name}.parquet"
        
        # Use efficient compression
        df.to_parquet(
            output_file,
            engine='pyarrow',
            compression='snappy',  # Fast compression, good ratio
            index=False
        )
        
        # Get file size
        size_mb = output_file.stat().st_size / (1024 * 1024)
        
        print(f"  ✅ Exported {len(df):,} rows, {size_mb:.2f} MB")
        print(f"     File: {output_file}")
        
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'size_mb': round(size_mb, 2),
            'file': str(output_file)
        }


def export_partitioned_table(
    table_name: str,
    partition_column: str,
    schema: str = 'public_public'
) -> dict:
    """
    Export table partitioned by a column (e.g., by date).
    Creates separate files for better Power BI performance.
    
    Args:
        table_name: Name of the table
        partition_column: Column to partition by
        schema: Database schema
    
    Returns:
        dict with export statistics
    """
    print(f"📦 Exporting {table_name} (partitioned by {partition_column})...")
    
    with get_db_session() as session:
        # Get distinct partition values
        query = f"""
            SELECT DISTINCT {partition_column} 
            FROM {schema}.{table_name} 
            WHERE {partition_column} IS NOT NULL
            ORDER BY {partition_column} DESC
            LIMIT 10
        """
        partitions = pd.read_sql(query, session.bind)
        
        if partitions.empty:
            print(f"  ⚠️  No data to partition, exporting as single file")
            return export_table_to_parquet(table_name, schema)
        
        total_rows = 0
        total_size = 0
        files = []
        
        # Create partition directory
        partition_dir = PARQUET_DIR / table_name
        partition_dir.mkdir(exist_ok=True)
        
        # Export each partition
        for partition_value in partitions[partition_column]:
            partition_str = str(partition_value).replace('-', '').replace(' ', '_')
            
            query = f"""
                SELECT * FROM {schema}.{table_name}
                WHERE {partition_column} = '{partition_value}'
            """
            df = pd.read_sql(query, session.bind)
            
            if not df.empty:
                output_file = partition_dir / f"{table_name}_{partition_str}.parquet"
                df.to_parquet(
                    output_file,
                    engine='pyarrow',
                    compression='snappy',
                    index=False
                )
                
                size_mb = output_file.stat().st_size / (1024 * 1024)
                total_rows += len(df)
                total_size += size_mb
                files.append(str(output_file))
        
        print(f"  ✅ Exported {total_rows:,} rows across {len(files)} files, {total_size:.2f} MB total")
        
        return {
            'rows': total_rows,
            'files': len(files),
            'size_mb': round(total_size, 2),
            'partitions': files
        }


def create_power_bi_schema_file():
    """
    Create a JSON file with table schemas for Power BI reference.
    """
    schema_file = PARQUET_DIR / "schema_reference.json"
    
    schemas = {}
    
    with get_db_session() as session:
        for table in GOLD_TABLES + SILVER_TABLES:
            query = f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'public_public' 
                AND table_name = '{table}'
                ORDER BY ordinal_position
            """
            result = session.execute(text(query)).fetchall()
            schemas[table] = [{'column': row[0], 'type': row[1]} for row in result]
    
    import json
    with open(schema_file, 'w') as f:
        json.dump(schemas, f, indent=2, default=str)
    
    print(f"  📄 Schema reference: {schema_file}")


def export_all_to_parquet():
    """
    Export all gold and silver tables to Parquet format.
    """
    print()
    print("=" * 80)
    print("🚀 EXPORTING DATA TO PARQUET FOR POWER BI")
    print("=" * 80)
    print()
    
    stats = {}
    
    # Export gold tables
    print("🥇 GOLD LAYER:")
    print("-" * 80)
    for table in GOLD_TABLES:
        try:
            stats[table] = export_table_to_parquet(table)
        except Exception as e:
            print(f"  ❌ Error exporting {table}: {e}")
            stats[table] = {'error': str(e)}
    print()
    
    # Export silver tables
    print("🥈 SILVER LAYER:")
    print("-" * 80)
    for table in SILVER_TABLES:
        try:
            stats[table] = export_table_to_parquet(table)
        except Exception as e:
            print(f"  ❌ Error exporting {table}: {e}")
            stats[table] = {'error': str(e)}
    print()
    
    # Create schema reference
    print("📋 METADATA:")
    print("-" * 80)
    try:
        create_power_bi_schema_file()
    except Exception as e:
        print(f"  ⚠️  Could not create schema file: {e}")
    print()
    
    # Summary
    print("=" * 80)
    print("✅ EXPORT COMPLETE")
    print("=" * 80)
    print()
    
    total_rows = sum(s.get('rows', 0) for s in stats.values())
    total_size = sum(s.get('size_mb', 0) for s in stats.values())
    success_count = sum(1 for s in stats.values() if 'error' not in s and s.get('rows', 0) > 0)
    
    print(f"📊 Summary:")
    print(f"  Tables exported: {success_count}/{len(stats)}")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Total size: {total_size:.2f} MB")
    print(f"  Location: {PARQUET_DIR}")
    print()
    
    print("🔌 POWER BI CONNECTION:")
    print(f"  1. In Power BI Desktop, click 'Get Data' > 'Folder'")
    print(f"  2. Browse to: {PARQUET_DIR}")
    print(f"  3. Click 'Combine & Transform'")
    print(f"  4. Select tables to import")
    print()
    
    print("💡 BENEFITS:")
    print("  • 10-100x faster queries than live database")
    print("  • No database load - queries run locally")
    print("  • Portable - can share files with team")
    print("  • Version control - snapshot of data at export time")
    print("  • Offline access - no network required")
    print()
    
    return stats


if __name__ == "__main__":
    export_all_to_parquet()
