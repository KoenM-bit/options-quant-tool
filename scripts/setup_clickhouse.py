#!/usr/bin/env python3
"""
Setup ClickHouse tables to read from MinIO parquet files.

This script creates ClickHouse tables that directly query MinIO parquet files
for fast analytical queries on the silver options data.
"""

import os
import logging
from src.utils.clickhouse_client import get_clickhouse_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_clickhouse_minio_connection():
    """
    Set up ClickHouse tables with MinIO S3 connection.
    Creates table functions and materialized views for fast analytics.
    """
    
    # MinIO credentials from environment
    minio_endpoint = os.getenv('MINIO_ENDPOINT', 'minio:9000')
    minio_access_key = os.getenv('MINIO_ACCESS_KEY', 'admin')
    minio_secret_key = os.getenv('MINIO_SECRET_KEY', 'miniopassword123')
    minio_bucket = os.getenv('MINIO_BUCKET', 'options-data')
    
    # Format endpoint for ClickHouse s3() function
    if not minio_endpoint.startswith('http'):
        minio_endpoint = f'http://{minio_endpoint}'
    
    client = get_clickhouse_client()
    
    try:
        logger.info("🚀 Setting up ClickHouse<->MinIO connection for silver layer...")
        
        # 1. Drop existing tables
        logger.info("Dropping existing tables...")
        client.drop_table('silver_options_enriched')
        client.drop_table('silver_options_by_expiry_mv')
        client.drop_table('silver_options_by_strike_mv')
        
        # 2. Create S3 table function for silver layer
        # This creates a table that reads directly from MinIO parquet files with glob pattern
        logger.info("Creating silver_options_enriched table...")
        
        create_silver_table = f"""
        CREATE TABLE IF NOT EXISTS silver_options_enriched
        (
            -- Primary identifiers
            ticker String,
            trade_date Date,
            option_type LowCardinality(String),
            strike Decimal(10, 2),
            expiry_date Date,
            symbol_code String,
            issue_id String,
            
            -- Pricing data
            bid Nullable(Decimal(10, 4)),
            ask Nullable(Decimal(10, 4)),
            mid_price Nullable(Decimal(10, 4)),
            last_price Nullable(Decimal(10, 4)),
            underlying_price Decimal(10, 4),
            
            -- Trading activity
            volume Nullable(UInt32),
            underlying_volume Nullable(UInt64),
            last_timestamp Nullable(DateTime),
            
            -- Calculated fields
            days_to_expiry Int32,
            moneyness LowCardinality(String),
            
            -- Greeks (populated after enrichment)
            delta Nullable(Decimal(10, 6)),
            gamma Nullable(Decimal(10, 6)),
            theta Nullable(Decimal(10, 6)),
            vega Nullable(Decimal(10, 6)),
            rho Nullable(Decimal(10, 6)),
            implied_volatility Nullable(Decimal(8, 4)),
            
            -- Metadata
            source_url Nullable(String),
            scraped_at DateTime,
            transformed_at DateTime
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(trade_date)
        ORDER BY (ticker, trade_date, option_type, strike, expiry_date)
        PRIMARY KEY (ticker, trade_date)
        SETTINGS index_granularity = 8192
        """
        
        client.execute_command(create_silver_table)
        logger.info("✅ Created silver_options_enriched table")
        
        # 3. Create function to load data from MinIO parquet files
        # This function will be called to refresh data
        logger.info("Creating s3cluster function for MinIO access...")
        
        # S3 function to read all silver parquet files
        s3_path = f"{minio_endpoint}/{minio_bucket}/silver/bd_options_enriched/**/*.parquet"
        
        logger.info(f"MinIO path: {s3_path}")
        
        # 4. Insert initial data from MinIO
        logger.info("Loading initial data from MinIO...")
        
        # Note: Parquet has extra 'date' column from partitioning
        # We need to explicitly select columns in the right order
        insert_query = f"""
        INSERT INTO silver_options_enriched
        SELECT 
            ticker, trade_date, option_type, strike, expiry_date, 
            symbol_code, issue_id,
            bid, ask, mid_price, last_price, underlying_price,
            volume, underlying_volume, last_timestamp,
            days_to_expiry, moneyness,
            delta, gamma, theta, vega, rho, implied_volatility,
            source_url, scraped_at, transformed_at
        FROM s3(
            '{s3_path}',
            '{minio_access_key}',
            '{minio_secret_key}',
            'Parquet'
        )
        """
        
        client.execute_command(insert_query)
        
        # Get count
        count = client.get_table_count('silver_options_enriched')
        logger.info(f"✅ Loaded {count} records into silver_options_enriched")
        
        # 5. Create materialized view for expiry aggregations
        logger.info("Creating materialized view for expiry aggregations...")
        
        create_expiry_mv = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS silver_options_by_expiry_mv
        ENGINE = AggregatingMergeTree()
        PARTITION BY toYYYYMM(trade_date)
        ORDER BY (ticker, trade_date, expiry_date, option_type)
        POPULATE
        AS SELECT
            ticker,
            trade_date,
            expiry_date,
            option_type,
            countState() AS contract_count,
            avgState(strike) AS avg_strike,
            avgState(mid_price) AS avg_mid_price,
            avgState(implied_volatility) AS avg_iv,
            avgState(delta) AS avg_delta,
            sumState(volume) AS total_volume
        FROM silver_options_enriched
        GROUP BY ticker, trade_date, expiry_date, option_type
        """
        
        client.execute_command(create_expiry_mv)
        logger.info("✅ Created materialized view: silver_options_by_expiry_mv")
        
        # 6. Create materialized view for strike aggregations
        logger.info("Creating materialized view for strike aggregations...")
        
        create_strike_mv = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS silver_options_by_strike_mv
        ENGINE = AggregatingMergeTree()
        PARTITION BY toYYYYMM(trade_date)
        ORDER BY (ticker, trade_date, strike, option_type)
        POPULATE
        AS SELECT
            ticker,
            trade_date,
            strike,
            option_type,
            countState() AS contract_count,
            avgState(mid_price) AS avg_mid_price,
            avgState(implied_volatility) AS avg_iv,
            avgState(delta) AS avg_delta,
            sumState(volume) AS total_volume,
            avgState(underlying_price) AS avg_underlying_price
        FROM silver_options_enriched
        GROUP BY ticker, trade_date, strike, option_type
        """
        
        client.execute_command(create_strike_mv)
        logger.info("✅ Created materialized view: silver_options_by_strike_mv")
        
        # 7. Optimize tables
        logger.info("Optimizing tables...")
        client.optimize_table('silver_options_enriched')
        
        # 8. Show table info
        logger.info("\n📊 Table Information:")
        tables = ['silver_options_enriched', 'silver_options_by_expiry_mv', 'silver_options_by_strike_mv']
        for table in tables:
            if client.table_exists(table):
                info = client.get_table_info(table)
                logger.info(f"\n{table}:")
                logger.info(f"  Rows: {info['row_count']:,}")
                logger.info(f"  Size: {info['size']}")
                logger.info(f"  Columns: {len(info['columns'])}")
        
        logger.info("\n✅ ClickHouse<->MinIO setup complete!")
        logger.info(f"\nQuery example:")
        logger.info(f"  SELECT * FROM silver_options_enriched WHERE trade_date = '2025-12-10' LIMIT 10")
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        raise
    finally:
        client.close()


def refresh_clickhouse_from_minio():
    """
    Refresh ClickHouse data from MinIO (incremental load).
    Run this after new data is exported to MinIO.
    """
    
    minio_endpoint = os.getenv('MINIO_ENDPOINT', 'minio:9000')
    minio_access_key = os.getenv('MINIO_ACCESS_KEY', 'admin')
    minio_secret_key = os.getenv('MINIO_SECRET_KEY', 'miniopassword123')
    minio_bucket = os.getenv('MINIO_BUCKET', 'options-data')
    
    # Format endpoint for ClickHouse s3() function
    if not minio_endpoint.startswith('http'):
        minio_endpoint = f'http://{minio_endpoint}'
    
    client = get_clickhouse_client()
    
    try:
        logger.info("🔄 Refreshing ClickHouse from MinIO...")
        
        # Get max trade_date in ClickHouse
        result = client.execute("SELECT max(trade_date) FROM silver_options_enriched")
        max_date = result.first_row[0] if result.first_row else None
        
        logger.info(f"Latest date in ClickHouse: {max_date}")
        
        # Insert new data from MinIO
        s3_path = f"{minio_endpoint}/{minio_bucket}/silver/bd_options_enriched/**/*.parquet"
        
        # Explicitly select columns (parquet has extra 'date' column from partitioning)
        insert_query = f"""
        INSERT INTO silver_options_enriched
        SELECT 
            ticker, trade_date, option_type, strike, expiry_date,
            symbol_code, issue_id,
            bid, ask, mid_price, last_price, underlying_price,
            volume, underlying_volume, last_timestamp,
            days_to_expiry, moneyness,
            delta, gamma, theta, vega, rho, implied_volatility,
            source_url, scraped_at, transformed_at
        FROM s3(
            '{s3_path}',
            '{minio_access_key}',
            '{minio_secret_key}',
            'Parquet'
        )
        WHERE trade_date > '{max_date}'
        """
        
        client.execute_command(insert_query)
        
        # Get new count
        count = client.get_table_count('silver_options_enriched')
        logger.info(f"✅ ClickHouse now has {count:,} total records")
        
        # Optimize
        logger.info("Optimizing table...")
        client.optimize_table('silver_options_enriched')
        
        logger.info("✅ Refresh complete!")
        
    except Exception as e:
        logger.error(f"❌ Refresh failed: {e}")
        raise
    finally:
        client.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'refresh':
        refresh_clickhouse_from_minio()
    else:
        setup_clickhouse_minio_connection()
