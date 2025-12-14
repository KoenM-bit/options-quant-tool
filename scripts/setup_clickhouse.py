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
    Set up ClickHouse star schema tables with MinIO S3 connection.
    Creates dimension and fact tables for fast analytics.
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
        logger.info("🚀 Setting up ClickHouse star schema from MinIO...")
        
        # 1. Drop existing tables (cleanup old schema)
        logger.info("Dropping existing tables...")
        client.drop_table('silver_options_enriched')
        client.drop_table('silver_options_by_expiry_mv')
        client.drop_table('silver_options_by_strike_mv')
        client.drop_table('fact_option_timeseries')
        client.drop_table('dim_option_contract')
        client.drop_table('dim_underlying')
        
        # ===================================================================
        # 2. Create dim_underlying table
        # ===================================================================
        logger.info("Creating dim_underlying table...")
        
        create_dim_underlying = """
        CREATE TABLE IF NOT EXISTS dim_underlying
        (
            underlying_id String,
            ticker String,
            name Nullable(String),
            asset_class LowCardinality(String),
            sector Nullable(String),
            exchange Nullable(String),
            currency LowCardinality(String),
            isin Nullable(String),
            created_at DateTime,
            updated_at DateTime
        )
        ENGINE = ReplacingMergeTree(updated_at)
        ORDER BY (underlying_id)
        PRIMARY KEY (underlying_id)
        """
        
        client.execute_command(create_dim_underlying)
        logger.info("✅ Created dim_underlying table")
        
        # Load dim_underlying from MinIO
        s3_path_underlying = f"{minio_endpoint}/{minio_bucket}/silver/dim_underlying/**/*.parquet"
        
        insert_underlying = f"""
        INSERT INTO dim_underlying
        SELECT *
        FROM s3(
            '{s3_path_underlying}',
            '{minio_access_key}',
            '{minio_secret_key}',
            'Parquet'
        )
        """
        
        client.execute_command(insert_underlying)
        count_underlying = client.get_table_count('dim_underlying')
        logger.info(f"✅ Loaded {count_underlying} underlying records")
        
        # ===================================================================
        # 3. Create dim_option_contract table
        # ===================================================================
        logger.info("Creating dim_option_contract table...")
        
        create_dim_contract = """
        CREATE TABLE IF NOT EXISTS dim_option_contract
        (
            option_id String,
            underlying_id String,
            ticker String,
            expiration_date Date,
            strike Decimal(10, 2),
            call_put LowCardinality(String),
            contract_size UInt32,
            style LowCardinality(String),
            symbol_code Nullable(String),
            issue_id Nullable(String),
            isin Nullable(String),
            created_at DateTime,
            updated_at DateTime
        )
        ENGINE = ReplacingMergeTree(updated_at)
        ORDER BY (option_id)
        PRIMARY KEY (option_id)
        """
        
        client.execute_command(create_dim_contract)
        logger.info("✅ Created dim_option_contract table")
        
        # Load dim_option_contract from MinIO
        s3_path_contract = f"{minio_endpoint}/{minio_bucket}/silver/dim_option_contract/**/*.parquet"
        
        insert_contract = f"""
        INSERT INTO dim_option_contract
        SELECT *
        FROM s3(
            '{s3_path_contract}',
            '{minio_access_key}',
            '{minio_secret_key}',
            'Parquet'
        )
        """
        
        client.execute_command(insert_contract)
        count_contract = client.get_table_count('dim_option_contract')
        logger.info(f"✅ Loaded {count_contract} option contract records")
        
        # ===================================================================
        # 4. Create fact_option_timeseries table
        # ===================================================================
        logger.info("Creating fact_option_timeseries table...")
        
        create_fact_table = """
        CREATE TABLE IF NOT EXISTS fact_option_timeseries
        (
            ts_id UInt64,
            trade_date Date,
            ts DateTime,
            option_id String,
            underlying_id String,
            
            -- Pricing
            underlying_price Decimal(10, 4),
            bid Nullable(Decimal(10, 4)),
            ask Nullable(Decimal(10, 4)),
            mid_price Nullable(Decimal(10, 4)),
            last_price Nullable(Decimal(10, 4)),
            
            -- Greeks
            iv Nullable(Decimal(10, 6)),
            delta Nullable(Decimal(10, 6)),
            gamma Nullable(Decimal(10, 6)),
            theta Nullable(Decimal(10, 6)),
            vega Nullable(Decimal(10, 6)),
            rho Nullable(Decimal(10, 6)),
            
            -- Trading activity
            volume Nullable(UInt32),
            open_interest Nullable(UInt32),
            
            -- Derived metrics
            intrinsic_value Nullable(Decimal(10, 4)),
            time_value Nullable(Decimal(10, 4)),
            moneyness LowCardinality(String),
            days_to_expiry Int32,
            
            -- Metadata
            source LowCardinality(String),
            created_at DateTime
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(trade_date)
        ORDER BY (trade_date, option_id, ts)
        PRIMARY KEY (trade_date, option_id)
        SETTINGS index_granularity = 8192
        """
        
        client.execute_command(create_fact_table)
        logger.info("✅ Created fact_option_timeseries table")
        
        # Load fact_option_timeseries from MinIO
        s3_path_fact = f"{minio_endpoint}/{minio_bucket}/silver/fact_option_timeseries/**/*.parquet"
        
        insert_fact = f"""
        INSERT INTO fact_option_timeseries
        SELECT *
        FROM s3(
            '{s3_path_fact}',
            '{minio_access_key}',
            '{minio_secret_key}',
            'Parquet'
        )
        """
        
        client.execute_command(insert_fact)
        count_fact = client.get_table_count('fact_option_timeseries')
        logger.info(f"✅ Loaded {count_fact} fact timeseries records")
        
        # ===================================================================
        # 4b. Create fact_market_overview table (daily market totals)
        # ===================================================================
        logger.info("Creating fact_market_overview table...")
        
        create_market_overview_table = """
        CREATE TABLE IF NOT EXISTS fact_market_overview
        (
            overview_id UInt64,
            trade_date Date,
            underlying_id String,
            
            -- Underlying price metrics
            underlying_price Nullable(Decimal(10, 4)),
            underlying_open Nullable(Decimal(10, 4)),
            underlying_high Nullable(Decimal(10, 4)),
            underlying_low Nullable(Decimal(10, 4)),
            underlying_volume Nullable(UInt32),
            underlying_change Nullable(Decimal(10, 4)),
            underlying_change_pct Nullable(Float32),
            
            -- Total volume
            total_volume Nullable(UInt32),
            total_call_volume Nullable(UInt32),
            total_put_volume Nullable(UInt32),
            
            -- Total open interest
            total_oi Nullable(UInt32),
            total_call_oi Nullable(UInt32),
            total_put_oi Nullable(UInt32),
            
            -- Ratios
            call_put_volume_ratio Nullable(Float32),
            call_put_oi_ratio Nullable(Float32),
            
            -- Metadata
            market_time LowCardinality(String),
            source LowCardinality(String),
            created_at DateTime
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(trade_date)
        ORDER BY (trade_date, underlying_id)
        PRIMARY KEY (trade_date, underlying_id)
        SETTINGS index_granularity = 8192
        """
        
        client.execute_command(create_market_overview_table)
        logger.info("✅ Created fact_market_overview table")
        
        # Load fact_market_overview from MinIO
        s3_path_overview = f"{minio_endpoint}/{minio_bucket}/silver/fact_market_overview/**/*.parquet"
        
        insert_overview = f"""
        INSERT INTO fact_market_overview
        SELECT *
        FROM s3(
            '{s3_path_overview}',
            '{minio_access_key}',
            '{minio_secret_key}',
            'Parquet'
        )
        """
        
        try:
            client.execute_command(insert_overview)
            count_overview = client.get_table_count('fact_market_overview')
            logger.info(f"✅ Loaded {count_overview} fact market overview records")
        except Exception as e:
            logger.warning(f"⚠️  No market overview data found in MinIO (this is OK if first run): {e}")
        
        # ===================================================================
        # 4c. Create fact_option_eod table (end-of-day FD data)
        # ===================================================================
        logger.info("Creating fact_option_eod table...")
        
        create_eod_table = """
        CREATE TABLE IF NOT EXISTS fact_option_eod
        (
            eod_id UInt64,
            trade_date Date,
            ts DateTime,
            option_id String,
            underlying_id String,
            
            -- EOD settlement price
            last_price Nullable(Decimal(10, 4)),
            
            -- Market activity (OFFICIAL EOD - the critical data!)
            volume Nullable(UInt32),
            open_interest Nullable(UInt32),
            
            -- Metadata
            source LowCardinality(String),
            created_at DateTime
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(trade_date)
        ORDER BY (trade_date, option_id)
        PRIMARY KEY (trade_date, option_id)
        SETTINGS index_granularity = 8192
        """
        
        client.execute_command(create_eod_table)
        logger.info("✅ Created fact_option_eod table")
        
        # Load fact_option_eod from MinIO
        s3_path_eod = f"{minio_endpoint}/{minio_bucket}/silver/fact_option_eod/**/*.parquet"
        
        insert_eod = f"""
        INSERT INTO fact_option_eod
        SELECT *
        FROM s3(
            '{s3_path_eod}',
            '{minio_access_key}',
            '{minio_secret_key}',
            'Parquet'
        )
        """
        
        try:
            client.execute_command(insert_eod)
            count_eod = client.get_table_count('fact_option_eod')
            logger.info(f"✅ Loaded {count_eod} fact EOD records")
        except Exception as e:
            logger.warning(f"⚠️  No EOD data found in MinIO (this is OK if first run): {e}")
        
        # ===================================================================
        # 4d. Create bronze_ohlcv table (stock OHLCV data)
        # ===================================================================
        logger.info("Creating bronze_ohlcv table...")
        
        create_ohlcv_table = """
        CREATE TABLE IF NOT EXISTS bronze_ohlcv
        (
            id UInt64,
            ticker LowCardinality(String),
            trade_date Date,
            open Decimal(10, 4),
            high Decimal(10, 4),
            low Decimal(10, 4),
            close Decimal(10, 4),
            volume UInt64,
            adj_close Decimal(10, 4),
            source LowCardinality(String),
            created_at DateTime,
            updated_at DateTime,
            scraped_at DateTime
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(trade_date)
        ORDER BY (ticker, trade_date)
        PRIMARY KEY (ticker, trade_date)
        SETTINGS index_granularity = 8192
        """
        
        client.execute_command(create_ohlcv_table)
        logger.info("✅ Created bronze_ohlcv table")
        
        # Load bronze_ohlcv from MinIO
        s3_path_ohlcv = f"{minio_endpoint}/{minio_bucket}/bronze/ohlcv/**/*.parquet"
        
        insert_ohlcv = f"""
        INSERT INTO bronze_ohlcv
        SELECT *
        FROM s3(
            '{s3_path_ohlcv}',
            '{minio_access_key}',
            '{minio_secret_key}',
            'Parquet'
        )
        """
        
        try:
            client.execute_command(insert_ohlcv)
            count_ohlcv = client.get_table_count('bronze_ohlcv')
            logger.info(f"✅ Loaded {count_ohlcv:,} OHLCV records from MinIO")
        except Exception as e:
            logger.warning(f"⚠️  No OHLCV data found in MinIO (this is OK if first run): {e}")
        
        # ===================================================================
        # 4e. Create fact_technical_indicators table
        # ===================================================================
        logger.info("Creating fact_technical_indicators table...")
        
        create_tech_indicators_table = """
        CREATE TABLE IF NOT EXISTS fact_technical_indicators
        (
            indicator_id UInt64,
            ticker LowCardinality(String),
            trade_date Date,
            close Nullable(Decimal(10, 4)),
            volume Nullable(UInt64),
            
            -- Trend Indicators
            sma_20 Nullable(Decimal(10, 4)),
            sma_50 Nullable(Decimal(10, 4)),
            sma_200 Nullable(Decimal(10, 4)),
            ema_12 Nullable(Decimal(10, 4)),
            ema_26 Nullable(Decimal(10, 4)),
            
            -- MACD
            macd Nullable(Decimal(10, 4)),
            macd_signal Nullable(Decimal(10, 4)),
            macd_histogram Nullable(Decimal(10, 4)),
            
            -- Momentum
            rsi_14 Nullable(Decimal(10, 4)),
            stochastic_k Nullable(Decimal(10, 4)),
            stochastic_d Nullable(Decimal(10, 4)),
            roc_20 Nullable(Decimal(10, 4)),
            
            -- Volatility
            atr_14 Nullable(Decimal(10, 4)),
            bollinger_upper Nullable(Decimal(10, 4)),
            bollinger_middle Nullable(Decimal(10, 4)),
            bollinger_lower Nullable(Decimal(10, 4)),
            bollinger_width Nullable(Decimal(10, 4)),
            realized_volatility_20 Nullable(Decimal(10, 4)),
            parkinson_volatility_20 Nullable(Decimal(10, 4)),
            
            -- Support/Resistance
            high_20d Nullable(Decimal(10, 4)),
            low_20d Nullable(Decimal(10, 4)),
            high_52w Nullable(Decimal(10, 4)),
            low_52w Nullable(Decimal(10, 4)),
            pct_from_high_20d Nullable(Decimal(10, 4)),
            pct_from_low_20d Nullable(Decimal(10, 4)),
            pct_from_high_52w Nullable(Decimal(10, 4)),
            pct_from_low_52w Nullable(Decimal(10, 4)),
            
            -- Volume
            volume_sma_20 Nullable(UInt64),
            volume_ratio Nullable(Decimal(10, 4)),
            obv Nullable(Int64),
            obv_sma_20 Nullable(Int64),
            
            -- ADX
            adx_14 Nullable(Decimal(10, 4)),
            plus_di_14 Nullable(Decimal(10, 4)),
            minus_di_14 Nullable(Decimal(10, 4)),
            
            -- Metadata
            created_at DateTime,
            updated_at DateTime,
            calculated_at DateTime
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(trade_date)
        ORDER BY (ticker, trade_date)
        PRIMARY KEY (ticker, trade_date)
        SETTINGS index_granularity = 8192
        """
        
        client.execute_command(create_tech_indicators_table)
        logger.info("✅ Created fact_technical_indicators table")
        
        # Load fact_technical_indicators from MinIO
        s3_path_tech = f"{minio_endpoint}/{minio_bucket}/silver/technical_indicators/**/*.parquet"
        
        insert_tech = f"""
        INSERT INTO fact_technical_indicators
        SELECT *
        FROM s3(
            '{s3_path_tech}',
            '{minio_access_key}',
            '{minio_secret_key}',
            'Parquet'
        )
        """
        
        try:
            client.execute_command(insert_tech)
            count_tech = client.get_table_count('fact_technical_indicators')
            logger.info(f"✅ Loaded {count_tech:,} technical indicator records from MinIO")
        except Exception as e:
            logger.warning(f"⚠️  No technical indicators found in MinIO (this is OK if first run): {e}")
        
        # 4f. Create fact_market_regime table (Gold Layer)
        logger.info("Creating fact_market_regime table...")
        
        create_market_regime_table = f"""
        CREATE TABLE IF NOT EXISTS fact_market_regime
        (
            regime_id Int64,
            ticker String,
            trade_date Date,
            
            -- Trend Regime
            trend_regime String,
            trend_strength Nullable(Float64),
            trend_signals Nullable(String),
            
            -- Volatility Regime
            volatility_regime String,
            volatility_percentile Nullable(Float64),
            volatility_signals Nullable(String),
            
            -- Market Phase
            market_phase String,
            phase_confidence Nullable(Float64),
            phase_signals Nullable(String),
            
            -- Support/Resistance
            primary_support Nullable(Float64),
            primary_resistance Nullable(Float64),
            support_strength Nullable(Float64),
            resistance_strength Nullable(Float64),
            
            -- Regime Change
            regime_change Nullable(String),
            days_in_regime Nullable(Int64),
            
            -- Strategy
            recommended_strategy Nullable(String),
            strategy_rationale Nullable(String),
            
            -- Timestamps
            calculated_at DateTime,
            updated_at DateTime
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(trade_date)
        ORDER BY (ticker, trade_date)
        """
        
        client.execute_command(create_market_regime_table)
        logger.info("✅ Created fact_market_regime table")
        
        # Load fact_market_regime from MinIO
        s3_path_regime = f"{minio_endpoint}/{minio_bucket}/gold/market_regime/**/*.parquet"
        
        insert_regime = f"""
        INSERT INTO fact_market_regime
        SELECT *
        FROM s3(
            '{s3_path_regime}',
            '{minio_access_key}',
            '{minio_secret_key}',
            'Parquet'
        )
        """
        
        try:
            client.execute_command(insert_regime)
            count_regime = client.get_table_count('fact_market_regime')
            logger.info(f"✅ Loaded {count_regime:,} market regime records from MinIO")
        except Exception as e:
            logger.warning(f"⚠️  No market regimes found in MinIO (this is OK if first run): {e}")
        
        # ===================================================================
        # 5. Create materialized views for daily aggregations
        # ===================================================================
        logger.info("Creating materialized view for daily aggregations...")
        
        create_daily_mv = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS fact_daily_summary_mv
        ENGINE = AggregatingMergeTree()
        PARTITION BY toYYYYMM(trade_date)
        ORDER BY (trade_date, underlying_id, expiration_date, call_put)
        POPULATE
        AS SELECT
            f.trade_date,
            f.underlying_id,
            c.expiration_date,
            c.call_put,
            countState() AS contract_count,
            avgState(c.strike) AS avg_strike,
            avgState(f.mid_price) AS avg_mid_price,
            avgState(f.iv) AS avg_iv,
            avgState(f.delta) AS avg_delta,
            sumState(f.volume) AS total_volume,
            avgState(f.underlying_price) AS avg_underlying_price
        FROM fact_option_timeseries f
        JOIN dim_option_contract c ON f.option_id = c.option_id
        GROUP BY f.trade_date, f.underlying_id, c.expiration_date, c.call_put
        """
        
        client.execute_command(create_daily_mv)
        logger.info("✅ Created materialized view: fact_daily_summary_mv")
        
        # ===================================================================
        # 6. Optimize tables
        # ===================================================================
        logger.info("Optimizing tables...")
        client.optimize_table('dim_underlying')
        client.optimize_table('dim_option_contract')
        client.optimize_table('fact_option_timeseries')
        client.optimize_table('fact_option_eod')
        client.optimize_table('fact_market_overview')
        client.optimize_table('bronze_ohlcv')
        client.optimize_table('fact_technical_indicators')
        client.optimize_table('fact_market_regime')
        
        # ===================================================================
        # 7. Show table info
        # ===================================================================
        logger.info("\n📊 Table Information:")
        tables = [
            'dim_underlying', 
            'dim_option_contract', 
            'fact_option_timeseries', 
            'fact_option_eod', 
            'fact_market_overview',
            'bronze_ohlcv',
            'fact_technical_indicators',
            'fact_market_regime',
            'fact_daily_summary_mv'
        ]
        for table in tables:
            if client.table_exists(table):
                info = client.get_table_info(table)
                logger.info(f"\n{table}:")
                logger.info(f"  Rows: {info['row_count']:,}")
                logger.info(f"  Size: {info['size']}")
                logger.info(f"  Columns: {len(info['columns'])}")
        
        logger.info("\n✅ ClickHouse star schema setup complete!")
        logger.info(f"\nQuery example:")
        logger.info(f"  SELECT f.trade_date, u.ticker, c.strike, c.call_put, f.mid_price, f.iv, f.delta")
        logger.info(f"  FROM fact_option_timeseries f")
        logger.info(f"  JOIN dim_option_contract c ON f.option_id = c.option_id")
        logger.info(f"  JOIN dim_underlying u ON f.underlying_id = u.underlying_id")
        logger.info(f"  WHERE f.trade_date = '2025-12-10' LIMIT 10")
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        raise
    finally:
        client.close()


def refresh_clickhouse_from_minio():
    """
    Refresh ClickHouse star schema from MinIO (incremental load).
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
        logger.info("🔄 Refreshing ClickHouse star schema from MinIO...")
        
        # ===================================================================
        # 1. Refresh dim_underlying (full replace - small table)
        # ===================================================================
        logger.info("Refreshing dim_underlying...")
        client.execute_command("TRUNCATE TABLE dim_underlying")
        
        s3_path_underlying = f"{minio_endpoint}/{minio_bucket}/silver/dim_underlying/**/*.parquet"
        
        insert_underlying = f"""
        INSERT INTO dim_underlying
        SELECT *
        FROM s3(
            '{s3_path_underlying}',
            '{minio_access_key}',
            '{minio_secret_key}',
            'Parquet'
        )
        """
        
        client.execute_command(insert_underlying)
        count_underlying = client.get_table_count('dim_underlying')
        logger.info(f"✅ Refreshed {count_underlying} underlying records")
        
        # ===================================================================
        # 2. Refresh dim_option_contract (full replace - relatively small)
        # ===================================================================
        logger.info("Refreshing dim_option_contract...")
        client.execute_command("TRUNCATE TABLE dim_option_contract")
        
        s3_path_contract = f"{minio_endpoint}/{minio_bucket}/silver/dim_option_contract/**/*.parquet"
        
        insert_contract = f"""
        INSERT INTO dim_option_contract
        SELECT *
        FROM s3(
            '{s3_path_contract}',
            '{minio_access_key}',
            '{minio_secret_key}',
            'Parquet'
        )
        """
        
        client.execute_command(insert_contract)
        count_contract = client.get_table_count('dim_option_contract')
        logger.info(f"✅ Refreshed {count_contract} option contract records")
        
        # ===================================================================
        # 3. Refresh fact_option_timeseries (incremental - only new dates)
        # ===================================================================
        logger.info("Refreshing fact_option_timeseries (incremental)...")
        
        # Get max trade_date in ClickHouse
        result = client.execute("SELECT max(trade_date) FROM fact_option_timeseries")
        max_date = result.first_row[0] if result.first_row else None
        
        logger.info(f"Latest date in ClickHouse: {max_date}")
        
        # Insert new data from MinIO
        s3_path_fact = f"{minio_endpoint}/{minio_bucket}/silver/fact_option_timeseries/**/*.parquet"
        
        if max_date:
            # Multi-ticker support: Insert data that doesn't already exist (by option_id + trade_date)
            # This allows new tickers to be added for the same date
            insert_fact = f"""
            INSERT INTO fact_option_timeseries
            SELECT s.*
            FROM s3(
                '{s3_path_fact}',
                '{minio_access_key}',
                '{minio_secret_key}',
                'Parquet'
            ) s
            WHERE s.trade_date >= '{max_date}'
            AND NOT EXISTS (
                SELECT 1 FROM fact_option_timeseries existing
                WHERE existing.option_id = s.option_id 
                AND existing.trade_date = s.trade_date
            )
            """
        else:
            # First load - get all data
            insert_fact = f"""
            INSERT INTO fact_option_timeseries
            SELECT *
            FROM s3(
                '{s3_path_fact}',
                '{minio_access_key}',
                '{minio_secret_key}',
                'Parquet'
            )
            """
        
        client.execute_command(insert_fact)
        count_fact = client.get_table_count('fact_option_timeseries')
        logger.info(f"✅ ClickHouse now has {count_fact:,} total fact records")
        
        # ===================================================================
        # 4. Optimize tables
        # ===================================================================
        logger.info("Optimizing tables...")
        client.optimize_table('dim_underlying')
        client.optimize_table('dim_option_contract')
        client.optimize_table('fact_option_timeseries')
        
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
