#!/usr/bin/env python3
"""
Backfill historical OHLCV data using direct Yahoo Finance API calls.

This version bypasses yfinance library and makes direct API calls with proper headers
to avoid rate limiting issues.

Usage:
    python scripts/backfill_ohlcv_direct.py --tickers AD.AS MT.AS --years 5
    python scripts/backfill_ohlcv_direct.py --start-date 2020-01-01 --end-date 2025-12-14
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import logging
import time
import random
from datetime import datetime, date, timedelta
from typing import List, Optional
import pandas as pd
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.config import settings
from src.models.bronze_ohlcv import BronzeOHLCV

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YahooFinanceAPI:
    """Direct Yahoo Finance API client with proper headers."""
    
    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://finance.yahoo.com/',
            'Origin': 'https://finance.yahoo.com',
            'Connection': 'keep-alive'
        })
    
    def get_historical_data(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        max_retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data from Yahoo Finance API.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
            max_retries: Maximum retry attempts
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        # Convert dates to Unix timestamps
        start_ts = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        end_ts = int(datetime.combine(end_date, datetime.min.time()).timestamp())
        
        url = f"{self.BASE_URL}/{ticker}"
        params = {
            'period1': start_ts,
            'period2': end_ts,
            'interval': '1d',
            'events': 'history'
        }
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 2 ** attempt
                    logger.info(f"⏳ Retry attempt {attempt + 1}/{max_retries} for {ticker}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                
                # Add random delay to avoid rate limiting
                time.sleep(random.uniform(1.0, 2.0))
                
                logger.info(f"📥 Fetching {ticker} data from {start_date} to {end_date}...")
                
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 429:
                    logger.warning(f"⚠️  Rate limited on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        continue
                    return None
                
                if response.status_code != 200:
                    logger.error(f"❌ HTTP {response.status_code}: {response.text[:100]}")
                    if attempt < max_retries - 1:
                        continue
                    return None
                
                data = response.json()
                
                # Check for errors in response
                if 'chart' not in data or 'result' not in data['chart']:
                    logger.error(f"❌ Invalid response structure for {ticker}")
                    if attempt < max_retries - 1:
                        continue
                    return None
                
                result = data['chart']['result']
                if not result or len(result) == 0:
                    logger.warning(f"⚠️  No data returned for {ticker}")
                    if attempt < max_retries - 1:
                        continue
                    return None
                
                result = result[0]
                
                # Extract data
                timestamps = result.get('timestamp', [])
                if not timestamps:
                    logger.warning(f"⚠️  No timestamps for {ticker}")
                    return None
                
                indicators = result.get('indicators', {})
                quote = indicators.get('quote', [{}])[0]
                adjclose = indicators.get('adjclose', [{}])[0]
                
                # Create DataFrame
                df = pd.DataFrame({
                    'timestamp': timestamps,
                    'open': quote.get('open', []),
                    'high': quote.get('high', []),
                    'low': quote.get('low', []),
                    'close': quote.get('close', []),
                    'volume': quote.get('volume', []),
                    'adj_close': adjclose.get('adjclose', quote.get('close', []))  # Fallback to close if no adjclose
                })
                
                # Convert timestamp to date
                df['trade_date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
                
                # Add ticker
                df['ticker'] = ticker
                
                # Select and reorder columns
                df = df[['ticker', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']]
                
                # Remove any rows with NaN in critical columns
                df = df.dropna(subset=['open', 'high', 'low', 'close'])
                
                logger.info(f"✅ Fetched {len(df)} rows for {ticker}")
                return df
                
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Request error for {ticker} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return None
            except Exception as e:
                logger.error(f"❌ Error processing {ticker} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return None
        
        return None


class OHLCVBackfill:
    """Backfill historical OHLCV data."""
    
    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        self.engine = create_engine(settings.database_url)
        self.Session = sessionmaker(bind=self.engine)
        self.api = YahooFinanceAPI()
    
    def create_table_if_not_exists(self):
        """Create bronze_ohlcv table if it doesn't exist."""
        from src.models.base import Base
        Base.metadata.create_all(self.engine, tables=[BronzeOHLCV.__table__])
        logger.info("✅ Ensured bronze_ohlcv table exists")
    
    def get_existing_dates(self, ticker: str) -> set:
        """Get dates that already exist for a ticker."""
        with self.Session() as session:
            result = session.execute(
                text("SELECT DISTINCT trade_date FROM bronze_ohlcv WHERE ticker = :ticker"),
                {"ticker": ticker}
            )
            return {row[0] for row in result}
    
    def insert_data(self, ticker: str, df: pd.DataFrame) -> int:
        """
        Insert OHLCV data into database.
        
        Args:
            ticker: Stock ticker
            df: DataFrame with OHLCV data
            
        Returns:
            Number of rows inserted
        """
        if df is None or df.empty:
            return 0
        
        # Get existing dates to avoid duplicates
        existing_dates = self.get_existing_dates(ticker)
        
        # Filter out dates that already exist
        df_new = df[~df['trade_date'].isin(existing_dates)]
        
        if df_new.empty:
            logger.info(f"ℹ️  All {len(df)} rows already exist for {ticker}")
            return 0
        
        # Prepare bulk insert query
        insert_query = text("""
            INSERT INTO bronze_ohlcv (
                ticker, trade_date, open, high, low, close, volume, adj_close, source
            ) VALUES (
                :ticker, :trade_date, :open, :high, :low, :close, :volume, :adj_close, 'yahoo_finance'
            )
            ON CONFLICT (ticker, trade_date) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                adj_close = EXCLUDED.adj_close,
                updated_at = CURRENT_TIMESTAMP
        """)
        
        # Convert DataFrame to list of dicts
        records = df_new.to_dict('records')
        
        # Insert in batches
        with self.Session() as session:
            session.execute(insert_query, records)
            session.commit()
        
        logger.info(f"✅ Inserted {len(df_new)} new rows for {ticker}")
        return len(df_new)
    
    def backfill_ticker(
        self,
        ticker: str,
        start_date: date,
        end_date: date
    ) -> tuple[int, int]:
        """
        Backfill data for a single ticker.
        
        Returns:
            Tuple of (rows_fetched, rows_inserted)
        """
        df = self.api.get_historical_data(ticker, start_date, end_date)
        
        if df is None or df.empty:
            return (0, 0)
        
        rows_inserted = self.insert_data(ticker, df)
        return (len(df), rows_inserted)
    
    def backfill_all(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ):
        """
        Backfill data for all configured tickers.
        
        Args:
            start_date: Start date (defaults to 5 years ago)
            end_date: End date (defaults to today)
        """
        if end_date is None:
            end_date = date.today()
        
        if start_date is None:
            start_date = end_date - timedelta(days=365 * 5)
        
        logger.info(f"📊 Processing {len(self.tickers)} ticker(s)...")
        
        total_fetched = 0
        total_inserted = 0
        
        for ticker in self.tickers:
            logger.info(f"\n📊 Processing {ticker}...")
            
            fetched, inserted = self.backfill_ticker(ticker, start_date, end_date)
            total_fetched += fetched
            total_inserted += inserted
            
            logger.info(f"✅ {ticker}: Fetched {fetched} rows, Inserted {inserted} rows")
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 BACKFILL SUMMARY")
        logger.info("=" * 80)
        for ticker in self.tickers:
            fetched, inserted = self.backfill_ticker(ticker, start_date, end_date)
            logger.info(f"  {ticker}: Fetched {fetched} rows, Inserted {inserted} rows")
        logger.info("")
        logger.info(f"Total: Fetched {total_fetched} rows, Inserted {total_inserted} rows")
        logger.info("=" * 80)
        logger.info("✅ Backfill complete!")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Backfill historical OHLCV data')
    parser.add_argument(
        '--tickers',
        nargs='+',
        default=['AD.AS', 'MT.AS'],
        help='Stock tickers to backfill (default: AD.AS MT.AS)'
    )
    parser.add_argument(
        '--years',
        type=int,
        help='Number of years of history to fetch (default: 5)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD format)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD format, default: today)'
    )
    
    args = parser.parse_args()
    
    # Parse dates
    end_date = date.today()
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    elif args.years:
        start_date = end_date - timedelta(days=365 * args.years)
    else:
        start_date = end_date - timedelta(days=365 * 5)  # Default 5 years
    
    logger.info("=" * 80)
    logger.info("🚀 Starting OHLCV Historical Backfill")
    logger.info("=" * 80)
    logger.info(f"Tickers: {' '.join(args.tickers)}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info("")
    
    # Create backfill instance
    backfill = OHLCVBackfill(args.tickers)
    
    # Ensure table exists
    backfill.create_table_if_not_exists()
    
    # Run backfill
    backfill.backfill_all(start_date, end_date)


if __name__ == "__main__":
    main()
