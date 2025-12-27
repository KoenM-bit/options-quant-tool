#!/usr/bin/env python3
"""
Backfill historical intraday (hourly) OHLCV data from Yahoo Finance.

Downloads historical hourly stock prices for configured tickers and loads into bronze_ohlcv_intraday.

Note: Yahoo Finance limits intraday data to approximately 730 days (2 years) for 1-hour intervals.

Usage:
    python scripts/backfill_ohlcv_hourly.py --tickers AD.AS MT.AS --days 60
    python scripts/backfill_ohlcv_hourly.py --start-date 2024-01-01 --end-date 2024-12-23
    python scripts/backfill_ohlcv_hourly.py --tickers AD.AS --period 3mo
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import logging
import time
import random
from datetime import datetime, timedelta
from typing import List, Optional
import yfinance as yf
import pandas as pd
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.config import settings
from src.models.bronze_ohlcv_intraday import BronzeOHLCVIntraday

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OHLCVHourlyBackfill:
    """Backfill historical intraday (hourly) OHLCV data from Yahoo Finance."""
    
    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        self.engine = create_engine(settings.database_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Create session with proper headers to avoid rate limiting
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://finance.yahoo.com/',
            'Origin': 'https://finance.yahoo.com'
        })
    
    def create_table_if_not_exists(self):
        """Create bronze_ohlcv_intraday table if it doesn't exist."""
        from src.models.base import Base
        Base.metadata.create_all(self.engine, tables=[BronzeOHLCVIntraday.__table__])
        logger.info("✅ Ensured bronze_ohlcv_intraday table exists")
    
    def get_existing_timestamps(self, ticker: str) -> set:
        """Get timestamps that already exist for a ticker."""
        with self.Session() as session:
            result = session.execute(
                text("SELECT timestamp FROM bronze_ohlcv_intraday WHERE ticker = :ticker"),
                {"ticker": ticker}
            )
            return set(row[0] for row in result)
    
    def fetch_yfinance_hourly_data(
        self, 
        ticker: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: Optional[str] = None,
        max_retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """
        Fetch hourly OHLCV data from Yahoo Finance with retry logic.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start datetime for data (optional if period is provided)
            end_date: End datetime for data (optional if period is provided)
            period: Period string like '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', 'max'
            max_retries: Maximum number of retry attempts
            
        Returns:
            DataFrame with hourly OHLCV data or None if fetch failed
            
        Note:
            Yahoo Finance limits intraday data to ~730 days for 1-hour intervals.
            If you request more than 730 days, it may return limited data or fail.
        """
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    # Exponential backoff: 2s, 4s, 8s...
                    wait_time = 2 ** attempt
                    logger.info(f"⏳ Retry attempt {attempt + 1}/{max_retries} for {ticker}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                
                if period:
                    logger.info(f"📥 Fetching {ticker} hourly data for period: {period}...")
                else:
                    logger.info(f"📥 Fetching {ticker} hourly data from {start_date} to {end_date}...")
                
                # Add delay between requests to avoid rate limiting
                time.sleep(random.uniform(1.0, 2.0))
                
                # Download hourly data from Yahoo Finance (without custom session for hourly data)
                stock = yf.Ticker(ticker)
                
                if period:
                    df = stock.history(
                        period=period,
                        interval='1h',
                        auto_adjust=False,
                        timeout=30
                    )
                else:
                    # Convert datetime to clean date string (YYYY-MM-DD) for Yahoo Finance API
                    start_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else str(start_date)
                    end_str = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime) else str(end_date)
                    
                    df = stock.history(
                        start=start_str,
                        end=end_str,
                        interval='1h',
                        auto_adjust=False,
                        timeout=30
                    )
                
                if df.empty:
                    logger.warning(f"⚠️  No data returned for {ticker}")
                    if attempt < max_retries - 1:
                        continue
                    return None
                
                # Reset index to make Datetime a column
                df = df.reset_index()
                
                # Rename columns to match our schema
                # Note: yfinance uses 'Datetime' for intraday data
                if 'Datetime' in df.columns:
                    df = df.rename(columns={'Datetime': 'timestamp'})
                elif 'Date' in df.columns:
                    df = df.rename(columns={'Date': 'timestamp'})
                
                df = df.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # Select only the columns we need
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                
                # Ensure timestamp is datetime (not just date)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Add ticker
                df['ticker'] = ticker
                
                # Add market based on ticker suffix
                if ticker.endswith('.AS'):
                    df['market'] = 'NL'
                elif ticker.endswith('.L'):
                    df['market'] = 'UK'
                elif ticker.endswith('.PA'):
                    df['market'] = 'FR'
                elif ticker.endswith('.DE'):
                    df['market'] = 'DE'
                else:
                    df['market'] = 'US'  # Default to US for tickers without suffix
                
                # Remove any NaN values
                df = df.dropna()
                
                logger.info(f"✅ Fetched {len(df)} hourly rows for {ticker}")
                logger.info(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                return df
                
            except Exception as e:
                logger.error(f"❌ Error fetching {ticker} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"💥 Failed to fetch {ticker} after {max_retries} attempts")
                    return None
        
        return None
    
    def insert_data(self, ticker: str, df: pd.DataFrame) -> int:
        """
        Insert hourly OHLCV data into database.
        
        Args:
            ticker: Stock ticker
            df: DataFrame with hourly OHLCV data
            
        Returns:
            Number of rows inserted
        """
        if df is None or df.empty:
            return 0
        
        # Get existing timestamps to avoid duplicates
        existing_timestamps = self.get_existing_timestamps(ticker)
        
        # Filter out existing timestamps
        df = df[~df['timestamp'].isin(existing_timestamps)]
        
        if df.empty:
            logger.info(f"ℹ️  All hourly data for {ticker} already exists")
            return 0
        
        # Convert DataFrame to list of dicts for bulk insert
        records = df.to_dict('records')
        
        # Add metadata
        now = datetime.now()
        for record in records:
            record['source'] = 'yahoo_finance'
            record['scraped_at'] = now
            record['created_at'] = now
            record['updated_at'] = now
        
        # Bulk insert
        with self.Session() as session:
            try:
                session.execute(
                    text("""
                        INSERT INTO bronze_ohlcv_intraday 
                        (ticker, timestamp, market, open, high, low, close, volume, 
                         source, scraped_at, created_at, updated_at)
                        VALUES 
                        (:ticker, :timestamp, :market, :open, :high, :low, :close, :volume,
                         :source, :scraped_at, :created_at, :updated_at)
                        ON CONFLICT (ticker, timestamp) 
                        DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume,
                            market = EXCLUDED.market,
                            updated_at = EXCLUDED.updated_at
                    """),
                    records
                )
                session.commit()
                logger.info(f"✅ Inserted {len(records)} hourly rows for {ticker}")
                return len(records)
            except Exception as e:
                session.rollback()
                logger.error(f"❌ Error inserting data for {ticker}: {e}")
                return 0
    
    def backfill_ticker(
        self, 
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: Optional[str] = None
    ) -> dict:
        """
        Backfill hourly data for a single ticker.
        
        Returns:
            Statistics dict with rows_fetched and rows_inserted
        """
        df = self.fetch_yfinance_hourly_data(ticker, start_date, end_date, period)
        
        if df is None or df.empty:
            return {'ticker': ticker, 'rows_fetched': 0, 'rows_inserted': 0}
        
        rows_inserted = self.insert_data(ticker, df)
        
        return {
            'ticker': ticker,
            'rows_fetched': len(df),
            'rows_inserted': rows_inserted
        }
    
    def backfill_all(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: Optional[str] = None
    ) -> List[dict]:
        """
        Backfill hourly data for all configured tickers.
        
        Returns:
            List of statistics dicts
        """
        logger.info("=" * 80)
        logger.info("🚀 Starting OHLCV Hourly Historical Backfill")
        logger.info("=" * 80)
        logger.info(f"Tickers: {', '.join(self.tickers)}")
        if period:
            logger.info(f"Period: {period}")
        else:
            logger.info(f"Date range: {start_date} to {end_date}")
        logger.info("⚠️  Note: Yahoo Finance limits intraday data to ~730 days")
        logger.info("")
        
        # Create table if needed
        self.create_table_if_not_exists()
        
        results = []
        for ticker in self.tickers:
            logger.info(f"\n📊 Processing {ticker}...")
            result = self.backfill_ticker(ticker, start_date, end_date, period)
            results.append(result)
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 BACKFILL SUMMARY")
        logger.info("=" * 80)
        
        total_fetched = sum(r['rows_fetched'] for r in results)
        total_inserted = sum(r['rows_inserted'] for r in results)
        
        for result in results:
            logger.info(
                f"  {result['ticker']}: "
                f"Fetched {result['rows_fetched']:,} hourly rows, "
                f"Inserted {result['rows_inserted']:,} rows"
            )
        
        logger.info("")
        logger.info(f"Total: Fetched {total_fetched:,} rows, Inserted {total_inserted:,} rows")
        logger.info("=" * 80)
        logger.info("✅ Hourly backfill complete!")
        logger.info("=" * 80)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Backfill historical intraday (hourly) OHLCV data from Yahoo Finance"
    )
    
    parser.add_argument(
        '--tickers',
        nargs='+',
        default=['AD.AS', 'MT.AS'],
        help='Ticker symbols to backfill (default: AD.AS MT.AS)'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        help='Number of days to backfill (from today, max ~730 for hourly data)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD, max ~730 days ago for hourly data)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD, default: now)'
    )
    
    parser.add_argument(
        '--period',
        type=str,
        choices=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '730d', 'max'],
        help='Period to fetch (alternative to start-date/end-date). Use 730d for maximum hourly data.'
    )
    
    args = parser.parse_args()
    
    # Validate that we have either period or date range
    if args.period:
        # Use period-based fetching
        backfill = OHLCVHourlyBackfill(tickers=args.tickers)
        backfill.backfill_all(period=args.period)
    else:
        # Calculate date range
        end_date = datetime.now()
        if args.end_date:
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        
        if args.start_date:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        elif args.days:
            start_date = end_date - timedelta(days=args.days)
        else:
            # Default: 60 days (reasonable for hourly data)
            start_date = end_date - timedelta(days=60)
        
        # Warn if date range is too large
        days_requested = (end_date - start_date).days
        if days_requested > 730:
            logger.warning(
                f"⚠️  Requesting {days_requested} days of hourly data. "
                f"Yahoo Finance typically limits to ~730 days. "
                f"You may get limited data or errors."
            )
        
        # Run backfill
        backfill = OHLCVHourlyBackfill(tickers=args.tickers)
        backfill.backfill_all(start_date, end_date)


if __name__ == "__main__":
    main()
