#!/usr/bin/env python3
"""
Backfill historical OHLCV data from Yahoo Finance.

Downloads historical stock prices for configured tickers and loads into bronze_ohlcv.

Usage:
    python scripts/backfill_ohlcv.py --tickers AD.AS MT.AS --years 5
    python scripts/backfill_ohlcv.py --start-date 2020-01-01 --end-date 2025-12-14
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
import yfinance as yf
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

# User agents to rotate for avoiding rate limits
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
]


class OHLCVBackfill:
    """Backfill historical OHLCV data from Yahoo Finance."""
    
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
        """Create bronze_ohlcv table if it doesn't exist."""
        from src.models.base import Base
        Base.metadata.create_all(self.engine, tables=[BronzeOHLCV.__table__])
        logger.info("✅ Ensured bronze_ohlcv table exists")
    
    def get_existing_dates(self, ticker: str) -> set:
        """Get dates that already exist for a ticker."""
        with self.Session() as session:
            result = session.execute(
                text("SELECT trade_date FROM bronze_ohlcv WHERE ticker = :ticker"),
                {"ticker": ticker}
            )
            return set(row[0] for row in result)
    
    def fetch_yfinance_data(
        self, 
        ticker: str, 
        start_date: date, 
        end_date: date,
        max_retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from Yahoo Finance with retry logic.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
            max_retries: Maximum number of retry attempts
            
        Returns:
            DataFrame with OHLCV data or None if fetch failed
        """
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    # Exponential backoff: 2s, 4s, 8s...
                    wait_time = 2 ** attempt
                    logger.info(f"⏳ Retry attempt {attempt + 1}/{max_retries} for {ticker}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                
                logger.info(f"📥 Fetching {ticker} data from {start_date} to {end_date}...")
                
                # Add delay between requests to avoid rate limiting
                time.sleep(random.uniform(1.0, 2.0))
                
                # Download data from Yahoo Finance with custom session
                stock = yf.Ticker(ticker, session=self.session)
                df = stock.history(
                    start=start_date, 
                    end=end_date, 
                    auto_adjust=False,
                    timeout=30
                )
                
                if df.empty:
                    logger.warning(f"⚠️  No data returned for {ticker}")
                    if attempt < max_retries - 1:
                        continue
                    return None
                
                # Reset index to make Date a column
                df = df.reset_index()
                
                # Rename columns to match our schema
                df = df.rename(columns={
                    'Date': 'trade_date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume',
                    'Adj Close': 'adj_close'
                })
                
                # Select only the columns we need
                df = df[['trade_date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']]
                
                # Convert date to date (not datetime)
                df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
                
                # Add ticker
                df['ticker'] = ticker
                
                # Remove any NaN values
                df = df.dropna()
                
                logger.info(f"✅ Fetched {len(df)} rows for {ticker}")
                return df
                
            except Exception as e:
                logger.error(f"❌ Error fetching {ticker} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"💥 Failed to fetch {ticker} after {max_retries} attempts")
                    return None
        
        return None
    
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
        
        # Filter out existing dates
        df = df[~df['trade_date'].isin(existing_dates)]
        
        if df.empty:
            logger.info(f"ℹ️  All data for {ticker} already exists")
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
                        INSERT INTO bronze_ohlcv 
                        (ticker, trade_date, open, high, low, close, volume, adj_close, 
                         source, scraped_at, created_at, updated_at)
                        VALUES 
                        (:ticker, :trade_date, :open, :high, :low, :close, :volume, :adj_close,
                         :source, :scraped_at, :created_at, :updated_at)
                        ON CONFLICT (ticker, trade_date) 
                        DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume,
                            adj_close = EXCLUDED.adj_close,
                            updated_at = EXCLUDED.updated_at
                    """),
                    records
                )
                session.commit()
                logger.info(f"✅ Inserted {len(records)} rows for {ticker}")
                return len(records)
            except Exception as e:
                session.rollback()
                logger.error(f"❌ Error inserting data for {ticker}: {e}")
                return 0
    
    def backfill_ticker(
        self, 
        ticker: str, 
        start_date: date, 
        end_date: date
    ) -> dict:
        """
        Backfill data for a single ticker.
        
        Returns:
            Statistics dict with rows_fetched and rows_inserted
        """
        df = self.fetch_yfinance_data(ticker, start_date, end_date)
        
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
        start_date: date, 
        end_date: date
    ) -> List[dict]:
        """
        Backfill data for all configured tickers.
        
        Returns:
            List of statistics dicts
        """
        logger.info("=" * 80)
        logger.info("🚀 Starting OHLCV Historical Backfill")
        logger.info("=" * 80)
        logger.info(f"Tickers: {', '.join(self.tickers)}")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info("")
        
        # Create table if needed
        self.create_table_if_not_exists()
        
        results = []
        for ticker in self.tickers:
            logger.info(f"\n📊 Processing {ticker}...")
            result = self.backfill_ticker(ticker, start_date, end_date)
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
                f"Fetched {result['rows_fetched']:,} rows, "
                f"Inserted {result['rows_inserted']:,} rows"
            )
        
        logger.info("")
        logger.info(f"Total: Fetched {total_fetched:,} rows, Inserted {total_inserted:,} rows")
        logger.info("=" * 80)
        logger.info("✅ Backfill complete!")
        logger.info("=" * 80)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Backfill historical OHLCV data from Yahoo Finance"
    )
    
    parser.add_argument(
        '--tickers',
        nargs='+',
        default=['AD.AS', 'MT.AS'],
        help='Ticker symbols to backfill (default: AD.AS MT.AS)'
    )
    
    parser.add_argument(
        '--years',
        type=int,
        help='Number of years to backfill (from today)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD, default: today)'
    )
    
    args = parser.parse_args()
    
    # Calculate date range
    end_date = date.today()
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    elif args.years:
        start_date = end_date - timedelta(days=365 * args.years)
    else:
        # Default: 5 years
        start_date = end_date - timedelta(days=365 * 5)
    
    # Run backfill
    backfill = OHLCVBackfill(tickers=args.tickers)
    backfill.backfill_all(start_date, end_date)


if __name__ == "__main__":
    main()
