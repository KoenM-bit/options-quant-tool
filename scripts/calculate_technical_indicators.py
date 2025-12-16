#!/usr/bin/env python3
"""
Calculate Technical Indicators

Transforms bronze_ohlcv data into fact_technical_indicators by calculating
various technical analysis indicators including trend, momentum, volatility,
and volume metrics.

Usage:
    python scripts/calculate_technical_indicators.py --tickers AD.AS MT.AS
    python scripts/calculate_technical_indicators.py --lookback-days 30
    python scripts/calculate_technical_indicators.py --all
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.config import settings
from src.models.fact_technical_indicators import FactTechnicalIndicators
from src.models.base import Base

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TechnicalIndicatorCalculator:
    """Calculate technical indicators from OHLCV data."""
    
    def __init__(self):
        self.engine = create_engine(settings.database_url)
        self.Session = sessionmaker(bind=self.engine)
    
    def create_table_if_not_exists(self):
        """Create fact_technical_indicators table if it doesn't exist."""
        Base.metadata.create_all(self.engine, tables=[FactTechnicalIndicators.__table__])
        logger.info("✅ Ensured fact_technical_indicators table exists")
    
    def get_ohlcv_data(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        lookback_days: int = 300
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a ticker with sufficient lookback for indicators.
        
        Args:
            ticker: Stock ticker
            start_date: Start date for calculations (None = all data)
            lookback_days: Extra days to fetch for indicator calculation
        """
        query = """
            SELECT 
                ticker,
                trade_date,
                open,
                high,
                low,
                close,
                volume,
                adj_close
            FROM bronze_ohlcv
            WHERE ticker = :ticker
        """
        
        params = {'ticker': ticker}
        
        if start_date:
            # Fetch extra data for indicator lookback
            lookback_date = start_date - timedelta(days=lookback_days)
            query += " AND trade_date >= :lookback_date"
            params['lookback_date'] = lookback_date
        
        query += " ORDER BY trade_date ASC"
        
        with self.engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn, params=params)
        
        if df.empty:
            logger.warning(f"⚠️  No OHLCV data found for {ticker}")
            return df
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'adj_close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        logger.info(f"📊 Loaded {len(df)} rows for {ticker} from {df['trade_date'].min()} to {df['trade_date'].max()}")
        return df
    
    def calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return series.rolling(window=period, min_periods=period).mean()
    
    def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return series.ewm(span=period, adjust=False, min_periods=period).mean()
    
    def calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple:
        """Calculate Stochastic Oscillator (%K and %D)."""
        lowest_low = low.rolling(window=period, min_periods=period).min()
        highest_high = high.rolling(window=period, min_periods=period).max()
        
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(window=3, min_periods=3).mean()
        
        return stoch_k, stoch_d
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period, min_periods=period).mean()
        
        return atr
    
    def calculate_bollinger_bands(self, series: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple:
        """Calculate Bollinger Bands."""
        middle = series.rolling(window=period, min_periods=period).mean()
        std = series.rolling(window=period, min_periods=period).std()
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        width = upper - lower
        
        return upper, middle, lower, width
    
    def calculate_realized_volatility(self, series: pd.Series, period: int = 20) -> pd.Series:
        """Calculate annualized realized volatility from returns."""
        returns = series.pct_change()
        vol = returns.rolling(window=period, min_periods=period).std() * np.sqrt(252) * 100
        return vol
    
    def calculate_parkinson_volatility(self, high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Parkinson volatility (uses high-low range)."""
        hl_ratio = np.log(high / low)
        parkinson = np.sqrt(hl_ratio.pow(2).rolling(window=period, min_periods=period).mean() / (4 * np.log(2))) * np.sqrt(252) * 100
        return parkinson
    
    def calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD, Signal, and Histogram."""
        ema_fast = self.calculate_ema(series, fast)
        ema_slow = self.calculate_ema(series, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple:
        """Calculate ADX, +DI, and -DI."""
        # True Range
        tr = pd.concat([
            high - low,
            np.abs(high - close.shift()),
            np.abs(low - close.shift())
        ], axis=1).max(axis=1)
        
        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        # Only keep positive values
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # Smoothed values
        tr_smooth = tr.rolling(window=period, min_periods=period).sum()
        plus_dm_smooth = plus_dm.rolling(window=period, min_periods=period).sum()
        minus_dm_smooth = minus_dm.rolling(window=period, min_periods=period).sum()
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period, min_periods=period).mean()
        
        return adx, plus_di, minus_di
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a DataFrame of OHLCV data.
        
        Args:
            df: DataFrame with columns: ticker, trade_date, open, high, low, close, volume
            
        Returns:
            DataFrame with all calculated indicators
        """
        if df.empty:
            return df
        
        result = df[['ticker', 'trade_date', 'close', 'volume']].copy()
        
        # Trend Indicators - SMAs
        result['sma_20'] = self.calculate_sma(df['close'], 20)
        result['sma_50'] = self.calculate_sma(df['close'], 50)
        result['sma_200'] = self.calculate_sma(df['close'], 200)
        
        # Trend Indicators - EMAs
        result['ema_12'] = self.calculate_ema(df['close'], 12)
        result['ema_26'] = self.calculate_ema(df['close'], 26)
        
        # MACD
        result['macd'], result['macd_signal'], result['macd_histogram'] = self.calculate_macd(df['close'])
        
        # Momentum Indicators
        result['rsi_14'] = self.calculate_rsi(df['close'], 14)
        result['stochastic_k'], result['stochastic_d'] = self.calculate_stochastic(df['high'], df['low'], df['close'])
        
        # Rate of Change
        result['roc_20'] = ((df['close'] - df['close'].shift(20)) / df['close'].shift(20)) * 100
        
        # Volatility Indicators
        result['atr_14'] = self.calculate_atr(df['high'], df['low'], df['close'])
        result['bollinger_upper'], result['bollinger_middle'], result['bollinger_lower'], result['bollinger_width'] = \
            self.calculate_bollinger_bands(df['close'])
        
        result['realized_volatility_20'] = self.calculate_realized_volatility(df['close'], 20)
        result['parkinson_volatility_20'] = self.calculate_parkinson_volatility(df['high'], df['low'], 20)
        
        # Support/Resistance
        result['high_20d'] = df['high'].rolling(window=20, min_periods=20).max()
        result['low_20d'] = df['low'].rolling(window=20, min_periods=20).min()
        result['high_52w'] = df['high'].rolling(window=252, min_periods=100).max()  # min 100 days for 52w
        result['low_52w'] = df['low'].rolling(window=252, min_periods=100).min()
        
        # Distance from highs/lows
        result['pct_from_high_20d'] = ((result['high_20d'] - df['close']) / result['high_20d']) * 100
        result['pct_from_low_20d'] = ((df['close'] - result['low_20d']) / result['low_20d']) * 100
        result['pct_from_high_52w'] = ((result['high_52w'] - df['close']) / result['high_52w']) * 100
        result['pct_from_low_52w'] = ((df['close'] - result['low_52w']) / result['low_52w']) * 100
        
        # Volume Indicators
        result['volume_sma_20'] = df['volume'].rolling(window=20, min_periods=20).mean()
        result['volume_ratio'] = df['volume'] / result['volume_sma_20']
        
        # On-Balance Volume (OBV)
        price_change = df['close'].diff()
        obv = (df['volume'] * price_change.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))).cumsum()
        result['obv'] = obv
        result['obv_sma_20'] = obv.rolling(window=20, min_periods=20).mean()
        
        # ADX
        result['adx_14'], result['plus_di_14'], result['minus_di_14'] = \
            self.calculate_adx(df['high'], df['low'], df['close'])
        
        return result
    
    def insert_indicators(self, df: pd.DataFrame, ticker: str) -> int:
        """
        Insert calculated indicators into database.
        
        Args:
            df: DataFrame with calculated indicators
            ticker: Stock ticker
            
        Returns:
            Number of rows inserted/updated
        """
        if df.empty:
            return 0
        
        # Drop rows with insufficient data (NaN in key indicators)
        # Keep only rows where at least SMA20 is available
        df_valid = df.dropna(subset=['sma_20']).copy()
        
        if df_valid.empty:
            logger.warning(f"⚠️  No valid indicator data for {ticker} (insufficient history)")
            return 0
        
        # Prepare bulk upsert
        insert_query = text("""
            INSERT INTO fact_technical_indicators (
                ticker, trade_date, close, volume,
                sma_20, sma_50, sma_200,
                ema_12, ema_26,
                macd, macd_signal, macd_histogram,
                rsi_14, stochastic_k, stochastic_d, roc_20,
                atr_14, bollinger_upper, bollinger_middle, bollinger_lower, bollinger_width,
                realized_volatility_20, parkinson_volatility_20,
                high_20d, low_20d, high_52w, low_52w,
                pct_from_high_20d, pct_from_low_20d, pct_from_high_52w, pct_from_low_52w,
                volume_sma_20, volume_ratio, obv, obv_sma_20,
                adx_14, plus_di_14, minus_di_14,
                calculated_at
            ) VALUES (
                :ticker, :trade_date, :close, :volume,
                :sma_20, :sma_50, :sma_200,
                :ema_12, :ema_26,
                :macd, :macd_signal, :macd_histogram,
                :rsi_14, :stochastic_k, :stochastic_d, :roc_20,
                :atr_14, :bollinger_upper, :bollinger_middle, :bollinger_lower, :bollinger_width,
                :realized_volatility_20, :parkinson_volatility_20,
                :high_20d, :low_20d, :high_52w, :low_52w,
                :pct_from_high_20d, :pct_from_low_20d, :pct_from_high_52w, :pct_from_low_52w,
                :volume_sma_20, :volume_ratio, :obv, :obv_sma_20,
                :adx_14, :plus_di_14, :minus_di_14,
                CURRENT_TIMESTAMP
            )
            ON CONFLICT (ticker, trade_date) DO UPDATE SET
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                sma_20 = EXCLUDED.sma_20,
                sma_50 = EXCLUDED.sma_50,
                sma_200 = EXCLUDED.sma_200,
                ema_12 = EXCLUDED.ema_12,
                ema_26 = EXCLUDED.ema_26,
                macd = EXCLUDED.macd,
                macd_signal = EXCLUDED.macd_signal,
                macd_histogram = EXCLUDED.macd_histogram,
                rsi_14 = EXCLUDED.rsi_14,
                stochastic_k = EXCLUDED.stochastic_k,
                stochastic_d = EXCLUDED.stochastic_d,
                roc_20 = EXCLUDED.roc_20,
                atr_14 = EXCLUDED.atr_14,
                bollinger_upper = EXCLUDED.bollinger_upper,
                bollinger_middle = EXCLUDED.bollinger_middle,
                bollinger_lower = EXCLUDED.bollinger_lower,
                bollinger_width = EXCLUDED.bollinger_width,
                realized_volatility_20 = EXCLUDED.realized_volatility_20,
                parkinson_volatility_20 = EXCLUDED.parkinson_volatility_20,
                high_20d = EXCLUDED.high_20d,
                low_20d = EXCLUDED.low_20d,
                high_52w = EXCLUDED.high_52w,
                low_52w = EXCLUDED.low_52w,
                pct_from_high_20d = EXCLUDED.pct_from_high_20d,
                pct_from_low_20d = EXCLUDED.pct_from_low_20d,
                pct_from_high_52w = EXCLUDED.pct_from_high_52w,
                pct_from_low_52w = EXCLUDED.pct_from_low_52w,
                volume_sma_20 = EXCLUDED.volume_sma_20,
                volume_ratio = EXCLUDED.volume_ratio,
                obv = EXCLUDED.obv,
                obv_sma_20 = EXCLUDED.obv_sma_20,
                adx_14 = EXCLUDED.adx_14,
                plus_di_14 = EXCLUDED.plus_di_14,
                minus_di_14 = EXCLUDED.minus_di_14,
                calculated_at = EXCLUDED.calculated_at,
                updated_at = CURRENT_TIMESTAMP
        """)
        
        # Convert DataFrame to records
        records = df_valid.to_dict('records')
        
        # Execute bulk insert
        with self.Session() as session:
            session.execute(insert_query, records)
            session.commit()
        
        logger.info(f"✅ Inserted/Updated {len(records)} indicator records for {ticker}")
        return len(records)
    
    def calculate_for_ticker(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        lookback_days: int = 300
    ) -> int:
        """
        Calculate and store technical indicators for a ticker.
        
        Args:
            ticker: Stock ticker
            start_date: Start date for calculations (None = all data)
            lookback_days: Extra days to fetch for indicator calculation
            
        Returns:
            Number of rows processed
        """
        logger.info(f"📊 Calculating indicators for {ticker}...")
        
        # Fetch OHLCV data
        df = self.get_ohlcv_data(ticker, start_date, lookback_days)
        
        if df.empty:
            return 0
        
        # Calculate indicators
        indicators = self.calculate_all_indicators(df)
        
        # Filter to requested date range if specified
        if start_date:
            indicators = indicators[indicators['trade_date'] >= start_date]
        
        # Insert into database
        rows_inserted = self.insert_indicators(indicators, ticker)
        
        return rows_inserted
    
    def calculate_all_tickers(
        self,
        tickers: List[str],
        start_date: Optional[date] = None,
        lookback_days: int = 300
    ):
        """Calculate indicators for all tickers."""
        logger.info("=" * 80)
        logger.info("🚀 Starting Technical Indicator Calculation")
        logger.info("=" * 80)
        logger.info(f"Tickers: {', '.join(tickers)}")
        if start_date:
            logger.info(f"Start date: {start_date} (with {lookback_days} days lookback)")
        else:
            logger.info("Processing all available data")
        logger.info("")
        
        total_rows = 0
        
        for ticker in tickers:
            try:
                rows = self.calculate_for_ticker(ticker, start_date, lookback_days)
                total_rows += rows
                logger.info(f"✅ {ticker}: Processed {rows} rows")
            except Exception as e:
                logger.error(f"❌ Error processing {ticker}: {e}", exc_info=True)
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 CALCULATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total rows processed: {total_rows}")
        logger.info("=" * 80)
        logger.info("✅ Technical indicator calculation complete!")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Calculate technical indicators from OHLCV data')
    parser.add_argument(
        '--tickers',
        nargs='+',
        default=['AD.AS', 'MT.AS'],
        help='Stock tickers to process (default: AD.AS MT.AS)'
    )
    parser.add_argument(
        '--lookback-days',
        type=int,
        help='Number of days to recalculate (default: all data)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all available data (ignores --lookback-days)'
    )
    
    args = parser.parse_args()
    
    # Determine start date
    start_date = None
    if args.lookback_days and not args.all:
        start_date = date.today() - timedelta(days=args.lookback_days)
    
    # Create calculator
    calculator = TechnicalIndicatorCalculator()
    
    # Ensure table exists
    calculator.create_table_if_not_exists()
    
    # Calculate indicators
    calculator.calculate_all_tickers(args.tickers, start_date)


if __name__ == "__main__":
    main()
