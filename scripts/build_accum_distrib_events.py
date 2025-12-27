#!/usr/bin/env python3
"""
Build Event-Based Accumulation/Distribution Dataset

Creates labeled events from hourly OHLCV data by:
1. Identifying prior moves (up/down trends)
2. Detecting consolidation/range periods
3. Labeling future resolution (breakout direction)
4. Computing event-window features for ML training

Usage:
    python scripts/build_accum_distrib_events.py --output data/ml_datasets/accum_distrib_events.parquet
    python scripts/build_accum_distrib_events.py --market NL --output data/ml_datasets/accum_distrib_nl.parquet
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import logging
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AccumDistribEventBuilder:
    """Build accumulation/distribution events from hourly OHLCV data."""
    
    # Hyperparameters (tunable)
    W = 60           # Range window for consolidation detection
    T_PRE = 120      # Prior trend context window
    H = 40           # Label lookahead window
    COOLDOWN = 0     # Hours cooldown after event (0 = no cooldown, segments already de-overlap)
    
    # Prior move thresholds (relaxed for hourly data)
    PRIOR_UP_THRESH = 0.03    # 3% move up (was 5%)
    PRIOR_DOWN_THRESH = -0.03  # 3% move down (was 5%)
    
    # ATR multiplier for breakout bands
    ATR_K = 1.5
    
    # Consolidation percentile thresholds (relaxed)
    RANGE_PERCENTILE = 55
    SLOPE_PERCENTILE = 55
    
    def __init__(self, market: Optional[str] = None):
        """
        Initialize builder.
        
        Args:
            market: Filter to specific market (e.g., 'NL', 'US') or None for all
        """
        self.market = market
        self.engine = create_engine(settings.database_url)
        
    def load_data(self) -> pd.DataFrame:
        """
        Load hourly OHLCV data from database.
        
        Returns:
            DataFrame with columns: ticker, market, timestamp, open, high, low, close, volume, is_training_data
        """
        logger.info("📥 Loading hourly OHLCV data from database...")
        
        query = """
            SELECT 
                ticker,
                market,
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                COALESCE(is_training_data, TRUE) as is_training_data
            FROM bronze_ohlcv_intraday
            WHERE 1=1
        """
        
        if self.market:
            query += f" AND market = '{self.market}'"
            
        query += """
            ORDER BY market, ticker, timestamp
        """
        
        df = pd.read_sql(query, self.engine)
        
        logger.info(f"✅ Loaded {len(df):,} rows for {df['ticker'].nunique()} tickers")
        logger.info(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def filter_trading_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to regular trading hours with proper timezone handling.
        Timestamps in database are stored as naive (no timezone), treated as CET.
        
        Args:
            df: DataFrame with timestamp column (naive, assumed CET)
            
        Returns:
            Filtered DataFrame
        """
        logger.info("🕐 Filtering to regular trading hours (timezone-aware)...")
        
        initial_count = len(df)
        
        # Localize naive timestamps as Europe/Amsterdam (CET/CEST)
        df['ts_local'] = pd.to_datetime(df['timestamp']).dt.tz_localize('Europe/Amsterdam', ambiguous='NaT', nonexistent='NaT')
        
        # Drop any ambiguous/nonexistent times (DST transitions)
        df = df[df['ts_local'].notna()].copy()
        dst_dropped = initial_count - len(df)
        if dst_dropped > 0:
            logger.info(f"   Dropped {dst_dropped} rows with ambiguous/nonexistent DST times")
        
        # Initialize keep flag to False (safe default)
        df['keep'] = False
        
        # For US market: convert to America/New_York and filter 9:30-16:00 ET
        us_mask = df['market'] == 'US'
        if us_mask.any():
            us_slice = df.loc[us_mask].copy()
            us_slice['ts_ny'] = us_slice['ts_local'].dt.tz_convert('America/New_York')
            h = us_slice['ts_ny'].dt.hour
            m = us_slice['ts_ny'].dt.minute
            us_keep = ((h == 9) & (m >= 30)) | ((h > 9) & (h < 16))
            df.loc[us_mask, 'keep'] = us_keep.values
            logger.info(f"   US market: {us_keep.sum():,} / {len(us_slice):,} rows kept ({us_keep.mean()*100:.1f}%)")
        
        # For NL/EU markets: filter 9:00-17:30 Amsterdam time (already localized)
        eu_mask = df['market'].isin(['NL', 'UK', 'FR', 'DE'])
        if eu_mask.any():
            eu_slice = df.loc[eu_mask].copy()
            h = eu_slice['ts_local'].dt.hour
            m = eu_slice['ts_local'].dt.minute
            eu_keep = ((h >= 9) & (h < 17)) | ((h == 17) & (m <= 30))
            df.loc[eu_mask, 'keep'] = eu_keep.values
            logger.info(f"   EU market: {eu_keep.sum():,} / {len(eu_slice):,} rows kept ({eu_keep.mean()*100:.1f}%)")
        
        # Check for markets without hours rules
        unknown_mask = ~(us_mask | eu_mask)
        if unknown_mask.any():
            unknown = df.loc[unknown_mask, 'market'].unique()
            logger.info(f"   Dropping {unknown_mask.sum():,} rows from markets with no hours rule: {unknown.tolist()}")
        
        # Filter and clean up
        df_filtered = df[df['keep'] == True].copy()
        df_filtered = df_filtered.drop(columns=['ts_local', 'keep'])
        
        # Sort by market, ticker, timestamp for consistency
        df_filtered = df_filtered.sort_values(['market', 'ticker', 'timestamp']).reset_index(drop=True)
        
        logger.info(f"✅ Filtered to {len(df_filtered):,} rows ({len(df_filtered)/initial_count*100:.1f}%)")
        
        # Sanity check: show sample ticker counts
        for market in df_filtered['market'].unique():
            market_df = df_filtered[df_filtered['market'] == market]
            sample_tickers = market_df['ticker'].value_counts().head(2)
            logger.info(f"   {market} sample: {dict(sample_tickers)}")
        
        return df_filtered
    
    def compute_base_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute base technical indicators per ticker.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicator columns
        """
        logger.info("📊 Computing base indicators...")
        
        # Data quality filters
        initial_count = len(df)
        
        # Fix null volumes
        df['volume'] = df['volume'].fillna(0)
        
        # Remove invalid OHLC data
        df = df[
            (df['high'] >= df['low']) &
            (df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0) &
            df['open'].notna() & df['high'].notna() & df['low'].notna() & df['close'].notna()
        ].copy()
        
        # Remove duplicate timestamps per ticker
        df = df.drop_duplicates(subset=['ticker', 'timestamp'], keep='first')
        
        quality_filtered = initial_count - len(df)
        if quality_filtered > 0:
            logger.info(f"   Filtered {quality_filtered:,} rows with data quality issues")
        
        results = []
        
        for ticker, group in df.groupby('ticker'):
            g = group.sort_values('timestamp').copy()
            
            # Log returns
            g['ret1'] = np.log(g['close'] / g['close'].shift(1))
            
            # True Range and ATR
            g['tr'] = np.maximum(
                g['high'] - g['low'],
                np.maximum(
                    abs(g['high'] - g['close'].shift(1)),
                    abs(g['low'] - g['close'].shift(1))
                )
            )
            g['atr14'] = g['tr'].rolling(14, min_periods=1).mean()
            
            # Range width (W-period)
            g['high_W'] = g['high'].rolling(self.W, min_periods=1).max()
            g['low_W'] = g['low'].rolling(self.W, min_periods=1).min()
            g['range_width_pct_W'] = (g['high_W'] - g['low_W']) / g['close']
            
            # Slope on log close (W-period)
            g['log_close'] = np.log(g['close'])
            g['slope_W'] = g['log_close'].rolling(self.W, min_periods=2).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 2 else 0
            )
            
            # Volatility (W-period)
            g['vol_W'] = g['ret1'].rolling(self.W, min_periods=2).std()
            
            # Volume stats
            g['vol_ma_W'] = g['volume'].rolling(self.W, min_periods=1).mean()
            g['rel_volume'] = g['volume'] / (g['vol_ma_W'] + 1e-9)
            
            # Prior move (T_PRE periods ago)
            g['close_T_PRE'] = g['close'].shift(self.T_PRE)
            g['R_pre'] = (g['close'] / g['close_T_PRE']) - 1
            
            results.append(g)
        
        df_result = pd.concat(results, ignore_index=True)
        
        logger.info(f"✅ Computed indicators for {df_result['ticker'].nunique()} tickers")
        
        return df_result
    
    def compute_consolidation_thresholds(self, df: pd.DataFrame) -> dict:
        """
        Compute per-ticker percentile thresholds for consolidation detection.
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            Dict mapping ticker to thresholds
        """
        logger.info("📏 Computing consolidation thresholds per ticker...")
        
        thresholds = {}
        
        for ticker, group in df.groupby('ticker'):
            valid = group.dropna(subset=['range_width_pct_W', 'slope_W'])
            
            if len(valid) > 0:
                thresholds[ticker] = {
                    'thr_range': np.percentile(valid['range_width_pct_W'], self.RANGE_PERCENTILE),
                    'thr_slope': np.percentile(np.abs(valid['slope_W']), self.SLOPE_PERCENTILE)
                }
            else:
                thresholds[ticker] = {
                    'thr_range': 0.05,
                    'thr_slope': 0.001
                }
        
        logger.info(f"✅ Computed thresholds for {len(thresholds)} tickers")
        
        return thresholds
    
    def detect_consolidation(self, df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
        """
        Detect consolidation periods using per-ticker thresholds (2 conditions: range + slope).
        
        Args:
            df: DataFrame with indicators
            thresholds: Per-ticker threshold dict
            
        Returns:
            DataFrame with consolidation flags
        """
        logger.info("🔍 Detecting consolidation periods...")
        
        df = df.copy()
        
        # Prior move conditions
        df['prior_up'] = df['R_pre'] > self.PRIOR_UP_THRESH
        df['prior_down'] = df['R_pre'] < self.PRIOR_DOWN_THRESH
        df['prior_dir'] = 0
        df.loc[df['prior_up'], 'prior_dir'] = 1
        df.loc[df['prior_down'], 'prior_dir'] = -1
        
        # Apply per-ticker consolidation thresholds (2 conditions only)
        df['is_range'] = False
        
        for ticker, thr in thresholds.items():
            mask = df['ticker'] == ticker
            df.loc[mask, 'is_range'] = (
                (df.loc[mask, 'range_width_pct_W'] <= thr['thr_range']) &
                (np.abs(df.loc[mask, 'slope_W']) <= thr['thr_slope'])
            )
        
        logger.info(f"✅ Detected consolidation in {df['is_range'].sum():,} bars")
        
        return df
    
    def create_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create de-overlapped events from consolidation segments.
        Attaches prior_dir computed at segment start.
        
        Args:
            df: DataFrame with consolidation flags
            
        Returns:
            DataFrame with one row per event
        """
        logger.info("🎯 Creating de-overlapped events from consolidation segments...")
        
        events = []
        
        for ticker, group in df.groupby('ticker'):
            g = group.sort_values('timestamp').copy()
            
            # Find where consolidation starts/ends
            g['range_start'] = (~g['is_range'].shift(1, fill_value=False)) & g['is_range']
            g['range_end'] = g['is_range'] & (~g['is_range'].shift(-1, fill_value=False))
            
            # Assign segment IDs to consolidation periods
            g['segment_id'] = g['range_start'].cumsum()
            g.loc[~g['is_range'], 'segment_id'] = 0  # Non-range bars get 0
            
            # Group by segments and process each
            for seg_id, segment in g[g['segment_id'] > 0].groupby('segment_id'):
                if len(segment) < 2:
                    continue
                
                # Get segment start and end
                seg_start_idx = segment.index[0]
                seg_end_idx = segment.index[-1]
                
                seg_start_row = g.loc[seg_start_idx]
                seg_end_row = g.loc[seg_end_idx]
                
                # Compute prior_dir at segment START (not end)
                # Use R_pre at the first bar of the segment
                R_pre_at_start = seg_start_row['R_pre']
                
                # Store prior_dir as a feature (not a gate)
                if pd.isna(R_pre_at_start):
                    # Not enough history - skip this event
                    continue
                
                if R_pre_at_start > self.PRIOR_UP_THRESH:
                    prior_dir = 1
                elif R_pre_at_start < self.PRIOR_DOWN_THRESH:
                    prior_dir = -1
                else:
                    prior_dir = 0  # No significant prior move
                
                # No longer skip events without prior move - include all consolidations!
                
                # Calculate window start time (W bars before segment end)
                # Use position-based indexing for robustness
                t_end = seg_end_row['timestamp']
                end_pos = g.index.get_loc(seg_end_idx)
                start_pos = max(0, end_pos - self.W + 1)
                t_start = g.iloc[start_pos]['timestamp']
                
                # Get is_training_data flag from the ticker data
                is_training = seg_end_row.get('is_training_data', True)
                
                events.append({
                    'ticker': ticker,
                    'market': seg_end_row['market'],
                    'is_training_data': is_training,
                    't_end': t_end,
                    't_start': t_start,
                    'prior_dir': prior_dir,
                    'R_pre': R_pre_at_start,
                    'close_end': seg_end_row['close'],
                    'atr14_end': seg_end_row['atr14']
                })
        
        events_df = pd.DataFrame(events)
        
        # Handle case where no events were created
        if len(events_df) == 0:
            logger.warning("⚠️  No events created - insufficient data or no consolidation patterns found")
            return pd.DataFrame()
        
        # Apply cooldown (de-overlap) per ticker if COOLDOWN > 0
        if self.COOLDOWN > 0:
            final_events = []
            for ticker in events_df['ticker'].unique():
                ticker_events = events_df[events_df['ticker'] == ticker].sort_values('t_end').copy()
                
                if len(ticker_events) == 0:
                    continue
                
                keep = [True]
                last_time = ticker_events.iloc[0]['t_end']
                
                for i in range(1, len(ticker_events)):
                    curr_time = ticker_events.iloc[i]['t_end']
                    hours_diff = (curr_time - last_time).total_seconds() / 3600
                    
                    if hours_diff >= self.COOLDOWN:
                        keep.append(True)
                        last_time = curr_time
                    else:
                        keep.append(False)
                
                final_events.append(ticker_events[keep])
            
            if len(final_events) > 0:
                events_df = pd.concat(final_events, ignore_index=True)
            else:
                logger.warning("⚠️  No events remained after cooldown filtering")
                return pd.DataFrame()
        
        if len(events_df) > 0:
            logger.info(f"✅ Created {len(events_df):,} events from {events_df['ticker'].nunique()} tickers")
        else:
            logger.warning("⚠️  No events in final dataset")
        
        return events_df
    
    def label_events(self, events_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label events by future resolution (up/down/none).
        
        Args:
            events_df: DataFrame with event metadata
            df: Full OHLCV DataFrame
            
        Returns:
            Events DataFrame with labels
        """
        logger.info("🏷️  Labeling events by future resolution...")
        
        labeled_events = []
        
        for _, event in events_df.iterrows():
            ticker = event['ticker']
            t_end = event['t_end']
            P0 = event['close_end']
            B = self.ATR_K * event['atr14_end']
            U = P0 + B  # Upper band
            D = P0 - B  # Lower band
            
            # Get future data (H hours)
            future_data = df[
                (df['ticker'] == ticker) & 
                (df['timestamp'] > t_end)
            ].sort_values('timestamp').head(self.H)
            
            # Track future data availability for label validity
            future_bars = len(future_data)
            label_valid = future_bars >= self.H
            
            if future_bars == 0:
                continue
            
            # Find first hit
            up_hit = future_data[future_data['high'] >= U]
            down_hit = future_data[future_data['low'] <= D]
            
            if len(up_hit) > 0 and len(down_hit) > 0:
                # Both hit - which came first?
                up_time = up_hit.iloc[0]['timestamp']
                down_time = down_hit.iloc[0]['timestamp']
                
                # Handle same-bar hits (wide candle)
                if up_time == down_time:
                    label = 'NO_RESOLVE'  # Both barriers hit in same bar - ambiguous
                elif up_time < down_time:
                    label = 'UP_RESOLVE'
                else:
                    label = 'DOWN_RESOLVE'
            elif len(up_hit) > 0:
                label = 'UP_RESOLVE'
            elif len(down_hit) > 0:
                label = 'DOWN_RESOLVE'
            else:
                label = 'NO_RESOLVE'
            
            # Map to accumulation/distribution (strict definition)
            if event['prior_dir'] == -1 and label == 'UP_RESOLVE':
                acc_dist_label = 'ACCUMULATION'
            elif event['prior_dir'] == 1 and label == 'DOWN_RESOLVE':
                acc_dist_label = 'DISTRIBUTION'
            else:
                acc_dist_label = 'OTHER'
            
            # Calculate band size for filtering
            band_pct = B / P0
            
            labeled_events.append({
                **event.to_dict(),
                'label_generic': label,
                'label_acc_dist': acc_dist_label,
                'breakout_band': B,
                'band_pct': band_pct,
                'future_bars': future_bars,
                'label_valid': label_valid
            })
        
        result_df = pd.DataFrame(labeled_events)
        
        # Filter absurd barriers
        initial_count = len(result_df)
        result_df = result_df[
            (result_df['band_pct'] >= 0.002) &  # Not too tiny (noise)
            (result_df['band_pct'] <= 0.05)     # Not too large (unlikely)
        ].copy()
        filtered_count = initial_count - len(result_df)
        
        logger.info(f"✅ Labeled {len(result_df):,} events")
        if filtered_count > 0:
            logger.info(f"   Filtered {filtered_count:,} events with absurd band sizes")
        logger.info(f"   Generic labels: {result_df['label_generic'].value_counts().to_dict()}")
        logger.info(f"   Acc/Dist labels: {result_df['label_acc_dist'].value_counts().to_dict()}")
        logger.info(f"   Label validity: {result_df['label_valid'].sum():,} valid / {len(result_df):,} total ({result_df['label_valid'].mean()*100:.1f}%)")
        
        return result_df
    
    def compute_event_features(self, events_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute event-window features for each event.
        
        Args:
            events_df: DataFrame with labeled events
            df: Full OHLCV DataFrame
            
        Returns:
            Events DataFrame with features
        """
        logger.info("🔢 Computing event-window features...")
        
        feature_list = []
        
        for _, event in events_df.iterrows():
            ticker = event['ticker']
            t_start = event['t_start']
            t_end = event['t_end']
            
            # Get window data
            window = df[
                (df['ticker'] == ticker) &
                (df['timestamp'] >= t_start) &
                (df['timestamp'] <= t_end)
            ].copy()
            
            if len(window) < 2:
                continue
            
            # === Compression Features ===
            range_width = (window['high'].max() - window['low'].min()) / event['close_end']
            atr_pct_mean = (window['atr14'] / window['close']).mean()
            atr_pct_last = event['atr14_end'] / event['close_end']
            
            vol15 = window['ret1'].tail(15).std() if len(window) >= 15 else window['ret1'].std()
            vol60 = window['ret1'].std()
            vol_ratio = vol15 / (vol60 + 1e-9)
            
            # === Volume Features ===
            rel_vol_mean = window['rel_volume'].mean()
            rel_vol_slope = np.polyfit(np.arange(len(window)), window['rel_volume'], 1)[0] if len(window) >= 2 else 0
            
            # Up volume share
            up_bars = window[window['close'] > window['open']]
            total_vol = window['volume'].sum()
            up_vol = up_bars['volume'].sum()
            up_vol_share = up_vol / (total_vol + 1) if total_vol > 0 else 0.5
            
            # Effort vs result
            price_change = abs(window['close'].iloc[-1] - window['close'].iloc[0])
            effort_vs_result = total_vol / (price_change + 1e-6)
            
            # === Rejection Features ===
            # Calculate event-range boundaries (not per-bar)
            range_low = window['low'].min()
            range_high = window['high'].max()
            rng = (range_high - range_low) + 1e-9
            
            # CLV (Close Location Value) - still per-bar for intrabar behavior
            window['clv'] = ((window['close'] - window['low']) - (window['high'] - window['close'])) / (window['high'] - window['low'] + 1e-9)
            clv_mean = window['clv'].mean()
            
            # Close position relative to EVENT RANGE (not last bar)
            close_pos_end = (window['close'].iloc[-1] - range_low) / rng
            
            # Distance to edges (very strong features for breakout direction)
            dist_to_top = (range_high - window['close'].iloc[-1]) / rng
            dist_to_bottom = (window['close'].iloc[-1] - range_low) / rng
            
            # % closes in top/bottom quartile of EVENT RANGE
            range_pos = (window['close'] - range_low) / rng
            pct_top_quartile = (range_pos >= 0.75).mean()
            pct_bottom_quartile = (range_pos <= 0.25).mean()
            
            # === Price Action Features ===
            # Consecutive closes direction
            window['close_dir'] = np.sign(window['close'].diff())
            max_consecutive_up = 0
            max_consecutive_down = 0
            current_streak = 0
            
            for val in window['close_dir'].dropna():
                if val > 0:
                    current_streak = current_streak + 1 if current_streak > 0 else 1
                    max_consecutive_up = max(max_consecutive_up, current_streak)
                elif val < 0:
                    current_streak = current_streak - 1 if current_streak < 0 else -1
                    max_consecutive_down = max(max_consecutive_down, abs(current_streak))
                else:
                    current_streak = 0
            
            # Volatility compression ratio (early vs late)
            half = len(window) // 2
            vol_early = window['ret1'].iloc[:half].std() if half > 1 else vol60
            vol_late = window['ret1'].iloc[half:].std() if half > 1 else vol60
            vol_compression = vol_late / (vol_early + 1e-9)
            
            # === NEW: Event Structure Features ===
            # Consolidation length (number of bars)
            event_len = len(window)
            
            # Range slope during window (trend of the range itself)
            range_midpoint = (window['high'] + window['low']) / 2
            range_slope = np.polyfit(np.arange(len(window)), range_midpoint, 1)[0] if len(window) >= 2 else 0
            range_slope_pct = range_slope / event['close_end']  # Normalize by price
            
            # Range volatility (how much the range bounced around)
            range_vol = range_midpoint.std() / event['close_end']  # Normalized
            
            # === NEW: Trend Inside Range ===
            # Slope of closes within consolidation (different from range_slope)
            slope_in_range = np.polyfit(np.arange(len(window)), np.log(window['close']), 1)[0] if len(window) >= 2 else 0
            
            # Net return inside consolidation
            net_return_in_range = (window['close'].iloc[-1] / window['close'].iloc[0]) - 1
            
            # === NEW: Wick Rejection Features ===
            # % bars with high near top but close below mid (rejection from top)
            mid_range = (range_low + range_high) / 2
            near_top_threshold = range_high - (rng * 0.1)  # Within 10% of top
            near_bottom_threshold = range_low + (rng * 0.1)  # Within 10% of bottom
            
            rejection_from_top = (
                (window['high'] >= near_top_threshold) & 
                (window['close'] < mid_range)
            ).mean()
            
            rejection_from_bottom = (
                (window['low'] <= near_bottom_threshold) & 
                (window['close'] > mid_range)
            ).mean()
            
            # === Context Features ===
            features = {
                'ticker': ticker,
                'is_training_data': event.get('is_training_data', True),
                't_start': t_start,
                't_end': t_end,
                
                # Compression
                'range_width_pct': range_width,
                'atr_pct_mean': atr_pct_mean,
                'atr_pct_last': atr_pct_last,
                'vol_ratio': vol_ratio,
                'vol_compression': vol_compression,
                
                # Volume
                'rel_vol_mean': rel_vol_mean,
                'rel_vol_slope': rel_vol_slope,
                'up_vol_share': up_vol_share,
                'effort_vs_result': effort_vs_result,
                
                # Rejection
                'clv_mean': clv_mean,
                'close_pos_end': close_pos_end,
                'dist_to_top': dist_to_top,
                'dist_to_bottom': dist_to_bottom,
                'pct_top_quartile': pct_top_quartile,
                'pct_bottom_quartile': pct_bottom_quartile,
                
                # Price action
                'max_consecutive_up': max_consecutive_up,
                'max_consecutive_down': max_consecutive_down,
                
                # Event structure
                'event_len': event_len,
                'range_slope_pct': range_slope_pct,
                'range_vol': range_vol,
                
                # Trend inside range
                'slope_in_range': slope_in_range,
                'net_return_in_range': net_return_in_range,
                
                # Wick rejection
                'rejection_from_top': rejection_from_top,
                'rejection_from_bottom': rejection_from_bottom,
                
                # Context
                'prior_dir': event['prior_dir'],
                'R_pre': event['R_pre'],
                'market': event['market'],
                
                # Label metadata (for filtering/audit)
                'breakout_band': event['breakout_band'],
                'band_pct': event['band_pct'],
                'future_bars': event['future_bars'],
                'label_valid': event['label_valid'],
                
                # Labels
                'label_generic': event['label_generic'],
                'label_acc_dist': event['label_acc_dist']
            }
            
            feature_list.append(features)
        
        features_df = pd.DataFrame(feature_list)
        
        logger.info(f"✅ Computed features for {len(features_df):,} events")
        logger.info(f"   Features per event: {len(features_df.columns) - 6}")  # Exclude meta columns
        
        return features_df
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize continuous features per ticker (z-score).
        Uses efficient groupby transform instead of nested loops.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with normalized features
        """
        logger.info("📐 Normalizing features per ticker...")
        
        continuous_cols = [
            'range_width_pct', 'atr_pct_mean', 'atr_pct_last', 'vol_ratio', 'vol_compression',
            'rel_vol_mean', 'rel_vol_slope', 'up_vol_share', 'effort_vs_result',
            'clv_mean', 'close_pos_end', 'dist_to_top', 'dist_to_bottom',
            'pct_top_quartile', 'pct_bottom_quartile', 'R_pre'
        ]
        
        for col in continuous_cols:
            if col in df.columns:
                # Efficient groupby transform instead of nested loop
                df[f'{col}_norm'] = df.groupby('ticker')[col].transform(
                    lambda s: (s - s.mean()) / (s.std() + 1e-9)
                )
        
        logger.info(f"✅ Normalized {len([c for c in df.columns if c.endswith('_norm')])} features")
        
        return df
    
    def build_dataset(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Build complete event dataset.
        
        Args:
            output_path: Optional path to save dataset
            
        Returns:
            Complete dataset DataFrame
        """
        logger.info("=" * 80)
        logger.info("🚀 Building Accumulation/Distribution Event Dataset")
        logger.info("=" * 80)
        logger.info(f"Market filter: {self.market or 'ALL'}")
        logger.info(f"Parameters: W={self.W}, T_PRE={self.T_PRE}, H={self.H}, COOLDOWN={self.COOLDOWN}")
        logger.info("")
        
        # Step 1: Load data
        df = self.load_data()
        
        # Step 2: Filter trading hours
        df = self.filter_trading_hours(df)
        
        # Step 3: Compute indicators
        df = self.compute_base_indicators(df)
        
        # Step 4: Compute thresholds
        thresholds = self.compute_consolidation_thresholds(df)
        
        # Step 5: Detect consolidation
        df = self.detect_consolidation(df, thresholds)
        
        # Step 6: Create events
        events_df = self.create_events(df)
        
        if len(events_df) == 0:
            logger.error("❌ No events created!")
            return pd.DataFrame()
        
        # Step 7: Label events
        events_df = self.label_events(events_df, df)
        
        # Step 8: Compute features
        features_df = self.compute_event_features(events_df, df)
        
        # Add time-split helper columns for training pipeline
        features_df['t_end_date'] = pd.to_datetime(features_df['t_end']).dt.date
        features_df['year_month'] = pd.to_datetime(features_df['t_end']).dt.to_period('M')
        
        # Step 9: Skip normalization - do it in training pipeline on train split only
        # features_df = self.normalize_features(features_df)
        logger.info("📐 Skipping normalization - will be done in training pipeline on train split only")
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 DATASET SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total events: {len(features_df):,}")
        logger.info(f"Tickers: {features_df['ticker'].nunique()}")
        logger.info(f"Markets: {features_df['market'].unique().tolist()}")
        logger.info(f"Date range: {features_df['t_start'].min()} to {features_df['t_end'].max()}")
        logger.info("")
        logger.info("Generic Labels:")
        for label, count in features_df['label_generic'].value_counts().items():
            logger.info(f"  {label:15s}: {count:5,} ({count/len(features_df)*100:5.1f}%)")
        logger.info("")
        logger.info("Acc/Dist Labels:")
        for label, count in features_df['label_acc_dist'].value_counts().items():
            logger.info(f"  {label:15s}: {count:5,} ({count/len(features_df)*100:5.1f}%)")
        logger.info("")
        logger.info("⚠️  Training Target Recommendations:")
        logger.info("   Option A: Breakout direction (binary) - Use label_generic in {UP_RESOLVE, DOWN_RESOLVE}")
        logger.info("             Filter: label_valid==True & label_generic!='NO_RESOLVE'")
        logger.info("   Option B: Accumulation/Distribution - Train only on ACCUMULATION vs DISTRIBUTION rows")
        logger.info("             Filter: label_valid==True & label_acc_dist in {'ACCUMULATION','DISTRIBUTION'}")
        logger.info("")
        logger.info("Time Split Columns Added: t_end_date, year_month")
        logger.info("  Suggested split: train <= 2024-12-31, val: 2025-01-01 to 2025-08-31, test: >= 2025-09-01")
        logger.info("=" * 80)
        
        # Save if requested
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.endswith('.parquet'):
                features_df.to_parquet(output_path, index=False)
            elif output_path.endswith('.csv'):
                features_df.to_csv(output_path, index=False)
            else:
                features_df.to_parquet(output_path + '.parquet', index=False)
            
            logger.info(f"💾 Saved dataset to {output_path}")
        
        logger.info("✅ Dataset build complete!")
        
        return features_df


def main():
    parser = argparse.ArgumentParser(
        description="Build event-based accumulation/distribution dataset from hourly OHLCV data"
    )
    
    parser.add_argument(
        '--market',
        type=str,
        help='Filter to specific market (NL, US, etc.)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/ml_datasets/accum_distrib_events.parquet',
        help='Output file path (parquet or csv)'
    )
    
    args = parser.parse_args()
    
    # Build dataset
    builder = AccumDistribEventBuilder(market=args.market)
    dataset = builder.build_dataset(output_path=args.output)
    
    if len(dataset) > 0:
        logger.info(f"\n✅ Success! Created {len(dataset):,} events")
    else:
        logger.warning("\n⚠️  No events created (insufficient data or no patterns found)")
        logger.info("   This is normal for new datasets with limited history")
        # Exit with 0 (success) since this is an expected condition
        sys.exit(0)


if __name__ == "__main__":
    main()
