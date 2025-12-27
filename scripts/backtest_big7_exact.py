#!/usr/bin/env python3
"""
Standalone Backtest for Big 7 (2025) - Exact Event Detection Pipeline

Fetches data from yfinance, applies EXACT event detection and feature engineering
from build_accum_distrib_events.py, runs calibrated ML model, simulates trades.

This tests how well the Ahold-trained model generalizes to US Big 7 stocks.

Usage:
    python scripts/backtest_big7_exact.py
    python scripts/backtest_big7_exact.py --tickers AAPL MSFT NVDA
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# === Exact Parameters from build_accum_distrib_events.py ===
W = 60           # Range window for consolidation detection
T_PRE = 120      # Prior trend context window  
H = 40           # Label lookahead window
PRIOR_UP_THRESH = 0.03    # 3% move up
PRIOR_DOWN_THRESH = -0.03  # 3% move down
ATR_K = 1.5      # ATR multiplier for breakout bands
RANGE_PERCENTILE = 55
SLOPE_PERCENTILE = 55

# Trading params (from paper_trade_tracker.py)
POSITION_SIZE = 10000
COMMISSION = 10
ATR_K_TARGET = 1.5
ATR_K_STOP = 1.0
MAX_BARS = 20

# Default Big 7 stocks
DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']


def download_hourly_data(tickers, start_date='2024-01-01', end_date='2025-12-24'):
    """Download hourly OHLCV data from yfinance."""
    print("="*80)
    print("📊 BIG 7 BACKTEST - Exact Event Detection Pipeline")
    print("="*80)
    print(f"\n📥 Downloading hourly data from yfinance...")
    print(f"   Tickers: {', '.join(tickers)}")
    print(f"   Period: {start_date} to {end_date}")
    
    all_data = []
    
    for ticker in tickers:
        print(f"   {ticker}...", end=' ', flush=True)
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval='1h', progress=False)
            if not df.empty:
                df = df.reset_index()
                
                # Flatten multi-index columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if col[0] != '' else col[1] for col in df.columns.values]
                
                df['ticker'] = ticker
                df['market'] = 'US'
                
                # Rename columns to lowercase
                df = df.rename(columns={
                    'Datetime': 'timestamp',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                all_data.append(df)
                print(f"✅ {len(df)} bars")
            else:
                print("❌ No data")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    if not all_data:
        raise ValueError("No data downloaded!")
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n✅ Downloaded {len(combined):,} bars for {len(all_data)} tickers")
    
    return combined


def filter_trading_hours(df):
    """Filter to US market hours 9:30-16:00 ET."""
    print("\n🕐 Filtering to US market hours (9:30-16:00 ET)...")
    
    initial_count = len(df)
    
    # Timestamps from yfinance are already timezone-aware
    df['ts_local'] = pd.to_datetime(df['timestamp']).dt.tz_convert('America/New_York')
    
    h = df['ts_local'].dt.hour
    m = df['ts_local'].dt.minute
    
    # Keep 9:30-16:00 ET
    keep = ((h == 9) & (m >= 30)) | ((h > 9) & (h < 16))
    
    df_filtered = df[keep].copy()
    df_filtered = df_filtered.drop(columns=['ts_local'])
    df_filtered = df_filtered.sort_values(['ticker', 'timestamp']).reset_index(drop=True)
    
    print(f"✅ Filtered to {len(df_filtered):,} bars ({len(df_filtered)/initial_count*100:.1f}%)")
    
    return df_filtered


def compute_base_indicators(df):
    """Compute base technical indicators per ticker (exact replica)."""
    print("\n📊 Computing base indicators...")
    
    # Data quality filters
    df['volume'] = df['volume'].fillna(0)
    df = df[
        (df['high'] >= df['low']) &
        (df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0) &
        df['open'].notna() & df['high'].notna() & df['low'].notna() & df['close'].notna()
    ].copy()
    
    df = df.drop_duplicates(subset=['ticker', 'timestamp'], keep='first')
    
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
        g['high_W'] = g['high'].rolling(W, min_periods=1).max()
        g['low_W'] = g['low'].rolling(W, min_periods=1).min()
        g['range_width_pct_W'] = (g['high_W'] - g['low_W']) / g['close']
        
        # Slope on log close (W-period)
        g['log_close'] = np.log(g['close'])
        g['slope_W'] = g['log_close'].rolling(W, min_periods=2).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 2 else 0
        )
        
        # Volatility (W-period)
        g['vol_W'] = g['ret1'].rolling(W, min_periods=2).std()
        
        # Volume stats
        g['vol_ma_W'] = g['volume'].rolling(W, min_periods=1).mean()
        g['rel_volume'] = g['volume'] / (g['vol_ma_W'] + 1e-9)
        
        # Prior move (T_PRE periods ago)
        g['close_T_PRE'] = g['close'].shift(T_PRE)
        g['R_pre'] = (g['close'] / g['close_T_PRE']) - 1
        
        results.append(g)
    
    df_result = pd.concat(results, ignore_index=True)
    
    print(f"✅ Computed indicators for {df_result['ticker'].nunique()} tickers")
    
    return df_result


def compute_consolidation_thresholds(df):
    """Compute per-ticker percentile thresholds."""
    print("\n📏 Computing consolidation thresholds per ticker...")
    
    thresholds = {}
    
    for ticker, group in df.groupby('ticker'):
        valid = group.dropna(subset=['range_width_pct_W', 'slope_W'])
        
        if len(valid) > 0:
            thresholds[ticker] = {
                'thr_range': np.percentile(valid['range_width_pct_W'], RANGE_PERCENTILE),
                'thr_slope': np.percentile(np.abs(valid['slope_W']), SLOPE_PERCENTILE)
            }
        else:
            thresholds[ticker] = {
                'thr_range': 0.05,
                'thr_slope': 0.001
            }
    
    print(f"✅ Computed thresholds for {len(thresholds)} tickers")
    
    return thresholds


def detect_consolidation(df, thresholds):
    """Detect consolidation periods (exact replica)."""
    print("\n🔍 Detecting consolidation periods...")
    
    df = df.copy()
    
    # Prior move conditions
    df['prior_up'] = df['R_pre'] > PRIOR_UP_THRESH
    df['prior_down'] = df['R_pre'] < PRIOR_DOWN_THRESH
    df['prior_dir'] = 0
    df.loc[df['prior_up'], 'prior_dir'] = 1
    df.loc[df['prior_down'], 'prior_dir'] = -1
    
    # Apply per-ticker consolidation thresholds
    df['is_range'] = False
    
    for ticker, thr in thresholds.items():
        mask = df['ticker'] == ticker
        df.loc[mask, 'is_range'] = (
            (df.loc[mask, 'range_width_pct_W'] <= thr['thr_range']) &
            (np.abs(df.loc[mask, 'slope_W']) <= thr['thr_slope'])
        )
    
    print(f"✅ Detected consolidation in {df['is_range'].sum():,} bars")
    
    return df


def create_events(df):
    """Create de-overlapped events from consolidation segments (exact replica)."""
    print("\n🎯 Creating de-overlapped events...")
    
    events = []
    
    for ticker, group in df.groupby('ticker'):
        g = group.sort_values('timestamp').copy()
        
        # Find where consolidation starts/ends
        g['range_start'] = (~g['is_range'].shift(1, fill_value=False)) & g['is_range']
        g['range_end'] = g['is_range'] & (~g['is_range'].shift(-1, fill_value=False))
        
        # Assign segment IDs
        g['segment_id'] = g['range_start'].cumsum()
        g.loc[~g['is_range'], 'segment_id'] = 0
        
        for seg_id, segment in g[g['segment_id'] > 0].groupby('segment_id'):
            if len(segment) < 2:
                continue
            
            seg_start_idx = segment.index[0]
            seg_end_idx = segment.index[-1]
            
            seg_start_row = g.loc[seg_start_idx]
            seg_end_row = g.loc[seg_end_idx]
            
            # Prior dir at segment START
            R_pre_at_start = seg_start_row['R_pre']
            
            if pd.isna(R_pre_at_start):
                continue
            
            if R_pre_at_start > PRIOR_UP_THRESH:
                prior_dir = 1
            elif R_pre_at_start < PRIOR_DOWN_THRESH:
                prior_dir = -1
            else:
                prior_dir = 0
            
            # Calculate window start time
            t_end = seg_end_row['timestamp']
            end_pos = g.index.get_loc(seg_end_idx)
            start_pos = max(0, end_pos - W + 1)
            t_start = g.iloc[start_pos]['timestamp']
            
            events.append({
                'ticker': ticker,
                'market': seg_end_row['market'],
                't_end': t_end,
                't_start': t_start,
                'prior_dir': prior_dir,
                'R_pre': R_pre_at_start,
                'close_end': seg_end_row['close'],
                'atr14_end': seg_end_row['atr14']
            })
    
    events_df = pd.DataFrame(events)
    
    print(f"✅ Created {len(events_df):,} events from {events_df['ticker'].nunique()} tickers")
    
    return events_df


def compute_event_features(events_df, df):
    """Compute event-window features (exact replica)."""
    print("\n🔢 Computing event-window features...")
    
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
        
        up_bars = window[window['close'] > window['open']]
        total_vol = window['volume'].sum()
        up_vol = up_bars['volume'].sum()
        up_vol_share = up_vol / (total_vol + 1) if total_vol > 0 else 0.5
        
        price_change = abs(window['close'].iloc[-1] - window['close'].iloc[0])
        effort_vs_result = total_vol / (price_change + 1e-6)
        
        # === Rejection Features ===
        range_low = window['low'].min()
        range_high = window['high'].max()
        rng = (range_high - range_low) + 1e-9
        
        window['clv'] = ((window['close'] - window['low']) - (window['high'] - window['close'])) / (window['high'] - window['low'] + 1e-9)
        clv_mean = window['clv'].mean()
        
        close_pos_end = (window['close'].iloc[-1] - range_low) / rng
        
        dist_to_top = (range_high - window['close'].iloc[-1]) / rng
        dist_to_bottom = (window['close'].iloc[-1] - range_low) / rng
        
        range_pos = (window['close'] - range_low) / rng
        pct_top_quartile = (range_pos >= 0.75).mean()
        pct_bottom_quartile = (range_pos <= 0.25).mean()
        
        # === Price Action Features ===
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
        
        half = len(window) // 2
        vol_early = window['ret1'].iloc[:half].std() if half > 1 else vol60
        vol_late = window['ret1'].iloc[half:].std() if half > 1 else vol60
        vol_compression = vol_late / (vol_early + 1e-9)
        
        # === Event Structure Features ===
        event_len = len(window)
        
        range_midpoint = (window['high'] + window['low']) / 2
        range_slope = np.polyfit(np.arange(len(window)), range_midpoint, 1)[0] if len(window) >= 2 else 0
        range_slope_pct = range_slope / event['close_end']
        
        range_vol = range_midpoint.std() / event['close_end']
        
        slope_in_range = np.polyfit(np.arange(len(window)), np.log(window['close']), 1)[0] if len(window) >= 2 else 0
        
        net_return_in_range = (window['close'].iloc[-1] / window['close'].iloc[0]) - 1
        
        # === Wick Rejection Features ===
        mid_range = (range_low + range_high) / 2
        near_top_threshold = range_high - (rng * 0.1)
        near_bottom_threshold = range_low + (rng * 0.1)
        
        rejection_from_top = (
            (window['high'] >= near_top_threshold) & 
            (window['close'] < mid_range)
        ).mean()
        
        rejection_from_bottom = (
            (window['low'] <= near_bottom_threshold) & 
            (window['close'] > mid_range)
        ).mean()
        
        features = {
            'ticker': ticker,
            't_start': t_start,
            't_end': t_end,
            'prior_dir': event['prior_dir'],
            'R_pre': event['R_pre'],
            'close_end': event['close_end'],
            'atr14_end': event['atr14_end'],
            
            # Features for ML
            'range_width_pct': range_width,
            'atr_pct_mean': atr_pct_mean,
            'atr_pct_last': atr_pct_last,
            'vol_ratio': vol_ratio,
            'vol_compression': vol_compression,
            'rel_vol_mean': rel_vol_mean,
            'rel_vol_slope': rel_vol_slope,
            'up_vol_share': up_vol_share,
            'effort_vs_result': effort_vs_result,
            'clv_mean': clv_mean,
            'close_pos_end': close_pos_end,
            'dist_to_top': dist_to_top,
            'dist_to_bottom': dist_to_bottom,
            'pct_top_quartile': pct_top_quartile,
            'pct_bottom_quartile': pct_bottom_quartile,
            'max_consecutive_up': max_consecutive_up,
            'max_consecutive_down': max_consecutive_down,
            'event_len': event_len,
            'range_slope_pct': range_slope_pct,
            'range_vol': range_vol,
            'slope_in_range': slope_in_range,
            'net_return_in_range': net_return_in_range,
            'rejection_from_top': rejection_from_top,
            'rejection_from_bottom': rejection_from_bottom,
        }
        
        feature_list.append(features)
    
    features_df = pd.DataFrame(feature_list)
    
    print(f"✅ Computed features for {len(features_df):,} events")
    
    return features_df


def load_model():
    """Load the calibrated ML model."""
    print("\n🤖 Loading ML model...")
    
    model_path = 'ML/production/v20251223_212702/model_calibrated.pkl'
    
    try:
        bundle = joblib.load(model_path)
        if isinstance(bundle, dict):
            model = bundle['model']
            calibrator = bundle.get('calibrator')
        else:
            model = bundle
            calibrator = None
        print(f"✅ Loaded model from {model_path}")
        return model, calibrator
    except:
        print(f"❌ Could not load {model_path}, searching...")
        pkl_files = list(Path('ML').rglob('*calibrated.pkl'))
        if pkl_files:
            model_path = str(pkl_files[0])
            print(f"   Found: {model_path}")
            bundle = joblib.load(model_path)
            if isinstance(bundle, dict):
                model = bundle['model']
                calibrator = bundle.get('calibrator')
            else:
                model = bundle
                calibrator = None
            print(f"✅ Loaded model")
            return model, calibrator
        else:
            raise FileNotFoundError("No calibrated model found!")


def generate_signals(features_df, model, calibrator):
    """Generate trading signals from events."""
    print("\n🔮 Generating trading signals...")
    
    # Features used by model (in order)
    feature_cols = [
        'close_pos_end', 'clv_mean', 'atr_pct_mean', 'event_len',
        'slope_in_range', 'net_return_in_range',
        'rejection_from_top', 'rejection_from_bottom'
    ]
    
    # Prepare feature matrix
    X = features_df[feature_cols].values
    
    # Get predictions
    if calibrator:
        raw_probs = model.predict_proba(X)[:, 1]
        p_up = calibrator.predict(raw_probs.reshape(-1, 1)).ravel()
    else:
        p_up = model.predict_proba(X)[:, 1]
    
    features_df['p_up'] = p_up
    
    # Generate directional signals (0.75/0.25 thresholds)
    features_df['direction'] = None
    features_df.loc[p_up >= 0.75, 'direction'] = 'LONG'
    features_df.loc[p_up <= 0.25, 'direction'] = 'SHORT'
    
    signals = features_df[features_df['direction'].notna()].copy()
    
    print(f"✅ Generated {len(signals):,} signals")
    print(f"   LONG: {len(signals[signals['direction']=='LONG']):,}")
    print(f"   SHORT: {len(signals[signals['direction']=='SHORT']):,}")
    
    return signals


def simulate_trades(signals, df):
    """Simulate trades with bar-based exits."""
    print("\n🎮 Simulating trades with bar-based exits...")
    
    results = []
    
    for _, signal in signals.iterrows():
        ticker = signal['ticker']
        entry_ts = signal['t_end']
        entry_price = float(signal['close_end'])
        direction = signal['direction']
        atr_abs = float(signal['atr14_end'])
        
        # Calculate target/stop
        if direction == 'LONG':
            target = entry_price + ATR_K_TARGET * atr_abs
            stop = entry_price - ATR_K_STOP * atr_abs
        else:
            target = entry_price - ATR_K_TARGET * atr_abs
            stop = entry_price + ATR_K_STOP * atr_abs
        
        # Get forward bars
        forward = df[
            (df['ticker'] == ticker) &
            (df['timestamp'] > entry_ts)
        ].sort_values('timestamp').head(60)
        
        if len(forward) == 0:
            continue
        
        # Simulate bar-by-bar
        exit_type = None
        exit_price = None
        bars_held = 0
        
        for idx, (_, bar) in enumerate(forward.iterrows(), 1):
            bars_held = idx
            high = float(bar['high'])
            low = float(bar['low'])
            close = float(bar['close'])
            
            if direction == 'LONG':
                if high >= target:
                    exit_type = 'TARGET'
                    exit_price = target
                    break
                elif low <= stop:
                    exit_type = 'STOP'
                    exit_price = stop
                    break
            else:
                if low <= target:
                    exit_type = 'TARGET'
                    exit_price = target
                    break
                elif high >= stop:
                    exit_type = 'STOP'
                    exit_price = stop
                    break
            
            if bars_held >= MAX_BARS:
                exit_type = 'TIME'
                exit_price = close
                break
        
        if exit_type is None:
            exit_type = 'OPEN'
            exit_price = float(forward.iloc[-1]['close'])
        
        # Calculate P&L
        shares = POSITION_SIZE / entry_price
        if direction == 'LONG':
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price
        
        pnl_eur = (shares * abs(exit_price - entry_price) * (1 if pnl_pct > 0 else -1)) - COMMISSION
        
        results.append({
            'ticker': ticker,
            'direction': direction,
            'exit_type': exit_type,
            'pnl_pct': pnl_pct,
            'pnl_eur': pnl_eur,
            'bars_held': bars_held,
            'p_up': signal['p_up'],
            'prior_dir': signal['prior_dir']
        })
    
    return pd.DataFrame(results)


def print_summary(results_df):
    """Print comprehensive summary."""
    closed = results_df[results_df['exit_type'] != 'OPEN']
    
    print("\n" + "="*80)
    print("📊 BACKTEST RESULTS")
    print("="*80)
    
    print(f"\n📈 Signal Summary:")
    print(f"   Total Signals: {len(results_df):,}")
    print(f"   Closed: {len(closed):,}")
    print(f"   Open: {len(results_df) - len(closed):,}")
    
    if closed.empty:
        print("\n⚠️  No closed trades to analyze")
        return
    
    wins = closed[closed['pnl_eur'] > 0]
    losses = closed[closed['pnl_eur'] <= 0]
    
    print(f"\n💰 Performance:")
    print(f"   Win Rate: {len(wins)/len(closed):.1%} ({len(wins)}W / {len(losses)}L)")
    if len(wins) > 0:
        print(f"   Avg Win:  {float(wins['pnl_pct'].mean()):.2%} (€{float(wins['pnl_eur'].mean()):.2f})")
    if len(losses) > 0:
        print(f"   Avg Loss: {float(losses['pnl_pct'].mean()):.2%} (€{float(losses['pnl_eur'].mean()):.2f})")
    print(f"   Total P&L: €{float(closed['pnl_eur'].sum()):.2f}")
    
    # Profit factor
    gross_profit = float(wins['pnl_eur'].sum()) if len(wins) > 0 else 0
    gross_loss = abs(float(losses['pnl_eur'].sum())) if len(losses) > 0 else 0
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
        print(f"   Profit Factor: {profit_factor:.2f}")
    
    # ROI
    capital = POSITION_SIZE * 2
    roi = (float(closed['pnl_eur'].sum()) / capital) * 100
    print(f"   ROI: {roi:+.2f}% (on €{capital:,} capital)")
    
    print(f"\n🎯 Exit Types:")
    for exit_type in ['TARGET', 'STOP', 'TIME']:
        count = len(closed[closed['exit_type'] == exit_type])
        if count > 0:
            print(f"   {exit_type:6s}: {count:3d} ({count/len(closed):.1%})")
    
    print(f"\n📊 By Ticker:")
    for ticker in sorted(closed['ticker'].unique()):
        ticker_trades = closed[closed['ticker'] == ticker]
        ticker_wins = ticker_trades[ticker_trades['pnl_eur'] > 0]
        ticker_pnl = float(ticker_trades['pnl_eur'].sum())
        wr = len(ticker_wins)/len(ticker_trades) if len(ticker_trades) > 0 else 0
        print(f"   {ticker:5s}: {len(ticker_trades):2d} trades | WR {wr:.1%} | P&L €{ticker_pnl:+7.2f}")
    
    # Direction analysis
    print(f"\n📈 By Direction:")
    for direction in ['LONG', 'SHORT']:
        dir_trades = closed[closed['direction'] == direction]
        if len(dir_trades) > 0:
            dir_wins = dir_trades[dir_trades['pnl_eur'] > 0]
            dir_pnl = float(dir_trades['pnl_eur'].sum())
            dir_wr = len(dir_wins)/len(dir_trades)
            print(f"   {direction:5s}: {len(dir_trades):2d} trades | WR {dir_wr:.1%} | P&L €{dir_pnl:+7.2f}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Backtest Big 7 with exact event detection')
    parser.add_argument('--tickers', nargs='+', default=DEFAULT_TICKERS, help='Tickers to backtest')
    parser.add_argument('--start-date', default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2025-12-24', help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # 1. Download data
    df = download_hourly_data(args.tickers, args.start_date, args.end_date)
    
    # 2. Filter trading hours
    df = filter_trading_hours(df)
    
    # 3. Compute indicators
    df = compute_base_indicators(df)
    
    # 4. Compute consolidation thresholds
    thresholds = compute_consolidation_thresholds(df)
    
    # 5. Detect consolidation
    df = detect_consolidation(df, thresholds)
    
    # 6. Create events
    events_df = create_events(df)
    
    if len(events_df) == 0:
        print("\n⚠️  No events detected!")
        return
    
    # 7. Compute features
    features_df = compute_event_features(events_df, df)
    
    if len(features_df) == 0:
        print("\n⚠️  No valid features computed!")
        return
    
    # Filter to 2025 only for signals
    features_df['t_end_dt'] = pd.to_datetime(features_df['t_end'])
    features_2025 = features_df[features_df['t_end_dt'] >= '2025-01-01'].copy()
    
    print(f"\n📅 Filtered to 2025: {len(features_2025):,} events")
    
    if len(features_2025) == 0:
        print("\n⚠️  No events in 2025!")
        return
    
    # 8. Load model
    model, calibrator = load_model()
    
    # 9. Generate signals
    signals = generate_signals(features_2025, model, calibrator)
    
    if len(signals) == 0:
        print("\n⚠️  No signals generated!")
        return
    
    # 10. Simulate trades
    results = simulate_trades(signals, df)
    
    # 11. Print summary
    print_summary(results)


if __name__ == '__main__':
    main()
