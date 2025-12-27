#!/usr/bin/env python3
"""
Backtest Paper Trading on Big 7 Stocks (2025)

Tests the ML model on historical 2025 data for the "Big 7" tech stocks:
AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA

Fetches data directly from yfinance (no database), generates events in-memory,
generates signals, simulates trades, and calculates P&L with real position sizing.

Usage:
    python scripts/backtest_2025_big7.py
    python scripts/backtest_2025_big7.py --start-date 2025-01-01 --end-date 2025-12-24
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import argparse
from typing import Dict, List
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Trading parameters (matching paper_trade_tracker.py)
ATR_K_TARGET = 1.5
ATR_K_STOP = 1.0
MAX_BARS = 20
POSITION_SIZE = 10000  # €10,000 per trade
COMMISSION = 10         # €10 per round-trip

# Big 7 tech stocks
BIG_7_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']

# Event detection parameters (from build_accum_distrib_events.py)
W = 60           # Range window for consolidation detection
T_PRE = 120      # Prior trend context window
H = 40           # Label lookahead window
ATR_K = 1.5      # ATR multiplier for breakout bands
PRIOR_UP_THRESH = 0.03
PRIOR_DOWN_THRESH = -0.03


class Big7Backtester:
    """Backtest paper trading on Big 7 stocks for 2025."""
    
    def __init__(self, events_path: str, model_bundle_path: str):
        self.events_path = events_path
        self.model_bundle_path = model_bundle_path
        self.engine = create_engine(settings.database_url)
        
        # Load events
        print(f"📥 Loading events from {events_path}...")
        self.events = pd.read_parquet(events_path)
        self.events['t_end'] = pd.to_datetime(self.events['t_end'])
        self.events['t_start'] = pd.to_datetime(self.events['t_start'])
        
        # Load model bundle
        print(f"🤖 Loading model from {model_bundle_path}...")
        bundle = joblib.load(model_bundle_path)
        if isinstance(bundle, dict):
            self.model = bundle['model']
            self.calibrator = bundle.get('calibrator')
            self.feature_cols = bundle.get('feature_cols', [
                'close_pos_end', 'clv_mean', 'atr_pct_mean', 'event_len',
                'slope_in_range', 'net_return_in_range',
                'rejection_from_top', 'rejection_from_bottom'
            ])
        else:
            self.model = bundle
            self.calibrator = None
            self.feature_cols = [
                'close_pos_end', 'clv_mean', 'atr_pct_mean', 'event_len',
                'slope_in_range', 'net_return_in_range',
                'rejection_from_top', 'rejection_from_bottom'
            ]
        
        print(f"✅ Loaded {len(self.events):,} events")
        print(f"   Tickers: {self.events['ticker'].nunique()}")
        print(f"   Date range: {self.events['t_end'].min()} to {self.events['t_end'].max()}")
    
    def filter_big7_2025(self, start_date: str = "2025-01-01", end_date: str = "2025-12-31") -> pd.DataFrame:
        """Filter events for Big 7 in 2025."""
        print(f"\n🔍 Filtering Big 7 events for {start_date} to {end_date}...")
        
        # Filter by tickers and date range
        filtered = self.events[
            (self.events['ticker'].isin(BIG_7_TICKERS)) &
            (self.events['t_end'] >= pd.to_datetime(start_date)) &
            (self.events['t_end'] <= pd.to_datetime(end_date)) &
            (self.events['label_valid'] == True) &
            (self.events['label_generic'].isin(['UP_RESOLVE', 'DOWN_RESOLVE']))
        ].copy()
        
        print(f"✅ Found {len(filtered):,} valid events")
        print(f"\n📊 Breakdown by ticker:")
        ticker_counts = filtered['ticker'].value_counts().sort_index()
        for ticker, count in ticker_counts.items():
            print(f"   {ticker}: {count:3d} events")
        
        return filtered
    
    def generate_signals(self, events: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from events using ML model."""
        print(f"\n🔮 Generating signals with ML model...")
        
        # Get entry prices from database
        events = self.add_entry_prices(events)
        
        # Make predictions
        X = events[self.feature_cols].values
        
        if self.calibrator:
            raw_scores = self.model.predict_proba(X)[:, 1]
            p_up = self.calibrator.predict(raw_scores)
        else:
            p_up = self.model.predict_proba(X)[:, 1]
        
        events['p_up'] = p_up
        
        # Determine direction based on thresholds (US market)
        events['direction'] = 'NONE'
        events.loc[events['p_up'] >= 0.75, 'direction'] = 'LONG'
        events.loc[events['p_up'] <= 0.25, 'direction'] = 'SHORT'
        
        # Filter to actual signals
        signals = events[events['direction'] != 'NONE'].copy()
        
        # Add confidence levels
        signals['confidence'] = 'MEDIUM'
        signals.loc[signals['p_up'] >= 0.85, 'confidence'] = 'HIGH'
        signals.loc[signals['p_up'] <= 0.15, 'confidence'] = 'HIGH'
        
        print(f"✅ Generated {len(signals):,} signals ({len(signals[signals['direction']=='LONG'])} LONG, {len(signals[signals['direction']=='SHORT'])} SHORT)")
        
        return signals
    
    def add_entry_prices(self, events: pd.DataFrame) -> pd.DataFrame:
        """Fetch entry prices (open of first bar after event end)."""
        print(f"   Fetching entry prices from database...")
        
        entry_prices = []
        
        for _, event in events.iterrows():
            query = text("""
                SELECT open
                FROM bronze_ohlcv_intraday
                WHERE ticker = :ticker
                  AND timestamp > :t_end
                ORDER BY timestamp ASC
                LIMIT 1
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {
                    "ticker": event['ticker'],
                    "t_end": event['t_end']
                }).fetchone()
            
            if result:
                entry_prices.append(float(result[0]))
            else:
                entry_prices.append(None)
        
        events['entry_open'] = entry_prices
        events['entry_ts'] = events['t_end'] + pd.Timedelta(hours=1)
        
        # Remove events without entry price
        before = len(events)
        events = events[events['entry_open'].notna()].copy()
        after = len(events)
        
        if before > after:
            print(f"   ⚠️  Removed {before - after} events without entry price")
        
        return events
    
    def fetch_forward_bars(self, ticker: str, entry_ts: pd.Timestamp, max_bars: int = 60) -> pd.DataFrame:
        """Fetch forward price bars for exit simulation."""
        query = text("""
            SELECT timestamp, open, high, low, close
            FROM bronze_ohlcv_intraday
            WHERE ticker = :ticker
              AND timestamp > :entry_ts
            ORDER BY timestamp ASC
            LIMIT :max_bars
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {
                "ticker": ticker,
                "entry_ts": entry_ts,
                "max_bars": max_bars
            }).fetchall()
        
        if not result:
            return pd.DataFrame()
        
        df = pd.DataFrame(result, columns=['timestamp', 'open', 'high', 'low', 'close'])
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)
        
        return df
    
    def simulate_trade(self, signal: pd.Series) -> Dict:
        """Simulate a single trade with bar-based exit logic."""
        ticker = signal['ticker']
        direction = signal['direction']
        entry_price = signal['entry_open']
        entry_ts = signal['entry_ts']
        atr_abs = entry_price * signal['atr_pct_mean']
        
        # Calculate target and stop
        if direction == 'LONG':
            target_price = entry_price + ATR_K_TARGET * atr_abs
            stop_price = entry_price - ATR_K_STOP * atr_abs
        else:  # SHORT
            target_price = entry_price - ATR_K_TARGET * atr_abs
            stop_price = entry_price + ATR_K_STOP * atr_abs
        
        # Calculate position size
        shares = POSITION_SIZE / entry_price
        
        # Fetch forward bars
        forward_bars = self.fetch_forward_bars(ticker, entry_ts, max_bars=60)
        
        if forward_bars.empty:
            return {
                'status': 'NO_DATA',
                'exit_type': None,
                'pnl_pct': 0,
                'pnl_eur': -COMMISSION,
                'bars_held': 0
            }
        
        # Simulate bar by bar
        for idx, bar in forward_bars.iterrows():
            bars_held = idx + 1
            
            if direction == 'LONG':
                if bar['high'] >= target_price:
                    pnl_pct = (target_price - entry_price) / entry_price
                    pnl_abs = target_price - entry_price
                    pnl_eur = (shares * pnl_abs) - COMMISSION
                    return {
                        'status': 'CLOSED',
                        'exit_type': 'TARGET',
                        'exit_price': target_price,
                        'pnl_pct': pnl_pct,
                        'pnl_eur': pnl_eur,
                        'bars_held': bars_held
                    }
                elif bar['low'] <= stop_price:
                    pnl_pct = (stop_price - entry_price) / entry_price
                    pnl_abs = stop_price - entry_price
                    pnl_eur = (shares * pnl_abs) - COMMISSION
                    return {
                        'status': 'CLOSED',
                        'exit_type': 'STOP',
                        'exit_price': stop_price,
                        'pnl_pct': pnl_pct,
                        'pnl_eur': pnl_eur,
                        'bars_held': bars_held
                    }
            else:  # SHORT
                if bar['low'] <= target_price:
                    pnl_pct = (entry_price - target_price) / entry_price
                    pnl_abs = entry_price - target_price
                    pnl_eur = (shares * pnl_abs) - COMMISSION
                    return {
                        'status': 'CLOSED',
                        'exit_type': 'TARGET',
                        'exit_price': target_price,
                        'pnl_pct': pnl_pct,
                        'pnl_eur': pnl_eur,
                        'bars_held': bars_held
                    }
                elif bar['high'] >= stop_price:
                    pnl_pct = (entry_price - stop_price) / entry_price
                    pnl_abs = entry_price - stop_price
                    pnl_eur = (shares * pnl_abs) - COMMISSION
                    return {
                        'status': 'CLOSED',
                        'exit_type': 'STOP',
                        'exit_price': stop_price,
                        'pnl_pct': pnl_pct,
                        'pnl_eur': pnl_eur,
                        'bars_held': bars_held
                    }
            
            # Time exit after MAX_BARS
            if bars_held >= MAX_BARS:
                exit_price = bar['close']
                if direction == 'LONG':
                    pnl_pct = (exit_price - entry_price) / entry_price
                    pnl_abs = exit_price - entry_price
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price
                    pnl_abs = entry_price - exit_price
                pnl_eur = (shares * pnl_abs) - COMMISSION
                return {
                    'status': 'CLOSED',
                    'exit_type': 'TIME',
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'pnl_eur': pnl_eur,
                    'bars_held': bars_held
                }
        
        # Still open (reached end of data)
        exit_price = forward_bars.iloc[-1]['close']
        if direction == 'LONG':
            pnl_pct = (exit_price - entry_price) / entry_price
            pnl_abs = exit_price - entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price
            pnl_abs = entry_price - exit_price
        pnl_eur = (shares * pnl_abs) - COMMISSION
        return {
            'status': 'OPEN',
            'exit_type': 'OPEN',
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'pnl_eur': pnl_eur,
            'bars_held': len(forward_bars)
        }
    
    def run_backtest(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Run backtest on all signals."""
        print(f"\n🎮 Running backtest on {len(signals)} signals...")
        
        results = []
        
        for idx, signal in signals.iterrows():
            result = self.simulate_trade(signal)
            
            results.append({
                'ticker': signal['ticker'],
                'direction': signal['direction'],
                'entry_ts': signal['entry_ts'],
                'entry_price': signal['entry_open'],
                'confidence': signal['confidence'],
                'p_up': signal['p_up'],
                **result
            })
            
            if (idx + 1) % 50 == 0:
                print(f"   Processed {idx + 1}/{len(signals)} trades...")
        
        results_df = pd.DataFrame(results)
        
        # Remove NO_DATA trades
        valid = results_df[results_df['status'] != 'NO_DATA']
        
        if len(valid) < len(results_df):
            print(f"   ⚠️  Removed {len(results_df) - len(valid)} trades without data")
        
        return valid
    
    def print_summary(self, results: pd.DataFrame):
        """Print comprehensive backtest summary."""
        print("\n" + "="*90)
        print("📊 BIG 7 BACKTEST SUMMARY (2025)")
        print("="*90)
        
        print(f"\n📈 Trade Statistics:")
        print(f"   Total Trades:  {len(results)}")
        print(f"   LONG:          {len(results[results['direction']=='LONG'])} ({len(results[results['direction']=='LONG'])/len(results):.1%})")
        print(f"   SHORT:         {len(results[results['direction']=='SHORT'])} ({len(results[results['direction']=='SHORT'])/len(results):.1%})")
        
        # Closed trades only
        closed = results[results['status'] == 'CLOSED']
        
        if not closed.empty:
            wins = closed[closed['pnl_eur'] > 0]
            losses = closed[closed['pnl_eur'] <= 0]
            
            win_rate = len(wins) / len(closed)
            avg_win_pct = wins['pnl_pct'].mean() if len(wins) > 0 else 0
            avg_loss_pct = losses['pnl_pct'].mean() if len(losses) > 0 else 0
            
            print(f"\n💰 Performance (Percentage):")
            print(f"   Win Rate:      {win_rate:.1%} ({len(wins)}W / {len(losses)}L)")
            print(f"   Avg Win:       {avg_win_pct:.2%}")
            print(f"   Avg Loss:      {avg_loss_pct:.2%}")
            print(f"   Avg P&L:       {closed['pnl_pct'].mean():.2%}")
            
            # Real P&L
            total_pnl = results['pnl_eur'].sum()
            total_commission = len(closed) * COMMISSION
            avg_win_eur = wins['pnl_eur'].mean() if len(wins) > 0 else 0
            avg_loss_eur = losses['pnl_eur'].mean() if len(losses) > 0 else 0
            
            gross_profit = wins['pnl_eur'].sum() + len(wins) * COMMISSION if len(wins) > 0 else 0
            gross_loss = abs(losses['pnl_eur'].sum() + len(losses) * COMMISSION) if len(losses) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            print(f"\n💶 Real P&L (EUR with €{POSITION_SIZE:,.0f}/trade, €{COMMISSION:.0f} commission):")
            print(f"   Avg Win:       €{avg_win_eur:,.2f}")
            print(f"   Avg Loss:      €{avg_loss_eur:,.2f}")
            print(f"   Total P&L:     €{total_pnl:,.2f}")
            print(f"   Commission:    €{total_commission:,.2f}")
            print(f"   Net P&L:       €{total_pnl:,.2f}")
            print(f"   Profit Factor: {profit_factor:.2f}")
            
            # ROI
            starting_capital = POSITION_SIZE * 2
            roi = (total_pnl / starting_capital) * 100
            print(f"   ROI:           {roi:+.2f}%")
            
            # Exit types
            print(f"\n🎯 Exit Types:")
            target_exits = len(closed[closed['exit_type'] == 'TARGET'])
            stop_exits = len(closed[closed['exit_type'] == 'STOP'])
            time_exits = len(closed[closed['exit_type'] == 'TIME'])
            print(f"   Target: {target_exits} ({target_exits/len(closed):.1%})")
            print(f"   Stop:   {stop_exits} ({stop_exits/len(closed):.1%})")
            print(f"   Time:   {time_exits} ({time_exits/len(closed):.1%})")
            
            # Holding time
            print(f"\n⏱️  Holding Time:")
            print(f"   Avg Bars: {closed['bars_held'].mean():.1f}")
            print(f"   Max Bars: {closed['bars_held'].max():.0f}")
            
            # By ticker breakdown
            print(f"\n📊 Performance by Ticker:")
            for ticker in sorted(results['ticker'].unique()):
                ticker_results = closed[closed['ticker'] == ticker]
                if len(ticker_results) > 0:
                    ticker_wins = ticker_results[ticker_results['pnl_eur'] > 0]
                    ticker_pnl = ticker_results['pnl_eur'].sum()
                    ticker_wr = len(ticker_wins) / len(ticker_results)
                    print(f"   {ticker:5s}: {len(ticker_results):3d} trades | WR {ticker_wr:5.1%} | P&L €{ticker_pnl:+8.2f}")
            
            # Top 10 winners
            print(f"\n🏆 Top 10 Winners:")
            top_wins = closed.nlargest(10, 'pnl_eur')
            for _, trade in top_wins.iterrows():
                print(f"   {trade['ticker']:5s} {trade['direction']:5s} {trade['exit_type']:6s} "
                      f"{trade['pnl_pct']:+6.2%} €{trade['pnl_eur']:+7.2f} @ {trade['entry_ts']:%Y-%m-%d}")
            
            # Worst 10 losers
            print(f"\n💸 Worst 10 Losers:")
            worst_losses = closed.nsmallest(10, 'pnl_eur')
            for _, trade in worst_losses.iterrows():
                print(f"   {trade['ticker']:5s} {trade['direction']:5s} {trade['exit_type']:6s} "
                      f"{trade['pnl_pct']:+6.2%} €{trade['pnl_eur']:+7.2f} @ {trade['entry_ts']:%Y-%m-%d}")
        
        print("\n" + "="*90)


def main():
    parser = argparse.ArgumentParser(description='Backtest Big 7 for 2025')
    parser.add_argument('--events', type=str, default='data/ml_datasets/accum_distrib_events.parquet',
                       help='Path to events parquet file')
    parser.add_argument('--model', type=str, default='ML/validated_models/production_bundle.pkl',
                       help='Path to model bundle')
    parser.add_argument('--start-date', type=str, default='2025-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2025-12-31',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default=None,
                       help='Save results to CSV')
    args = parser.parse_args()
    
    # Initialize backtester
    backtester = Big7Backtester(args.events, args.model)
    
    # Filter Big 7 events
    big7_events = backtester.filter_big7_2025(args.start_date, args.end_date)
    
    if big7_events.empty:
        print("❌ No events found for Big 7 in specified date range")
        return
    
    # Generate signals
    signals = backtester.generate_signals(big7_events)
    
    if signals.empty:
        print("❌ No signals generated")
        return
    
    # Run backtest
    results = backtester.run_backtest(signals)
    
    # Print summary
    backtester.print_summary(results)
    
    # Save results if requested
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()
