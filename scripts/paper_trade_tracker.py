#!/usr/bin/env python3
"""
Paper Trading Tracker

Tracks open paper trades from signals and updates their status based on actual market data.
Logs entries, exits, and P&L to create a complete trading journal with real position sizing.

Position Sizing:
- Fixed €10,000 per trade
- €10 commission per round-trip trade
- Calculates shares, real EUR P&L, and ROI

Usage:
    # Add signals from a specific date
    python scripts/paper_trade_tracker.py --date 2025-12-24
    
    # Update open trades only
    python scripts/paper_trade_tracker.py --update-only
    
    # View summary for specific period
    python scripts/paper_trade_tracker.py --start-date 2025-12-01 --end-date 2025-12-31 --update-only
    
    # View summary from start date onwards
    python scripts/paper_trade_tracker.py --start-date 2025-12-01 --update-only
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from src.config import settings
import json
import argparse

# Optimized exit parameters (validated via exit sensitivity analysis)
ATR_K_TARGET = 1.5
ATR_K_STOP = 1.0
MAX_BARS = 20  # Maximum bars before time exit (NOT hours!)

# Position sizing and costs
POSITION_SIZE = 10000  # €10,000 per trade
COMMISSION = 10        # €10 fixed commission per trade (round-trip)


class PaperTradeTracker:
    """Track paper trades and log results."""
    
    def __init__(self):
        self.engine = create_engine(settings.database_url)
        self.trades_log_path = Path("data/paper_trades/trade_log.csv")
        self.trades_log_path.parent.mkdir(exist_ok=True)
        
        # Initialize trade log if doesn't exist
        if not self.trades_log_path.exists():
            self._init_trade_log()
    
    def _init_trade_log(self):
        """Create initial trade log CSV."""
        df = pd.DataFrame(columns=[
            'trade_id', 'signal_date', 'ticker', 'market', 'direction',
            'entry_ts', 'entry_price', 'target_price', 'stop_price',
            'exit_ts', 'exit_price', 'exit_type', 'pnl_pct', 'pnl_abs',
            'pnl_eur', 'shares', 'commission_eur',
            'bars_held', 'status', 'confidence', 'p_up'
        ])
        df.to_csv(self.trades_log_path, index=False)
        print(f"✅ Initialized trade log: {self.trades_log_path}")
    
    def load_signals(self, signal_date: str) -> pd.DataFrame:
        """Load signals from a specific date."""
        signal_path = Path(f"data/signals/breakout_signals_{signal_date}.csv")
        
        if not signal_path.exists():
            print(f"❌ No signals file found for {signal_date}")
            return pd.DataFrame()
        
        # Check if empty
        if signal_path.stat().st_size < 10:
            print(f"📊 No signals generated on {signal_date}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(signal_path)
            if df.empty:
                print(f"📊 No signals on {signal_date}")
                return pd.DataFrame()
            
            # Parse timestamps
            df['entry_ts'] = pd.to_datetime(df['entry_ts'])
            return df
        except Exception as e:
            print(f"❌ Error loading signals: {e}")
            return pd.DataFrame()
    
    def load_trade_log(self) -> pd.DataFrame:
        """Load existing trade log."""
        df = pd.read_csv(self.trades_log_path)
        if not df.empty and 'entry_ts' in df.columns:
            df['entry_ts'] = pd.to_datetime(df['entry_ts'])
            df['exit_ts'] = pd.to_datetime(df['exit_ts'], errors='coerce')
        return df
    
    def save_trade_log(self, df: pd.DataFrame):
        """Save trade log to CSV."""
        df.to_csv(self.trades_log_path, index=False)
    
    def fetch_forward_bars(self, ticker: str, entry_ts: pd.Timestamp, max_bars: int = 40) -> pd.DataFrame:
        """Fetch forward price bars after entry."""
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
            })
            rows = result.fetchall()
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close'])
        # Convert Decimal to float
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)
        return df
    
    def simulate_trade_exit(self, trade: pd.Series, forward_bars: pd.DataFrame) -> dict:
        """
        Simulate trade exit based on actual price movement.
        Uses bar-count-based time exit (MAX_BARS) to match sensitivity analysis.
        Calculates real P&L with fixed position size and commission.
        """
        if forward_bars.empty:
            return {
                'status': 'OPEN',
                'exit_ts': None,
                'exit_price': None,
                'exit_type': None,
                'pnl_pct': None,
                'pnl_abs': None,
                'pnl_eur': None,
                'shares': None,
                'commission_eur': None,
                'bars_held': 0
            }
        
        entry_price = trade['entry_price']
        target_price = trade['target_price']
        stop_price = trade['stop_price']
        direction = trade['direction']
        
        # Calculate position size (shares)
        shares = POSITION_SIZE / entry_price
        
        # Simulate bar by bar
        for idx, bar in forward_bars.iterrows():
            bars_held = idx + 1
            
            if direction == 'LONG':
                # Check target hit (high >= target)
                if bar['high'] >= target_price:
                    exit_price = target_price
                    exit_type = 'TARGET'
                    pnl_pct = (exit_price - entry_price) / entry_price
                    pnl_abs = exit_price - entry_price
                    pnl_eur = (shares * pnl_abs) - COMMISSION
                    return {
                        'status': 'CLOSED',
                        'exit_ts': bar['timestamp'],
                        'exit_price': exit_price,
                        'exit_type': exit_type,
                        'pnl_pct': pnl_pct,
                        'pnl_abs': pnl_abs,
                        'pnl_eur': pnl_eur,
                        'shares': shares,
                        'commission_eur': COMMISSION,
                        'bars_held': bars_held
                    }
                # Check stop hit (low <= stop)
                elif bar['low'] <= stop_price:
                    exit_price = stop_price
                    exit_type = 'STOP'
                    pnl_pct = (exit_price - entry_price) / entry_price
                    pnl_abs = exit_price - entry_price
                    pnl_eur = (shares * pnl_abs) - COMMISSION
                    return {
                        'status': 'CLOSED',
                        'exit_ts': bar['timestamp'],
                        'exit_price': exit_price,
                        'exit_type': exit_type,
                        'pnl_pct': pnl_pct,
                        'pnl_abs': pnl_abs,
                        'pnl_eur': pnl_eur,
                        'shares': shares,
                        'commission_eur': COMMISSION,
                        'bars_held': bars_held
                    }
            else:  # SHORT
                # Check target hit (low <= target)
                if bar['low'] <= target_price:
                    exit_price = target_price
                    exit_type = 'TARGET'
                    pnl_pct = (entry_price - exit_price) / entry_price
                    pnl_abs = entry_price - exit_price
                    pnl_eur = (shares * pnl_abs) - COMMISSION
                    return {
                        'status': 'CLOSED',
                        'exit_ts': bar['timestamp'],
                        'exit_price': exit_price,
                        'exit_type': exit_type,
                        'pnl_pct': pnl_pct,
                        'pnl_abs': pnl_abs,
                        'pnl_eur': pnl_eur,
                        'shares': shares,
                        'commission_eur': COMMISSION,
                        'bars_held': bars_held
                    }
                # Check stop hit (high >= stop)
                elif bar['high'] >= stop_price:
                    exit_price = stop_price
                    exit_type = 'STOP'
                    pnl_pct = (entry_price - exit_price) / entry_price
                    pnl_abs = entry_price - exit_price
                    pnl_eur = (shares * pnl_abs) - COMMISSION
                    return {
                        'status': 'CLOSED',
                        'exit_ts': bar['timestamp'],
                        'exit_price': exit_price,
                        'exit_type': exit_type,
                        'pnl_pct': pnl_pct,
                        'pnl_abs': pnl_abs,
                        'pnl_eur': pnl_eur,
                        'shares': shares,
                        'commission_eur': COMMISSION,
                        'bars_held': bars_held
                    }
            
            # Check time exit (bar count based, not timestamp)
            if bars_held >= MAX_BARS:
                exit_price = bar['close']
                exit_type = 'TIME'
                if direction == 'LONG':
                    pnl_pct = (exit_price - entry_price) / entry_price
                    pnl_abs = exit_price - entry_price
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price
                    pnl_abs = entry_price - exit_price
                pnl_eur = (shares * pnl_abs) - COMMISSION
                return {
                    'status': 'CLOSED',
                    'exit_ts': bar['timestamp'],
                    'exit_price': exit_price,
                    'exit_type': exit_type,
                    'pnl_pct': pnl_pct,
                    'pnl_abs': pnl_abs,
                    'pnl_eur': pnl_eur,
                    'shares': shares,
                    'commission_eur': COMMISSION,
                    'bars_held': bars_held
                }
        
        # Still open
        return {
            'status': 'OPEN',
            'exit_ts': None,
            'exit_price': None,
            'exit_type': None,
            'pnl_pct': None,
            'pnl_abs': None,
            'pnl_eur': None,
            'shares': None,
            'commission_eur': None,
            'bars_held': len(forward_bars)
        }
    
    def add_signals_to_log(self, signals: pd.DataFrame, signal_date: str):
        """Add new signals to trade log."""
        trade_log = self.load_trade_log()
        
        new_trades = []
        for _, signal in signals.iterrows():
            # Generate trade ID
            trade_id = f"{signal_date}_{signal['ticker']}_{signal['direction']}"
            
            # Check if already logged
            if not trade_log.empty and trade_id in trade_log['trade_id'].values:
                print(f"⚠️  Trade {trade_id} already in log, skipping")
                continue
            
            # Calculate exit prices
            entry_price = signal['entry_open']
            atr_abs = entry_price * signal['atr_pct_mean']
            
            if signal['direction'] == 'LONG':
                target_price = entry_price + ATR_K_TARGET * atr_abs
                stop_price = entry_price - ATR_K_STOP * atr_abs
            else:
                target_price = entry_price - ATR_K_TARGET * atr_abs
                stop_price = entry_price + ATR_K_STOP * atr_abs
            
            new_trade = {
                'trade_id': trade_id,
                'signal_date': signal_date,
                'ticker': signal['ticker'],
                'market': signal['market'],
                'direction': signal['direction'],
                'entry_ts': signal['entry_ts'],
                'entry_price': entry_price,
                'target_price': target_price,
                'stop_price': stop_price,
                'exit_ts': None,
                'exit_price': None,
                'exit_type': None,
                'pnl_pct': None,
                'pnl_abs': None,
                'pnl_eur': None,
                'shares': POSITION_SIZE / entry_price,
                'commission_eur': None,
                'bars_held': None,
                'status': 'OPEN',
                'confidence': signal.get('confidence', ''),
                'p_up': signal['p_up']
            }
            new_trades.append(new_trade)
            print(f"📝 Added new trade: {trade_id}")
        
        if new_trades:
            new_df = pd.DataFrame(new_trades)
            trade_log = pd.concat([trade_log, new_df], ignore_index=True)
            self.save_trade_log(trade_log)
            print(f"✅ Added {len(new_trades)} new trades to log")
    
    def update_open_trades(self):
        """Update status of all open trades."""
        trade_log = self.load_trade_log()
        
        if trade_log.empty:
            print("📊 No trades in log")
            return
        
        open_trades = trade_log[trade_log['status'] == 'OPEN']
        
        if open_trades.empty:
            print("📊 No open trades to update")
            return
        
        print(f"\n🔄 Updating {len(open_trades)} open trades...")
        
        updated_count = 0
        for idx, trade in open_trades.iterrows():
            # Fetch forward bars
            forward_bars = self.fetch_forward_bars(
                trade['ticker'],
                trade['entry_ts'],
                max_bars=60  # Look ahead more to catch exits
            )
            
            # Simulate exit
            result = self.simulate_trade_exit(trade, forward_bars)
            
            # Update trade log
            if result['status'] == 'CLOSED':
                trade_log.loc[idx, 'status'] = result['status']
                trade_log.loc[idx, 'exit_ts'] = result['exit_ts']
                trade_log.loc[idx, 'exit_price'] = result['exit_price']
                trade_log.loc[idx, 'exit_type'] = result['exit_type']
                trade_log.loc[idx, 'pnl_pct'] = result['pnl_pct']
                trade_log.loc[idx, 'pnl_abs'] = result['pnl_abs']
                trade_log.loc[idx, 'pnl_eur'] = result['pnl_eur']
                trade_log.loc[idx, 'commission_eur'] = result['commission_eur']
                trade_log.loc[idx, 'bars_held'] = result['bars_held']
                updated_count += 1
                
                win_loss = "🟢 WIN" if result['pnl_pct'] > 0 else "🔴 LOSS"
                print(f"   {win_loss} {trade['trade_id']}: {result['exit_type']} @ {result['exit_price']:.2f} ({result['pnl_pct']:.2%})")
        
        if updated_count > 0:
            self.save_trade_log(trade_log)
            print(f"✅ Updated {updated_count} trades")
        else:
            print("📊 No trades closed yet")
    
    def print_summary(self, start_date: str = None, end_date: str = None):
        """
        Print trading summary statistics.
        
        Args:
            start_date: Optional filter start date (YYYY-MM-DD)
            end_date: Optional filter end date (YYYY-MM-DD)
        """
        trade_log = self.load_trade_log()
        
        if trade_log.empty:
            print("\n📊 No trades yet")
            return
        
        # Apply date filters if provided
        if start_date or end_date:
            trade_log['signal_date_dt'] = pd.to_datetime(trade_log['signal_date'])
            if start_date:
                trade_log = trade_log[trade_log['signal_date_dt'] >= pd.to_datetime(start_date)]
            if end_date:
                trade_log = trade_log[trade_log['signal_date_dt'] <= pd.to_datetime(end_date)]
        
        closed_trades = trade_log[trade_log['status'] == 'CLOSED']
        open_trades = trade_log[trade_log['status'] == 'OPEN']
        
        print("\n" + "="*90)
        print("📊 PAPER TRADING SUMMARY")
        if start_date or end_date:
            period = f" ({start_date or 'start'} to {end_date or 'present'})"
            print(f"   Period: {period}")
        print("="*90)
        
        print(f"\n📈 Trade Status:")
        print(f"   Open:   {len(open_trades)}")
        print(f"   Closed: {len(closed_trades)}")
        print(f"   Total:  {len(trade_log)}")
        
        if not closed_trades.empty:
            wins = closed_trades[closed_trades['pnl_eur'] > 0]
            losses = closed_trades[closed_trades['pnl_eur'] <= 0]
            
            win_rate = len(wins) / len(closed_trades)
            avg_win_pct = wins['pnl_pct'].mean() if len(wins) > 0 else 0
            avg_loss_pct = losses['pnl_pct'].mean() if len(losses) > 0 else 0
            avg_pnl_pct = closed_trades['pnl_pct'].mean()
            
            # Real P&L in EUR
            avg_win_eur = wins['pnl_eur'].mean() if len(wins) > 0 else 0
            avg_loss_eur = losses['pnl_eur'].mean() if len(losses) > 0 else 0
            total_pnl_eur = closed_trades['pnl_eur'].sum()
            total_commission = closed_trades['commission_eur'].sum()
            
            # Profit factor
            gross_profit = wins['pnl_eur'].sum() + total_commission if len(wins) > 0 else 0
            gross_loss = abs(losses['pnl_eur'].sum() + (len(losses) * COMMISSION)) if len(losses) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Exit type distribution
            target_exits = len(closed_trades[closed_trades['exit_type'] == 'TARGET'])
            stop_exits = len(closed_trades[closed_trades['exit_type'] == 'STOP'])
            time_exits = len(closed_trades[closed_trades['exit_type'] == 'TIME'])
            
            print(f"\n💰 Performance (Percentage):")
            print(f"   Win Rate:      {win_rate:.1%} ({len(wins)}W / {len(losses)}L)")
            print(f"   Avg Win:       {avg_win_pct:.2%}")
            print(f"   Avg Loss:      {avg_loss_pct:.2%}")
            print(f"   Avg P&L:       {avg_pnl_pct:.2%}")
            
            print(f"\n💶 Real P&L (EUR with €{POSITION_SIZE:,.0f}/trade, €{COMMISSION:.0f} commission):")
            print(f"   Avg Win:       €{avg_win_eur:,.2f}")
            print(f"   Avg Loss:      €{avg_loss_eur:,.2f}")
            print(f"   Total P&L:     €{total_pnl_eur:,.2f}")
            print(f"   Commission:    €{total_commission:,.2f}")
            print(f"   Net P&L:       €{total_pnl_eur:,.2f}")
            print(f"   Profit Factor: {profit_factor:.2f}")
            
            # Account growth
            starting_capital = POSITION_SIZE * 2  # Assuming we can have 2 positions
            roi = (total_pnl_eur / starting_capital) * 100
            print(f"   ROI (on €{starting_capital:,.0f}): {roi:+.2f}%")
            
            print(f"\n🎯 Exit Types:")
            print(f"   Target: {target_exits} ({target_exits/len(closed_trades):.1%})")
            print(f"   Stop:   {stop_exits} ({stop_exits/len(closed_trades):.1%})")
            print(f"   Time:   {time_exits} ({time_exits/len(closed_trades):.1%})")
            
            print(f"\n⏱️  Holding Time:")
            print(f"   Avg Bars: {closed_trades['bars_held'].mean():.1f}")
            print(f"   Max Bars: {closed_trades['bars_held'].max():.0f}")
            
            # Recent trades
            print(f"\n📋 Recent Closed Trades:")
            recent = closed_trades.sort_values('exit_ts', ascending=False).head(10)
            for _, trade in recent.iterrows():
                win_loss = "🟢" if trade['pnl_eur'] > 0 else "🔴"
                currency = "$" if trade['market'] == 'US' else "€"
                # Calculate actual hours held
                entry_dt = pd.to_datetime(trade['entry_ts'])
                exit_dt = pd.to_datetime(trade['exit_ts'])
                hours_held = (exit_dt - entry_dt).total_seconds() / 3600
                print(f"   {win_loss} {trade['ticker']:6s} {trade['direction']:5s} "
                      f"{trade['exit_type']:6s} {currency}{trade['exit_price']:7.2f} "
                      f"{trade['pnl_pct']:+6.2%} €{trade['pnl_eur']:+7.2f} "
                      f"({hours_held:.1f}h / {trade['bars_held']:.0f} bars)")
        
        if not open_trades.empty:
            print(f"\n⏳ Open Trades:")
            for _, trade in open_trades.iterrows():
                currency = "$" if trade['market'] == 'US' else "€"
                print(f"   📊 {trade['ticker']:6s} {trade['direction']:5s} "
                      f"Entry: {currency}{trade['entry_price']:7.2f} @ {trade['entry_ts']}")
        
        print("="*90 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Paper Trading Tracker')
    parser.add_argument('--start-date', type=str, help='Start date for analysis (YYYY-MM-DD)', default=None)
    parser.add_argument('--end-date', type=str, help='End date for analysis (YYYY-MM-DD)', default=None)
    parser.add_argument('--date', type=str, help='Single signal date to add (YYYY-MM-DD)', default=None)
    parser.add_argument('--update-only', action='store_true', help='Only update open trades, do not add new')
    args = parser.parse_args()
    
    tracker = PaperTradeTracker()
    
    # Add new signals if date provided
    if args.date and not args.update_only:
        signals = tracker.load_signals(args.date)
        if not signals.empty:
            tracker.add_signals_to_log(signals, args.date)
    
    # Update all open trades
    tracker.update_open_trades()
    
    # Print summary with optional date filtering
    tracker.print_summary(start_date=args.start_date, end_date=args.end_date)


if __name__ == "__main__":
    main()
