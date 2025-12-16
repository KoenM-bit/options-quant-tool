"""
Calculate market regime classifications from technical indicators.

This script analyzes technical indicators to classify market conditions into:
1. Trend Regime (uptrend/downtrend/ranging)
2. Volatility Regime (high/normal/low)
3. Market Phase (accumulation/markup/distribution/markdown)

These classifications help select appropriate options strategies.
"""

import sys
import os
import logging
import argparse
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.models.fact_market_regime import FactMarketRegime
from src.models.fact_technical_indicators import FactTechnicalIndicators

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarketRegimeCalculator:
    """Calculate market regime classifications from technical indicators."""
    
    def __init__(self):
        """Initialize calculator with database connection."""
        db_url = (
            f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
            f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
        )
        self.engine = create_engine(db_url)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def ensure_table_exists(self):
        """Ensure the fact_market_regime table exists."""
        from src.models.base import Base
        Base.metadata.create_all(self.engine, tables=[FactMarketRegime.__table__])
        logger.info("✅ Ensured fact_market_regime table exists")
    
    def load_indicators(self, ticker: str, start_date: Optional[date] = None, 
                       lookback_days: int = 252) -> pd.DataFrame:
        """
        Load technical indicators for a ticker with sufficient lookback.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Optional start date (if None, loads all data)
            lookback_days: Days of lookback for percentile calculations
            
        Returns:
            DataFrame with technical indicators
        """
        query = """
            SELECT 
                ticker, trade_date, close, volume,
                sma_20, sma_50, sma_200,
                ema_12, ema_26,
                macd, macd_signal, macd_histogram,
                rsi_14, stochastic_k, stochastic_d,
                atr_14, bollinger_width,
                realized_volatility_20, parkinson_volatility_20,
                high_20d, low_20d, high_52w, low_52w,
                pct_from_high_20d, pct_from_low_20d,
                pct_from_high_52w, pct_from_low_52w,
                volume_sma_20, volume_ratio,
                obv, obv_sma_20,
                adx_14, plus_di_14, minus_di_14
            FROM fact_technical_indicators
            WHERE ticker = %(ticker)s
        """
        
        params = {'ticker': ticker}
        
        if start_date:
            # Load lookback data for percentile calculations
            lookback_start = start_date - timedelta(days=lookback_days + 30)
            query += " AND trade_date >= %(lookback_start)s"
            params['lookback_start'] = lookback_start
        
        df = pd.read_sql(query, self.engine, params=params)
        
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date')
        
        logger.info(f"📊 Loaded {len(df)} indicator rows for {ticker} from {df['trade_date'].min().date()} to {df['trade_date'].max().date()}")
        return df
    
    def classify_trend_regime(self, row: pd.Series, df: pd.DataFrame, idx: int) -> Tuple[str, float, Dict]:
        """
        Classify trend regime based on SMA alignment, ADX, and price position.
        
        Returns:
            (regime, strength, signals) tuple
        """
        signals = {}
        score = 0
        max_score = 0
        
        # 1. SMA Alignment (40 points)
        max_score += 40
        if pd.notna(row['sma_20']) and pd.notna(row['sma_50']) and pd.notna(row['sma_200']):
            # Uptrend: 20 > 50 > 200
            if row['sma_20'] > row['sma_50'] > row['sma_200']:
                score += 40
                signals['sma_alignment'] = 'bullish'
            # Downtrend: 20 < 50 < 200
            elif row['sma_20'] < row['sma_50'] < row['sma_200']:
                score -= 40
                signals['sma_alignment'] = 'bearish'
            else:
                signals['sma_alignment'] = 'mixed'
        
        # 2. Price vs SMA200 (20 points)
        max_score += 20
        if pd.notna(row['sma_200']):
            price_vs_sma200 = ((row['close'] - row['sma_200']) / row['sma_200']) * 100
            signals['price_vs_sma200_pct'] = round(price_vs_sma200, 2)
            
            if price_vs_sma200 > 2:
                score += 20
            elif price_vs_sma200 < -2:
                score -= 20
            else:
                score += 10 * (price_vs_sma200 / 2)  # Gradual scaling
        
        # 3. ADX Trend Strength (30 points)
        max_score += 30
        if pd.notna(row['adx_14']):
            signals['adx'] = round(row['adx_14'], 2)
            
            # ADX indicates trend strength, use +DI/-DI for direction
            if pd.notna(row['plus_di_14']) and pd.notna(row['minus_di_14']):
                di_diff = row['plus_di_14'] - row['minus_di_14']
                signals['di_difference'] = round(di_diff, 2)
                
                # Strong trend (ADX > 25)
                if row['adx_14'] > 25:
                    if di_diff > 5:
                        score += 30
                    elif di_diff < -5:
                        score -= 30
                    else:
                        score += 15 * (di_diff / 5)
                # Weak trend (ADX < 20)
                elif row['adx_14'] < 20:
                    score += 5 * (di_diff / 10)  # Minimal impact
                # Medium trend (20-25)
                else:
                    score += 20 * (di_diff / 10)
        
        # 4. MACD Signal (10 points)
        max_score += 10
        if pd.notna(row['macd']) and pd.notna(row['macd_signal']):
            macd_diff = row['macd'] - row['macd_signal']
            signals['macd_position'] = 'above_signal' if macd_diff > 0 else 'below_signal'
            
            if macd_diff > 0:
                score += 10
            else:
                score -= 10
        
        # Calculate normalized strength (0-100)
        strength = min(100, max(0, ((score / max_score) + 1) * 50))
        
        # Classify regime based on score
        if score > 30:
            regime = 'uptrend'
        elif score < -30:
            regime = 'downtrend'
        else:
            regime = 'ranging'
        
        signals['raw_score'] = round(score, 2)
        signals['max_score'] = max_score
        
        return regime, strength, signals
    
    def classify_volatility_regime(self, row: pd.Series, df: pd.DataFrame, idx: int) -> Tuple[str, float, Dict]:
        """
        Classify volatility regime based on ATR, realized vol, and Bollinger width.
        
        Returns:
            (regime, percentile, signals) tuple
        """
        signals = {}
        
        # Calculate ATR percentile over lookback period
        if pd.notna(row['atr_14']) and idx >= 252:
            lookback_df = df.iloc[max(0, idx-252):idx+1]
            atr_percentile = (lookback_df['atr_14'] < row['atr_14']).sum() / len(lookback_df) * 100
            signals['atr_percentile'] = round(atr_percentile, 2)
        else:
            atr_percentile = 50  # Default to middle
        
        # Calculate realized vol percentile
        if pd.notna(row['realized_volatility_20']) and idx >= 252:
            lookback_df = df.iloc[max(0, idx-252):idx+1]
            vol_percentile = (lookback_df['realized_volatility_20'] < row['realized_volatility_20']).sum() / len(lookback_df) * 100
            signals['realized_vol_percentile'] = round(vol_percentile, 2)
        else:
            vol_percentile = 50
        
        # Bollinger Band width percentile
        if pd.notna(row['bollinger_width']) and idx >= 252:
            lookback_df = df.iloc[max(0, idx-252):idx+1]
            bb_percentile = (lookback_df['bollinger_width'] < row['bollinger_width']).sum() / len(lookback_df) * 100
            signals['bb_width_percentile'] = round(bb_percentile, 2)
        else:
            bb_percentile = 50
        
        # Average percentile across metrics
        avg_percentile = np.mean([atr_percentile, vol_percentile, bb_percentile])
        
        # Add current values to signals
        if pd.notna(row['atr_14']):
            signals['atr'] = round(row['atr_14'], 4)
        if pd.notna(row['realized_volatility_20']):
            signals['realized_vol'] = round(row['realized_volatility_20'], 2)
        if pd.notna(row['bollinger_width']):
            signals['bb_width'] = round(row['bollinger_width'], 4)
        
        # Classify volatility regime
        if avg_percentile > 70:
            regime = 'high'
        elif avg_percentile < 30:
            regime = 'low'
        else:
            regime = 'normal'
        
        return regime, avg_percentile, signals
    
    def classify_market_phase(self, row: pd.Series, df: pd.DataFrame, idx: int) -> Tuple[str, float, Dict]:
        """
        Classify market phase using price action, volume, and momentum.
        
        Phases:
        - Accumulation: Ranging + increasing volume + oversold
        - Markup: Uptrend + strong volume + momentum
        - Distribution: Ranging + high volume + overbought
        - Markdown: Downtrend + declining momentum
        
        Returns:
            (phase, confidence, signals) tuple
        """
        signals = {}
        
        # Get trend regime (simplified check)
        is_uptrend = pd.notna(row['sma_20']) and pd.notna(row['sma_50']) and row['sma_20'] > row['sma_50']
        is_downtrend = pd.notna(row['sma_20']) and pd.notna(row['sma_50']) and row['sma_20'] < row['sma_50']
        is_ranging = not is_uptrend and not is_downtrend
        
        # Volume analysis
        volume_increasing = pd.notna(row['volume_ratio']) and row['volume_ratio'] > 1.2
        volume_high = pd.notna(row['volume_ratio']) and row['volume_ratio'] > 1.5
        volume_declining = pd.notna(row['volume_ratio']) and row['volume_ratio'] < 0.8
        
        signals['volume_ratio'] = round(row['volume_ratio'], 2) if pd.notna(row['volume_ratio']) else None
        
        # Momentum analysis
        rsi_oversold = pd.notna(row['rsi_14']) and row['rsi_14'] < 40
        rsi_overbought = pd.notna(row['rsi_14']) and row['rsi_14'] > 60
        rsi_neutral = not rsi_oversold and not rsi_overbought
        
        signals['rsi'] = round(row['rsi_14'], 2) if pd.notna(row['rsi_14']) else None
        
        # MACD momentum
        macd_bullish = pd.notna(row['macd']) and pd.notna(row['macd_signal']) and row['macd'] > row['macd_signal']
        macd_bearish = pd.notna(row['macd']) and pd.notna(row['macd_signal']) and row['macd'] < row['macd_signal']
        
        signals['macd_position'] = 'bullish' if macd_bullish else 'bearish' if macd_bearish else 'neutral'
        
        # OBV trend (is money flowing in or out?)
        if idx >= 20 and pd.notna(row['obv']) and pd.notna(row['obv_sma_20']):
            obv_rising = row['obv'] > row['obv_sma_20']
            signals['obv_vs_sma'] = 'rising' if obv_rising else 'falling'
        else:
            obv_rising = None
        
        # Phase classification logic
        confidence_scores = {}
        
        # Accumulation: Ranging + volume building + oversold + OBV rising
        accumulation_score = 0
        if is_ranging:
            accumulation_score += 30
        if volume_increasing:
            accumulation_score += 25
        if rsi_oversold:
            accumulation_score += 25
        if obv_rising:
            accumulation_score += 20
        confidence_scores['accumulation'] = accumulation_score
        
        # Markup: Uptrend + strong volume + momentum + OBV rising
        markup_score = 0
        if is_uptrend:
            markup_score += 30
        if volume_high or volume_increasing:
            markup_score += 25
        if macd_bullish:
            markup_score += 25
        if obv_rising:
            markup_score += 20
        confidence_scores['markup'] = markup_score
        
        # Distribution: Ranging/topping + high volume + overbought + OBV falling
        distribution_score = 0
        if is_ranging or (is_uptrend and rsi_overbought):
            distribution_score += 30
        if volume_high:
            distribution_score += 25
        if rsi_overbought:
            distribution_score += 25
        if obv_rising is False:
            distribution_score += 20
        confidence_scores['distribution'] = distribution_score
        
        # Markdown: Downtrend + declining volume + bearish momentum
        markdown_score = 0
        if is_downtrend:
            markdown_score += 30
        if volume_declining or not volume_increasing:
            markdown_score += 25
        if macd_bearish:
            markdown_score += 25
        if obv_rising is False:
            markdown_score += 20
        confidence_scores['markdown'] = markdown_score
        
        # Select phase with highest confidence
        phase = max(confidence_scores, key=confidence_scores.get)
        confidence = confidence_scores[phase]
        
        signals['phase_scores'] = confidence_scores
        
        return phase, confidence, signals
    
    def identify_support_resistance(self, row: pd.Series, df: pd.DataFrame, idx: int) -> Tuple[float, float, float, float]:
        """
        Identify primary support and resistance levels with strength scores.
        
        Returns:
            (support, resistance, support_strength, resistance_strength) tuple
        """
        # Use 52-week high/low as primary resistance/support
        resistance = row['high_52w'] if pd.notna(row['high_52w']) else row['high_20d']
        support = row['low_52w'] if pd.notna(row['low_52w']) else row['low_20d']
        
        # Calculate strength based on distance from current price
        if pd.notna(resistance):
            resistance_distance = abs((resistance - row['close']) / row['close']) * 100
            # Closer levels are stronger (max 100, decays with distance)
            resistance_strength = max(0, 100 - resistance_distance * 2)
        else:
            resistance_strength = 0
        
        if pd.notna(support):
            support_distance = abs((row['close'] - support) / row['close']) * 100
            support_strength = max(0, 100 - support_distance * 2)
        else:
            support_strength = 0
        
        return support, resistance, support_strength, resistance_strength
    
    def recommend_strategy(self, trend_regime: str, volatility_regime: str, 
                          market_phase: str, signals: Dict) -> Tuple[str, Dict]:
        """
        Recommend options strategy based on regime analysis.
        
        Returns:
            (strategy, rationale) tuple
        """
        rationale = {
            'trend': trend_regime,
            'volatility': volatility_regime,
            'phase': market_phase
        }
        
        # Strategy selection matrix
        if trend_regime == 'ranging' and volatility_regime == 'low':
            strategy = 'iron_condor'
            rationale['reason'] = 'Low volatility ranging market ideal for selling premium with defined risk'
        
        elif trend_regime == 'uptrend' and market_phase in ['accumulation', 'markup']:
            if volatility_regime == 'low':
                strategy = 'bull_call_spread'
                rationale['reason'] = 'Uptrend with low volatility - buy calls with spread to reduce cost'
            else:
                strategy = 'long_call'
                rationale['reason'] = 'Strong uptrend in markup phase - directional call position'
        
        elif trend_regime == 'downtrend' and market_phase in ['distribution', 'markdown']:
            if volatility_regime == 'low':
                strategy = 'bear_put_spread'
                rationale['reason'] = 'Downtrend with low volatility - buy puts with spread'
            else:
                strategy = 'long_put'
                rationale['reason'] = 'Strong downtrend in markdown phase - directional put position'
        
        elif volatility_regime == 'high':
            if trend_regime == 'ranging':
                strategy = 'short_straddle'
                rationale['reason'] = 'High volatility ranging market - sell premium expecting vol crush'
            else:
                strategy = 'long_straddle'
                rationale['reason'] = 'High volatility trending market - expect big move'
        
        elif market_phase == 'accumulation':
            strategy = 'bull_call_spread'
            rationale['reason'] = 'Accumulation phase - positioned for potential breakout'
        
        elif market_phase == 'distribution':
            strategy = 'bear_put_spread'
            rationale['reason'] = 'Distribution phase - positioned for potential breakdown'
        
        else:
            strategy = 'neutral_strategy'
            rationale['reason'] = 'Mixed signals - wait for clearer regime'
        
        return strategy, rationale
    
    def calculate_for_ticker(self, ticker: str, start_date: Optional[date] = None) -> int:
        """
        Calculate regime classifications for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Optional start date (if None, processes all data)
            
        Returns:
            Number of rows inserted/updated
        """
        # Load technical indicators with lookback
        df = self.load_indicators(ticker, start_date, lookback_days=252)
        
        if df.empty:
            logger.warning(f"⚠️  No indicators found for {ticker}")
            return 0
        
        # Calculate regimes for each row
        regimes = []
        prev_regime_key = None
        days_in_regime = 0
        
        for idx, row in df.iterrows():
            # Skip if we don't have enough data
            if pd.isna(row['close']) or pd.isna(row['rsi_14']):
                continue
            
            # Skip lookback period if start_date specified
            if start_date and row['trade_date'].date() < start_date:
                continue
            
            # Classify regimes
            trend_regime, trend_strength, trend_signals = self.classify_trend_regime(row, df, idx)
            vol_regime, vol_percentile, vol_signals = self.classify_volatility_regime(row, df, idx)
            phase, phase_confidence, phase_signals = self.classify_market_phase(row, df, idx)
            
            # Identify support/resistance
            support, resistance, support_strength, resistance_strength = \
                self.identify_support_resistance(row, df, idx)
            
            # Recommend strategy
            strategy, strategy_rationale = self.recommend_strategy(
                trend_regime, vol_regime, phase, 
                {**trend_signals, **vol_signals, **phase_signals}
            )
            
            # Detect regime changes
            current_regime_key = f"{trend_regime}_{vol_regime}_{phase}"
            if prev_regime_key is None or current_regime_key != prev_regime_key:
                regime_change = 'yes'
                days_in_regime = 1
            else:
                regime_change = 'no'
                days_in_regime += 1
            prev_regime_key = current_regime_key
            
            regimes.append({
                'ticker': ticker,
                'trade_date': row['trade_date'].date(),
                'trend_regime': trend_regime,
                'trend_strength': trend_strength,
                'trend_signals': trend_signals,
                'volatility_regime': vol_regime,
                'volatility_percentile': vol_percentile,
                'volatility_signals': vol_signals,
                'market_phase': phase,
                'phase_confidence': phase_confidence,
                'phase_signals': phase_signals,
                'primary_support': support,
                'primary_resistance': resistance,
                'support_strength': support_strength,
                'resistance_strength': resistance_strength,
                'regime_change': regime_change,
                'days_in_regime': days_in_regime,
                'recommended_strategy': strategy,
                'strategy_rationale': strategy_rationale
            })
        
        if not regimes:
            logger.warning(f"⚠️  No regimes calculated for {ticker}")
            return 0
        
        # Bulk insert/update
        rows_inserted = self.insert_regimes(regimes, ticker)
        return rows_inserted
    
    def insert_regimes(self, regimes: List[Dict], ticker: str) -> int:
        """
        Insert regime records with ON CONFLICT handling.
        
        Args:
            regimes: List of regime dictionaries
            ticker: Stock ticker symbol
            
        Returns:
            Number of rows inserted/updated
        """
        insert_query = text("""
            INSERT INTO fact_market_regime (
                ticker, trade_date,
                trend_regime, trend_strength, trend_signals,
                volatility_regime, volatility_percentile, volatility_signals,
                market_phase, phase_confidence, phase_signals,
                primary_support, primary_resistance,
                support_strength, resistance_strength,
                regime_change, days_in_regime,
                recommended_strategy, strategy_rationale,
                calculated_at
            ) VALUES (
                :ticker, :trade_date,
                :trend_regime, :trend_strength, :trend_signals,
                :volatility_regime, :volatility_percentile, :volatility_signals,
                :market_phase, :phase_confidence, :phase_signals,
                :primary_support, :primary_resistance,
                :support_strength, :resistance_strength,
                :regime_change, :days_in_regime,
                :recommended_strategy, :strategy_rationale,
                CURRENT_TIMESTAMP
            )
            ON CONFLICT (ticker, trade_date) DO UPDATE SET
                trend_regime = EXCLUDED.trend_regime,
                trend_strength = EXCLUDED.trend_strength,
                trend_signals = EXCLUDED.trend_signals,
                volatility_regime = EXCLUDED.volatility_regime,
                volatility_percentile = EXCLUDED.volatility_percentile,
                volatility_signals = EXCLUDED.volatility_signals,
                market_phase = EXCLUDED.market_phase,
                phase_confidence = EXCLUDED.phase_confidence,
                phase_signals = EXCLUDED.phase_signals,
                primary_support = EXCLUDED.primary_support,
                primary_resistance = EXCLUDED.primary_resistance,
                support_strength = EXCLUDED.support_strength,
                resistance_strength = EXCLUDED.resistance_strength,
                regime_change = EXCLUDED.regime_change,
                days_in_regime = EXCLUDED.days_in_regime,
                recommended_strategy = EXCLUDED.recommended_strategy,
                strategy_rationale = EXCLUDED.strategy_rationale,
                calculated_at = EXCLUDED.calculated_at,
                updated_at = CURRENT_TIMESTAMP
        """)
        
        # Convert JSON fields to string
        for regime in regimes:
            import json
            regime['trend_signals'] = json.dumps(regime['trend_signals'])
            regime['volatility_signals'] = json.dumps(regime['volatility_signals'])
            regime['phase_signals'] = json.dumps(regime['phase_signals'])
            regime['strategy_rationale'] = json.dumps(regime['strategy_rationale'])
        
        self.session.execute(insert_query, regimes)
        self.session.commit()
        
        logger.info(f"✅ Inserted/Updated {len(regimes)} regime records for {ticker}")
        return len(regimes)
    
    def calculate_all_tickers(self, tickers: List[str], start_date: Optional[date] = None) -> int:
        """
        Calculate regimes for all tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Optional start date
            
        Returns:
            Total rows processed
        """
        total_rows = 0
        
        for ticker in tickers:
            logger.info(f"📊 Calculating regimes for {ticker}...")
            try:
                rows = self.calculate_for_ticker(ticker, start_date)
                total_rows += rows
                logger.info(f"✅ {ticker}: Processed {rows} rows")
            except Exception as e:
                logger.error(f"❌ Error processing {ticker}: {e}")
                continue
        
        return total_rows


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Calculate market regime classifications')
    parser.add_argument('--tickers', type=str, required=True,
                       help='Comma-separated list of tickers (e.g., AD.AS,MT.AS)')
    parser.add_argument('--start-date', type=str,
                       help='Start date (YYYY-MM-DD). If omitted, processes all data')
    parser.add_argument('--lookback-days', type=int, default=252,
                       help='Days of lookback for percentile calculations (default: 252)')
    parser.add_argument('--all', action='store_true',
                       help='Process all available data (ignores start-date)')
    
    args = parser.parse_args()
    
    # Parse tickers
    tickers = [t.strip() for t in args.tickers.split(',')]
    
    # Parse start date
    start_date = None
    if args.start_date and not args.all:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    
    # Initialize calculator
    calculator = MarketRegimeCalculator()
    calculator.ensure_table_exists()
    
    logger.info("=" * 80)
    logger.info("🚀 Starting Market Regime Classification")
    logger.info("=" * 80)
    logger.info(f"Tickers: {', '.join(tickers)}")
    if start_date:
        logger.info(f"Start date: {start_date}")
    else:
        logger.info("Processing all available data")
    logger.info("")
    
    # Calculate regimes
    total_rows = calculator.calculate_all_tickers(tickers, start_date)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 CLASSIFICATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total rows processed: {total_rows}")
    logger.info("=" * 80)
    logger.info("✅ Market regime classification complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
