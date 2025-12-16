"""
Market Regime Classification Model (Gold Layer).

This model classifies market conditions into regime types based on technical indicators:
- Trend Regime: Identifies market trend direction (uptrend, downtrend, ranging)
- Volatility Regime: Classifies volatility levels (high, normal, low)
- Market Phase: Identifies market cycle phase (accumulation, markup, distribution, markdown)

These regimes are used to select appropriate options strategies.
"""

from sqlalchemy import Column, BigInteger, String, Date, Numeric, DateTime, JSON, UniqueConstraint
from sqlalchemy.sql import func
from src.models.base import Base


class FactMarketRegime(Base):
    """Fact table for market regime classification (Gold Layer)."""
    
    __tablename__ = 'fact_market_regime'
    
    # Primary Key
    regime_id = Column(BigInteger, primary_key=True, autoincrement=True)
    
    # Dimensions
    ticker = Column(String(20), nullable=False, index=True, comment='Stock ticker symbol')
    trade_date = Column(Date, nullable=False, index=True, comment='Trading date')
    
    # Trend Regime Classification
    trend_regime = Column(
        String(20), 
        nullable=False,
        comment='Trend regime: uptrend, downtrend, ranging'
    )
    trend_strength = Column(
        Numeric(5, 2),
        nullable=True,
        comment='Trend strength score (0-100, higher = stronger trend)'
    )
    trend_signals = Column(
        JSON,
        nullable=True,
        comment='JSON with trend signals: sma_alignment, adx_strength, price_position, etc.'
    )
    
    # Volatility Regime Classification
    volatility_regime = Column(
        String(20),
        nullable=False,
        comment='Volatility regime: high, normal, low'
    )
    volatility_percentile = Column(
        Numeric(5, 2),
        nullable=True,
        comment='Current volatility percentile vs 252-day history (0-100)'
    )
    volatility_signals = Column(
        JSON,
        nullable=True,
        comment='JSON with volatility signals: atr_percentile, bb_width, realized_vol, etc.'
    )
    
    # Market Phase Classification
    market_phase = Column(
        String(20),
        nullable=False,
        comment='Market cycle phase: accumulation, markup, distribution, markdown'
    )
    phase_confidence = Column(
        Numeric(5, 2),
        nullable=True,
        comment='Confidence in phase classification (0-100)'
    )
    phase_signals = Column(
        JSON,
        nullable=True,
        comment='JSON with phase signals: volume_trend, rsi_divergence, macd_signal, etc.'
    )
    
    # Support/Resistance Levels
    primary_support = Column(
        Numeric(10, 4),
        nullable=True,
        comment='Primary support level identified from technical analysis'
    )
    primary_resistance = Column(
        Numeric(10, 4),
        nullable=True,
        comment='Primary resistance level identified from technical analysis'
    )
    support_strength = Column(
        Numeric(5, 2),
        nullable=True,
        comment='Support level strength score (0-100, based on touch count and hold)'
    )
    resistance_strength = Column(
        Numeric(5, 2),
        nullable=True,
        comment='Resistance level strength score (0-100, based on touch count and rejection)'
    )
    
    # Regime Change Detection
    regime_change = Column(
        String(3),
        nullable=True,
        comment='Flag indicating if regime changed from previous day: yes/no'
    )
    days_in_regime = Column(
        BigInteger,
        nullable=True,
        comment='Number of consecutive days in current regime'
    )
    
    # Options Strategy Recommendation
    recommended_strategy = Column(
        String(50),
        nullable=True,
        comment='Recommended options strategy based on regime: iron_condor, bull_spread, straddle, etc.'
    )
    strategy_rationale = Column(
        JSON,
        nullable=True,
        comment='JSON with strategy rationale: why this strategy fits current regime'
    )
    
    # Timestamps
    calculated_at = Column(DateTime, server_default=func.now(), comment='When regime was calculated')
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('ticker', 'trade_date', name='uq_regime_ticker_date'),
        {'comment': 'Market regime classification for options strategy selection (Gold Layer)'}
    )
    
    def __repr__(self):
        return f"<FactMarketRegime(ticker={self.ticker}, date={self.trade_date}, trend={self.trend_regime}, vol={self.volatility_regime}, phase={self.market_phase})>"
