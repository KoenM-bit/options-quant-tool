"""
Fact Technical Indicators Model

Silver layer table containing calculated technical indicators from OHLCV data.
Includes trend, momentum, volatility, and volume-based indicators.
"""

from sqlalchemy import Column, BigInteger, String, Date, Numeric, DateTime, Index, ForeignKey
from sqlalchemy.sql import func
from src.models.base import Base


class FactTechnicalIndicators(Base):
    """
    Technical indicators calculated from daily OHLCV data.
    
    Includes:
    - Trend: SMA (20, 50, 200), EMA (12, 26), MACD
    - Momentum: RSI (14), Stochastic, ROC
    - Volatility: ATR (14), Bollinger Bands, Historical Volatility
    - Support/Resistance: 20-day high/low, 52-week high/low
    - Volume: Volume SMA, Volume ratio
    """
    
    __tablename__ = 'fact_technical_indicators'
    
    # Primary key
    indicator_id = Column(BigInteger, primary_key=True, autoincrement=True)
    
    # Natural key
    ticker = Column(String(20), nullable=False)
    trade_date = Column(Date, nullable=False)
    
    # Price reference (for convenience)
    close = Column(Numeric(10, 4))
    volume = Column(BigInteger)
    
    # Trend Indicators - Simple Moving Averages
    sma_20 = Column(Numeric(10, 4), comment='20-day Simple Moving Average')
    sma_50 = Column(Numeric(10, 4), comment='50-day Simple Moving Average')
    sma_200 = Column(Numeric(10, 4), comment='200-day Simple Moving Average')
    
    # Trend Indicators - Exponential Moving Averages
    ema_12 = Column(Numeric(10, 4), comment='12-day Exponential Moving Average')
    ema_26 = Column(Numeric(10, 4), comment='26-day Exponential Moving Average')
    
    # MACD (Moving Average Convergence Divergence)
    macd = Column(Numeric(10, 4), comment='MACD Line (EMA12 - EMA26)')
    macd_signal = Column(Numeric(10, 4), comment='MACD Signal Line (9-day EMA of MACD)')
    macd_histogram = Column(Numeric(10, 4), comment='MACD Histogram (MACD - Signal)')
    
    # Momentum Indicators
    rsi_14 = Column(Numeric(10, 4), comment='14-day Relative Strength Index')
    stochastic_k = Column(Numeric(10, 4), comment='Stochastic %K')
    stochastic_d = Column(Numeric(10, 4), comment='Stochastic %D (3-day SMA of %K)')
    roc_20 = Column(Numeric(10, 4), comment='20-day Rate of Change (%)')
    
    # Volatility Indicators
    atr_14 = Column(Numeric(10, 4), comment='14-day Average True Range')
    bollinger_upper = Column(Numeric(10, 4), comment='Bollinger Band Upper (SMA20 + 2*std)')
    bollinger_middle = Column(Numeric(10, 4), comment='Bollinger Band Middle (SMA20)')
    bollinger_lower = Column(Numeric(10, 4), comment='Bollinger Band Lower (SMA20 - 2*std)')
    bollinger_width = Column(Numeric(10, 4), comment='Bollinger Band Width (upper - lower)')
    
    # Historical Volatility
    realized_volatility_20 = Column(Numeric(10, 4), comment='20-day Realized Volatility (annualized %)')
    parkinson_volatility_20 = Column(Numeric(10, 4), comment='20-day Parkinson Volatility (annualized %)')
    
    # Support and Resistance Levels
    high_20d = Column(Numeric(10, 4), comment='20-day High')
    low_20d = Column(Numeric(10, 4), comment='20-day Low')
    high_52w = Column(Numeric(10, 4), comment='52-week (252-day) High')
    low_52w = Column(Numeric(10, 4), comment='52-week (252-day) Low')
    
    # Distance from highs/lows
    pct_from_high_20d = Column(Numeric(10, 4), comment='% below 20-day high')
    pct_from_low_20d = Column(Numeric(10, 4), comment='% above 20-day low')
    pct_from_high_52w = Column(Numeric(10, 4), comment='% below 52-week high')
    pct_from_low_52w = Column(Numeric(10, 4), comment='% above 52-week low')
    
    # Volume Indicators
    volume_sma_20 = Column(BigInteger, comment='20-day Volume SMA')
    volume_ratio = Column(Numeric(10, 4), comment='Current Volume / 20-day Avg Volume')
    obv = Column(BigInteger, comment='On-Balance Volume (cumulative)')
    obv_sma_20 = Column(BigInteger, comment='20-day SMA of OBV')
    
    # ADX (Average Directional Index) - Trend Strength
    adx_14 = Column(Numeric(10, 4), comment='14-day Average Directional Index')
    plus_di_14 = Column(Numeric(10, 4), comment='14-day Plus Directional Indicator')
    minus_di_14 = Column(Numeric(10, 4), comment='14-day Minus Directional Indicator')
    
    # Audit columns
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    calculated_at = Column(DateTime, nullable=False, server_default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_fact_ti_ticker_date', 'ticker', 'trade_date', unique=True),
        Index('idx_fact_ti_ticker', 'ticker'),
        Index('idx_fact_ti_date', 'trade_date'),
        Index('idx_fact_ti_calculated', 'calculated_at'),
    )
    
    def __repr__(self):
        return (
            f"<FactTechnicalIndicators("
            f"ticker={self.ticker}, "
            f"date={self.trade_date}, "
            f"close={self.close}, "
            f"sma_20={self.sma_20}, "
            f"rsi={self.rsi_14}"
            f")>"
        )
