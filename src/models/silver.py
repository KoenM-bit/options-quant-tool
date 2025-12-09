"""
Silver layer models - Cleaned and validated data.
Ready for analytics with calculated fields.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Index, Boolean, Numeric
from decimal import Decimal

from src.models.base import Base, TimestampMixin


class SilverUnderlyingPrice(Base, TimestampMixin):
    """
    Cleaned underlying price history.
    One record per trading day.
    """
    __tablename__ = "silver_underlying_price"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Identifiers
    ticker = Column(String(20), nullable=False, index=True)
    trade_date = Column(Date, nullable=False, index=True)
    
    # Price data
    open_price = Column(Numeric(10, 2), nullable=True)
    high_price = Column(Numeric(10, 2), nullable=True)
    low_price = Column(Numeric(10, 2), nullable=True)
    close_price = Column(Numeric(10, 2), nullable=False)
    volume = Column(Integer, nullable=True)
    
    # Calculated fields
    daily_return = Column(Float, nullable=True)
    daily_return_pct = Column(Float, nullable=True)
    
    # Data quality
    is_validated = Column(Boolean, default=False)
    source_id = Column(Integer, nullable=True)  # Reference to bronze table
    
    __table_args__ = (
        Index("ix_silver_underlying_unique", "ticker", "trade_date", unique=True),
    )
    
    def __repr__(self):
        return f"<SilverUnderlyingPrice(ticker={self.ticker}, date={self.trade_date}, close={self.close_price})>"


class SilverOption(Base, TimestampMixin):
    """
    Cleaned options data with calculated Greeks and metrics.
    """
    __tablename__ = "silver_options"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Identifiers
    ticker = Column(String(20), nullable=False, index=True)
    isin = Column(String(20), nullable=True, index=True)
    option_type = Column(String(10), nullable=False, index=True)
    strike = Column(Numeric(10, 2), nullable=False)
    expiry_date = Column(Date, nullable=False, index=True)
    trade_date = Column(Date, nullable=False, index=True)
    
    # Price data
    bid = Column(Numeric(10, 4), nullable=True)
    ask = Column(Numeric(10, 4), nullable=True)
    mid_price = Column(Numeric(10, 4), nullable=True)
    last_price = Column(Numeric(10, 4), nullable=True)
    
    # Volume and OI
    volume = Column(Integer, nullable=True)
    open_interest = Column(Integer, nullable=True)
    oi_change = Column(Integer, nullable=True)
    
    # Underlying
    underlying_price = Column(Numeric(10, 2), nullable=True)
    
    # Calculated metrics
    moneyness = Column(Float, nullable=True)  # S/K for calls, K/S for puts
    intrinsic_value = Column(Numeric(10, 4), nullable=True)
    time_value = Column(Numeric(10, 4), nullable=True)
    days_to_expiry = Column(Integer, nullable=True)
    
    # Greeks
    delta = Column(Float, nullable=True)
    gamma = Column(Float, nullable=True)
    theta = Column(Float, nullable=True)
    vega = Column(Float, nullable=True)
    rho = Column(Float, nullable=True)
    implied_volatility = Column(Float, nullable=True)

    # Greeks metadata
    risk_free_rate_used = Column(Float, nullable=True)
    greeks_valid = Column(Boolean, default=False, nullable=False)
    greeks_status = Column(String(50), nullable=True)
    
    # Spread metrics
    bid_ask_spread = Column(Numeric(10, 4), nullable=True)
    bid_ask_spread_pct = Column(Float, nullable=True)
    
    # Data quality
    is_validated = Column(Boolean, default=False)
    has_volume = Column(Boolean, default=False)
    is_liquid = Column(Boolean, default=False)
    source_id = Column(Integer, nullable=True)
    
    __table_args__ = (
        Index("ix_silver_options_ticker_date", "ticker", "trade_date"),
        Index("ix_silver_options_expiry_strike", "expiry_date", "strike"),
        Index("ix_silver_options_moneyness", "moneyness"),
    )
    
    def __repr__(self):
        return (
            f"<SilverOption(ticker={self.ticker}, type={self.option_type}, "
            f"strike={self.strike}, expiry={self.expiry_date})>"
        )
