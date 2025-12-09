"""
Gold layer models - Business-ready aggregations and metrics.
Optimized for reporting and analytics.
"""

from sqlalchemy import Column, Integer, String, Float, Date, Index, Numeric, Text
from datetime import date

from src.models.base import Base, TimestampMixin


class GoldOptionsSummaryDaily(Base, TimestampMixin):
    """
    Daily summary of options market activity.
    Used for high-level dashboards.
    """
    __tablename__ = "gold_options_summary_daily"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Identifiers
    ticker = Column(String(20), nullable=False, index=True)
    trade_date = Column(Date, nullable=False, index=True)
    
    # Underlying
    underlying_close = Column(Numeric(10, 2), nullable=False)
    underlying_return_pct = Column(Float, nullable=True)
    
    # Volume metrics
    total_volume = Column(Integer, nullable=True)
    call_volume = Column(Integer, nullable=True)
    put_volume = Column(Integer, nullable=True)
    call_put_volume_ratio = Column(Float, nullable=True)
    
    # Open interest metrics
    total_oi = Column(Integer, nullable=True)
    call_oi = Column(Integer, nullable=True)
    put_oi = Column(Integer, nullable=True)
    call_put_oi_ratio = Column(Float, nullable=True)
    
    # OI changes
    total_oi_change = Column(Integer, nullable=True)
    call_oi_change = Column(Integer, nullable=True)
    put_oi_change = Column(Integer, nullable=True)
    
    # Strike distribution
    atm_strike = Column(Numeric(10, 2), nullable=True)
    min_strike = Column(Numeric(10, 2), nullable=True)
    max_strike = Column(Numeric(10, 2), nullable=True)
    num_strikes = Column(Integer, nullable=True)
    
    # Volatility metrics
    avg_implied_vol = Column(Float, nullable=True)
    atm_implied_vol = Column(Float, nullable=True)
    vol_skew = Column(Float, nullable=True)
    
    # Greek exposures
    total_call_delta = Column(Float, nullable=True)
    total_put_delta = Column(Float, nullable=True)
    net_delta = Column(Float, nullable=True)
    total_gamma = Column(Float, nullable=True)
    total_vega = Column(Float, nullable=True)
    
    __table_args__ = (
        Index("ix_gold_summary_unique", "ticker", "trade_date", unique=True),
    )
    
    def __repr__(self):
        return f"<GoldOptionsSummaryDaily(ticker={self.ticker}, date={self.trade_date})>"


class GoldVolatilitySurface(Base, TimestampMixin):
    """
    Implied volatility surface by strike and expiry.
    Used for volatility analysis and visualization.
    """
    __tablename__ = "gold_volatility_surface"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Identifiers
    ticker = Column(String(20), nullable=False, index=True)
    trade_date = Column(Date, nullable=False, index=True)
    expiry_date = Column(Date, nullable=False, index=True)
    strike = Column(Numeric(10, 2), nullable=False)
    
    # Metrics
    days_to_expiry = Column(Integer, nullable=False)
    moneyness = Column(Float, nullable=False, index=True)
    
    # Volatility
    call_iv = Column(Float, nullable=True)
    put_iv = Column(Float, nullable=True)
    avg_iv = Column(Float, nullable=True)
    
    # Volume and OI
    call_volume = Column(Integer, nullable=True)
    put_volume = Column(Integer, nullable=True)
    call_oi = Column(Integer, nullable=True)
    put_oi = Column(Integer, nullable=True)
    
    # Price data
    underlying_price = Column(Numeric(10, 2), nullable=True)
    call_mid = Column(Numeric(10, 4), nullable=True)
    put_mid = Column(Numeric(10, 4), nullable=True)
    
    __table_args__ = (
        Index("ix_gold_vol_surface_key", "ticker", "trade_date", "expiry_date", "strike"),
        Index("ix_gold_vol_surface_moneyness", "moneyness"),
    )
    
    def __repr__(self):
        return (
            f"<GoldVolatilitySurface(ticker={self.ticker}, strike={self.strike}, "
            f"expiry={self.expiry_date}, iv={self.avg_iv})>"
        )


class GoldGreekAnalytics(Base, TimestampMixin):
    """
    Greek exposures and risk metrics aggregated by expiry.
    Used for risk management dashboards.
    """
    __tablename__ = "gold_greek_analytics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Identifiers
    ticker = Column(String(20), nullable=False, index=True)
    trade_date = Column(Date, nullable=False, index=True)
    expiry_date = Column(Date, nullable=False, index=True)
    
    # Expiry details
    days_to_expiry = Column(Integer, nullable=False)
    
    # Delta exposure
    call_delta_total = Column(Float, nullable=True)
    put_delta_total = Column(Float, nullable=True)
    net_delta = Column(Float, nullable=True)
    
    # Gamma exposure
    call_gamma_total = Column(Float, nullable=True)
    put_gamma_total = Column(Float, nullable=True)
    net_gamma = Column(Float, nullable=True)
    
    # Vega exposure
    call_vega_total = Column(Float, nullable=True)
    put_vega_total = Column(Float, nullable=True)
    net_vega = Column(Float, nullable=True)
    
    # Theta exposure
    call_theta_total = Column(Float, nullable=True)
    put_theta_total = Column(Float, nullable=True)
    net_theta = Column(Float, nullable=True)
    
    # Open interest
    call_oi = Column(Integer, nullable=True)
    put_oi = Column(Integer, nullable=True)
    total_oi = Column(Integer, nullable=True)
    
    __table_args__ = (
        Index("ix_gold_greeks_unique", "ticker", "trade_date", "expiry_date", unique=True),
    )
    
    def __repr__(self):
        return (
            f"<GoldGreekAnalytics(ticker={self.ticker}, expiry={self.expiry_date}, "
            f"net_delta={self.net_delta})>"
        )


class GoldOpenInterestFlow(Base, TimestampMixin):
    """
    Open interest flow analysis by strike.
    Tracks changes in positioning over time.
    """
    __tablename__ = "gold_open_interest_flow"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Identifiers
    ticker = Column(String(20), nullable=False, index=True)
    trade_date = Column(Date, nullable=False, index=True)
    strike = Column(Numeric(10, 2), nullable=False, index=True)
    
    # Aggregated across all expiries for this strike
    call_oi = Column(Integer, nullable=True)
    put_oi = Column(Integer, nullable=True)
    total_oi = Column(Integer, nullable=True)
    
    # Daily changes
    call_oi_change = Column(Integer, nullable=True)
    put_oi_change = Column(Integer, nullable=True)
    total_oi_change = Column(Integer, nullable=True)
    
    # Volume
    call_volume = Column(Integer, nullable=True)
    put_volume = Column(Integer, nullable=True)
    total_volume = Column(Integer, nullable=True)
    
    # Metrics
    call_put_ratio = Column(Float, nullable=True)
    underlying_price = Column(Numeric(10, 2), nullable=True)
    distance_from_price = Column(Numeric(10, 2), nullable=True)
    distance_pct = Column(Float, nullable=True)
    
    # Classification
    strike_type = Column(String(20), nullable=True)  # 'ITM', 'ATM', 'OTM'
    
    __table_args__ = (
        Index("ix_gold_oi_flow_unique", "ticker", "trade_date", "strike"),
        Index("ix_gold_oi_flow_strike_type", "strike_type"),
    )
    
    def __repr__(self):
        return (
            f"<GoldOpenInterestFlow(ticker={self.ticker}, date={self.trade_date}, "
            f"strike={self.strike}, oi={self.total_oi})>"
        )
