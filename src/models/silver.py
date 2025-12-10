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
    
    # Identifiers (composite primary key - managed by DBT)
    ticker = Column(String(20), nullable=False, primary_key=True)
    trade_date = Column(Date, nullable=False, primary_key=True)
    
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
    
    # Identifiers (composite primary key - managed by DBT)
    ticker = Column(String(20), nullable=False, primary_key=True)
    isin = Column(String(20), nullable=True, index=True)
    option_type = Column(String(10), nullable=False, primary_key=True)
    strike = Column(Numeric(10, 2), nullable=False, primary_key=True)
    expiry_date = Column(Date, nullable=False, primary_key=True)
    trade_date = Column(Date, nullable=False, primary_key=True)
    
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


class SilverOptionsChain(Base):
    """
    Silver layer: Options chain with merged BD (pricing) + FD (metrics) data.
    High-quality Greeks calculation with synchronized underlying price.
    """
    
    __tablename__ = "silver_options_chain"
    
    # A) Identity / time / lineage
    ticker = Column(String, primary_key=True, nullable=False)
    trade_date = Column(Date, primary_key=True, nullable=False, comment="Trading day this data represents")
    symbol_code = Column(String)
    as_of_ts = Column(DateTime, nullable=False, comment="Timestamp of data snapshot")
    as_of_date = Column(Date, nullable=False, comment="Date of data snapshot")
    source = Column(String, comment="Data source (e.g., 'beursduivel_primary_fd_secondary')")
    source_url = Column(String)
    scrape_run_id = Column(String)
    ingested_at = Column(DateTime)
    
    # B) Contract fields
    option_type = Column(String, primary_key=True, nullable=False, comment="Call or Put")
    expiry_date = Column(Date, primary_key=True, nullable=False)
    strike = Column(Float, primary_key=True, nullable=False)
    contract_key = Column(String, nullable=False, comment="ticker|expiry|strike|type")
    
    # C) Quote fields
    last_price = Column(Float)
    bid = Column(Float)
    ask = Column(Float)
    mid_price = Column(Float, comment="(bid+ask)/2")
    spread_abs = Column(Float, comment="ask - bid")
    spread_pct = Column(Float, comment="spread / mid_price")
    volume = Column(Integer, comment="Intraday volume")
    open_interest = Column(Integer, comment="Total open contracts")
    underlying_price = Column(Float, comment="Underlying stock price (synchronized)")
    
    # D) Quality helpers
    is_valid_quote = Column(Boolean)
    has_bd_data = Column(Boolean, comment="Has Beursduivel pricing")
    has_fd_data = Column(Boolean, comment="Has FD metrics")
    row_hash = Column(String)
    
    # E) Greeks
    iv = Column(Float, comment="Implied volatility (annual)")
    delta = Column(Float)
    gamma = Column(Float)
    vega = Column(Float)
    theta = Column(Float)
    rho = Column(Float)
    
    # F) Derived fields
    dte = Column(Integer, comment="Days to expiry")
    moneyness = Column(Float, comment="S/K ratio")
    is_itm = Column(Boolean, comment="In the money flag")
    
    # Metadata
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    
    def __repr__(self):
        return (
            f"<SilverOptionsChain(ticker={self.ticker}, type={self.option_type}, "
            f"strike={self.strike}, expiry={self.expiry_date}, iv={self.iv})>"
        )
