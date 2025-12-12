"""
fact_option_timeseries: Fact table for option time series data.
"""

from sqlalchemy import Column, String, BigInteger, Numeric, Date, DateTime, Integer, Index, ForeignKey, TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.models.base import Base


class FactOptionTimeseries(Base):
    """
    Fact table for option time series data.
    
    Grain: One row per option per trade date
    """
    
    __tablename__ = "fact_option_timeseries"
    
    # Surrogate key
    ts_id = Column(BigInteger, primary_key=True, autoincrement=True)
    
    # Time dimension
    trade_date = Column(Date, nullable=False, index=True)
    ts = Column(TIMESTAMP, nullable=False, server_default=func.now())
    
    # Foreign keys to dimensions
    option_id = Column(String(64), ForeignKey("dim_option_contract.option_id"), nullable=False, index=True)
    underlying_id = Column(String(64), ForeignKey("dim_underlying.underlying_id"), nullable=False, index=True)
    
    # Relationships
    option_contract = relationship("DimContract")
    underlying = relationship("DimUnderlying")
    
    # === UNDERLYING PRICE ===
    underlying_price = Column(Numeric(10, 4))
    underlying_bid = Column(Numeric(10, 4))
    underlying_ask = Column(Numeric(10, 4))
    
    # === OPTION PRICING ===
    bid = Column(Numeric(10, 4))
    ask = Column(Numeric(10, 4))
    mid_price = Column(Numeric(10, 4))
    last_price = Column(Numeric(10, 4))
    
    # === GREEKS ===
    iv = Column(Numeric(10, 6))  # Implied volatility
    delta = Column(Numeric(10, 6))
    gamma = Column(Numeric(10, 6))
    vega = Column(Numeric(10, 6))
    theta = Column(Numeric(10, 6))
    rho = Column(Numeric(10, 6))
    
    # === MARKET ACTIVITY ===
    volume = Column(Integer)
    open_interest = Column(Integer)
    
    # === DERIVED MEASURES ===
    intrinsic_value = Column(Numeric(10, 4))
    time_value = Column(Numeric(10, 4))
    moneyness = Column(Numeric(10, 6))
    days_to_expiry = Column(Integer)
    
    # === SOURCE ===
    source = Column(String(50), default="beursduivel")
    
    # Audit
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    
    # Indexes
    __table_args__ = (
        Index("idx_fact_ts_date", "trade_date"),
        Index("idx_fact_ts_option", "option_id", "trade_date"),
        Index("idx_fact_ts_underlying", "underlying_id", "trade_date"),
        Index("idx_fact_ts_date_option", "trade_date", "option_id"),
    )
    
    def __repr__(self):
        return (
            f"<FactOptionTimeseries(ts_id={self.ts_id}, "
            f"option_id={self.option_id}, "
            f"trade_date={self.trade_date}, "
            f"mid_price={self.mid_price})>"
        )
