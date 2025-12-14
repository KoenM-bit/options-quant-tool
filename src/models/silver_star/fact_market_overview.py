"""
Fact table for daily market overview data.
Stores aggregate volume and open interest metrics per ticker per day.
"""

from sqlalchemy import Column, String, Date, Integer, Float, TIMESTAMP, Index, BigInteger, ForeignKey
from sqlalchemy.dialects.postgresql import NUMERIC as Numeric
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from src.models.base import Base


class FactMarketOverview(Base):
    """
    Fact table for daily market overview metrics.
    
    Grain: One row per ticker per trade date
    Source: bronze_fd_overview (FD.nl overview page)
    
    Contains:
    - Total volume (calls + puts)
    - Total open interest (calls + puts) 
    - Call/Put ratios
    - Underlying price at close
    """
    
    __tablename__ = "fact_market_overview"
    
    # Surrogate key
    overview_id = Column(BigInteger, primary_key=True, autoincrement=True)
    
    # Time dimension
    trade_date = Column(Date, nullable=False, index=True)
    
    # Foreign key to underlying
    underlying_id = Column(String(64), ForeignKey("dim_underlying.underlying_id"), nullable=False, index=True)
    
    # Relationship
    underlying = relationship("DimUnderlying")
    
    # === UNDERLYING PRICE AT CLOSE ===
    underlying_price = Column(Numeric(10, 4), comment="Closing price of underlying")
    underlying_open = Column(Numeric(10, 4), comment="Opening price")
    underlying_high = Column(Numeric(10, 4), comment="Daily high")
    underlying_low = Column(Numeric(10, 4), comment="Daily low")
    underlying_volume = Column(Integer, comment="Underlying stock volume")
    underlying_change = Column(Numeric(10, 4), comment="Price change vs previous day")
    underlying_change_pct = Column(Float, comment="% change vs previous day")
    
    # === TOTAL VOLUME ===
    total_volume = Column(Integer, comment="Total options volume (calls + puts)")
    total_call_volume = Column(Integer, comment="Total call options volume")
    total_put_volume = Column(Integer, comment="Total put options volume")
    
    # === TOTAL OPEN INTEREST ===
    total_oi = Column(Integer, comment="Total open interest (calls + puts)")
    total_call_oi = Column(Integer, comment="Total call open interest")
    total_put_oi = Column(Integer, comment="Total put open interest")
    
    # === RATIOS ===
    call_put_volume_ratio = Column(Float, comment="Call volume / Put volume")
    call_put_oi_ratio = Column(Float, comment="Call OI / Put OI")
    
    # === METADATA ===
    market_time = Column(String(50), comment="Market timestamp from source")
    source = Column(String(50), default="fd.nl", comment="Data source")
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
    
    # Indexes
    __table_args__ = (
        Index("idx_fact_overview_date_underlying", "trade_date", "underlying_id"),
        Index("idx_fact_overview_underlying_date", "underlying_id", "trade_date"),
    )
    
    def __repr__(self):
        return (
            f"<FactMarketOverview("
            f"underlying_id={self.underlying_id}, "
            f"trade_date={self.trade_date}, "
            f"total_volume={self.total_volume}, "
            f"total_oi={self.total_oi}"
            f")>"
        )
