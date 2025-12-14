"""
Fact table for end-of-day option settlement data.
Source: FD.nl (scraped next day, represents previous trading day)
"""

from sqlalchemy import Column, String, Date, Integer, TIMESTAMP, Index, BigInteger, ForeignKey
from sqlalchemy.dialects.postgresql import NUMERIC as Numeric
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from src.models.base import Base


class FactOptionEOD(Base):
    """
    Fact table for end-of-day option settlement data from FD.nl
    
    Grain: One row per option per trade date
    Source: bronze_fd_options (scraped on T+1, represents T data)
    
    Contains:
    - Official closing prices
    - End-of-day volume
    - Open interest (critical for analysis!)
    
    Different from fact_option_timeseries (intraday BD data):
    - This is SETTLEMENT data (T-1 closing)
    - Has open interest
    - Scraped AFTER market close
    """
    
    __tablename__ = "fact_option_eod"
    
    # Surrogate key
    eod_id = Column(BigInteger, primary_key=True, autoincrement=True)
    
    # Time dimension
    trade_date = Column(Date, nullable=False, index=True, comment="Trading day this data represents")
    ts = Column(TIMESTAMP, nullable=False, server_default=func.now())
    
    # Foreign keys to dimensions
    option_id = Column(String(64), ForeignKey("dim_option_contract.option_id"), nullable=False, index=True)
    underlying_id = Column(String(64), ForeignKey("dim_underlying.underlying_id"), nullable=False, index=True)
    
    # Relationships
    option_contract = relationship("DimContract")
    underlying = relationship("DimUnderlying")
    
    # === EOD SETTLEMENT PRICE ===
    last_price = Column(Numeric(10, 4), comment="Official settlement/closing price")
    
    # === MARKET ACTIVITY (OFFICIAL EOD) ===
    volume = Column(Integer, comment="Total daily volume")
    open_interest = Column(Integer, comment="Open interest at close (CRITICAL!)")
    
    # === SOURCE ===
    source = Column(String(50), default="fd.nl")
    
    # Audit
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
    
    # Indexes
    __table_args__ = (
        Index("idx_fact_eod_date", "trade_date"),
        Index("idx_fact_eod_option", "option_id", "trade_date"),
        Index("idx_fact_eod_underlying", "underlying_id", "trade_date"),
        Index("idx_fact_eod_date_option", "trade_date", "option_id"),
    )
    
    def __repr__(self):
        return (
            f"<FactOptionEOD("
            f"option_id={self.option_id}, "
            f"trade_date={self.trade_date}, "
            f"volume={self.volume}, "
            f"oi={self.open_interest}"
            f")>"
        )
