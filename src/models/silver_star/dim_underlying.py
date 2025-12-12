"""
dim_underlying: Dimension table for underlying assets.
"""

from sqlalchemy import Column, String, DateTime, Index
from sqlalchemy.sql import func
from src.models.base import Base


class DimUnderlying(Base):
    """
    Dimension table for underlying assets.
    
    Natural key: ticker
    Primary key: underlying_id (hash of ticker)
    """
    
    __tablename__ = "dim_underlying"
    
    # Primary key - hash of ticker
    underlying_id = Column(String(64), primary_key=True)  # MD5/SHA hash of ticker
    
    # Natural key
    ticker = Column(String(20), nullable=False, unique=True, index=True)
    
    # Asset metadata
    name = Column(String(200))
    asset_class = Column(String(50), default="Stock")
    sector = Column(String(100))
    exchange = Column(String(50), default="Euronext Amsterdam")
    currency = Column(String(3), default="EUR")
    isin = Column(String(20))
    
    # Audit fields
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index("idx_dim_underlying_ticker", "ticker"),
    )
    
    def __repr__(self):
        return (
            f"<DimUnderlying(underlying_id={self.underlying_id}, "
            f"ticker={self.ticker}, "
            f"name={self.name})>"
        )
