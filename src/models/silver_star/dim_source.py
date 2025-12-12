"""
dim_source: Dimension table for data sources.
Tracks where data comes from and source characteristics.
"""

from sqlalchemy import Column, Integer, String, Boolean, SmallInteger, DateTime, Index
from sqlalchemy.sql import func
from src.models.base import Base


class DimSource(Base):
    """
    Dimension table for data sources.
    
    Tracks data lineage and source characteristics:
    - Beursduivel: Has Greeks, intraday data, typically available at 16:30 UTC
    - FD (Financieele Dagblad): Has open interest, end-of-day data, typically available at 17:00 UTC
    - Manual: Human-entered corrections or enrichments
    
    Natural key: source_name
    Surrogate key: source_key (auto-increment)
    """
    
    __tablename__ = "dim_source"
    
    # Surrogate key
    source_key = Column(Integer, primary_key=True, autoincrement=True)
    
    # Natural key - source name
    source_name = Column(String(50), nullable=False, unique=True, index=True)
    
    # Source characteristics
    source_description = Column(String(500))
    has_greeks = Column(Boolean, nullable=False, default=False)
    has_open_interest = Column(Boolean, nullable=False, default=False)
    has_volume = Column(Boolean, nullable=False, default=True)
    has_intraday = Column(Boolean, nullable=False, default=False)
    
    # Typical availability time (minutes after market close)
    typical_latency_minutes = Column(SmallInteger, nullable=True)
    
    # Source URL pattern (for traceability)
    source_url_pattern = Column(String(500))
    
    # Audit fields
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    
    # Index on source_name for fast lookups
    __table_args__ = (
        Index("idx_dim_source_name", "source_name"),
    )
    
    def __repr__(self):
        return (
            f"<DimSource(source_key={self.source_key}, "
            f"name={self.source_name}, "
            f"greeks={self.has_greeks}, "
            f"oi={self.has_open_interest})>"
        )
