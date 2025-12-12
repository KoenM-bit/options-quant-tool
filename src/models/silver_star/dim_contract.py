"""
dim_option_contract: Dimension table for option contract definitions.
"""

from sqlalchemy import Column, String, Date, Integer, DateTime, Index, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.models.base import Base


class DimContract(Base):
    """
    Dimension table for option contracts.
    
    Natural key: ticker + expiration_date + strike + call_put
    Primary key: option_id (hash of natural key)
    """
    
    __tablename__ = "dim_option_contract"
    
    # Primary key - hash of (ticker, expiry, strike, call_put)
    option_id = Column(String(64), primary_key=True)  # MD5/SHA hash
    
    # Foreign key to underlying
    underlying_id = Column(String(64), ForeignKey("dim_underlying.underlying_id"), nullable=False, index=True)
    
    # Natural key components
    ticker = Column(String(20), nullable=False, index=True)
    expiration_date = Column(Date, nullable=False)
    strike = Column(Integer, nullable=False)
    call_put = Column(String(1), nullable=False)  # 'C' or 'P'
    
    # Contract specifications
    contract_size = Column(Integer, default=100)  # Standard size
    style = Column(String(20), default="European")  # European or American
    
    # Source identifiers (for traceability)
    symbol_code = Column(String(50))  # Beursduivel symbol code
    issue_id = Column(String(50))     # Beursduivel issue ID
    isin = Column(String(20))          # ISIN code
    
    # Relationship
    underlying = relationship("DimUnderlying")
    
    # Audit fields
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index("idx_dim_contract_underlying", "underlying_id"),
        Index("idx_dim_contract_natural_key", "ticker", "expiration_date", "strike", "call_put"),
        Index("idx_dim_contract_expiry", "expiration_date"),
    )
    
    def __repr__(self):
        return (
            f"<DimContract(option_id={self.option_id}, "
            f"ticker={self.ticker}, type={self.call_put}, "
            f"strike={self.strike}, expiry={self.expiration_date})>"
        )
