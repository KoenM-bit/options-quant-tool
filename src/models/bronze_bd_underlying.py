"""
SQLAlchemy model for Beursduivel underlying stock data (bronze layer).
"""

from sqlalchemy import Column, String, Float, Integer, DateTime, func
from src.models.base import Base


class BronzeBDUnderlying(Base):
    """
    Bronze layer table for underlying stock price data from Beursduivel.
    Synchronized with the options data scrape.
    """
    
    __tablename__ = "bronze_bd_underlying"
    
    # Identity
    ticker = Column(String, primary_key=True, nullable=False, comment="Stock ticker (e.g. AD.AS)")
    trade_date = Column(DateTime, primary_key=True, nullable=False, comment="Trading day this data represents")
    isin = Column(String, comment="ISIN code (e.g. NL0011794037)")
    name = Column(String, comment="Stock name")
    
    # Pricing
    last_price = Column(Float, comment="Last traded price")
    bid = Column(Float, comment="Current bid price")
    ask = Column(Float, comment="Current ask price")
    
    # Volume
    volume = Column(Integer, comment="Daily volume")
    
    # Timestamps
    last_timestamp_text = Column(String, comment="Timestamp text from website (e.g. '10 dec 2025 13:10')")
    scraped_at = Column(DateTime, primary_key=True, nullable=False, comment="When this data was scraped")
    
    # Lineage
    source_url = Column(String, comment="Source URL")
    
    # Metadata
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
