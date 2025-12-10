"""
Database models for Beursduivel bronze layer
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Text, func, UniqueConstraint
from src.models.base import Base


class BronzeBDOptions(Base):
    """
    Bronze layer table for Beursduivel options data.
    Stores raw scraped data with live bid/ask/last prices.
    
    Unique Constraint: Prevents duplicate scrapes of the same contract on the same trade_date.
    This protects against holiday scraping (where scrapers might fetch stale data).
    """
    __tablename__ = 'bronze_bd_options'
    
    __table_args__ = (
        UniqueConstraint(
            'ticker', 'trade_date', 'option_type', 'strike', 'expiry_date',
            name='uq_bronze_bd_options_contract_date'
        ),
    )
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Identity
    ticker = Column(String(20), nullable=False, index=True)
    symbol_code = Column(String(20), nullable=False)  # e.g., 'AH', 'AH9', '2AH'
    issue_id = Column(String(50), nullable=False)  # Beursduivel internal ID
    trade_date = Column(Date, nullable=False, index=True, comment="Trading day this data represents")
    
    # Contract details
    option_type = Column(String(10), nullable=False)  # 'Call' or 'Put'
    expiry_date = Column(Date, nullable=False, index=True)
    expiry_text = Column(String(100))  # Original expiry text from website
    strike = Column(Float, nullable=False)
    
    # Pricing data (from overview page - very high coverage)
    bid = Column(Float)
    ask = Column(Float)
    
    # Live data (from detail pages - requires extra requests)
    last_price = Column(Float)  # Most recent trade price
    volume = Column(Integer)  # Intraday volume for this contract
    last_timestamp = Column(DateTime)  # When last_price was updated
    last_date_text = Column(String(50))  # Original timestamp text
    
    # Source metadata
    source = Column(String(50), default='beursduivel')
    source_url = Column(Text)  # Link to detail page
    scraped_at = Column(DateTime, nullable=False, index=True)
    
    # Audit
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return (f"<BronzeBDOptions(ticker={self.ticker}, type={self.option_type}, "
                f"strike={self.strike}, expiry={self.expiry_date}, bid={self.bid}, ask={self.ask})>")
