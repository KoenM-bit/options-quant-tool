"""
Bronze layer model for Euronext options data.
Stores raw scraped data from live.euronext.com including open interest.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Index, Text
from datetime import datetime

from src.models.base import Base, TimestampMixin


class BronzeEuronextOptions(Base, TimestampMixin):
    """
    Raw Euronext options data including open interest.
    Data scraped from live.euronext.com individual option pages.
    """
    __tablename__ = "bronze_euronext_options"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Identifiers
    ticker = Column(String(20), nullable=False, index=True, comment="Ticker symbol (e.g., AH-DAMS)")
    option_type = Column(String(10), nullable=False, index=True, comment="C or P")
    strike = Column(Float, nullable=False, comment="Strike price in euros")
    expiration_date = Column(String(20), nullable=False, index=True, comment="Expiration date from URL parameter")
    
    # Timestamps
    scraped_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True, comment="When data was scraped")
    trade_date = Column(Date, nullable=True, index=True, comment="Trading date (derived from scraped_at)")
    
    # Source
    source_url = Column(Text, nullable=True, comment="Full Euronext URL")
    
    # Contract specifications
    isin = Column(String(20), nullable=True, index=True, comment="ISIN code (NLEX...)")
    amr_code = Column(String(50), nullable=True, comment="AMR code")
    external_instrument_id = Column(String(50), nullable=True, comment="External instrument ID")
    contract_size = Column(Integer, nullable=True, comment="Contract multiplier (usually 100)")
    expiry_cycle = Column(String(50), nullable=True, comment="Expiry cycle (e.g., Maandelijks)")
    actual_expiration_date = Column(Date, nullable=True, comment="Actual expiration date from characteristics")
    
    # Open Interest - PRIMARY FIELD
    open_interest = Column(Integer, nullable=True, index=True, comment="Open interest")
    open_interest_date = Column(Date, nullable=True, index=True, comment="Date of open interest data")
    
    # Volume and transactions
    volume = Column(Integer, nullable=True, comment="Daily volume")
    volume_date = Column(Date, nullable=True, comment="Date of volume data")
    num_transactions = Column(Integer, nullable=True, comment="Number of transactions")
    quantity_on_exchange = Column(Integer, nullable=True, comment="Quantity traded on exchange")
    quantity_off_exchange = Column(Integer, nullable=True, comment="Quantity traded off exchange")
    
    # Option prices (in euros)
    settlement_price = Column(Float, nullable=True, comment="Settlement price")
    settlement_date = Column(Date, nullable=True, comment="Settlement date")
    opening_price = Column(Float, nullable=True, comment="Opening price")
    day_high = Column(Float, nullable=True, comment="Day high")
    day_low = Column(Float, nullable=True, comment="Day low")
    week_high = Column(Float, nullable=True, comment="Week high")
    week_low = Column(Float, nullable=True, comment="Week low")
    
    # Underlying stock data
    underlying_name = Column(String(100), nullable=True, comment="Underlying stock name")
    underlying_last_price = Column(Float, nullable=True, comment="Underlying last price")
    
    # Indexes for common queries
    __table_args__ = (
        Index("ix_bronze_euronext_ticker_strike_exp", "ticker", "strike", "expiration_date"),
        Index("ix_bronze_euronext_type_exp", "option_type", "expiration_date"),
        Index("ix_bronze_euronext_isin", "isin"),
        Index("ix_bronze_euronext_oi_date", "open_interest_date"),
        Index("ix_bronze_euronext_scraped", "scraped_at"),
        # Unique constraint to prevent duplicate scrapes
        Index("ix_bronze_euronext_unique", "ticker", "option_type", "strike", "expiration_date", "scraped_at", unique=True),
    )
    
    def __repr__(self):
        return (f"<BronzeEuronextOptions(ticker={self.ticker}, {self.option_type} "
                f"Strike={self.strike} Exp={self.expiration_date} OI={self.open_interest})>")
