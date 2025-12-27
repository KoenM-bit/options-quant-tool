"""
Bronze OHLCV Intraday Model - Raw hourly stock price data
Source: Yahoo Finance (yfinance)
"""

from sqlalchemy import Column, String, DateTime, Integer, TIMESTAMP, Index, BigInteger, UniqueConstraint, Boolean
from sqlalchemy.dialects.postgresql import NUMERIC as Numeric
from sqlalchemy.sql import func
from src.models.base import Base


class BronzeOHLCVIntraday(Base):
    """
    Bronze table for raw intraday OHLCV (Open, High, Low, Close, Volume) stock data.
    
    Grain: One row per ticker per timestamp (hourly)
    Source: Yahoo Finance via yfinance API
    Updates: Can be fetched multiple times per day for recent data
    """
    
    __tablename__ = "bronze_ohlcv_intraday"
    
    # Primary key
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    
    # Natural key
    ticker = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True, comment="Trading timestamp (hourly)")
    
    # Market/Exchange identifier
    market = Column(String(10), index=True, comment="Market identifier (e.g., NL, US, UK)")
    
    # OHLCV data
    open = Column(Numeric(10, 4), nullable=False, comment="Opening price for the hour")
    high = Column(Numeric(10, 4), nullable=False, comment="Highest price during the hour")
    low = Column(Numeric(10, 4), nullable=False, comment="Lowest price during the hour")
    close = Column(Numeric(10, 4), nullable=False, comment="Closing price for the hour")
    volume = Column(BigInteger, comment="Trading volume during the hour")
    
    # Metadata
    source = Column(String(50), default="yahoo_finance")
    scraped_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
    is_training_data = Column(Boolean, default=True, nullable=False, 
                             comment="Whether this data should be used for training (vs validation/test)")
    
    # Audit
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP, nullable=False, server_default=func.now(), onupdate=func.now())
    
    # Indexes and constraints
    __table_args__ = (
        UniqueConstraint('ticker', 'timestamp', name='uq_bronze_ohlcv_intraday_ticker_timestamp'),
        Index('idx_bronze_ohlcv_intraday_ticker', 'ticker'),
        Index('idx_bronze_ohlcv_intraday_timestamp', 'timestamp'),
        Index('idx_bronze_ohlcv_intraday_ticker_timestamp', 'ticker', 'timestamp'),
    )
    
    def __repr__(self):
        return (
            f"<BronzeOHLCVIntraday("
            f"ticker={self.ticker}, "
            f"timestamp={self.timestamp}, "
            f"close={self.close}, "
            f"volume={self.volume}"
            f")>"
        )
