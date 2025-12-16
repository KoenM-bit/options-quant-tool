"""
Bronze OHLCV Model - Raw daily stock price data
Source: Yahoo Finance (yfinance)
"""

from sqlalchemy import Column, String, Date, Integer, TIMESTAMP, Index, BigInteger, UniqueConstraint
from sqlalchemy.dialects.postgresql import NUMERIC as Numeric
from sqlalchemy.sql import func
from src.models.base import Base


class BronzeOHLCV(Base):
    """
    Bronze table for raw OHLCV (Open, High, Low, Close, Volume) stock data.
    
    Grain: One row per ticker per trade date
    Source: Yahoo Finance via yfinance API
    Updates: Daily, fetched before market open
    """
    
    __tablename__ = "bronze_ohlcv"
    
    # Primary key
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    
    # Natural key
    ticker = Column(String(20), nullable=False, index=True)
    trade_date = Column(Date, nullable=False, index=True)
    
    # OHLCV data
    open = Column(Numeric(10, 4), nullable=False, comment="Opening price")
    high = Column(Numeric(10, 4), nullable=False, comment="Highest price of the day")
    low = Column(Numeric(10, 4), nullable=False, comment="Lowest price of the day")
    close = Column(Numeric(10, 4), nullable=False, comment="Closing price")
    volume = Column(BigInteger, comment="Trading volume")
    
    # Adjusted close for splits/dividends
    adj_close = Column(Numeric(10, 4), comment="Adjusted closing price")
    
    # Metadata
    source = Column(String(50), default="yahoo_finance")
    scraped_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
    
    # Audit
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP, nullable=False, server_default=func.now(), onupdate=func.now())
    
    # Indexes and constraints
    __table_args__ = (
        UniqueConstraint('ticker', 'trade_date', name='uq_bronze_ohlcv_ticker_date'),
        Index('idx_bronze_ohlcv_ticker', 'ticker'),
        Index('idx_bronze_ohlcv_date', 'trade_date'),
        Index('idx_bronze_ohlcv_ticker_date', 'ticker', 'trade_date'),
    )
    
    def __repr__(self):
        return (
            f"<BronzeOHLCV("
            f"ticker={self.ticker}, "
            f"date={self.trade_date}, "
            f"close={self.close}, "
            f"volume={self.volume}"
            f")>"
        )
