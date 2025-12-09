"""
Bronze layer models - Raw data from scrapers.
Minimal processing, preserving original structure.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Index, Text
from datetime import datetime

from src.models.base import Base, TimestampMixin


class BronzeFDOverview(Base, TimestampMixin):
    """
    Raw FD.nl overview data.
    Stores daily market summary and underlying price.
    """
    __tablename__ = "bronze_fd_overview"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Identifiers
    ticker = Column(String(20), nullable=False, index=True)
    symbol_code = Column(String(50), nullable=False)
    scraped_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    source_url = Column(Text, nullable=True)
    
    # Underlying price data
    onderliggende_waarde = Column(String(100), nullable=True)
    koers = Column(Float, nullable=True)
    vorige = Column(Float, nullable=True)
    delta = Column(Float, nullable=True)
    delta_pct = Column(Float, nullable=True)
    hoog = Column(Float, nullable=True)
    laag = Column(Float, nullable=True)
    volume_underlying = Column(Integer, nullable=True)
    tijd = Column(String(50), nullable=True)
    
    # Market totals
    peildatum = Column(Date, nullable=True, index=True)
    totaal_volume = Column(Integer, nullable=True)
    totaal_volume_calls = Column(Integer, nullable=True)
    totaal_volume_puts = Column(Integer, nullable=True)
    totaal_oi = Column(Integer, nullable=True)
    totaal_oi_calls = Column(Integer, nullable=True)
    totaal_oi_puts = Column(Integer, nullable=True)
    call_put_ratio = Column(Float, nullable=True)
    
    # Indexes for common queries
    __table_args__ = (
        Index("ix_bronze_fd_overview_ticker_date", "ticker", "peildatum"),
        Index("ix_bronze_fd_overview_scraped", "scraped_at"),
    )
    
    def __repr__(self):
        return f"<BronzeFDOverview(ticker={self.ticker}, date={self.peildatum}, koers={self.koers})>"


class BronzeFDOptions(Base, TimestampMixin):
    """
    Raw FD.nl options chain data.
    Stores individual option contracts.
    """
    __tablename__ = "bronze_fd_options"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Identifiers
    ticker = Column(String(20), nullable=False, index=True)
    symbol_code = Column(String(50), nullable=False)
    scraped_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    source_url = Column(Text, nullable=True)
    
    # Option details
    option_type = Column(String(10), nullable=False, index=True)  # 'Call' or 'Put'
    expiry_date = Column(Date, nullable=False, index=True)
    strike = Column(Float, nullable=False)
    
    # Option data
    naam = Column(String(100), nullable=True)
    isin = Column(String(20), nullable=True)
    laatste = Column(Float, nullable=True)
    bid = Column(Float, nullable=True)
    ask = Column(Float, nullable=True)
    volume = Column(Integer, nullable=True)
    open_interest = Column(Integer, nullable=True)
    
    # Underlying at scrape time
    underlying_price = Column(Float, nullable=True)
    
    # Note: Greeks are NOT stored in Bronze (raw data only)
    # Greeks are calculated and stored in Silver layer after deduplication
    
    # Indexes for common queries
    __table_args__ = (
        Index("ix_bronze_fd_options_ticker_expiry", "ticker", "expiry_date"),
        Index("ix_bronze_fd_options_type_strike", "option_type", "strike"),
        Index("ix_bronze_fd_options_isin", "isin"),
    )
    
    def __repr__(self):
        return (
            f"<BronzeFDOptions(ticker={self.ticker}, type={self.option_type}, "
            f"strike={self.strike}, expiry={self.expiry_date})>"
        )
