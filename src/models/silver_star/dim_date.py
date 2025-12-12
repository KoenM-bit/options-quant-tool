"""
dim_date: Date dimension table with pre-computed calendar attributes.
Populated once with historical and future dates.
"""

from sqlalchemy import Column, Integer, Date, String, Boolean, SmallInteger, Index
from src.models.base import Base


class DimDate(Base):
    """
    Date dimension table with pre-computed calendar attributes.
    
    Supports date-based filtering and aggregation without complex calculations:
    - Business calendar: weekday/weekend, month-end, quarter-end, year-end
    - Date parts: year, quarter, month, week, day
    - Display formats: YYYY-MM-DD, readable formats
    
    Uses integer date_key (YYYYMMDD) as primary key for performance.
    """
    
    __tablename__ = "dim_date"
    
    # Surrogate key as integer (YYYYMMDD format, e.g., 20251212)
    date_key = Column(Integer, primary_key=True)
    
    # Full date value
    date_value = Column(Date, nullable=False, unique=True, index=True)
    
    # Year attributes
    year = Column(SmallInteger, nullable=False, index=True)
    year_name = Column(String(4), nullable=False)  # '2025'
    
    # Quarter attributes
    quarter = Column(SmallInteger, nullable=False)  # 1-4
    quarter_name = Column(String(6), nullable=False)  # 'Q1', 'Q2', etc.
    year_quarter = Column(String(7), nullable=False)  # '2025-Q4'
    
    # Month attributes
    month = Column(SmallInteger, nullable=False, index=True)  # 1-12
    month_name = Column(String(20), nullable=False)  # 'January', 'February', etc.
    month_short = Column(String(3), nullable=False)  # 'Jan', 'Feb', etc.
    year_month = Column(String(7), nullable=False)  # '2025-12'
    
    # Week attributes
    week_of_year = Column(SmallInteger, nullable=False)  # 1-53
    year_week = Column(String(8), nullable=False)  # '2025-W50'
    
    # Day attributes
    day_of_month = Column(SmallInteger, nullable=False)  # 1-31
    day_of_week = Column(SmallInteger, nullable=False)  # 0=Monday, 6=Sunday
    day_name = Column(String(10), nullable=False)  # 'Monday', 'Tuesday', etc.
    day_short = Column(String(3), nullable=False)  # 'Mon', 'Tue', etc.
    day_of_year = Column(SmallInteger, nullable=False)  # 1-366
    
    # Business calendar flags
    is_weekday = Column(Boolean, nullable=False, index=True)  # Mon-Fri
    is_weekend = Column(Boolean, nullable=False)  # Sat-Sun
    is_month_end = Column(Boolean, nullable=False)
    is_quarter_end = Column(Boolean, nullable=False)
    is_year_end = Column(Boolean, nullable=False)
    
    # Display formats
    date_display = Column(String(10), nullable=False)  # '2025-12-12'
    date_display_long = Column(String(30), nullable=False)  # 'Thursday, December 12, 2025'
    
    # Indexes for common queries
    __table_args__ = (
        Index("idx_dim_date_year_month", "year", "month"),
        Index("idx_dim_date_year_quarter", "year", "quarter"),
        Index("idx_dim_date_weekday", "is_weekday"),
    )
    
    def __repr__(self):
        return (
            f"<DimDate(date_key={self.date_key}, "
            f"date={self.date_value}, "
            f"day={self.day_name}, "
            f"is_weekday={self.is_weekday})>"
        )
