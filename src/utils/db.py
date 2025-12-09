"""
Database utilities for connection management.
"""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool, QueuePool

from src.config import settings

logger = logging.getLogger(__name__)


# Create engine
def create_db_engine(pool_size: int = 5, max_overflow: int = 10) -> Engine:
    """
    Create SQLAlchemy engine with connection pooling.
    
    Args:
        pool_size: Number of connections to maintain
        max_overflow: Maximum overflow connections
    
    Returns:
        SQLAlchemy Engine
    """
    engine = create_engine(
        settings.database_url,
        poolclass=QueuePool,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,  # Verify connections before using
        echo=settings.log_level == "DEBUG",
    )
    
    # Add connection event listeners for logging
    @event.listens_for(engine, "connect")
    def receive_connect(dbapi_conn, connection_record):
        logger.debug("Database connection established")
    
    @event.listens_for(engine, "close")
    def receive_close(dbapi_conn, connection_record):
        logger.debug("Database connection closed")
    
    return engine


# Global engine instance
engine = create_db_engine()

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    
    Usage:
        with get_db_session() as session:
            session.query(Model).all()
    
    Yields:
        SQLAlchemy Session
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI/Airflow to get DB session.
    
    Yields:
        SQLAlchemy Session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database tables.
    Should be called once during setup.
    Note: Only creates Bronze tables. Silver and Gold are managed by DBT.
    """
    from src.models.base import Base
    from src.models import bronze
    # Don't import silver/gold - DBT manages those schemas
    
    logger.info("Creating database tables (Bronze layer only)...")
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables created successfully")


def drop_all_tables():
    """
    Drop all tables. Use with caution!
    """
    from src.models.base import Base
    
    logger.warning("Dropping all database tables...")
    Base.metadata.drop_all(bind=engine)
    logger.info("✅ All tables dropped")


def test_connection() -> bool:
    """
    Test database connection.
    
    Returns:
        True if connection successful
    """
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        logger.info("✅ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False
