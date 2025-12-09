"""
Database initialization script.
Creates all tables and optionally seeds with sample data.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.db import init_db, test_connection, drop_all_tables
from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Initialize database."""
    logger.info("🚀 Starting database initialization...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Database: {settings.postgres_db}")
    
    # Test connection
    if not test_connection():
        logger.error("❌ Database connection failed. Exiting.")
        sys.exit(1)
    
    # Ask for confirmation if dropping tables
    if len(sys.argv) > 1 and sys.argv[1] == "--drop":
        confirm = input("⚠️  This will DROP all existing tables. Continue? (yes/no): ")
        if confirm.lower() == 'yes':
            drop_all_tables()
        else:
            logger.info("Aborted.")
            sys.exit(0)
    
    # Create tables
    try:
        init_db()
        logger.info("✅ Database initialized successfully!")
        logger.info("Tables created:")
        logger.info("  - Bronze: bronze_fd_overview, bronze_fd_options")
        logger.info("  - Silver: silver_underlying_price, silver_options")
        logger.info("  - Gold: gold_options_summary_daily, gold_volatility_surface, "
                   "gold_greek_analytics, gold_open_interest_flow")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
