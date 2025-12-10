"""
Create bronze_bd_options table for Beursduivel data
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from src.utils.db import get_db_session
from src.models.bronze_bd import BronzeBDOptions
from src.models.base import Base
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_bronze_bd_table():
    """Create bronze_bd_options table"""
    
    logger.info("Creating bronze_bd_options table...")
    
    with get_db_session() as session:
        # Drop existing table if exists
        session.execute(text("DROP TABLE IF EXISTS bronze_bd_options CASCADE"))
        session.commit()
        
        # Create table using SQLAlchemy model
        Base.metadata.create_all(bind=session.get_bind(), tables=[BronzeBDOptions.__table__])
        
        logger.info("✅ bronze_bd_options table created successfully")
        
        # Show table structure
        result = session.execute(text("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'bronze_bd_options'
            ORDER BY ordinal_position
        """))
        
        logger.info("\nTable structure:")
        for row in result:
            logger.info(f"  {row[0]:20s} {row[1]:20s} {'NULL' if row[2] == 'YES' else 'NOT NULL'}")

if __name__ == "__main__":
    create_bronze_bd_table()
