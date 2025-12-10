"""
Create silver_options_chain table in database.
"""

import sys
import os
from pathlib import Path

# Support both Docker and local execution
if os.path.exists('/opt/airflow'):
    sys.path.insert(0, '/opt/airflow/dags/..')
else:
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

from sqlalchemy import text
from src.utils.db import get_db_session
from src.models.silver import SilverOptionsChain
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_silver_table():
    """Create silver_options_chain table."""
    
    logger.info("Creating silver_options_chain table...")
    
    with get_db_session() as session:
        # Drop existing table if requested
        if '--drop' in sys.argv:
            logger.info("Dropping existing table...")
            session.execute(text("DROP TABLE IF EXISTS silver_options_chain CASCADE"))
            session.commit()
        
        # Create table
        SilverOptionsChain.__table__.create(session.get_bind(), checkfirst=True)
        session.commit()
        logger.info("✅ Table created successfully")
        
        # Show table info
        result = session.execute(text("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'silver_options_chain'
            ORDER BY ordinal_position
        """))
        
        logger.info("\n📋 Table schema:")
        for row in result:
            logger.info(f"   {row[0]}: {row[1]}")

if __name__ == "__main__":
    create_silver_table()
