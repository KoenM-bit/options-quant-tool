"""
Rebuild Beursduivel bronze tables with underlying stock data support.
Drops and recreates bronze_bd_options and bronze_bd_underlying tables.
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
from src.models.bronze_bd import BronzeBDOptions
from src.models.bronze_bd_underlying import BronzeBDUnderlying
from src.scrapers.bd_options_scraper import scrape_all_options
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rebuild_bd_tables():
    """Drop and recreate BD bronze tables, then populate with fresh scrape."""
    
    logger.info("="*60)
    logger.info("Rebuilding Beursduivel bronze tables")
    logger.info("="*60)
    
    with get_db_session() as session:
        # Drop existing tables
        logger.info("1️⃣  Dropping existing tables...")
        session.execute(text("DROP TABLE IF EXISTS bronze_bd_options CASCADE"))
        session.execute(text("DROP TABLE IF EXISTS bronze_bd_underlying CASCADE"))
        session.commit()
        logger.info("   ✅ Tables dropped")
        
        # Create new tables
        logger.info("2️⃣  Creating new tables...")
        BronzeBDOptions.__table__.create(session.get_bind(), checkfirst=True)
        BronzeBDUnderlying.__table__.create(session.get_bind(), checkfirst=True)
        session.commit()
        logger.info("   ✅ Tables created:")
        logger.info("      - bronze_bd_options")
        logger.info("      - bronze_bd_underlying")
    
    # Scrape fresh data
    logger.info("3️⃣  Scraping live data from Beursduivel...")
    options, underlying = scrape_all_options(ticker="AD.AS", fetch_live=False)
    
    if not options:
        logger.error("   ❌ No options data scraped")
        return
    
    logger.info(f"   ✅ Scraped {len(options)} options")
    
    # Insert options data
    logger.info("4️⃣  Inserting options data...")
    
    with get_db_session() as session:
        inserted = 0
        errors = 0
        
        for opt in options:
            try:
                # Extract trade_date from scraped_at (the day the market data represents)
                trade_date = opt.get('scraped_at').date() if opt.get('scraped_at') else datetime.now().date()
                
                record = BronzeBDOptions(
                    ticker=opt.get('ticker', 'AD.AS'),
                    trade_date=trade_date,
                    symbol_code=opt.get('symbol_code'),
                    issue_id=opt.get('issue_id'),
                    option_type=opt.get('type'),
                    expiry_date=opt.get('expiry_date'),
                    expiry_text=opt.get('expiry'),
                    strike=opt.get('strike'),
                    bid=opt.get('bid'),
                    ask=opt.get('ask'),
                    last_price=opt.get('last_price'),
                    volume=opt.get('volume'),
                    last_timestamp=opt.get('last_timestamp'),
                    last_date_text=opt.get('last_date_text'),
                    source=opt.get('source', 'beursduivel'),
                    source_url=opt.get('url'),
                    scraped_at=opt.get('scraped_at')
                )
                session.add(record)
                inserted += 1
            except Exception as e:
                logger.error(f"   Error inserting option {opt.get('issue_id')}: {e}")
                errors += 1
        
        session.commit()
        logger.info(f"   ✅ Inserted {inserted} options ({errors} errors)")
    
    # Insert underlying data
    if underlying:
        logger.info("5️⃣  Inserting underlying stock data...")
        
        with get_db_session() as session:
            try:
                # Extract trade_date from scraped_at
                trade_date = underlying['scraped_at'].date() if underlying.get('scraped_at') else datetime.now().date()
                
                record = BronzeBDUnderlying(
                    ticker=underlying['ticker'],
                    trade_date=trade_date,
                    isin=underlying.get('isin'),
                    name=underlying.get('name'),
                    last_price=underlying.get('last_price'),
                    bid=underlying.get('bid'),
                    ask=underlying.get('ask'),
                    volume=underlying.get('volume'),
                    last_timestamp_text=underlying.get('last_timestamp_text'),
                    scraped_at=underlying['scraped_at'],
                    source_url=underlying['source_url']
                )
                session.add(record)
                session.commit()
                logger.info(f"   ✅ Inserted underlying: {underlying.get('name')} @ €{underlying.get('last_price')}")
            except Exception as e:
                logger.error(f"   ❌ Error inserting underlying: {e}")
    else:
        logger.warning("5️⃣  No underlying data to insert")
    
    # Verification
    logger.info("6️⃣  Verifying data...")
    
    with get_db_session() as session:
        opt_count = session.execute(text("SELECT COUNT(*) FROM bronze_bd_options")).scalar()
        und_count = session.execute(text("SELECT COUNT(*) FROM bronze_bd_underlying")).scalar()
        
        logger.info(f"   📊 bronze_bd_options: {opt_count} contracts")
        logger.info(f"   📊 bronze_bd_underlying: {und_count} records")
        
        # Sample underlying data
        if und_count > 0:
            result = session.execute(text("""
                SELECT ticker, last_price, bid, ask, volume, last_timestamp_text
                FROM bronze_bd_underlying
                ORDER BY scraped_at DESC
                LIMIT 1
            """)).fetchone()
            
            logger.info(f"   📈 Latest underlying: {result[0]} @ €{result[1]} (bid: €{result[2]}, ask: €{result[3]}) vol: {result[4]:,}")
    
    logger.info("="*60)
    logger.info("✅ Database rebuild complete!")
    logger.info("="*60)


if __name__ == "__main__":
    rebuild_bd_tables()
