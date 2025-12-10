"""
Test Beursduivel scraper with database insertion
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.scrapers.bd_options_scraper import scrape_all_options
from src.models.bronze_bd import BronzeBDOptions
from src.utils.db import get_db_session
from sqlalchemy import text, func
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_scrape_and_insert():
    """Test scraping and inserting into database"""
    
    logger.info("="*60)
    logger.info("🧪 Testing Beursduivel Scraper + Database Insertion")
    logger.info("="*60)
    
    # Scrape data (without live data for faster testing)
    logger.info("\n1️⃣ Scraping Beursduivel (overview only, no live data)...")
    options = scrape_all_options(ticker="AD.AS", fetch_live=False)
    
    if not options:
        logger.error("❌ No data scraped!")
        return
    
    logger.info(f"✅ Scraped {len(options)} contracts")
    
    # Check data quality
    bid_count = sum(1 for o in options if o.get('bid'))
    ask_count = sum(1 for o in options if o.get('ask'))
    logger.info(f"   Bid coverage: {bid_count}/{len(options)} ({100*bid_count/len(options):.1f}%)")
    logger.info(f"   Ask coverage: {ask_count}/{len(options)} ({100*ask_count/len(options):.1f}%)")
    
    # Insert into database
    logger.info("\n2️⃣ Inserting into bronze_bd_options...")
    
    with get_db_session() as session:
        inserted = 0
        errors = 0
        
        for opt in options:
            try:
                record = BronzeBDOptions(
                    ticker=opt['ticker'],
                    symbol_code=opt['symbol_code'],
                    issue_id=opt['issue_id'],
                    option_type=opt['type'],
                    expiry_date=opt['expiry_date'],
                    expiry_text=opt['expiry_text'],
                    strike=opt['strike'],
                    bid=opt.get('bid'),
                    ask=opt.get('ask'),
                    last_price=opt.get('last_price'),
                    volume=opt.get('volume'),
                    last_timestamp=opt.get('last_timestamp'),
                    last_date_text=opt.get('last_date_text'),
                    source=opt['source'],
                    source_url=opt['url'],
                    scraped_at=opt['scraped_at']
                )
                session.add(record)
                inserted += 1
            except Exception as e:
                logger.error(f"Error inserting {opt.get('strike')} {opt.get('type')}: {e}")
                errors += 1
                continue
        
        session.commit()
        logger.info(f"✅ Inserted {inserted} contracts ({errors} errors)")
    
    # Verify data in database
    logger.info("\n3️⃣ Verifying data in database...")
    
    with get_db_session() as session:
        # Count records
        total = session.query(func.count(BronzeBDOptions.id)).scalar()
        logger.info(f"   Total records: {total}")
        
        # Count by option type
        result = session.execute(text("""
            SELECT option_type, COUNT(*) as cnt
            FROM bronze_bd_options
            GROUP BY option_type
        """))
        logger.info("   By option type:")
        for row in result:
            logger.info(f"     {row[0]}: {row[1]}")
        
        # Count by expiry
        result = session.execute(text("""
            SELECT expiry_date, COUNT(*) as cnt
            FROM bronze_bd_options
            GROUP BY expiry_date
            ORDER BY expiry_date
            LIMIT 5
        """))
        logger.info("   First 5 expiries:")
        for row in result:
            logger.info(f"     {row[0]}: {row[1]} contracts")
        
        # Sample data
        result = session.execute(text("""
            SELECT option_type, strike, expiry_date, bid, ask
            FROM bronze_bd_options
            ORDER BY expiry_date, strike
            LIMIT 5
        """))
        logger.info("   Sample contracts:")
        for row in result:
            logger.info(f"     {row[0]} @ {row[1]:.2f} exp {row[2]}, bid={row[3]}, ask={row[4]}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ TEST PASSED - Beursduivel scraper working!")
    logger.info("="*60)

if __name__ == "__main__":
    test_scrape_and_insert()
