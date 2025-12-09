#!/usr/bin/env python
"""
Test script to verify scraper functionality.
Run: python scripts/test_scraper.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scrapers.fd_overview_scraper import scrape_fd_overview
from src.config import settings
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Test the FD overview scraper."""
    logger.info("🧪 Testing FD Overview Scraper")
    logger.info(f"Ticker: {settings.ahold_ticker}")
    logger.info(f"Symbol Code: {settings.ahold_symbol_code}")
    
    try:
        # Scrape data
        logger.info("📡 Scraping data...")
        data = scrape_fd_overview()
        
        # Pretty print results
        logger.info("✅ Scrape successful!")
        logger.info("\n" + "="*50)
        logger.info("SCRAPED DATA:")
        logger.info("="*50)
        print(json.dumps(data, indent=2, default=str))
        logger.info("="*50)
        
        # Validate data
        logger.info("\n🔍 Validating data...")
        checks = {
            "Has ticker": data.get('ticker') is not None,
            "Has price": data.get('koers') is not None,
            "Has volume": data.get('totaal_volume') is not None,
            "Has OI": data.get('totaal_oi') is not None,
            "Has date": data.get('peildatum') is not None,
        }
        
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            logger.info(f"{status} {check}")
        
        if all(checks.values()):
            logger.info("\n🎉 All checks passed!")
            return 0
        else:
            logger.warning("\n⚠️  Some checks failed")
            return 1
    
    except Exception as e:
        logger.error(f"\n❌ Scraping failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
