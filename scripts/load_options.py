import sys
sys.path.insert(0, '/opt/airflow/dags/..')

from src.scrapers.fd_options_scraper import scrape_fd_options
from src.utils.db import get_db_session
from src.models.bronze import BronzeFDOptions
from src.config import settings

print("Scraping options...")
options = scrape_fd_options(settings.ahold_ticker, settings.ahold_symbol_code)
print(f"Scraped {len(options)} options")

print("Loading to database...")
with get_db_session() as session:
    for option in options:
        record = BronzeFDOptions(**option)
        session.add(record)
    session.commit()
    print(f"✅ Loaded {len(options)} options to bronze_fd_options")
