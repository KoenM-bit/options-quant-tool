"""
Professional FD.nl Options Scraper
Scrapes call and put options from Financieele Dagblad (FD.nl)
"""

from datetime import datetime, date
from typing import List, Dict, Any, Optional
import logging

from src.config import settings
from src.utils.http_client import fetch_html
from src.utils.parsers import parse_float_nl, parse_int_nl, parse_date_nl

logger = logging.getLogger(__name__)


class FDOptionsScraper:
    """
    Scraper for FD.nl options data (calls and puts).
    Returns raw data ready for Bronze layer storage.
    """
    
    def __init__(self):
        self.base_url = "https://beurs.fd.nl/derivaten/opties/"
    
    def scrape_options(
        self, 
        ticker: str = "AD.AS",
        symbol_code: str = "AEX.AH/O",
        peildatum: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """
        Scrape all options (calls + puts) for a given ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., "AD.AS" for Ahold Delhaize)
            symbol_code: FD-specific symbol code
            peildatum: Reference date (defaults to today)
        
        Returns:
            List of dictionaries with raw option data
        """
        if peildatum is None:
            peildatum = date.today()
        
        logger.info(f"Scraping FD options for {ticker} ({symbol_code})")
        
        # Scrape calls and puts
        calls = self._scrape_option_type(symbol_code, "call", ticker, peildatum)
        puts = self._scrape_option_type(symbol_code, "put", ticker, peildatum)
        
        all_options = calls + puts
        
        logger.info(f"✅ Scraped {len(all_options)} options ({len(calls)} calls, {len(puts)} puts)")
        
        return all_options
    
    def _scrape_option_type(
        self,
        symbol_code: str,
        option_type: str,
        ticker: str,
        peildatum: date
    ) -> List[Dict[str, Any]]:
        """Scrape specific option type (call or put)."""
        
        url = f"{self.base_url}?{option_type}={symbol_code}"
        
        try:
            soup = fetch_html(url)
            table = soup.find("table", {"id": "m_Content_GridViewIssues"})
            
            if table is None:
                logger.warning(f"No option table found for {option_type}")
                return []
            
            rows = table.find_all("tr")[1:]  # Skip header
            options = []
            
            for tr in rows:
                cols = [c.get_text(strip=True).replace("\xa0", "") for c in tr.find_all("td")]
                
                if len(cols) < 13:
                    continue
                
                try:
                    # Map scraper fields to database model fields
                    # Only include fields that exist in BronzeFDOptions model
                    option_data = {
                        # Identifiers
                        "ticker": ticker,
                        "symbol_code": symbol_code,
                        "scraped_at": datetime.now(),
                        "source_url": url,
                        
                        # Option details (mapped to model)
                        "option_type": option_type.capitalize(),  # 'Call' or 'Put'
                        "expiry_date": parse_date_nl(cols[0]) if cols[0] else None,
                        "strike": parse_float_nl(cols[2]) if cols[2] else None,
                        
                        # Option data (only fields in model)
                        "laatste": parse_float_nl(cols[3]) if cols[3] else None,  # last price
                        "bid": parse_float_nl(cols[7]) if cols[7] else None,
                        "ask": parse_float_nl(cols[8]) if cols[8] else None,
                        "volume": parse_int_nl(cols[11]) if cols[11] else 0,
                        "open_interest": parse_int_nl(cols[1]) if cols[1] else 0,
                        
                        # naam and isin not available from this table
                        "naam": None,
                        "isin": None,
                        
                        # Greeks NOT stored in Bronze - calculated in Silver layer (dbt)
                        # "delta": None,
                        # "gamma": None,
                        # "theta": None,
                        # "vega": None,
                        # "implied_volatility": None,
                        
                        # Underlying price for reference (from scrape context)
                        "underlying_price": None,
                    }
                    
                    # Only add if we have essential data
                    if option_data["strike"] and option_data["expiry_date"]:
                        options.append(option_data)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse row: {e}")
                    continue
            
            return options
            
        except Exception as e:
            logger.error(f"Failed to scrape {option_type} options: {e}")
            return []


# Convenience function for backward compatibility
def scrape_fd_options(ticker: str = "AD.AS", symbol_code: str = "AEX.AH/O") -> List[Dict[str, Any]]:
    """
    Convenience function to scrape FD options.
    
    Returns:
        List of dictionaries ready for Bronze layer storage
    """
    scraper = FDOptionsScraper()
    return scraper.scrape_options(ticker=ticker, symbol_code=symbol_code)
