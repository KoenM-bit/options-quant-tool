"""
Professional FD.nl Overview Scraper
Scrapes market summary data (underlying price, volume, open interest totals)
"""

import re
from datetime import datetime, date
from typing import Dict, Any, Optional
import logging

from src.utils.http_client import fetch_html
from src.utils.parsers import parse_float_nl, parse_int_nl
from src.config import settings

logger = logging.getLogger(__name__)


class FDOverviewScraper:
    """
    Scraper for FD.nl overview/summary data.
    Returns market summary including underlying price and volume/OI totals.
    """
    
    def __init__(self):
        self.base_url = settings.ahold_fd_base_url
    
    def scrape_overview(
        self, 
        ticker: Optional[str] = None,
        symbol_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Scrape overview data for a ticker.
        
        Args:
            ticker: Stock ticker symbol (defaults to config)
            symbol_code: FD-specific symbol code (defaults to config)
        
        Returns:
            Dictionary with overview data ready for Bronze layer
        """
        ticker = ticker or settings.ahold_ticker
        symbol_code = symbol_code or settings.ahold_symbol_code
        
        url = f"{self.base_url}?call={symbol_code}"
        
        logger.info(f"Scraping FD overview for {ticker} ({symbol_code})")
        
        try:
            soup = fetch_html(url)
            
            # Extract header data (underlying price info)
            header_data = self._extract_header(soup)
            
            # Extract totals data (volume, open interest)
            totals_data = self._extract_totals(soup)
            
            # Combine into single record
            overview = {
                # Identifiers
                "ticker": ticker,
                "symbol_code": symbol_code,
                "scraped_at": datetime.now(),
                "source_url": url,
                
                # Underlying price data
                "onderliggende_waarde": header_data.get("name"),
                "koers": header_data.get("koers"),
                "vorige": header_data.get("vorige"),
                "delta": header_data.get("delta"),
                "delta_pct": header_data.get("delta_pct"),
                "hoog": header_data.get("hoog"),
                "laag": header_data.get("laag"),
                "volume_underlying": header_data.get("volume_ul"),
                "tijd": header_data.get("tijd"),
                
                # Market totals
                "peildatum": totals_data.get("peildatum"),
                "totaal_volume": totals_data.get("totaal_volume"),
                "totaal_volume_calls": totals_data.get("totaal_volume_calls"),
                "totaal_volume_puts": totals_data.get("totaal_volume_puts"),
                "totaal_oi": totals_data.get("totaal_oi_opening"),
                "totaal_oi_calls": totals_data.get("totaal_oi_calls"),
                "totaal_oi_puts": totals_data.get("totaal_oi_puts"),
                "call_put_ratio": totals_data.get("call_put_ratio"),
            }
            
            logger.info(f"✅ Scraped overview: {ticker} @ {header_data.get('koers')}")
            return overview
            
        except Exception as e:
            logger.error(f"Failed to scrape overview for {ticker}: {e}")
            raise
    
    def _extract_header(self, soup) -> Dict[str, Any]:
        """Extract underlying price header data."""
        header_tbl = soup.find("table", id="m_Content_GridViewSingleUnderlyingIssue")
        
        if not header_tbl:
            logger.warning("Header table not found")
            return {}
        
        rows = header_tbl.find_all("tr")
        if len(rows) < 2:
            return {}
        
        data_tds = rows[-1].find_all("td")
        
        if len(data_tds) < 9:
            logger.warning(f"Expected 9 columns in header, got {len(data_tds)}")
            return {}
        
        return {
            "name": data_tds[0].get_text(strip=True),
            "koers": parse_float_nl(data_tds[1].get_text()),
            "vorige": parse_float_nl(data_tds[2].get_text()),
            "delta": parse_float_nl(data_tds[3].get_text()),
            "delta_pct": parse_float_nl(data_tds[4].get_text(strip=True).replace("%", "")),
            "hoog": parse_float_nl(data_tds[5].get_text()),
            "laag": parse_float_nl(data_tds[6].get_text()),
            "volume_ul": parse_int_nl(data_tds[7].get_text()),
            "tijd": data_tds[8].get_text(strip=True),
        }
    
    def _extract_totals(self, soup) -> Dict[str, Any]:
        """Extract volume and open interest totals."""
        totals_tbl = soup.find("table", class_="fAr11 mb10 mt10")
        
        if not totals_tbl:
            logger.warning("Totals table not found")
            return {}
        
        totals = {
            "totaal_volume": None,
            "totaal_volume_calls": None,
            "totaal_volume_puts": None,
            "totaal_oi_opening": None,
            "totaal_oi_calls": None,
            "totaal_oi_puts": None,
            "call_put_ratio": None,
            "peildatum": None,
        }
        
        # Extract peildatum from first row
        first_row = totals_tbl.find("tr")
        if first_row:
            subtitle_td = first_row.find("td")
            if subtitle_td:
                subtitle_text = subtitle_td.get_text()
                # Look for date in format DD-MM-YYYY
                m = re.search(r"(\d{1,2}-\d{1,2}-\d{4})", subtitle_text)
                if m:
                    date_str = m.group(1)
                    try:
                        totals["peildatum"] = datetime.strptime(date_str, "%d-%m-%Y").date()
                    except ValueError:
                        logger.warning(f"Could not parse date: {date_str}")
        
        # Parse data rows
        trs = totals_tbl.find_all("tr")[1:]
        for tr in trs:
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue
            
            label = tds[0].get_text(strip=True).lower()
            val = tds[1].get_text(" ", strip=True)
            
            if "totaal volume" in label:
                # Format: "1.234 (567 Calls, 678 Puts)"
                m = re.search(r"([\d\.\s]+)\s*\(\s*([\d\.\s]+)\s*Calls,\s*([\d\.\s]+)\s*Puts\)", val)
                if m:
                    totals["totaal_volume"] = parse_int_nl(m.group(1))
                    totals["totaal_volume_calls"] = parse_int_nl(m.group(2))
                    totals["totaal_volume_puts"] = parse_int_nl(m.group(3))
            
            elif "totaal open interest" in label:
                # Format: "12.345 (6.789 Calls, 5.556 Puts)"
                m = re.search(r"([\d\.\s]+)\s*\(\s*([\d\.\s]+)\s*Calls,\s*([\d\.\s]+)\s*Puts\)", val)
                if m:
                    totals["totaal_oi_opening"] = parse_int_nl(m.group(1))
                    totals["totaal_oi_calls"] = parse_int_nl(m.group(2))
                    totals["totaal_oi_puts"] = parse_int_nl(m.group(3))
            
            elif "call" in label and "put" in label:
                totals["call_put_ratio"] = parse_float_nl(val)
        
        return totals


# Convenience function
def scrape_fd_overview(
    ticker: Optional[str] = None,
    symbol_code: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to scrape FD overview.
    
    Returns:
        Dictionary ready for Bronze layer storage
    """
    scraper = FDOverviewScraper()
    return scraper.scrape_overview(ticker=ticker, symbol_code=symbol_code)
