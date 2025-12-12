"""
Beursduivel Options Scraper
============================
Scrapes live option pricing data from Beursduivel.be
Provides superior bid/ask coverage (96%) vs FD (1-5%)

Target: Run at 17:30 CET before market close
Data: Real-time bid/ask/last prices with volume
"""

import requests
from bs4 import BeautifulSoup
import datetime as dt
from typing import List, Dict, Optional, Tuple
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

BASE = "https://www.beursduivel.be"
# Legacy default - can be overridden via url parameter
DEFAULT_AHOLD_URL = f"{BASE}/Aandeel-Koers/11755/Ahold-Delhaize-Koninklijke/opties-expiratiedatum.aspx"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

# Expiry month mapping (Dutch -> English)
MONTH_MAP = {
    'januari': 'January',
    'februari': 'February',
    'maart': 'March',
    'april': 'April',
    'mei': 'May',
    'juni': 'June',
    'juli': 'July',
    'augustus': 'August',
    'september': 'September',
    'oktober': 'October',
    'november': 'November',
    'december': 'December'
}


def clean_href(href: str) -> str:
    """Convert relative href to absolute URL."""
    href = href.replace("../../../", "/")
    return f"{BASE}{href}"


def parse_eu_number(s: str) -> Optional[float]:
    """
    Parse European number format to float.
    Examples: '1.234,56' -> 1234.56, '1,900' -> 1.9, '0,15' -> 0.15
    """
    if not s:
        return None
    
    s = str(s).strip().replace("\xa0", "")
    
    # Remove thousands separator (.)
    s = s.replace(".", "")
    # Replace decimal separator (,) with .
    s = s.replace(",", ".")
    
    try:
        return float(s)
    except ValueError:
        return None


def parse_expiry_date(expiry_text: str) -> Optional[Tuple[str, str]]:
    """
    Parse expiry text to standard date format.
    
    Examples:
        'December 2025 (AEX / AH)' -> ('2025-12-19', 'AH')
        'Januari 2026 (AEX / AH9)' -> ('2026-01-17', 'AH9')
    
    Returns:
        Tuple of (date_string, symbol_code) or None
    """
    try:
        # Extract month, year, and symbol
        # Format: "Month YYYY (AEX / SYMBOL)"
        match = re.match(r'(\w+)\s+(\d{4})\s+\(AEX\s+/\s+([^)]+)\)', expiry_text)
        if not match:
            logger.warning(f"Could not parse expiry: {expiry_text}")
            return None
        
        month_dutch = match.group(1).lower()
        year = int(match.group(2))
        symbol = match.group(3).strip()
        
        # Convert Dutch month to English
        month_eng = MONTH_MAP.get(month_dutch)
        if not month_eng:
            logger.warning(f"Unknown month: {month_dutch}")
            return None
        
        # AEX options expire on 3rd Friday of the month
        # Find 3rd Friday
        from calendar import monthrange
        _, last_day = monthrange(year, list(MONTH_MAP.keys()).index(month_dutch) + 1)
        
        month_num = list(MONTH_MAP.keys()).index(month_dutch) + 1
        
        # Find first day of month and its weekday
        first_day = dt.date(year, month_num, 1)
        first_weekday = first_day.weekday()  # 0=Monday, 4=Friday
        
        # Calculate days to first Friday
        days_to_friday = (4 - first_weekday) % 7
        first_friday = 1 + days_to_friday
        
        # Third Friday is 14 days later
        third_friday = first_friday + 14
        
        expiry_date = dt.date(year, month_num, third_friday)
        
        return (expiry_date.strftime('%Y-%m-%d'), symbol)
        
    except Exception as e:
        logger.error(f"Error parsing expiry '{expiry_text}': {e}")
        return None


def fetch_option_overview(ticker: str = "AD.AS", url: str = None, expand_all: bool = True) -> Tuple[List[Dict], Optional[Dict]]:
    """
    Fetch all options from Beursduivel overview page along with underlying stock data.
    Returns list of options with bid/ask from overview table + underlying price data.
    
    Args:
        ticker: Stock ticker (default: AD.AS)
        url: Beursduivel options page URL (if None, uses DEFAULT_AHOLD_URL)
        expand_all: If True, click "Show More" buttons to get all strikes (recommended)
    
    Returns:
        Tuple of (options_list, underlying_data):
        - options_list: List of dicts with option data
        - underlying_data: Dict with underlying stock price, bid, ask, volume, timestamp
    """
    if url is None:
        url = DEFAULT_AHOLD_URL
        
    logger.info(f"Fetching Beursduivel options for {ticker} from {url}...")
    
    # Start a session to maintain state
    session = requests.Session()
    session.headers.update(HEADERS)
    
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        logger.error(f"Failed to fetch overview page: {e}")
        return [], None
    
    # === SCRAPE UNDERLYING STOCK DATA ===
    underlying_data = None
    try:
        dashboard = soup.find("section", class_="dashboard")
        if dashboard:
            # Get stock name and ISIN
            name_span = dashboard.find("span", class_="dashboard__name")
            isin_span = dashboard.find("span", class_="dashboard__isn")
            stock_name = name_span.get_text(strip=True) if name_span else None
            isin = isin_span.get_text(strip=True) if isin_span else None
            
            # Get last price - need to find span.dashboard__val (the value, not the label)
            price_el = dashboard.find("span", {"id": re.compile(r"\d+LastPrice$"), "class": "dashboard__val"})
            if price_el:
                price_text = price_el.get_text(strip=True)
                logger.debug(f"Price element text: '{price_text}'")
                price = parse_eu_number(price_text)
            else:
                logger.warning("Could not find LastPrice value element")
                price = None
            
            # Get timestamp
            time_el = dashboard.find("time", id=re.compile(r"\d+LastDateTime"))
            time_text = time_el.get_text(strip=True) if time_el else None
            
            # Get bid
            bid_el = dashboard.find("span", id=re.compile(r"\d+BidPrice"))
            bid = parse_eu_number(bid_el.get_text(strip=True)) if bid_el else None
            
            # Get ask
            ask_el = dashboard.find("span", id=re.compile(r"\d+AskPrice"))
            ask = parse_eu_number(ask_el.get_text(strip=True)) if ask_el else None
            
            # Get volume
            volume_el = dashboard.find("span", id=re.compile(r"\d+Volume"))
            volume_text = volume_el.get_text(strip=True) if volume_el else None
            volume = parse_eu_number(volume_text.replace(".", "")) if volume_text else None  # Remove thousands separator
            
            underlying_data = {
                "ticker": ticker,
                "isin": isin,
                "name": stock_name,
                "last_price": price,
                "bid": bid,
                "ask": ask,
                "volume": int(volume) if volume else None,
                "last_timestamp_text": time_text,
                "scraped_at": datetime.now(),
                "source_url": url
            }
            
            logger.info(f"📊 Underlying: {stock_name} @ €{price} (bid: €{bid}, ask: €{ask}) at {time_text}")
        else:
            logger.warning("Could not find dashboard section for underlying data")
    except Exception as e:
        logger.error(f"Failed to parse underlying data: {e}")
    
    # If expand_all is True, click all "Meer opties" (Show More) buttons
    # Note: Each POST replaces the page, so we need to collect data from each expansion separately
    if expand_all:
        # Collect all section data (initial visible + expanded sections)
        all_section_soups = []
        
        # First, get the initial page sections (already visible)
        sections = soup.find_all("section", class_="contentblock", id=re.compile(r"opties-"))
        logger.info(f"Found {len(sections)} sections on initial page")
        
        # Find all "Show More" buttons
        more_buttons = soup.find_all("a", class_="morelink")
        logger.info(f"Found {len(more_buttons)} 'Show More' buttons to expand...")
        
        # Extract ASP.NET form data for POST requests
        viewstate = soup.find("input", {"name": "__VIEWSTATE"})
        viewstate_val = viewstate["value"] if viewstate else ""
        
        viewstate_gen = soup.find("input", {"name": "__VIEWSTATEGENERATOR"})
        viewstate_gen_val = viewstate_gen["value"] if viewstate_gen else ""
        
        event_validation = soup.find("input", {"name": "__EVENTVALIDATION"})
        event_validation_val = event_validation["value"] if event_validation else ""
        
        # Expand each section individually and collect the HTML
        expanded_sections = []
        for i, button in enumerate(more_buttons, 1):
            button_id = button.get("id")
            if not button_id:
                continue
            
            # Extract the control name from the href
            href = button.get("href", "")
            match = re.search(r'WebForm_PostBackOptions\("([^"]+)"', href)
            if not match:
                match = re.search(r"__doPostBack\('([^']+)'", href)
                if not match:
                    continue
            
            event_target = match.group(1)
            
            logger.debug(f"  Expanding group {i}/{len(more_buttons)}...")
            
            # Prepare POST data
            post_data = {
                "__EVENTTARGET": event_target,
                "__EVENTARGUMENT": "",
                "__VIEWSTATE": viewstate_val,
                "__VIEWSTATEGENERATOR": viewstate_gen_val,
                "__EVENTVALIDATION": event_validation_val
            }
            
            try:
                r = session.post(url, data=post_data, timeout=30)
                r.raise_for_status()
                
                # Parse the expanded section
                expanded_soup = BeautifulSoup(r.text, "html.parser")
                
                # Get the expanded section (there should be exactly one)
                expanded_section = expanded_soup.find("section", class_="contentblock", id=re.compile(r"opties-"))
                if expanded_section:
                    expanded_sections.append(expanded_section)
                    logger.debug(f"    ✓ Collected expanded section")
                
            except Exception as e:
                logger.warning(f"  Failed to expand group {i}: {e}")
                continue
        
        logger.info(f"✅ Successfully expanded {len(expanded_sections)}/{len(more_buttons)} sections")
        
        # Combine initial sections with expanded sections
        # Remove sections that have "Show More" buttons (they're collapsed)
        filtered_initial = [s for s in sections if not s.find("a", class_="morelink")]
        logger.info(f"   Using {len(filtered_initial)} already-visible sections + {len(expanded_sections)} expanded sections")
        
        all_section_soups = filtered_initial + expanded_sections
        
        # Replace soup with combined sections for parsing
        # Create a new soup with all sections
        combined_html = '<div>'
        for section in all_section_soups:
            combined_html += str(section)
        combined_html += '</div>'
        soup = BeautifulSoup(combined_html, "html.parser")
    
    options = []
    
    for section in soup.select("section.contentblock"):
        # Get expiry header
        expiry_el = section.find("h3", class_="titlecontent")
        expiry_text = expiry_el.get_text(strip=True) if expiry_el else None
        
        if not expiry_text:
            continue
        
        # Parse expiry to standard date format
        expiry_info = parse_expiry_date(expiry_text)
        if not expiry_info:
            continue
        
        expiry_date, symbol_code = expiry_info
        
        # Process each row in the options table
        for row in section.select("tr"):
            # Get strike price
            strike_cell = row.select_one(".optiontable__focus")
            if not strike_cell:
                continue
            
            strike_text = strike_cell.get_text(strip=True).split()[0]
            # The strike text contains issue_id appended (e.g., "34,00012" where 12 is issue_id)
            # Clean it by rounding to nearest 0.50 (typical strike increments)
            strike_raw = parse_eu_number(strike_text)
            if not strike_raw:
                continue
            
            # Round to nearest 0.50 to remove embedded issue_id digits
            strike = round(strike_raw * 2) / 2  # e.g., 34.00012 -> 34.0, 34.50016 -> 34.5
            
            # Get bid/ask columns for both calls and puts
            bid_call = row.select_one(".optiontable__bidcall")
            ask_call = row.select_one(".optiontable__askcall")
            bid_put = row.select_one(".optiontable__bid")
            ask_put = row.select_one(".optiontable__askput")
            
            # Process both Call and Put
            for opt_type in ["Call", "Put"]:
                link = row.select_one(f"a.optionlink.{opt_type}")
                if not link or "href" not in link.attrs:
                    continue
                
                href = link["href"]
                parts = href.split("/")
                issue_id = next((p for p in parts if p.isdigit()), None)
                if not issue_id:
                    continue
                
                full_url = clean_href(href)
                
                # Choose correct bid/ask based on option type
                if opt_type == "Call":
                    bid_el, ask_el = bid_call, ask_call
                else:
                    bid_el, ask_el = bid_put, ask_put
                
                bid_val = (
                    parse_eu_number(bid_el.get_text(strip=True))
                    if bid_el and bid_el.get_text(strip=True)
                    else None
                )
                ask_val = (
                    parse_eu_number(ask_el.get_text(strip=True))
                    if ask_el and ask_el.get_text(strip=True)
                    else None
                )
                
                # FILTER: Only include main monthly options (symbol_code = short ticker code)
                # Skip weekly/daily options (AH9, 2AH, 3AH, 4AH, etc. for Ahold)
                # For Ahold: AH, For ArcelorMittal: MT, etc.
                # Rule: symbol_code should be 2-3 letters without numbers
                if symbol_code and re.match(r'^[A-Z]{1,3}$', symbol_code):
                    options.append({
                        "ticker": ticker,
                        "type": opt_type,
                        "expiry_text": expiry_text,
                        "expiry_date": expiry_date,
                        "symbol_code": symbol_code,
                        "strike": strike,
                        "issue_id": issue_id,
                        "url": full_url,
                        "bid": bid_val,
                        "ask": ask_val,
                    })
    
    logger.info(f"Fetched {len(options)} options from overview page")
    return options, underlying_data


def fetch_live_price(issue_id: str, detail_url: str) -> Optional[Dict]:
    """
    Fetch live price, volume and timestamp from option detail page.
    
    Returns:
        Dict with keys: last, volume, timestamp, date_text
    """
    try:
        r = requests.get(detail_url, headers=HEADERS, timeout=10)
        if not r.ok:
            logger.debug(f"Failed to fetch detail page for {issue_id}: {r.status_code}")
            return None
        
        soup = BeautifulSoup(r.text, "html.parser")
        
        # Find price, volume, date elements
        price_el = soup.find("span", id=f"{issue_id}LastPrice")
        date_el = soup.find("time", id=f"{issue_id}LastDateTime")
        vol_el = soup.find("span", id=f"{issue_id}Volume")
        
        if not price_el:
            logger.debug(f"No price element found for {issue_id}")
            return None
        
        # Parse price
        last_raw = price_el.get_text(strip=True)
        last_val = parse_eu_number(last_raw)
        
        # Parse date/time
        date_text = date_el.get_text(strip=True) if date_el else None
        timestamp = None
        if date_text:
            # Try to parse timestamp (format: "10 dec 2025 11:33" or "9 dec 2025")
            try:
                # Handle both with and without time
                if len(date_text.split()) == 4:  # Has time
                    timestamp = datetime.strptime(date_text, "%d %b %Y %H:%M")
                else:  # Date only
                    timestamp = datetime.strptime(date_text, "%d %b %Y")
            except ValueError:
                logger.debug(f"Could not parse timestamp: {date_text}")
        
        # Parse volume
        volume_text = vol_el.get_text(strip=True).replace("\xa0", "") if vol_el else None
        volume = None
        if volume_text and volume_text.replace(".", "").isdigit():
            volume = int(volume_text.replace(".", ""))
        
        return {
            "last": last_val,
            "volume": volume,
            "timestamp": timestamp,
            "date_text": date_text
        }
        
    except Exception as e:
        logger.debug(f"Error fetching live price for {issue_id}: {e}")
        return None


def scrape_all_options(ticker: str = "AD.AS", url: str = None, fetch_live: bool = True) -> Tuple[List[Dict], Optional[Dict]]:
    """
    Main scraper function: fetch all options with pricing data + underlying stock data.
    
    Args:
        ticker: Stock ticker (default: AD.AS)
        url: Beursduivel options page URL (if None, uses DEFAULT_AHOLD_URL)
        fetch_live: If True, fetch live data from detail pages (slower but more complete)
    
    Returns:
        Tuple of (options_list, underlying_data):
        - options_list: List of option contracts with all available data
        - underlying_data: Dict with underlying stock price synchronized with options
    """
    if url is None:
        url = DEFAULT_AHOLD_URL
        
    logger.info("="*60)
    logger.info(f"🚀 Starting Beursduivel scraper for {ticker} from {url}")
    logger.info("="*60)
    
    # Step 1: Get overview data (fast, has bid/ask for all contracts + underlying price)
    options, underlying_data = fetch_option_overview(ticker, url=url)
    
    if not options:
        logger.error("No options fetched from overview page")
        return [], underlying_data
    
    logger.info(f"✅ Fetched {len(options)} contracts from overview")
    
    # Step 2: Optionally fetch live data (slow, 1 request per contract)
    if fetch_live:
        logger.info(f"📡 Fetching live data for {len(options)} contracts...")
        
        enriched_count = 0
        for i, opt in enumerate(options, 1):
            if i % 50 == 0:
                logger.info(f"  Progress: {i}/{len(options)} contracts...")
            
            live = fetch_live_price(opt['issue_id'], opt['url'])
            if live:
                opt['last_price'] = live['last']
                opt['volume'] = live['volume']
                opt['last_timestamp'] = live['timestamp']
                opt['last_date_text'] = live['date_text']
                enriched_count += 1
            else:
                opt['last_price'] = None
                opt['volume'] = None
                opt['last_timestamp'] = None
                opt['last_date_text'] = None
        
        logger.info(f"✅ Enriched {enriched_count}/{len(options)} contracts with live data")
    
    # Add scrape metadata
    scrape_time = datetime.now()
    for opt in options:
        opt['scraped_at'] = scrape_time
        opt['source'] = 'beursduivel'
    
    logger.info(f"🎉 Scraping complete: {len(options)} contracts")
    logger.info(f"📊 Underlying: €{underlying_data.get('last_price')} @ {underlying_data.get('last_timestamp_text')}")
    return options, underlying_data


if __name__ == "__main__":
    # Test scraper
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Scrape without live data (fast)
    options, underlying = scrape_all_options(fetch_live=False)
    
    print(f"\n📊 Options Summary:")
    print(f"Total contracts: {len(options)}")
    print(f"Bid coverage: {sum(1 for o in options if o['bid']) / len(options) * 100:.1f}%")
    print(f"Ask coverage: {sum(1 for o in options if o['ask']) / len(options) * 100:.1f}%")
    
    if underlying:
        print(f"\n📈 Underlying Stock:")
        print(f"  {underlying.get('name')} ({underlying.get('isin')})")
        print(f"  Price: €{underlying.get('last_price')}")
        print(f"  Bid: €{underlying.get('bid')}, Ask: €{underlying.get('ask')}")
        print(f"  Volume: {underlying.get('volume'):,}")
        print(f"  Timestamp: {underlying.get('last_timestamp_text')}")
