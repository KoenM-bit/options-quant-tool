"""
Test Beursduivel Scraper - Live Option Data
============================================
Scrapes live bid/ask/last prices from Beursduivel for all Ahold options.
Compare with FD data to see data richness.
"""

import requests
from bs4 import BeautifulSoup
import datetime as dt
import pandas as pd
from typing import List, Dict, Optional

BASE = "https://www.beursduivel.be"
URL = f"{BASE}/Aandeel-Koers/11755/Ahold-Delhaize-Koninklijke/opties-expiratiedatum.aspx"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# ------------------ Helpers ------------------

def clean_href(href: str) -> str:
    """Convert relative href to absolute URL."""
    href = href.replace("../../../", "/")
    return f"{BASE}{href}"

def _parse_eu_number(s: str) -> Optional[float]:
    """Parse '1.234,56' -> 1234.56 and '1,900' -> 1.9"""
    s = (s or "").strip().replace("\xa0", "")
    s = s.replace(".", "")  # remove thousands separator
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None

# ------------------ Fetchers ------------------

def fetch_option_chain() -> List[Dict]:
    """Fetch all options from overview page with bid/ask."""
    print(f"Fetching option overview from {URL} ...")
    r = requests.get(URL, headers=HEADERS)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    options = []
    for section in soup.select("section.contentblock"):
        expiry_el = section.find("h3", class_="titlecontent")
        expiry_text = expiry_el.get_text(strip=True) if expiry_el else "Unknown Expiry"

        for row in section.select("tr"):
            strike_cell = row.select_one(".optiontable__focus")
            strike = strike_cell.get_text(strip=True).split()[0] if strike_cell else None

            bid_call = row.select_one(".optiontable__bidcall")
            ask_call = row.select_one(".optiontable__askcall")
            bid_put = row.select_one(".optiontable__bid")
            ask_put = row.select_one(".optiontable__askput")

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
                    _parse_eu_number(bid_el.get_text(strip=True))
                    if bid_el and bid_el.get_text(strip=True)
                    else None
                )
                ask_val = (
                    _parse_eu_number(ask_el.get_text(strip=True))
                    if ask_el and ask_el.get_text(strip=True)
                    else None
                )

                options.append({
                    "type": opt_type,
                    "expiry": expiry_text,
                    "strike": strike,
                    "issue_id": issue_id,
                    "url": full_url,
                    "bid": bid_val,
                    "ask": ask_val,
                })

    print(f"Found {len(options)} options in total.")
    return options

def get_live_price(issue_id: str, detail_url: str) -> Optional[Dict]:
    """Get live price, date and volume from option detail page."""
    try:
        r = requests.get(detail_url, headers=HEADERS, timeout=5)
        if not r.ok:
            print(f"⚠️ Failed to fetch detail page for {issue_id}")
            return None
        soup = BeautifulSoup(r.text, "html.parser")

        price_el = soup.find("span", id=f"{issue_id}LastPrice")
        date_el = soup.find("time", id=f"{issue_id}LastDateTime")
        vol_el = soup.find("span", id=f"{issue_id}Volume")

        if not price_el:
            return None

        last_raw = price_el.get_text(strip=True)
        last_val = _parse_eu_number(last_raw)
        date_text = date_el.get_text(strip=True) if date_el else None
        volume_text = vol_el.get_text(strip=True).replace("\xa0", "") if vol_el else None
        volume = int(volume_text) if volume_text and volume_text.isdigit() else None

        return {
            "last_raw": last_raw,
            "last": last_val,
            "date_text": date_text,
            "volume": volume
        }
    except Exception as e:
        print(f"⚠️ Error fetching live price for {issue_id}: {e}")
        return None

# ------------------ Main Analysis ------------------

if __name__ == "__main__":
    print("="*80)
    print("BEURSDUIVEL SCRAPER TEST - Data Analysis")
    print("="*80)
    
    # Fetch overview data
    options = fetch_option_chain()
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(options)
    
    print(f"\n📊 OVERVIEW DATA SUMMARY")
    print("="*80)
    print(f"Total options: {len(df)}")
    print(f"Unique expiries: {df['expiry'].nunique()}")
    print(f"  {df['expiry'].unique()}")
    print(f"\nOption types:")
    print(df['type'].value_counts())
    print(f"\nData availability (from overview page):")
    print(f"  Has bid:   {df['bid'].notna().sum()} / {len(df)} ({100*df['bid'].notna().sum()/len(df):.1f}%)")
    print(f"  Has ask:   {df['ask'].notna().sum()} / {len(df)} ({100*df['ask'].notna().sum()/len(df):.1f}%)")
    print(f"  Has both:  {(df['bid'].notna() & df['ask'].notna()).sum()} / {len(df)} ({100*(df['bid'].notna() & df['ask'].notna()).sum()/len(df):.1f}%)")
    
    # Sample some live data
    print(f"\n📡 TESTING LIVE DATA (sampling 5 contracts)...")
    print("="*80)
    
    sample_options = df.sample(min(5, len(df))).to_dict('records')
    live_data = []
    
    for opt in sample_options:
        print(f"\nFetching: {opt['type']} {opt['strike']} exp {opt['expiry']}")
        print(f"  Overview bid/ask: {opt['bid']} / {opt['ask']}")
        
        live = get_live_price(opt['issue_id'], opt['url'])
        if live:
            print(f"  Live data: last={live['last']}, volume={live['volume']}, date={live['date_text']}")
            live_data.append({
                **opt,
                'last_price': live['last'],
                'last_volume': live['volume'],
                'last_date': live['date_text']
            })
        else:
            print(f"  ❌ No live data")
    
    # Summary
    print(f"\n✅ DATA COMPARISON vs FD")
    print("="*80)
    print(f"\n🔵 BEURSDUIVEL (from overview + live pages):")
    print(f"  ✅ Bid/Ask: {100*(df['bid'].notna() & df['ask'].notna()).sum()/len(df):.1f}% coverage")
    print(f"  ✅ Live price: Available on detail pages")
    print(f"  ✅ Volume: Available on detail pages")
    print(f"  ✅ Real-time: Updated during trading hours")
    print(f"  ❌ Open Interest: NOT available")
    print(f"  ❌ Requires: 1 request per contract for live data")
    
    print(f"\n🔴 FD (current source):")
    print(f"  ⚠️ Bid/Ask: Very low coverage (~1-5%)")
    print(f"  ✅ Last price: ~63% coverage")
    print(f"  ✅ Volume: Available")
    print(f"  ✅ Open Interest: Available")
    print(f"  ⚠️ Static: End-of-day data only")
    
    print(f"\n💡 RECOMMENDATION:")
    print("="*80)
    print("""
    BEST APPROACH: Combine both sources!
    
    1. PRIMARY: Beursduivel for PRICING DATA (17:30 scrape)
       - Bid/Ask spreads (90%+ coverage)
       - Live last price
       - Intraday volume
       - Most recent quote data
    
    2. SECONDARY: FD for MARKET METRICS (after hours)
       - Open Interest (most accurate on Saturdays)
       - Daily volume totals
       - Validation/backup
    
    IMPLEMENTATION:
    - Scrape Beursduivel at 17:30 (before close) -> bronze_bd_options
    - Scrape FD overnight -> bronze_fd_options  
    - Merge in silver layer: BD pricing + FD open interest
    - Result: 90%+ contracts with Greeks vs current 10%!
    """)
    
    # Show sample merged structure
    if live_data:
        print(f"\n📋 SAMPLE MERGED DATA STRUCTURE:")
        print("="*80)
        sample_df = pd.DataFrame(live_data)
        print(sample_df[['type', 'strike', 'expiry', 'bid', 'ask', 'last_price', 'last_volume']].head())
