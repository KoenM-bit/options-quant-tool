#!/usr/bin/env python3
"""
Quick test of the FD options scraper.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.scrapers.fd_options_scraper import scrape_fd_options

if __name__ == "__main__":
    print("Testing FD Options Scraper...")
    print("=" * 60)
    
    options = scrape_fd_options(ticker="AD.AS", symbol_code="AEX.AH/O")
    
    print(f"\n✅ Total options scraped: {len(options)}")
    
    if options:
        # Show first few options
        print("\n📊 Sample options:")
        for i, opt in enumerate(options[:5]):
            print(f"\n{i+1}. {opt['option_type']} @ Strike {opt['strike']}")
            print(f"   Expiry: {opt['expiry_date']}")
            print(f"   Last: {opt.get('laatste', 'N/A')}")
            print(f"   Bid/Ask: {opt.get('bid', 'N/A')} / {opt.get('ask', 'N/A')}")
            print(f"   Volume: {opt.get('volume', 0)}, OI: {opt.get('open_interest', 0)}")
        
        # Summary by type
        calls = [o for o in options if o['option_type'] == 'Call']
        puts = [o for o in options if o['option_type'] == 'Put']
        print(f"\n📈 Calls: {len(calls)}")
        print(f"📉 Puts: {len(puts)}")
    else:
        print("\n⚠️  No options data retrieved")
