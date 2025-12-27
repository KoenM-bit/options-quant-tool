#!/usr/bin/env python3
"""Quick test to fetch hourly data"""

import yfinance as yf
import sys
from pathlib import Path

# Ensure we're using the project venv
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print(f"Python: {sys.executable}")
print(f"yfinance version: {yf.__version__}")

ticker = "AD.AS"
print(f"\nFetching hourly data for {ticker}...")

stock = yf.Ticker(ticker)
df = stock.history(period='730d', interval='1h')

print(f"Got {len(df)} rows")
if len(df) > 0:
    print(f"From: {df.index.min()} to {df.index.max()}")
    print(df.head())
