#!/usr/bin/env python3
"""
Scrape Euronext options chain with open interest data.
Selects the previous trading day to get OI data.
Extracts expiration dates from table headers and filters out summary rows.
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from bs4 import BeautifulSoup
import time
import pandas as pd
from datetime import datetime, timedelta
import re
import calendar
import os

def parse_expiration_date(header_text):
    """
    Parse expiration date from table header text.
    Example: "Dec 2025 Prijzen vanaf16 Dec 2025 17:27 CET" -> "2025-12-19" (3rd Friday)
    
    Args:
        header_text: Text from the table header (e.g., h3 or caption)
    
    Returns:
        Expiration date string in YYYY-MM-DD format, or None if not found
    """
    if not header_text:
        return None
    
    # Month mapping (Dutch and English)
    month_map = {
        'jan': 1, 'januari': 1, 'january': 1,
        'feb': 2, 'februari': 2, 'february': 2,
        'mrt': 3, 'maart': 3, 'mar': 3, 'march': 3,
        'apr': 4, 'april': 4,
        'mei': 5, 'may': 5,
        'jun': 6, 'juni': 6, 'june': 6,
        'jul': 7, 'juli': 7, 'july': 7,
        'aug': 8, 'augustus': 8, 'august': 8,
        'sep': 9, 'september': 9,
        'okt': 10, 'oktober': 10, 'oct': 10, 'october': 10,
        'nov': 11, 'november': 11,
        'dec': 12, 'december': 12
    }
    
    # Try to find "Dec 2025" or "Jan 2026" or "Mar 2026" pattern
    # Support both Dutch (mrt) and English (mar)
    pattern = r'(jan|feb|mrt|mar|apr|mei|may|jun|jul|aug|sep|okt|oct|nov|dec)\w*\s+(\d{4})'
    match = re.search(pattern, header_text.lower())
    
    if match:
        month_str = match.group(1)
        year = int(match.group(2))
        month = month_map.get(month_str)
        
        if month:
            # Calculate 3rd Friday of the month
            third_friday = calculate_third_friday(year, month)
            return third_friday.strftime('%Y-%m-%d')
    
    return None

def calculate_third_friday(year, month):
    """Calculate the 3rd Friday of a given month (options expiration day)."""
    # Find the first day of the month
    first_day = datetime(year, month, 1)
    
    # Find the first Friday
    days_until_friday = (4 - first_day.weekday()) % 7  # Friday is weekday 4
    first_friday = first_day + timedelta(days=days_until_friday)
    
    # Add 2 weeks to get the 3rd Friday
    third_friday = first_friday + timedelta(weeks=2)
    
    return third_friday

def scrape_options_chain(target_date_str=None, headless=True):
    """
    Scrape the full options chain from Euronext settlement prices page.
    
    Args:
        target_date_str: Date string like "15 december 2025" or None for previous day
        headless: Whether to run browser in headless mode
    
    Returns:
        DataFrame with all options data
    """
    print("="*80)
    print("EURONEXT OPTIONS CHAIN SCRAPER")
    print("="*80)
    
    # Setup Chrome options
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080')
    
    # Add anti-detection
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    driver = None
    all_options = []
    
    try:
        print("\nInitializing Chromium driver...")
        
        # Use system chromium and chromedriver
        options.binary_location = "/usr/bin/chromium"
        
        # Check if chromedriver is in PATH or use system location
        chromedriver_path = "/usr/bin/chromedriver"
        if os.path.exists(chromedriver_path):
            service = Service(chromedriver_path)
            driver = webdriver.Chrome(service=service, options=options)
        else:
            # Fallback to default driver (will search in PATH)
            driver = webdriver.Chrome(options=options)
        
        # Anti-detection
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {
            "userAgent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        url = "https://live.euronext.com/nl/product/stock-options/AH-DAMS/settlement-prices"
        print(f"\nLoading: {url}")
        driver.get(url)
        
        print("Waiting for page to load...")
        time.sleep(5)
        
        # Find the date selector dropdown
        print("\nLooking for date selector...")
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # Find select element with date options
        select_elements = soup.find_all('select')
        date_select = None
        
        for select in select_elements:
            options_list = select.find_all('option')
            # Check if this select contains date options
            for opt in options_list[:3]:
                if 'december' in opt.get_text().lower() or 'januari' in opt.get_text().lower():
                    date_select = select
                    print(f"✅ Found date selector with {len(options_list)} date options")
                    break
            if date_select:
                break
        
        if not date_select:
            print("❌ Could not find date selector dropdown")
            return pd.DataFrame()
        
        # Get the select element's ID or name for Selenium interaction
        select_id = date_select.get('id')
        select_name = date_select.get('name')
        select_class = date_select.get('class')
        
        print(f"Date selector - ID: {select_id}, Name: {select_name}, Class: {select_class}")
        
        # List available dates
        print("\nAvailable dates:")
        for i, opt in enumerate(date_select.find_all('option')[:10]):
            value = opt.get('value')
            text = opt.get_text(strip=True)
            print(f"  {i+1}. {text} (value={value})")
        
        # Find the date to select
        if not target_date_str:
            # Default to "15 december 2025" (previous day)
            target_date_str = "15 december 2025"
        
        print(f"\nAttempting to select date: {target_date_str}")
        
        # Use Selenium to interact with the select element
        try:
            # Try to find by ID first
            if select_id:
                selenium_select = Select(driver.find_element(By.ID, select_id))
            elif select_name:
                selenium_select = Select(driver.find_element(By.NAME, select_name))
            else:
                # Try to find by tag and option content
                selenium_select = Select(driver.find_element(By.XPATH, "//select[.//option[contains(text(), 'december')]]"))
            
            # Select by visible text
            selenium_select.select_by_visible_text(target_date_str)
            print(f"✅ Selected date: {target_date_str}")
            
            # Find and click the "Toepassen" (Apply) button
            print("Looking for 'Toepassen' button...")
            try:
                # Try different possible selectors for the Apply button
                apply_button = None
                possible_selectors = [
                    "//button[contains(text(), 'Toepassen')]",
                    "//input[@type='submit' and contains(@value, 'Toepassen')]",
                    "//button[@type='submit']",
                    "//*[contains(text(), 'Toepassen')]",
                ]
                
                for selector in possible_selectors:
                    try:
                        apply_button = driver.find_element(By.XPATH, selector)
                        if apply_button:
                            print(f"✅ Found 'Toepassen' button using: {selector}")
                            break
                    except:
                        continue
                
                if apply_button:
                    apply_button.click()
                    print("✅ Clicked 'Toepassen' button")
                else:
                    print("⚠️  Could not find 'Toepassen' button, trying alternative methods...")
                    # Try to submit the form
                    try:
                        form = driver.find_element(By.XPATH, "//form")
                        form.submit()
                        print("✅ Submitted form")
                    except:
                        print("❌ Could not submit form either")
                
            except Exception as e:
                print(f"⚠️  Error clicking apply button: {e}")
            
            # Wait for page to reload with new data
            print("Waiting for data to load...")
            time.sleep(5)
            
        except Exception as e:
            print(f"❌ Error selecting date: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
        
        # Now scrape the tables
        print("\n" + "="*80)
        print("SCRAPING OPTIONS DATA")
        print("="*80)
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        tables = soup.find_all('table')
        
        print(f"\nFound {len(tables)} tables")
        
        for table_idx, table in enumerate(tables, 1):
            print(f"\nProcessing table {table_idx}...")
            
            # Try to find expiration date from nearby heading
            expiration_date = None
            
            # Look for h3 heading before this table
            prev_sibling = table.find_previous(['h3', 'h2', 'h4', 'caption'])
            if prev_sibling:
                header_text = prev_sibling.get_text(strip=True)
                expiration_date = parse_expiration_date(header_text)
                if expiration_date:
                    print(f"  ✅ Found expiration date: {expiration_date} from '{header_text[:60]}'")
            
            # If not found, try to find it in table caption
            if not expiration_date:
                caption = table.find('caption')
                if caption:
                    caption_text = caption.get_text(strip=True)
                    expiration_date = parse_expiration_date(caption_text)
                    if expiration_date:
                        print(f"  ✅ Found expiration date: {expiration_date} from caption")
            
            rows = table.find_all('tr')
            if len(rows) < 2:
                continue
            
            # Get headers
            header_row = rows[0]
            headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
            
            # Map Dutch headers to English
            header_map = {
                'Uitoefenprijs': 'strike',
                'Soort': 'option_type',
                'Openingskoers': 'open',
                'Hoog': 'high',
                'Laag': 'low',
                'Laatste': 'last',
                'Wijzigen': 'change',
                'Settle': 'settle',
                'Aantal': 'volume',
                'Openstaande positie': 'open_interest'
            }
            
            # Process data rows
            options_count = 0
            for row in rows[1:]:
                cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
                
                if len(cells) != len(headers):
                    continue
                
                # Create dictionary for this option
                option_data = {}
                for header, cell in zip(headers, cells):
                    english_name = header_map.get(header, header)
                    option_data[english_name] = cell
                
                # Filter out "Totaal" summary rows
                strike_value = option_data.get('strike', '')
                if not strike_value or strike_value == 'Totaal' or strike_value == '':
                    continue
                
                # Add metadata
                option_data['scrape_date'] = datetime.now().strftime('%Y-%m-%d')
                option_data['data_date'] = target_date_str
                option_data['expiration_date'] = expiration_date
                option_data['table_number'] = table_idx
                
                # Check if open interest is populated
                if option_data.get('open_interest') and option_data['open_interest'] != '-':
                    options_count += 1
                
                all_options.append(option_data)
            
            print(f"  ✅ Extracted {len([o for o in all_options if o['table_number'] == table_idx])} options, {options_count} with OI data")
        
        print(f"\n{'='*80}")
        print(f"TOTAL: Scraped {len(all_options)} options from {len(tables)} tables")
        print(f"{'='*80}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_options)
        
        # Show summary
        if not df.empty:
            print("\nData summary:")
            print(f"  Total options: {len(df)}")
            print(f"  Strikes: {df['strike'].nunique()} unique")
            print(f"  Option types: {df['option_type'].unique()}")
            print(f"  Expirations: {df['expiration_date'].nunique()} unique")
            print(f"  Expiration dates: {sorted(df['expiration_date'].dropna().unique())}")
            
            # Count options with OI data
            oi_populated = df[df['open_interest'] != '-']
            print(f"  Options with OI data: {len(oi_populated)} / {len(df)}")
            
            if not oi_populated.empty:
                print("\nSample data with OI:")
                print(oi_populated[['strike', 'option_type', 'expiration_date', 'settle', 'volume', 'open_interest']].head(10))
        
        return df
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
        
    finally:
        if driver:
            print("\nClosing browser...")
            driver.quit()

def main():
    """Main function to scrape options chain for the previous trading day."""
    # Calculate previous trading day
    today = datetime.now()
    
    # If today is Monday, go back 3 days (to Friday)
    # If today is Sunday, go back 2 days (to Friday)
    # Otherwise, go back 1 day
    if today.weekday() == 0:  # Monday
        days_back = 3
    elif today.weekday() == 6:  # Sunday
        days_back = 2
    else:
        days_back = 1
    
    previous_day = today - timedelta(days=days_back)
    
    # Format as "15 december 2025"
    month_names_nl = {
        1: 'januari', 2: 'februari', 3: 'maart', 4: 'april',
        5: 'mei', 6: 'juni', 7: 'juli', 8: 'augustus',
        9: 'september', 10: 'oktober', 11: 'november', 12: 'december'
    }
    
    target_date_str = f"{previous_day.day} {month_names_nl[previous_day.month]} {previous_day.year}"
    
    print(f"Target date: {target_date_str} (previous trading day)")
    
    # Scrape with previous trading day
    df = scrape_options_chain(target_date_str=target_date_str, headless=True)
    
    if not df.empty:
        # Save to CSV
        output_file = 'data/euronext_options_chain.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✅ Saved {len(df)} options to {output_file}")
        
        # Show breakdown by expiration
        print("\nBreakdown by expiration:")
        expiration_counts = df.groupby('expiration_date').size()
        for exp_date, count in expiration_counts.items():
            print(f"  {exp_date}: {count} options")
        
        # Save to database
        print("\n" + "="*80)
        print("SAVING TO DATABASE")
        print("="*80)
        
        saved_count = save_chain_to_database(df)
        
        if saved_count > 0:
            print(f"\n✅ Successfully saved {saved_count} options to bronze_euronext_options table")
        else:
            print("\n⚠️  No options saved to database")
    else:
        print("\n❌ No data scraped")


def save_chain_to_database(df: pd.DataFrame) -> int:
    """
    Save scraped options chain data to PostgreSQL bronze layer.
    
    Args:
        df: DataFrame with scraped options data
        
    Returns:
        int: Number of records saved
    """
    import sys
    import os
    
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.models.bronze_euronext import BronzeEuronextOptions
    from src.utils.db import SessionLocal
    
    session = SessionLocal()
    saved_count = 0
    
    try:
        print(f"Connecting to database...")
        
        for idx, row in df.iterrows():
            try:
                # Convert values
                strike = float(row['strike']) if pd.notna(row['strike']) else None
                
                # Parse volume
                volume_str = str(row.get('volume', ''))
                if volume_str and volume_str not in ['', 'N/A', '-']:
                    try:
                        volume = int(volume_str)
                    except (ValueError, TypeError):
                        volume = None
                else:
                    volume = None
                
                # Parse open interest
                oi_value = row.get('open_interest', '-')
                if oi_value and oi_value != '-':
                    try:
                        open_interest = int(oi_value)
                    except (ValueError, TypeError):
                        open_interest = None
                else:
                    open_interest = None
                
                # Parse prices - helper function
                def safe_float(val):
                    if pd.isna(val) or val in ['', 'N/A', '-']:
                        return None
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return None
                
                opening_price = safe_float(row.get('open'))
                high_price = safe_float(row.get('high'))
                low_price = safe_float(row.get('low'))
                last_price = safe_float(row.get('last'))
                settlement_price = safe_float(row.get('settle'))
                
                # Parse actual expiration date (3rd Friday)
                actual_expiration_date = row.get('expiration_date')
                if pd.notna(actual_expiration_date):
                    try:
                        actual_expiration_date = datetime.strptime(str(actual_expiration_date), '%Y-%m-%d').date()
                    except:
                        actual_expiration_date = None
                else:
                    actual_expiration_date = None
                
                # Parse trading date (data date like "15 december 2025")
                data_date_str = row.get('data_date', '')
                if data_date_str:
                    # Convert "15 december 2025" to date object
                    try:
                        # Parse Dutch month names
                        month_map = {
                            'januari': 1, 'februari': 2, 'maart': 3, 'april': 4,
                            'mei': 5, 'juni': 6, 'juli': 7, 'augustus': 8,
                            'september': 9, 'oktober': 10, 'november': 11, 'december': 12
                        }
                        parts = data_date_str.split()
                        if len(parts) == 3:
                            day = int(parts[0])
                            month = month_map.get(parts[1].lower())
                            year = int(parts[2])
                            if month:
                                trade_date = datetime(year, month, day).date()
                            else:
                                trade_date = datetime.now().date()
                        else:
                            trade_date = datetime.now().date()
                    except:
                        trade_date = datetime.now().date()
                else:
                    trade_date = datetime.now().date()
                
                # Create expiration_date string in format like "12-2025" (month-year)
                expiration_str = actual_expiration_date.strftime('%m-%Y') if actual_expiration_date else ''
                
                # Check if record already exists for this trade_date + option combination
                existing = session.query(BronzeEuronextOptions).filter(
                    BronzeEuronextOptions.ticker == 'AH',
                    BronzeEuronextOptions.option_type == row['option_type'],
                    BronzeEuronextOptions.strike == strike,
                    BronzeEuronextOptions.actual_expiration_date == actual_expiration_date,
                    BronzeEuronextOptions.trade_date == trade_date
                ).first()
                
                if existing:
                    # Update existing record
                    existing.expiration_date = expiration_str
                    existing.open_interest = open_interest
                    existing.volume = volume
                    existing.opening_price = opening_price
                    existing.day_high = high_price
                    existing.day_low = low_price
                    existing.settlement_price = settlement_price
                    existing.scraped_at = datetime.now()
                else:
                    # Create new record
                    option_record = BronzeEuronextOptions(
                        ticker='AH',
                        option_type=row['option_type'],
                        strike=strike,
                        expiration_date=expiration_str,  # String format: "12-2025" (month-year)
                        actual_expiration_date=actual_expiration_date,  # Date: 3rd Friday (e.g., 2025-12-19)
                        open_interest=open_interest,
                        volume=volume,
                        opening_price=opening_price,
                        day_high=high_price,
                        day_low=low_price,
                        settlement_price=settlement_price,
                        scraped_at=datetime.now(),  # Timestamp: when we scraped it
                        trade_date=trade_date,  # Date: the trading/data date (e.g., 2025-12-15)
                    )
                    session.add(option_record)
                
                saved_count += 1
                
                # Commit in batches of 50
                if saved_count % 50 == 0:
                    session.commit()
                    print(f"  Saved {saved_count} options...")
                
            except Exception as e:
                print(f"  ⚠️  Error saving row {idx}: {e}")
                continue
        
        # Commit remaining
        session.commit()
        print(f"  Committed final batch...")
        
        return saved_count
        
    except Exception as e:
        session.rollback()
        print(f"❌ Database error: {e}")
        import traceback
        traceback.print_exc()
        return 0
    finally:
        session.close()

if __name__ == "__main__":
    main()
