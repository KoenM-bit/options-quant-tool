"""
Analyze option_prices_live to identify the last complete snapshot per day
"""

import mysql.connector
import os
from datetime import datetime

def connect_mysql():
    return mysql.connector.connect(
        host=os.getenv('MYSQL_HOST', '192.168.1.201'),
        user=os.getenv('MYSQL_USER', 'remoteuser'),
        password=os.getenv('MYSQL_PASSWORD'),
        database=os.getenv('MYSQL_DATABASE', 'optionsdb')
    )


def analyze_daily_pattern():
    """Get overview of daily scraping pattern"""
    conn = connect_mysql()
    cursor = conn.cursor()
    
    query = '''
    SELECT 
        DATE(created_at) as date,
        COUNT(*) as total_records,
        COUNT(DISTINCT CONCAT(type, '-', strike, '-', expiry)) as unique_contracts,
        MIN(TIME(created_at)) as first_scrape_time,
        MAX(TIME(created_at)) as last_scrape_time,
        COUNT(DISTINCT HOUR(created_at)) as num_hours_with_data,
        MAX(created_at) as latest_timestamp
    FROM option_prices_live
    GROUP BY DATE(created_at)
    ORDER BY date DESC
    LIMIT 20
    '''
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    print('\n' + '='*120)
    print('DAILY SCRAPING PATTERN - option_prices_live')
    print('='*120)
    print(f"{'Date':<12} {'Total':<8} {'Unique':<8} {'First':<10} {'Last':<10} {'Hours':<8} {'Latest Timestamp':<20}")
    print('-'*120)
    
    for row in rows:
        date, total, unique, first, last, hours, latest = row
        print(f"{date!s:<12} {total:<8} {unique:<8} {str(first):<10} {str(last):<10} {hours:<8} {latest!s:<20}")
    
    cursor.close()
    conn.close()
    
    return rows


def get_last_snapshot_per_day():
    """Get the last complete snapshot for each day"""
    conn = connect_mysql()
    cursor = conn.cursor(dictionary=True)
    
    # Find the latest timestamp per day
    query = '''
    WITH daily_latest AS (
        SELECT 
            DATE(created_at) as trade_date,
            MAX(created_at) as last_scrape_time
        FROM option_prices_live
        GROUP BY DATE(created_at)
    )
    SELECT 
        dl.trade_date,
        dl.last_scrape_time,
        COUNT(*) as num_contracts,
        COUNT(CASE WHEN opl.type = 'Call' THEN 1 END) as calls,
        COUNT(CASE WHEN opl.type = 'Put' THEN 1 END) as puts,
        MIN(opl.strike) as min_strike,
        MAX(opl.strike) as max_strike,
        AVG(opl.spot_price) as underlying_price,
        COUNT(CASE WHEN opl.bid IS NOT NULL AND opl.ask IS NOT NULL THEN 1 END) as with_quotes
    FROM daily_latest dl
    INNER JOIN option_prices_live opl 
        ON DATE(opl.created_at) = dl.trade_date 
        AND opl.created_at = dl.last_scrape_time
    GROUP BY dl.trade_date, dl.last_scrape_time
    ORDER BY dl.trade_date DESC
    LIMIT 20
    '''
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    print('\n' + '='*120)
    print('LAST COMPLETE SNAPSHOT PER DAY')
    print('='*120)
    print(f"{'Date':<12} {'Timestamp':<20} {'Contracts':<10} {'Calls':<7} {'Puts':<7} {'Min$':<8} {'Max$':<8} {'Spot$':<8} {'w/Quote':<8}")
    print('-'*120)
    
    for row in rows:
        print(f"{row['trade_date']!s:<12} {row['last_scrape_time']!s:<20} {row['num_contracts']:<10} "
              f"{row['calls']:<7} {row['puts']:<7} {row['min_strike']:<8.2f} {row['max_strike']:<8.2f} "
              f"{row['underlying_price']:<8.2f} {row['with_quotes']:<8}")
    
    cursor.close()
    conn.close()
    
    return rows


def show_sample_from_last_snapshot():
    """Show sample records from the most recent complete snapshot"""
    conn = connect_mysql()
    cursor = conn.cursor(dictionary=True)
    
    # Get latest snapshot
    query = '''
    SELECT MAX(created_at) as latest FROM option_prices_live
    '''
    cursor.execute(query)
    latest = cursor.fetchone()['latest']
    
    # Get records from that exact timestamp
    query = '''
    SELECT 
        ticker, type, expiry, strike, bid, ask, price as mid_price, 
        last_price, volume, spot_price, issue_id,
        iv, delta, gamma, theta, vega,
        created_at
    FROM option_prices_live
    WHERE created_at = %s
    ORDER BY type, strike
    LIMIT 10
    '''
    
    cursor.execute(query, (latest,))
    rows = cursor.fetchall()
    
    print('\n' + '='*120)
    print(f'SAMPLE FROM LATEST SNAPSHOT: {latest}')
    print('='*120)
    
    for i, row in enumerate(rows, 1):
        print(f"\n{i}. {row['type']:<4} Strike={row['strike']:>7.2f} Expiry={row['expiry']:<20}")
        print(f"   Bid={row['bid']:>7.4f} Ask={row['ask']:>7.4f} Last={row['last_price']!s:>7} Vol={row['volume']!s:>6}")
        print(f"   Spot={row['spot_price']:>7.2f} IV={row['iv']!s:>8} Delta={row['delta']!s:>8}")
        print(f"   Issue ID={row['issue_id']}")
    
    cursor.close()
    conn.close()


def main():
    print("\n🔍 ANALYZING option_prices_live TABLE...")
    print(f"Timestamp: {datetime.now()}")
    
    try:
        # 1. Daily pattern overview
        analyze_daily_pattern()
        
        # 2. Last snapshot per day
        get_last_snapshot_per_day()
        
        # 3. Sample from latest
        show_sample_from_last_snapshot()
        
        print('\n' + '='*120)
        print('✅ ANALYSIS COMPLETE')
        print('='*120)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
