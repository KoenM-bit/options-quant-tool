"""Check scraping times to find best snapshot (17:15-17:30 range)"""
import mysql.connector
import os

conn = mysql.connector.connect(
    host=os.getenv('MYSQL_HOST', '192.168.1.201'),
    user=os.getenv('MYSQL_USER', 'remoteuser'),
    password=os.getenv('MYSQL_PASSWORD'),
    database=os.getenv('MYSQL_DATABASE', 'optionsdb')
)

cursor = conn.cursor()

# Get last 3 timestamps per day to see the pattern
query = '''
SELECT 
    DATE(created_at) as trade_date,
    created_at as timestamp,
    TIME(created_at) as time_only,
    COUNT(*) as num_records
FROM option_prices_live
WHERE DATE(created_at) >= '2025-12-08'
GROUP BY DATE(created_at), created_at
ORDER BY DATE(created_at) DESC, created_at DESC
'''

cursor.execute(query)
rows = cursor.fetchall()

print('\nALL SCRAPE TIMES (Recent 3 days):')
print('='*100)

current_date = None
for row in rows:
    trade_date, timestamp, time_only, num_records = row
    
    if current_date != trade_date:
        if current_date is not None:
            print('-'*100)
        current_date = trade_date
        print(f'\n📅 {trade_date}:')
    
    print(f'   {timestamp!s:<20} (Time: {time_only!s:<10}) - {num_records:>4} records')

print('\n' + '='*100)

# Now find the 2nd-to-last timestamp per day (around 17:15-17:30)
# Strategy: Get all distinct timestamps, rank them, pick rank=2
query2 = '''
WITH all_timestamps AS (
    SELECT DISTINCT
        DATE(created_at) as trade_date,
        created_at as ts
    FROM option_prices_live
),
ranked AS (
    SELECT 
        trade_date,
        ts,
        ROW_NUMBER() OVER (PARTITION BY trade_date ORDER BY ts DESC) as rn
    FROM all_timestamps
)
SELECT 
    r.trade_date,
    r.ts as selected_timestamp,
    TIME(r.ts) as time_only,
    COUNT(opl.id) as num_records
FROM ranked r
INNER JOIN option_prices_live opl 
    ON DATE(opl.created_at) = r.trade_date 
    AND opl.created_at = r.ts
WHERE r.rn = 2  -- 2nd to last timestamp
GROUP BY r.trade_date, r.ts
ORDER BY r.trade_date DESC
LIMIT 10
'''

cursor.execute(query2)
rows2 = cursor.fetchall()

print('\n🎯 RECOMMENDED SNAPSHOT (2nd-to-last scrape per day):')
print('='*100)
print(f"{'Date':<12} {'Timestamp':<20} {'Time':<10} {'Records':<10}")
print('-'*100)

for row in rows2:
    trade_date, timestamp, time_only, num_records = row
    print(f"{trade_date!s:<12} {timestamp!s:<20} {time_only!s:<10} {num_records:<10}")

print('\n✅ This should give us market-hours data (17:15-17:30) before market close!')

cursor.close()
conn.close()
