#!/usr/bin/env python3
"""Check specific record from MariaDB"""
import mysql.connector
import sys

# MariaDB connection
conn = mysql.connector.connect(
    host='192.168.1.201',
    user='remoteuser',
    password='Secure2024!',
    database='optionsdb'
)

cursor = conn.cursor(dictionary=True)

issue_id = '520070160'
trade_date = '2025-12-10'

# Check all scrapes for this record on this date
query = """
WITH ranked_scrapes AS (
    SELECT 
        ticker, issue_id, type, strike, bid, ask,
        last_price, last_time, volume, 
        DATE(created_at) as trade_date,
        created_at as scrape_time,
        ROW_NUMBER() OVER (PARTITION BY DATE(created_at) ORDER BY created_at DESC) as rn
    FROM option_prices_live
    WHERE issue_id = %s
      AND DATE(created_at) = %s
)
SELECT * FROM ranked_scrapes
ORDER BY scrape_time DESC
LIMIT 10
"""

cursor.execute(query, (issue_id, trade_date))
results = cursor.fetchall()

print(f"\nFound {len(results)} scrapes for issue_id={issue_id} on {trade_date}:")
print("=" * 100)

for row in results:
    print(f"\nScrape #{row['rn']} at {row['scrape_time']}")
    print(f"  Strike: {row['strike']}, Bid: {row['bid']}, Ask: {row['ask']}")
    print(f"  Last Price: {row['last_price']}, Last Time: {row['last_time']}, Volume: {row['volume']}")

# Now check the rn=2 record specifically
print("\n" + "=" * 100)
print("\nFocusing on rn=2 (2nd-to-last scrape) which migration would use:")
print("=" * 100)

query_rn2 = """
WITH ranked_scrapes AS (
    SELECT 
        ticker, issue_id, type, strike, bid, ask,
        last_price, last_time, volume, 
        DATE(created_at) as trade_date,
        created_at as scrape_time,
        ROW_NUMBER() OVER (PARTITION BY DATE(created_at) ORDER BY created_at DESC) as rn
    FROM option_prices_live
    WHERE issue_id = %s
      AND DATE(created_at) = %s
)
SELECT * FROM ranked_scrapes
WHERE rn = 2
"""

cursor.execute(query_rn2, (issue_id, trade_date))
rn2 = cursor.fetchone()

if rn2:
    print(f"\nrn=2 Scrape at {rn2['scrape_time']}")
    print(f"  Strike: {rn2['strike']}, Bid: {rn2['bid']}, Ask: {rn2['ask']}")
    print(f"  Last Price: {rn2['last_price']}")
    print(f"  Last Time: {rn2['last_time']}")
    print(f"  Volume: {rn2['volume']}")
    
    if rn2['last_time'] is None:
        print("\n⚠️  last_time IS NULL in MariaDB for this rn=2 record")
        print("   This means NULL last_timestamp in PostgreSQL is CORRECT")
    else:
        print(f"\n✅ last_time has value: {rn2['last_time']}")
        print("   Migration should populate last_timestamp with this value")
else:
    print("\n❌ No rn=2 record found!")

cursor.close()
conn.close()
