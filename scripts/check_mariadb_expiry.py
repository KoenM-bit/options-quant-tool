import mysql.connector
import os

conn = mysql.connector.connect(
    host=os.getenv('MYSQL_HOST', '192.168.1.201'),
    user=os.getenv('MYSQL_USER', 'remoteuser'),
    password=os.getenv('MYSQL_PASSWORD'),
    database=os.getenv('MYSQL_DATABASE', 'optionsdb')
)

cursor = conn.cursor(dictionary=True)

query = """
SELECT 
    ticker, issue_id, expiry, 
    last_price, last_time, volume,
    created_at
FROM option_prices_live 
WHERE last_time IS NOT NULL 
ORDER BY created_at DESC 
LIMIT 3
"""

cursor.execute(query)
rows = cursor.fetchall()

print('\nSample records with last_time data:')
print('='*80)
for row in rows:
    print(f"Issue: {row['issue_id']}")
    print(f"  expiry: '{row['expiry']}'")
    print(f"  last_price: {row['last_price']}")
    print(f"  last_time: {row['last_time']}")
    print(f"  volume: {row['volume']}")
    print()

cursor.close()
conn.close()
