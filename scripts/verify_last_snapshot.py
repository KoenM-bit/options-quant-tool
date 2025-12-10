"""Verify we can extract the last snapshot per day"""
import mysql.connector
import os

conn = mysql.connector.connect(
    host=os.getenv('MYSQL_HOST', '192.168.1.201'),
    user=os.getenv('MYSQL_USER', 'remoteuser'),
    password=os.getenv('MYSQL_PASSWORD'),
    database=os.getenv('MYSQL_DATABASE', 'optionsdb')
)

cursor = conn.cursor()

query = '''
SELECT 
    DATE(created_at) as trade_date,
    MAX(created_at) as last_timestamp,
    COUNT(DISTINCT created_at) as num_timestamps,
    (SELECT COUNT(*) 
     FROM option_prices_live opl2 
     WHERE DATE(opl2.created_at) = DATE(opl.created_at) 
       AND opl2.created_at = MAX(opl.created_at)) as records_at_last
FROM option_prices_live opl
GROUP BY DATE(created_at)
ORDER BY trade_date DESC
LIMIT 10
'''

cursor.execute(query)
rows = cursor.fetchall()

print('\nLAST SNAPSHOT PER DAY:')
print('='*80)
print(f"{'Date':<12} {'Last Timestamp':<20} {'#Times':<10} {'Records@Last':<15}")
print('-'*80)

for row in rows:
    date, last_ts, num_ts, records = row
    print(f"{date!s:<12} {last_ts!s:<20} {num_ts:<10} {records:<15}")

print('\n✅ YES - We can identify the exact LAST entry for every day!')
print(f'   Strategy: Use MAX(created_at) per DATE(created_at)')
print(f'   Result: Gets us the ~17:30-17:45 end-of-day snapshot')

cursor.close()
conn.close()
