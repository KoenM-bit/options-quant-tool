#!/bin/bash
# Verification script for Synology ClickHouse deployment
# Run this on your Synology via SSH

echo "=== ClickHouse Tables ==="
docker exec ahold-options-clickhouse clickhouse-client --user default --password clickhouse123 --query "SELECT table, total_rows FROM system.tables WHERE database = 'ahold_options' ORDER BY table FORMAT PrettyCompact"

echo ""
echo "=== MinIO Buckets ==="
docker exec ahold-options-minio mc ls local/

echo ""
echo "=== MinIO Parquet Files (Gold) ==="
docker exec ahold-options-minio mc ls local/options-data/parquet/gold/

echo ""
echo "=== MinIO Parquet Files (Silver) ==="
docker exec ahold-options-minio mc ls local/options-data/parquet/silver/

echo ""
echo "=== PostgreSQL Gold Tables ==="
docker exec ahold-options-postgres psql -U airflow -d ahold_options -c "SELECT schemaname, tablename FROM pg_tables WHERE schemaname = 'gold' ORDER BY tablename"

echo ""
echo "=== Last Sync Task Output ==="
docker exec ahold-options-airflow-webserver-1 python /opt/airflow/scripts/sync_to_clickhouse.py
