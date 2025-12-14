#!/bin/bash
# Cleanup script to delete all data from Synology remote instance
# This connects directly to exposed ports on 192.168.1.201

set -e

SYNOLOGY_HOST="192.168.1.201"
POSTGRES_PORT="5433"
POSTGRES_USER="airflow"
POSTGRES_PASSWORD="airflow"
POSTGRES_DB="ahold_options"

MINIO_ENDPOINT="192.168.1.201:9000"
MINIO_ACCESS_KEY="admin"
MINIO_SECRET_KEY="miniopassword123"

CLICKHOUSE_HOST="192.168.1.201"
CLICKHOUSE_PORT="8123"
CLICKHOUSE_USER="default"
CLICKHOUSE_PASSWORD="clickhouse123"

echo "=========================================="
echo "🧹 SYNOLOGY REMOTE DATA CLEANUP"
echo "=========================================="
echo ""
echo "⚠️  WARNING: This will delete ALL data from Synology (${SYNOLOGY_HOST}):"
echo "  - PostgreSQL bronze tables"
echo "  - PostgreSQL silver tables"
echo "  - MinIO buckets"
echo "  - ClickHouse tables"
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "=========================================="
echo "1️⃣  Cleaning PostgreSQL Bronze Tables"
echo "=========================================="

PGPASSWORD=${POSTGRES_PASSWORD} psql -h ${SYNOLOGY_HOST} -p ${POSTGRES_PORT} -U ${POSTGRES_USER} -d ${POSTGRES_DB} <<EOF
TRUNCATE TABLE bronze_bd_options CASCADE;
TRUNCATE TABLE bronze_bd_underlying CASCADE;
TRUNCATE TABLE bronze_fd_options CASCADE;
TRUNCATE TABLE bronze_fd_overview CASCADE;
SELECT 'Bronze tables cleaned' as status;
EOF

echo ""
echo "=========================================="
echo "2️⃣  Cleaning PostgreSQL Silver Tables"
echo "=========================================="

PGPASSWORD=${POSTGRES_PASSWORD} psql -h ${SYNOLOGY_HOST} -p ${POSTGRES_PORT} -U ${POSTGRES_USER} -d ${POSTGRES_DB} <<EOF
TRUNCATE TABLE dim_underlying CASCADE;
TRUNCATE TABLE dim_option_contract CASCADE;
TRUNCATE TABLE fact_option_timeseries CASCADE;
SELECT 'Silver tables cleaned' as status;
EOF

echo ""
echo "=========================================="
echo "3️⃣  Cleaning MinIO Buckets"
echo "=========================================="
echo ""
echo "⚠️  Please clean MinIO manually via web UI at http://${MINIO_ENDPOINT}"
echo "   1. Login with credentials: ${MINIO_ACCESS_KEY}"
echo "   2. Navigate to 'options-data' bucket"
echo "   3. Delete bronze/, silver/, and gold/ folders"
echo ""
read -p "Press Enter once MinIO is cleaned..."

echo ""
echo "=========================================="
echo "4️⃣  Cleaning ClickHouse Tables"
echo "=========================================="

echo "Truncating dim_underlying..."
curl -X POST "http://${CLICKHOUSE_HOST}:${CLICKHOUSE_PORT}/?user=${CLICKHOUSE_USER}&password=${CLICKHOUSE_PASSWORD}" --data-binary "TRUNCATE TABLE ahold_options.dim_underlying" 2>/dev/null
echo "✓ dim_underlying truncated"

echo "Truncating dim_option_contract..."
curl -X POST "http://${CLICKHOUSE_HOST}:${CLICKHOUSE_PORT}/?user=${CLICKHOUSE_USER}&password=${CLICKHOUSE_PASSWORD}" --data-binary "TRUNCATE TABLE ahold_options.dim_option_contract" 2>/dev/null
echo "✓ dim_option_contract truncated"

echo "Truncating fact_option_timeseries..."
curl -X POST "http://${CLICKHOUSE_HOST}:${CLICKHOUSE_PORT}/?user=${CLICKHOUSE_USER}&password=${CLICKHOUSE_PASSWORD}" --data-binary "TRUNCATE TABLE ahold_options.fact_option_timeseries" 2>/dev/null
echo "✓ fact_option_timeseries truncated"

echo ""
echo "=========================================="
echo "5️⃣  Verification - Checking Record Counts"
echo "=========================================="

echo ""
echo "PostgreSQL Bronze:"
PGPASSWORD=${POSTGRES_PASSWORD} psql -h ${SYNOLOGY_HOST} -p ${POSTGRES_PORT} -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c "
SELECT 
    'bronze_bd_options' as table_name, COUNT(*) as count FROM bronze_bd_options
UNION ALL
SELECT 'bronze_bd_underlying', COUNT(*) FROM bronze_bd_underlying
UNION ALL
SELECT 'bronze_fd_options', COUNT(*) FROM bronze_fd_options
UNION ALL
SELECT 'bronze_fd_overview', COUNT(*) FROM bronze_fd_overview;
"

echo ""
echo "PostgreSQL Silver:"
PGPASSWORD=${POSTGRES_PASSWORD} psql -h ${SYNOLOGY_HOST} -p ${POSTGRES_PORT} -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c "
SELECT 
    'dim_underlying' as table_name, COUNT(*) as count FROM dim_underlying
UNION ALL
SELECT 'dim_option_contract', COUNT(*) FROM dim_option_contract
UNION ALL
SELECT 'fact_option_timeseries', COUNT(*) FROM fact_option_timeseries;
"

echo ""
echo "ClickHouse:"
curl "http://${CLICKHOUSE_HOST}:${CLICKHOUSE_PORT}/?user=${CLICKHOUSE_USER}&password=${CLICKHOUSE_PASSWORD}" --data-binary "SELECT 'dim_underlying' as table_name, count() as count FROM ahold_options.dim_underlying UNION ALL SELECT 'dim_option_contract', count() FROM ahold_options.dim_option_contract UNION ALL SELECT 'fact_option_timeseries', count() FROM ahold_options.fact_option_timeseries FORMAT PrettyCompact"

echo ""
echo "=========================================="
echo "✅ CLEANUP COMPLETE"
echo "=========================================="
echo ""
echo "All data layers are now clean on Synology!"
echo "You can now run a fresh end-to-end test."
echo ""
echo "To trigger the pipeline on Synology:"
echo "  1. Go to Airflow UI: http://${SYNOLOGY_HOST}:8081"
echo "  2. Enable and trigger DAG: options_bronze_silver_pipeline"
echo ""
