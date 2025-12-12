#!/bin/bash
# Cleanup script to delete all data from bronze, silver, MinIO, and ClickHouse
# Run this to test the full pipeline from scratch

set -e

echo "=========================================="
echo "🧹 FULL DATA CLEANUP"
echo "=========================================="
echo ""
echo "⚠️  WARNING: This will delete ALL data from:"
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

docker compose exec -T postgres psql -U airflow -d ahold_options <<EOF
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

docker compose exec -T postgres psql -U airflow -d ahold_options <<EOF
TRUNCATE TABLE dim_underlying CASCADE;
TRUNCATE TABLE dim_option_contract CASCADE;
TRUNCATE TABLE fact_option_timeseries CASCADE;
SELECT 'Silver tables cleaned' as status;
EOF

echo ""
echo "=========================================="
echo "3️⃣  Cleaning MinIO Buckets"
echo "=========================================="

# Use MinIO client (mc) inside minio container
docker compose exec -T minio sh <<'EOF'
# Configure mc alias
mc alias set local http://localhost:9000 $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD 2>/dev/null || true

# Remove all objects from bronze and silver
echo "Removing bronze data..."
mc rm --recursive --force local/options-data/bronze/ 2>/dev/null || echo "No bronze data to remove"

echo "Removing silver data..."
mc rm --recursive --force local/options-data/silver/ 2>/dev/null || echo "No silver data to remove"

echo "Removing gold data..."
mc rm --recursive --force local/options-data/gold/ 2>/dev/null || echo "No gold data to remove"

echo "MinIO cleaned"
EOF

echo ""
echo "=========================================="
echo "4️⃣  Cleaning ClickHouse Tables"
echo "=========================================="

docker compose exec -T clickhouse clickhouse-client --query "TRUNCATE TABLE ahold_options.dim_underlying" 2>/dev/null || echo "dim_underlying already empty"
docker compose exec -T clickhouse clickhouse-client --query "TRUNCATE TABLE ahold_options.dim_option_contract" 2>/dev/null || echo "dim_option_contract already empty"
docker compose exec -T clickhouse clickhouse-client --query "TRUNCATE TABLE ahold_options.fact_option_timeseries" 2>/dev/null || echo "fact_option_timeseries already empty"

echo ""
echo "=========================================="
echo "5️⃣  Verification - Checking Record Counts"
echo "=========================================="

echo ""
echo "PostgreSQL Bronze:"
docker compose exec -T postgres psql -U airflow -d ahold_options -c "
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
docker compose exec -T postgres psql -U airflow -d ahold_options -c "
SELECT 
    'dim_underlying' as table_name, COUNT(*) as count FROM dim_underlying
UNION ALL
SELECT 'dim_option_contract', COUNT(*) FROM dim_option_contract
UNION ALL
SELECT 'fact_option_timeseries', COUNT(*) FROM fact_option_timeseries;
"

echo ""
echo "ClickHouse:"
docker compose exec -T clickhouse clickhouse-client --query "
SELECT 'dim_underlying' as table_name, COUNT(*) as count FROM ahold_options.dim_underlying
UNION ALL
SELECT 'dim_option_contract', COUNT(*) FROM ahold_options.dim_option_contract
UNION ALL
SELECT 'fact_option_timeseries', COUNT(*) FROM ahold_options.fact_option_timeseries
FORMAT Pretty
"

echo ""
echo "=========================================="
echo "✅ CLEANUP COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Go to Airflow UI: http://localhost:8081"
echo "2. Trigger the 'options_bronze_silver_pipeline' DAG"
echo "3. Watch it scrape, transform, export, and sync data for both AD.AS and MT.AS"
echo ""
