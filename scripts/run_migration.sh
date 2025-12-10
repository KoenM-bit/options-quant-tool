#!/bin/bash
# Simple migration runner to copy historical data from MariaDB to PostgreSQL
# Usage:
#   ./run_migration.sh                    # Dry run (show what would be migrated)
#   ./run_migration.sh --execute          # Actually migrate all data
#   ./run_migration.sh --execute --limit 10   # Migrate only last 10 dates
#   ./run_migration.sh --execute --date 2025-12-09  # Migrate specific date

set -e

echo "=================================================="
echo "Historical Data Migration Tool"
echo "MariaDB (192.168.1.201) → PostgreSQL (Docker)"
echo "=================================================="
echo ""

# Check if .env.migration exists
if [ ! -f ".env.migration" ]; then
    echo "❌ Error: .env.migration file not found"
    echo "Please create it with your MariaDB and PostgreSQL credentials"
    exit 1
fi

# Check if migration script exists
if [ ! -f "scripts/migrate_historical_data.py" ]; then
    echo "❌ Error: scripts/migrate_historical_data.py not found"
    exit 1
fi

# Parse arguments
DRY_RUN="--dry-run"
ARGS=""

for arg in "$@"; do
    if [ "$arg" = "--execute" ]; then
        DRY_RUN=""
    else
        ARGS="$ARGS $arg"
    fi
done

# Show what we're doing
if [ -z "$DRY_RUN" ]; then
    echo "⚠️  EXECUTE MODE: Data will be written to PostgreSQL"
    echo "Press Ctrl+C within 3 seconds to cancel..."
    sleep 3
else
    echo "ℹ️  DRY-RUN MODE: No data will be written"
    echo ""
fi

# Run the migration
echo "Starting migration..."
echo ""

python scripts/migrate_historical_data.py $DRY_RUN $ARGS

echo ""
echo "✅ Migration complete!"
