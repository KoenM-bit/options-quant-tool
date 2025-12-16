#!/bin/bash
###############################################################################
# Sync Local Database from Synology Production
#
# This script dumps the production PostgreSQL database from Synology and
# restores it to your local Docker environment for testing.
#
# Usage:
#   ./scripts/sync_local_db_from_synology.sh
#
# Prerequisites:
#   - Local Docker containers running (docker-compose up -d)
#   - Network access to Synology (192.168.1.201)
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}   SYNC LOCAL DATABASE FROM SYNOLOGY PRODUCTION${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Configuration
SYNOLOGY_HOST="192.168.1.201"
SYNOLOGY_PORT="5433"
SYNOLOGY_USER="airflow"
SYNOLOGY_PASSWORD="airflow"
SYNOLOGY_DB="ahold_options"

LOCAL_CONTAINER="ahold-options-postgres-1"  # Local postgres container name
LOCAL_PORT="5432"
LOCAL_USER="airflow"
LOCAL_PASSWORD="airflow"
LOCAL_DB="ahold_options"

DUMP_FILE="/tmp/synology_db_dump_$(date +%Y%m%d_%H%M%S).sql"

# Check if local Docker is running
echo -e "${YELLOW}📋 Checking local Docker containers...${NC}"
if ! docker ps | grep -q "${LOCAL_CONTAINER}"; then
    echo -e "${RED}❌ Local PostgreSQL container not running!${NC}"
    echo -e "${YELLOW}   Please start your local Docker environment first:${NC}"
    echo -e "${YELLOW}   docker-compose up -d${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Local Docker containers are running${NC}"
echo ""

# Step 1: Dump Synology database
echo -e "${YELLOW}📦 Step 1: Dumping Synology production database...${NC}"
echo -e "   Host: ${SYNOLOGY_HOST}:${SYNOLOGY_PORT}"
echo -e "   Database: ${SYNOLOGY_DB}"

# Use Docker's pg_dump to match PostgreSQL version
docker exec -i "${LOCAL_CONTAINER}" pg_dump \
    -h "${SYNOLOGY_HOST}" \
    -p "${SYNOLOGY_PORT}" \
    -U "${SYNOLOGY_USER}" \
    -d "${SYNOLOGY_DB}" \
    --clean \
    --if-exists \
    --no-owner \
    --no-privileges \
    > "${DUMP_FILE}"

export PGPASSWORD="${SYNOLOGY_PASSWORD}"

if [ $? -eq 0 ]; then
    DUMP_SIZE=$(du -h "${DUMP_FILE}" | cut -f1)
    echo -e "${GREEN}✅ Database dumped successfully (${DUMP_SIZE})${NC}"
    echo -e "   Saved to: ${DUMP_FILE}"
else
    echo -e "${RED}❌ Failed to dump database!${NC}"
    exit 1
fi
echo ""

# Step 2: Drop and recreate local database
echo -e "${YELLOW}🔄 Step 2: Preparing local database...${NC}"
echo -e "${YELLOW}   ⚠️  This will DROP and recreate the local ${LOCAL_DB} database!${NC}"
read -p "   Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}❌ Aborted by user${NC}"
    rm -f "${DUMP_FILE}"
    exit 1
fi

# Drop all connections to the database
docker exec -i "${LOCAL_CONTAINER}" psql -U "${LOCAL_USER}" -d postgres <<EOF
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE datname = '${LOCAL_DB}' AND pid <> pg_backend_pid();
EOF

# Drop and recreate database
docker exec -i "${LOCAL_CONTAINER}" psql -U "${LOCAL_USER}" -d postgres <<EOF
DROP DATABASE IF EXISTS ${LOCAL_DB};
CREATE DATABASE ${LOCAL_DB} OWNER ${LOCAL_USER};
EOF

echo -e "${GREEN}✅ Local database prepared${NC}"
echo ""

# Step 3: Restore dump to local database
echo -e "${YELLOW}📥 Step 3: Restoring database to local Docker...${NC}"

docker exec -i "${LOCAL_CONTAINER}" psql \
    -U "${LOCAL_USER}" \
    -d "${LOCAL_DB}" \
    < "${DUMP_FILE}"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Database restored successfully${NC}"
else
    echo -e "${RED}❌ Failed to restore database!${NC}"
    exit 1
fi
echo ""

# Step 4: Verify restoration
echo -e "${YELLOW}🔍 Step 4: Verifying restoration...${NC}"

# Get table counts from local DB
docker exec -i "${LOCAL_CONTAINER}" psql -U "${LOCAL_USER}" -d "${LOCAL_DB}" <<EOF
SELECT 
    schemaname,
    relname as tablename,
    n_live_tup as row_count
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY n_live_tup DESC
LIMIT 10;
EOF

echo -e "${GREEN}✅ Top 10 tables restored${NC}"
echo ""

# Cleanup
echo -e "${YELLOW}🧹 Cleaning up dump file...${NC}"
rm -f "${DUMP_FILE}"
echo -e "${GREEN}✅ Cleanup complete${NC}"
echo ""

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}   ✅ DATABASE SYNC COMPLETE!${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""
echo -e "${BLUE}Your local database now contains a copy of Synology production data.${NC}"
echo -e "${BLUE}You can now test DAGs and features locally without affecting production.${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Test your changes locally"
echo -e "  2. Commit and push when ready"
echo -e "  3. Deploy to Synology production"
echo ""
