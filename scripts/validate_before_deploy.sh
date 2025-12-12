#!/bin/bash
# Pre-deployment validation script
# Run this BEFORE deploying to Synology to catch issues early

set -e

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                      PRE-DEPLOYMENT VALIDATION                             ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

FAILED=0

echo -e "${BLUE}1️⃣  Python Syntax Check${NC}"
echo "────────────────────────────────────────────────────────────────────────────"
python3 -m py_compile dags/*.py src/**/*.py scripts/*.py 2>/dev/null && {
    echo -e "${GREEN}✅ All Python files have valid syntax${NC}"
} || {
    echo -e "${RED}❌ Python syntax errors found${NC}"
    FAILED=1
}
echo ""

echo -e "${BLUE}2️⃣  Critical Regression Checks${NC}"
echo "────────────────────────────────────────────────────────────────────────────"

# Check for execution_date usage (common mistake)
if grep -r "execution_date.date()" dags/ --include="*.py" | grep -v "^#" | grep "trade_date" > /dev/null; then
    echo -e "${RED}❌ CRITICAL: Found execution_date.date() used for trade_date${NC}"
    echo "   Must use datetime.now().date() for today's data"
    FAILED=1
else
    echo -e "${GREEN}✅ Date logic correct (using datetime.now())${NC}"
fi

# Check DAG IDs don't conflict
echo -e "${GREEN}✅ Checking DAG ID uniqueness...${NC}"
DAG_IDS=$(grep -h "DAG(" dags/*.py | grep -o "'[^']*'" | sort | uniq -d)
if [ ! -z "$DAG_IDS" ]; then
    echo -e "${RED}❌ Duplicate DAG IDs found: $DAG_IDS${NC}"
    FAILED=1
else
    echo -e "${GREEN}✅ All DAG IDs are unique${NC}"
fi
echo ""

echo -e "${BLUE}3️⃣  End-to-End Pipeline Test${NC}"
echo "────────────────────────────────────────────────────────────────────────────"
python3 scripts/test_pipeline_e2e.py || {
    echo -e "${RED}❌ Pipeline test failed${NC}"
    FAILED=1
}
echo ""

echo -e "${BLUE}4️⃣  DAG File Validation${NC}"
echo "────────────────────────────────────────────────────────────────────────────"

# Check required DAGs exist
REQUIRED_DAGS=(
    "dags/options_bronze_silver_pipeline.py"
    "dags/gold_daily_analytics.py"
    "dags/gold_weekly_analytics.py"
)

for dag in "${REQUIRED_DAGS[@]}"; do
    if [ -f "$dag" ]; then
        echo -e "${GREEN}✅ Found: $dag${NC}"
    else
        echo -e "${RED}❌ Missing: $dag${NC}"
        FAILED=1
    fi
done
echo ""

echo -e "${BLUE}5️⃣  Database Model Checks${NC}"
echo "────────────────────────────────────────────────────────────────────────────"

# Check models have required fields
if grep -q "class BronzeBDOptions" src/models/bronze_bd.py && \
   grep -q "trade_date" src/models/bronze_bd.py && \
   grep -q "UniqueConstraint" src/models/bronze_bd.py; then
    echo -e "${GREEN}✅ BronzeBDOptions model looks good${NC}"
else
    echo -e "${RED}❌ BronzeBDOptions model missing required fields${NC}"
    FAILED=1
fi

if grep -q "open_interest" src/models/bronze.py; then
    echo -e "${GREEN}✅ FD models have open_interest field${NC}"
else
    echo -e "${RED}❌ FD models missing open_interest${NC}"
    FAILED=1
fi
echo ""

# Final result
echo "╔════════════════════════════════════════════════════════════════════════════╗"
if [ $FAILED -eq 0 ]; then
    echo -e "║  ${GREEN}✅ ALL VALIDATION CHECKS PASSED${NC}                                          ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo -e "${GREEN}🚀 SAFE TO DEPLOY TO SYNOLOGY${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. git add ."
    echo "  2. git commit -m \"Your commit message\""
    echo "  3. git push origin master"
    echo "  4. SSH to Synology and git pull"
    echo "  5. Run: docker compose up -d --build"
    echo ""
    exit 0
else
    echo -e "║  ${RED}❌ VALIDATION FAILED${NC}                                                      ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo -e "${RED}🛑 DO NOT DEPLOY - FIX ERRORS FIRST${NC}"
    echo ""
    exit 1
fi
