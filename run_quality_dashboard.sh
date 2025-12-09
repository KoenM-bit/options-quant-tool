#!/bin/bash
# Launch Data Quality Monitoring Dashboard (connects directly to PostgreSQL)

echo "=================================================="
echo "Ahold Options - Data Quality Monitor"
echo "=================================================="
echo ""
echo "This dashboard connects directly to PostgreSQL"
echo "Make sure Docker containers are running!"
echo ""
echo "Starting dashboard on http://localhost:8501"
echo "Press Ctrl+C to stop"
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run Streamlit
streamlit run dashboards/data_quality_monitor.py \
    --server.port=8501 \
    --server.address=localhost \
    --browser.gatherUsageStats=false
