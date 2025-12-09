#!/bin/bash
# Script to run the Streamlit dashboard locally
# This copies the Parquet files from Docker and launches the dashboard

set -e

echo "🚀 Starting Streamlit Dashboard Setup"
echo ""

# Create local data directory
echo "📁 Creating local data directory..."
mkdir -p data/parquet

# Copy Parquet files from Docker container
echo "📦 Copying Parquet files from Docker..."
docker compose cp airflow-webserver:/opt/airflow/data/parquet/. data/parquet/

# Check if files were copied
if [ ! -f "data/parquet/gold_gex_positioning_trends.parquet" ]; then
    echo "❌ Error: Parquet files not found!"
    echo "   Please ensure the export script has been run."
    exit 1
fi

echo "✅ Parquet files copied successfully"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "📦 Installing Streamlit..."
    pip install streamlit plotly pyarrow pandas
fi

echo ""
echo "🎨 Launching Streamlit Dashboard..."
echo "   Dashboard will open in your browser at http://localhost:8501"
echo ""
echo "   Press Ctrl+C to stop the dashboard"
echo ""

# Set environment variable for data directory
export PARQUET_DATA_DIR="data/parquet"

# Launch Streamlit
streamlit run streamlit_app.py
