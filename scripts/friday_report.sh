#!/bin/bash
# Quick Friday OI & Volume Report
# 
# This script runs Friday-specific OI and volume analysis for ATM strikes
# Usage:
#   ./scripts/friday_report.sh                      # ATM ±6 strikes analysis
#   ./scripts/friday_report.sh --strike-range 8     # Custom strike range
#   ./scripts/friday_report.sh --ticker AH-DAMS     # Specific ticker

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run the Friday ATM analysis
python scripts/friday_atm_oi_analysis.py "$@"
