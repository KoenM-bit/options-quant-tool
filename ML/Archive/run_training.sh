#!/bin/bash
# Run ML training script with proper environment setup

set -e

# Load environment variables if .env exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set default model version if not provided
export MODEL_VERSION=${MODEL_VERSION:-"range10_xgb_v1_$(date +%Y-%m-%d)"}

echo "=========================================="
echo "ML Training: Range 10-Day Prediction"
echo "=========================================="
echo "Model Version: $MODEL_VERSION"
echo "Database: $POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB"
echo "=========================================="

# Run training
python ML/train_range10.py

echo ""
echo "✅ Training complete!"
echo "Check predictions in table: gold_predictions_range10_ad"
