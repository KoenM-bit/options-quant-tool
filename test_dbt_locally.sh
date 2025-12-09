#!/bin/bash
# Test DBT transformation locally using Docker

echo "🧪 Testing DBT Silver transformation locally..."
echo ""

# Build the image
echo "📦 Building Docker image..."
docker build -t options-test . || exit 1

echo ""
echo "🔄 Running DBT silver models..."
docker run --rm \
  --network host \
  -e POSTGRES_HOST=127.0.0.1 \
  -e POSTGRES_PORT=5432 \
  -e POSTGRES_USER=airflow \
  -e POSTGRES_PASSWORD=airflow \
  -e POSTGRES_DB=ahold_options \
  options-test \
  bash -c "cd /opt/airflow/dbt/ahold_options && dbt run --models tag:silver --profiles-dir /opt/airflow/dbt"

echo ""
echo "✅ Test complete! Check the output above for results."
