#!/bin/bash
# Simple startup script for Ahold Options Platform

set -e

echo "🚀 Starting Ahold Options Platform..."
echo ""

# Check if images are built
if ! docker images | grep -q "ahold-options-airflow"; then
    echo "📦 Docker images not found. Building now (this takes 3-5 minutes)..."
    docker compose build
    echo "✅ Build complete!"
    echo ""
fi

# Start services
echo "🚀 Starting services..."
docker compose up -d

echo "⏳ Waiting for services to initialize (30 seconds)..."
sleep 30

# Check status
echo ""
echo "📊 Service Status:"
docker compose ps

echo ""
echo "✅ Platform started!"
echo ""
echo "📍 Access Points:"
echo "   • Airflow UI: http://localhost:8080"
echo "   • Login: admin / admin"
echo "   • PostgreSQL: localhost:5432"
echo ""
echo "🎯 Next Steps:"
echo "   1. Open Airflow UI: open http://localhost:8080"
echo "   2. Initialize database: docker compose exec airflow-webserver python /opt/airflow/scripts/init_db.py"
echo "   3. Enable DAGs in Airflow UI"
echo "   4. Trigger first scrape: docker compose exec airflow-webserver airflow dags trigger ahold_options_daily"
echo ""
echo "📚 See START_HERE.md for detailed instructions"
echo ""
