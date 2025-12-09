#!/bin/bash

# Setup script for local development
# Run: bash scripts/setup.sh

set -e  # Exit on error

echo "🚀 Ahold Options Platform - Setup Script"
echo "=========================================="

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose found"

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env and set:"
    echo "  1. AIRFLOW__CORE__FERNET_KEY"
    echo "  2. Database passwords"
    echo "  3. Admin passwords"
    echo ""
    echo "Generate Fernet key with:"
    echo "  python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
    echo ""
    read -p "Press Enter after editing .env to continue..."
else
    echo "✅ .env file exists"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs plugins data/raw data/processed

# Set permissions
echo "🔐 Setting permissions..."
chmod -R 755 logs plugins

# Build images
echo "🏗️  Building Docker images..."
docker compose build

# Start services
echo "🚀 Starting services..."
docker compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 30

# Check service status
echo "🔍 Checking service status..."
docker compose ps

# Initialize database
echo "💾 Initializing database..."
docker compose exec -T airflow-webserver python /opt/airflow/scripts/init_db.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "📊 Access Airflow UI: http://localhost:8080"
echo "🔐 Default credentials: admin/admin (change in .env)"
echo ""
echo "🎯 Next steps:"
echo "  1. Log in to Airflow UI"
echo "  2. Enable DAGs (toggle switches)"
echo "  3. Trigger 'ahold_options_daily' DAG manually"
echo "  4. Check execution in Graph view"
echo ""
echo "📚 Documentation: docs/QUICKSTART.md"
echo ""
