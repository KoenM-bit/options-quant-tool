MAKEFLAGS += --silent

.PHONY: help setup start stop restart logs test clean

help:
	@echo "Ahold Options Platform - Make Commands"
	@echo "======================================="
	@echo ""
	@echo "Setup:"
	@echo "  make setup        - Initial setup (run once)"
	@echo "  make start        - Start all services"
	@echo "  make stop         - Stop all services"
	@echo "  make restart      - Restart all services"
	@echo ""
	@echo "Development:"
	@echo "  make logs         - View logs (all services)"
	@echo "  make logs-airflow - View Airflow logs"
	@echo "  make logs-db      - View database logs"
	@echo "  make shell        - Open shell in webserver"
	@echo "  make db-shell     - Open PostgreSQL shell"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run tests"
	@echo "  make test-scraper - Test scraper"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo ""
	@echo "Database:"
	@echo "  make init-db      - Initialize database"
	@echo "  make backup-db    - Backup database"
	@echo "  make dbt-run      - Run DBT models"
	@echo "  make dbt-test     - Run DBT tests"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        - Clean up containers and volumes"
	@echo "  make clean-logs   - Clean log files"
	@echo ""

setup:
	@echo "🚀 Setting up Ahold Options Platform..."
	bash scripts/setup.sh

start:
	@echo "🚀 Starting services..."
	docker compose up -d
	@echo "✅ Services started"
	@echo "📊 Airflow UI: http://localhost:8080"

stop:
	@echo "🛑 Stopping services..."
	docker compose down
	@echo "✅ Services stopped"

restart:
	@echo "♻️  Restarting services..."
	docker compose restart
	@echo "✅ Services restarted"

logs:
	docker compose logs -f

logs-airflow:
	docker compose logs -f airflow-webserver airflow-scheduler

logs-db:
	docker compose logs -f postgres

shell:
	docker compose exec airflow-webserver bash

db-shell:
	docker compose exec postgres psql -U airflow -d ahold_options

test:
	@echo "🧪 Running tests..."
	docker compose exec airflow-webserver pytest tests/ -v

test-scraper:
	@echo "🧪 Testing scraper..."
	python scripts/test_scraper.py

lint:
	@echo "🔍 Running linters..."
	flake8 src/ dags/
	mypy src/

format:
	@echo "✨ Formatting code..."
	black src/ dags/ tests/
	isort src/ dags/ tests/

init-db:
	@echo "💾 Initializing database..."
	docker compose exec airflow-webserver python /opt/airflow/scripts/init_db.py

backup-db:
	@echo "💾 Backing up database..."
	docker compose exec postgres pg_dump -U airflow ahold_options > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "✅ Backup created"

dbt-run:
	@echo "🔄 Running DBT models..."
	docker compose exec airflow-webserver bash -c "cd /opt/airflow/dbt/ahold_options && dbt run"

dbt-test:
	@echo "🧪 Running DBT tests..."
	docker compose exec airflow-webserver bash -c "cd /opt/airflow/dbt/ahold_options && dbt test"

dbt-docs:
	@echo "📚 Generating DBT docs..."
	docker compose exec airflow-webserver bash -c "cd /opt/airflow/dbt/ahold_options && dbt docs generate"

clean:
	@echo "🧹 Cleaning up..."
	docker compose down -v
	rm -rf logs/* plugins/*
	@echo "✅ Cleanup complete"

clean-logs:
	@echo "🧹 Cleaning logs..."
	rm -rf logs/*
	@echo "✅ Logs cleaned"

build:
	@echo "🏗️  Building images..."
	docker compose build --no-cache

trigger-scrape:
	@echo "▶️  Triggering scrape DAG..."
	docker compose exec airflow-webserver airflow dags trigger ahold_options_daily

trigger-dbt:
	@echo "▶️  Triggering DBT DAG..."
	docker compose exec airflow-webserver airflow dags trigger ahold_dbt_transform

status:
	@echo "📊 Service status:"
	docker compose ps
