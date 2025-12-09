# Synology Portainer Deployment Guide

This guide explains how to deploy the Ahold Options Quantitative Tool on Synology NAS using Portainer.

## Overview

The deployment consists of:
- **PostgreSQL 15**: Database for Airflow metadata and options data
- **Redis 7**: Cache and rate limiting storage
- **Airflow Webserver**: Web UI (port 8091)
- **Airflow Scheduler**: Background job scheduler
- **Adminer**: Database management UI (port 8092)

## Prerequisites

1. **Synology NAS** with Docker support
2. **Portainer** installed and running
3. **Network ports available**:
   - 5433: PostgreSQL
   - 8091: Airflow Web UI
   - 8092: Adminer (Database GUI)

## Deployment Steps

### 1. Generate Fernet Key

On your local machine, generate a Fernet key for Airflow:

```bash
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Copy the output - you'll need it for the environment variables.

### 2. Create Stack in Portainer

1. Log into **Portainer**
2. Go to **Stacks** → **Add stack**
3. Name it: `ahold-options`
4. Choose **Web editor**
5. Copy the contents of `docker-compose.synology.yml` into the editor

### 3. Configure Environment Variables

In the **Environment variables** section, add these variables:

#### Required Variables

```bash
# Airflow Configuration
AIRFLOW_UID=50000
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__CORE__FERNET_KEY=<your-generated-fernet-key>

# Admin Credentials (CHANGE THESE!)
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=<your-secure-password>

# Database Configuration (CHANGE PASSWORD!)
POSTGRES_USER=airflow
POSTGRES_PASSWORD=<your-secure-database-password>
POSTGRES_DB=ahold_options

# Application Settings
ENVIRONMENT=production
LOG_LEVEL=INFO

# Scraper Configuration
SCRAPER_USER_AGENT=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36
SCRAPER_TIMEOUT=30
SCRAPER_RETRY_ATTEMPTS=3

# Ahold Settings
AHOLD_TICKER=AD.AS
AHOLD_SYMBOL_CODE=AEX.AH/O
```

### 4. Deploy the Stack

1. Click **Deploy the stack**
2. Wait for all containers to start (this may take 2-3 minutes)
3. Check the **Containers** view to ensure all services are running

### 5. Verify Deployment

Check that all containers are healthy:

- ✅ `ahold-options-postgres` - Running
- ✅ `ahold-options-redis` - Running
- ✅ `ahold-options-airflow-init` - Completed (exits after initialization)
- ✅ `ahold-options-airflow-webserver` - Running
- ✅ `ahold-options-airflow-scheduler` - Running
- ✅ `ahold-options-adminer` - Running

### 6. Access the Application

#### Airflow Web UI
- URL: `http://<your-synology-ip>:8091`
- Username: `admin` (or what you configured)
- Password: Your configured password

#### Adminer (Database GUI)
- URL: `http://<your-synology-ip>:8092`
- System: PostgreSQL
- Server: postgres
- Username: airflow (or what you configured)
- Password: Your configured database password
- Database: ahold_options

## Key Changes from Local Development

### 1. **No Local Volume Mounts**
- Code is baked into the Docker image `ghcr.io/koenm-bit/options-quant-tool:latest`
- Only persistent data (logs, database, cache) uses volumes
- No need to mount source code directories

### 2. **Security Improvements**
- All containers run as non-root users (UID 50000)
- Uses `_AIRFLOW_DB_MIGRATE` instead of deprecated `_AIRFLOW_DB_UPGRADE`
- Redis configured with rate limiting for production use
- Explicit network isolation

### 3. **Production-Ready Configuration**
- Automatic restart policies (`unless-stopped`)
- Health checks on all services
- Proper dependency ordering
- Resource limits (Redis: 256MB max memory)

### 4. **Port Mappings for Synology**
- PostgreSQL: `5433:5432` (avoids conflicts with system PostgreSQL)
- Airflow UI: `8091:8080` (custom port)
- Adminer: `8092:8080` (custom port)

## Troubleshooting

### Container Warnings

If you see these warnings, they're already addressed:

✅ **"Container runs as root"** - Fixed: All services use non-root user (UID 50000)
✅ **"_AIRFLOW_DB_UPGRADE is deprecated"** - Fixed: Using `_AIRFLOW_DB_MIGRATE`
✅ **"Flask-Limiter in-memory storage"** - Fixed: Using Redis backend

### Container Logs

To view logs in Portainer:
1. Go to **Containers**
2. Click on the container name
3. Click **Logs**

### Common Issues

#### Issue: airflow-init fails
**Solution**: Check PostgreSQL and Redis are healthy first

#### Issue: Can't access web UI
**Solution**: 
- Check firewall rules on Synology
- Verify port 8091 is not already in use
- Check container logs for errors

#### Issue: Database connection errors
**Solution**: 
- Verify environment variables are set correctly
- Check PostgreSQL container is running and healthy
- Verify POSTGRES_PASSWORD matches in all places

## Updating the Application

To update to a new version:

1. Pull the latest image:
   ```bash
   docker pull ghcr.io/koenm-bit/options-quant-tool:latest
   ```

2. In Portainer, go to your stack
3. Click **Update the stack**
4. Check "Re-pull image"
5. Click **Update**

## Backup Strategy

### Database Backup

Connect to the PostgreSQL container:
```bash
docker exec -it ahold-options-postgres pg_dump -U airflow ahold_options > backup.sql
```

### Volume Backup

Backup these volumes in Portainer:
- `postgres-db-volume` - Database data
- `airflow-logs` - Airflow logs
- `airflow-data` - Processed data
- `redis-volume` - Redis cache (optional)

## Monitoring

### Key Metrics to Monitor

1. **Airflow Scheduler**: Should always be running
2. **Database Size**: Monitor PostgreSQL volume usage
3. **DAG Execution**: Check Airflow UI for failed runs
4. **Container Health**: All containers should be "healthy"

### Health Check Endpoints

- Airflow Webserver: `http://localhost:8091/health`
- Airflow Scheduler: `http://localhost:8974/health` (internal)

## Security Recommendations

1. ✅ **Change default passwords** in environment variables
2. ✅ **Use strong Fernet key** (generated with cryptography library)
3. ✅ **Restrict network access** using Synology firewall
4. ✅ **Regular backups** of PostgreSQL database
5. ✅ **Monitor logs** for suspicious activity
6. 🔐 **Consider using HTTPS** with reverse proxy (Nginx/Traefik)
7. 🔐 **Use Synology's built-in firewall** to restrict port access

## Support

For issues or questions:
- Check container logs in Portainer
- Review Airflow logs at `http://<synology-ip>:8091` → Browse → Logs
- Check the project documentation in the `docs/` folder
