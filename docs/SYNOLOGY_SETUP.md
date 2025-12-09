# Synology NAS Deployment Guide

Complete guide for deploying the Ahold Options platform on Synology NAS.

## Prerequisites

- Synology NAS with DSM 7.0+
- Container Manager package installed
- At least 8GB RAM
- 20GB free disk space
- SSH access enabled (optional, for advanced setup)

## Step-by-Step Setup

### 1. Prepare Synology

1. **Install Container Manager**
   - Open Package Center
   - Search for "Container Manager"
   - Click Install

2. **Create Project Folder**
   - Open File Station
   - Navigate to `/volume1/docker/`
   - Create folder: `ahold-options`

### 2. Upload Project Files

#### Option A: Via File Station (Recommended for beginners)

1. Open File Station
2. Navigate to `/volume1/docker/ahold-options/`
3. Upload all project files:
   - `docker-compose.synology.yml` → rename to `docker-compose.yml`
   - `Dockerfile`
   - `requirements.txt`
   - `.env.example` → rename to `.env`
   - Folders: `src/`, `dags/`, `dbt/`, `scripts/`

#### Option B: Via Git (SSH required)

```bash
# SSH into Synology
ssh admin@your-synology-ip

# Navigate to docker folder
cd /volume1/docker/

# Clone repository
git clone <your-repo-url> ahold-options
cd ahold-options

# Copy and configure environment
cp .env.example .env
nano .env  # Edit with your settings
```

### 3. Configure Environment

Edit `/volume1/docker/ahold-options/.env`:

```bash
# Generate Fernet key first (on your local machine):
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Then edit .env:
ENVIRONMENT=prod
POSTGRES_USER=airflow
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_DB=ahold_options
AIRFLOW__CORE__FERNET_KEY=your_generated_key_here
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=your_admin_password_here

# Ahold settings
AHOLD_TICKER=AD.AS
AHOLD_SYMBOL_CODE=AEX.AH/O

# Logging
LOG_LEVEL=INFO
```

### 4. Adjust Volume Paths

Edit `docker-compose.yml` if your Synology uses different volume paths:

```yaml
volumes:
  # Change /volume1/ to your volume number if different
  - /volume1/docker/ahold-options/dags:/opt/airflow/dags:ro
  - /volume1/docker/ahold-options/src:/opt/airflow/src:ro
  # ... etc
```

### 5. Deploy with Container Manager

#### Using Container Manager UI:

1. **Open Container Manager**
2. **Go to Project**
3. **Click "Create"**
4. **Set Project Name**: `ahold-options`
5. **Set Path**: `/volume1/docker/ahold-options`
6. **Select Source**: Upload or paste `docker-compose.yml`
7. **Environment Variables**:
   - Either upload `.env` file
   - Or manually enter each variable
8. **Click "Next"** then **"Done"**

#### Using SSH (Advanced):

```bash
cd /volume1/docker/ahold-options

# Build and start
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### 6. Initial Setup

1. **Wait for Services to Start** (~2-3 minutes)

2. **Check Service Health**:
   ```bash
   docker-compose ps
   # All services should show "healthy"
   ```

3. **Access Airflow Web UI**:
   - Open browser: `http://your-synology-ip:8080`
   - Login: admin / (your password from .env)

4. **Initialize Database** (first time only):
   ```bash
   # SSH into webserver container
   docker-compose exec airflow-webserver bash
   
   # Run init script
   python /opt/airflow/scripts/init_db.py
   
   # Exit container
   exit
   ```

5. **Enable DAGs**:
   - In Airflow UI, toggle DAGs to ON:
     - `ahold_options_daily`
     - `ahold_dbt_transform`
     - `ahold_data_quality_checks`

6. **Test Manual Run**:
   - Click on `ahold_options_daily`
   - Click "Trigger DAG" (play button)
   - Monitor execution in Graph view

## Resource Optimization for Synology

The Synology docker-compose file is already optimized, but you can adjust:

### For 8GB RAM systems:
```yaml
deploy:
  resources:
    limits:
      cpus: '1'
      memory: 1G
```

### For 16GB+ RAM systems:
```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 4G
```

## Networking

### Access Services

- **Airflow UI**: `http://synology-ip:8080`
- **PostgreSQL**: `synology-ip:5432`

### Firewall Rules (if needed)

1. Open Control Panel → Security → Firewall
2. Add rules for ports:
   - 8080 (Airflow)
   - 5432 (PostgreSQL, if external access needed)

## Backup Strategy

### Database Backups

Create scheduled task in DSM:

1. Control Panel → Task Scheduler
2. Create → Scheduled Task → User-defined script
3. Schedule: Daily at 2:00 AM
4. Script:
   ```bash
   #!/bin/bash
   cd /volume1/docker/ahold-options
   docker-compose exec -T postgres pg_dump -U airflow ahold_options > /volume1/backups/ahold-options-$(date +%Y%m%d).sql
   # Keep last 30 days
   find /volume1/backups/ahold-options-*.sql -mtime +30 -delete
   ```

### Full Backup

Use Hyper Backup to backup:
- `/volume1/docker/ahold-options/` (all files)
- `/volume1/docker/ahold-options/postgres-data/` (database)

## Monitoring

### Check Service Status
```bash
docker-compose ps
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f airflow-scheduler
docker-compose logs -f postgres
```

### Resource Usage

Monitor in DSM:
- Main Menu → Resource Monitor
- Look for Docker containers

## Troubleshooting

### Services Won't Start

```bash
# Check logs
docker-compose logs

# Restart services
docker-compose restart

# Full restart
docker-compose down
docker-compose up -d
```

### Out of Memory

1. Stop other Docker containers
2. Reduce resource limits in docker-compose.yml
3. Increase Synology swap space

### Database Connection Errors

```bash
# Check PostgreSQL logs
docker-compose logs postgres

# Test connection
docker-compose exec postgres psql -U airflow -d ahold_options -c "SELECT 1"
```

### Permission Errors

```bash
# Fix ownership
sudo chown -R 50000:root /volume1/docker/ahold-options/logs
sudo chown -R 50000:root /volume1/docker/ahold-options/plugins
```

## Maintenance

### Update Platform

```bash
cd /volume1/docker/ahold-options

# Pull latest code
git pull

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Clean Up Old Data

```bash
# Run cleanup (respects retention policies in .env)
docker-compose exec airflow-webserver airflow dags trigger cleanup_old_data
```

## Performance Tips

1. **Enable SSD cache** if available
2. **Schedule DAGs during off-peak hours**
3. **Monitor disk I/O** in Resource Monitor
4. **Use PostgreSQL on SSD volume** if possible
5. **Regular database vacuuming**:
   ```bash
   docker-compose exec postgres psql -U airflow -d ahold_options -c "VACUUM ANALYZE"
   ```

## Security

1. **Change default passwords** immediately
2. **Use strong Fernet key**
3. **Enable HTTPS** (reverse proxy with DSM)
4. **Restrict network access** via firewall
5. **Regular updates** of base images

## Support

For issues specific to Synology deployment:
- Check DSM logs: Log Center → Container
- Synology Community Forums
- Project GitHub Issues

---

**Last Updated**: 2024-12-07
