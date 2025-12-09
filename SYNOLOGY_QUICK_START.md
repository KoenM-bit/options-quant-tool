# Quick Deployment Guide for Synology

## What's Fixed

✅ **Security Warning**: All containers run as non-root user (UID 50000)
✅ **Deprecation Warning**: Using `_AIRFLOW_DB_MIGRATE` instead of `_AIRFLOW_DB_UPGRADE`
✅ **Flask-Limiter Warning**: Rate limiting disabled (not needed for private NAS)

## Deploy in Portainer

### 1. Copy docker-compose.synology.yml

Use the file `docker-compose.synology.yml` - it's ready for Synology/Portainer.

### 2. Set Environment Variables in Portainer

**Minimum Required:**

```bash
# Generate this first: python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
AIRFLOW__CORE__FERNET_KEY=your-generated-fernet-key-here

# Change these passwords!
_AIRFLOW_WWW_USER_PASSWORD=your-secure-password
POSTGRES_PASSWORD=your-secure-database-password

# Database config
POSTGRES_USER=airflow
POSTGRES_DB=ahold_options

# User ID
AIRFLOW_UID=50000

# Admin user
_AIRFLOW_WWW_USER_USERNAME=admin

# Scraper config
SCRAPER_USER_AGENT=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36
```

### 3. Deploy

1. Portainer → Stacks → Add stack
2. Name: `ahold-options`
3. Paste docker-compose.synology.yml
4. Add environment variables above
5. Click **Deploy**

### 4. Access

- **Airflow UI**: http://your-synology-ip:8091
- **Adminer**: http://your-synology-ip:8092
- **PostgreSQL**: Port 5433

## Expected Startup Sequence

1. ✅ `postgres` - Starts first, waits for health check (~10s)
2. ✅ `airflow-init` - Runs migrations, creates admin user, **creates application tables**, then **exits successfully** (this is normal! takes ~30s)
3. ✅ `airflow-webserver` - Starts after init completes (~30s to be healthy)
4. ✅ `airflow-scheduler` - Starts after init completes (~30s to be healthy)
5. ✅ `adminer` - Starts with postgres

**Total startup time: ~2-3 minutes**

## Important Notes

### airflow-init Container

⚠️ **The `airflow-init` container is SUPPOSED to exit!**

This is normal behavior. It:
1. Initializes Airflow metadata database (migrations)
2. Creates the admin user
3. **Creates application database tables** (bronze_fd_overview, bronze_fd_options, etc.)
4. Then exits with status "Completed"

If you see it in "Exited (0)" state, that's correct. Don't try to restart it.

You can check the logs to verify all tables were created:
```
Portainer → Containers → ahold-options-airflow-init → Logs
```

Look for:
```
✅ Database initialized successfully!
Tables created:
  - Bronze: bronze_fd_overview, bronze_fd_options
  - Silver: silver_underlying_price, silver_options
  - Gold: gold_options_summary_daily, gold_volatility_surface, ...
```

### No More Warnings

After deployment, you should see:
- ✅ No "running as root" warning
- ✅ No "_AIRFLOW_DB_UPGRADE deprecated" warning  
- ✅ No "Flask-Limiter in-memory storage" warning

## Troubleshooting

### "admin already exists"
This is just an info message, not an error. It means the admin user is already created.

### DAGs fail with "relation bronze_fd_overview does not exist"
**Cause**: Application tables weren't created during initialization

**Solution**: 
1. Check `airflow-init` container logs
2. Look for "✅ Database initialized successfully!" and the list of tables
3. If missing, check for errors during table creation
4. If init failed, fix the issue and restart the stack

### Init container keeps restarting
Check the logs:
1. Portainer → Containers → ahold-options-airflow-init → Logs
2. Look for actual error messages
3. Common issues:
   - Can't connect to postgres (check postgres is healthy)
   - Invalid Fernet key (must be generated properly)
   - Wrong database credentials
   - Table creation failed (check Python errors in logs)

### Can't access web UI
- Check firewall on Synology
- Verify port 8091 is not in use
- Wait 2-3 minutes for full startup
- Check webserver container logs for errors

## Update Application

When a new image is pushed to ghcr.io:

1. Portainer → Stacks → ahold-options
2. Click **Pull and redeploy**
3. Or manually: `docker pull ghcr.io/koenm-bit/options-quant-tool:latest`

## Services Overview

| Service | Purpose | Port | Status |
|---------|---------|------|--------|
| postgres | Database | 5433 | Always running |
| airflow-init | One-time setup | - | Exits after completion |
| airflow-webserver | Web UI | 8091 | Always running |
| airflow-scheduler | Job scheduler | - | Always running |
| adminer | DB admin | 8092 | Always running |

## Next Steps After Deployment

1. Log into Airflow: http://your-synology-ip:8091
2. Check DAGs are loaded
3. Enable the DAGs you want to run
4. Monitor the first run in the Airflow UI
