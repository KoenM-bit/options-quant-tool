# Fix for "relation bronze_fd_overview does not exist" Error

## Problem

When running DAGs on Synology, you get errors like:
```
psycopg2.errors.UndefinedTable: relation "bronze_fd_overview" does not exist
psycopg2.errors.UndefinedTable: relation "bronze_fd_options" does not exist
```

## Root Cause

The `airflow-init` container was only creating **Airflow's metadata tables** (for Airflow itself), but not creating **your application's data tables** (bronze, silver, gold layers).

## Solution Applied

Updated `docker-compose.synology.yml` to run the database initialization script during startup:

```yaml
airflow-init:
  command:
    - -c
    - |
      echo "Initializing Airflow database..."
      /entrypoint airflow version
      echo "Initializing application database tables..."
      python /opt/airflow/scripts/init_db.py  # ← This creates your tables!
      echo "✅ Initialization complete!"
```

This now creates all required tables:
- **Bronze**: `bronze_fd_overview`, `bronze_fd_options`
- **Silver**: `silver_underlying_price`, `silver_options`
- **Gold**: `gold_options_summary_daily`, `gold_volatility_surface`, `gold_greek_analytics`, `gold_open_interest_flow`

## How to Apply the Fix

### Option 1: Fresh Deployment (Recommended)

1. **Stop and remove the current stack** in Portainer
2. **Use the updated `docker-compose.synology.yml`**
3. **Deploy the stack** - tables will be created automatically
4. **Verify** in airflow-init logs:
   ```
   ✅ Database initialized successfully!
   Tables created:
     - Bronze: bronze_fd_overview, bronze_fd_options
     ...
   ```

### Option 2: Manual Table Creation (If Stack Already Running)

If you don't want to recreate the stack, manually create tables:

1. **Open a shell in any Airflow container:**
   ```bash
   docker exec -it ahold-options-airflow-webserver bash
   ```

2. **Run the init script:**
   ```bash
   python /opt/airflow/scripts/init_db.py
   ```

3. **Verify tables exist** in Adminer (http://your-synology-ip:8092)

## Verification

After applying the fix, check:

1. ✅ **Init container logs** show table creation success
2. ✅ **Adminer** shows all tables in `ahold_options` database
3. ✅ **DAGs run successfully** without "relation does not exist" errors

## Tables That Should Exist

In Adminer, you should see these tables in the `ahold_options` database:

**Bronze Layer (Raw scraped data):**
- `bronze_fd_overview`
- `bronze_fd_options`

**Silver Layer (Cleaned/enriched):**
- `silver_underlying_price`
- `silver_options`

**Gold Layer (Analytics):**
- `gold_options_summary_daily`
- `gold_volatility_surface`
- `gold_greek_analytics`
- `gold_open_interest_flow`

**Airflow Metadata (managed by Airflow):**
- Various `ab_*` and other Airflow tables

## What Changed

| File | Change |
|------|--------|
| `docker-compose.synology.yml` | Added `python /opt/airflow/scripts/init_db.py` to airflow-init |
| `docker-compose.yml` | Same update for local development |
| `SYNOLOGY_QUICK_START.md` | Added troubleshooting for this error |

## Why This Happened

The original setup assumed tables were already migrated from your old MySQL database or would be created manually. For a fresh Synology deployment, we need to explicitly create them during the initialization phase.

## Prevention

This is now part of the automated setup. Any new deployment will:
1. Create Airflow metadata database
2. Create admin user
3. **Create all application tables** ← Fixed!
4. Exit successfully

The `airflow-init` container running this is a one-time initialization step and will exit after completion (this is normal behavior).
