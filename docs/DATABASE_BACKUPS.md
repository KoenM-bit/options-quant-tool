# Database Backup System

Automated daily backups of the PostgreSQL database to MinIO with 7-day retention policy.

## Overview

- **Schedule**: Daily at 23:00 CET (after all data pipelines complete)
- **Retention**: 7 days (automatically deletes older backups)
- **Storage**: MinIO bucket under `backups/postgres/`
- **Format**: Compressed PostgreSQL dump (`.sql.gz`)
- **Naming**: `ahold_options_YYYY-MM-DD_HHMMSS.sql.gz`

## Components

### 1. Backup DAG (`database_backup_pipeline`)

Automated Airflow DAG that runs daily:

```
create_backup → verify_backup → send_notification
```

**Tasks:**
- `create_backup`: Creates compressed pg_dump and uploads to MinIO
- `verify_backup`: Verifies backup file exists and has reasonable size
- `send_notification`: Logs backup summary

### 2. Backup Script

**Location**: `scripts/backup_database_to_minio.py`

**Manual Usage:**
```bash
# Create backup for today
python scripts/backup_database_to_minio.py

# Create backup for specific date
python scripts/backup_database_to_minio.py --date 2025-12-16

# Custom retention period (14 days)
python scripts/backup_database_to_minio.py --retention-days 14
```

**Features:**
- Compressed backups (gzip)
- Automatic cleanup of old backups
- Includes DROP commands for clean restore
- No ownership/privilege commands (for portability)

### 3. Restore Script

**Location**: `scripts/restore_database_from_minio.py`

**Usage:**

```bash
# List available backups
python scripts/restore_database_from_minio.py --list

# Restore latest backup (with confirmation prompt)
python scripts/restore_database_from_minio.py --latest

# Restore specific backup
python scripts/restore_database_from_minio.py --backup ahold_options_2025-12-16_120000.sql.gz

# Restore without confirmation (dangerous!)
python scripts/restore_database_from_minio.py --latest --no-confirm
```

**Safety Features:**
- Confirmation prompt before restore
- Displays backup date, size before restoring
- Shows database connection details for verification

## MinIO Storage Structure

```
s3://options-data/
└── backups/
    └── postgres/
        ├── ahold_options_2025-12-16_120000.sql.gz
        ├── ahold_options_2025-12-15_120000.sql.gz
        ├── ahold_options_2025-12-14_120000.sql.gz
        └── ... (7 days total)
```

## Retention Policy

- **Default**: 7 days
- **Configurable**: Can be changed via `--retention-days` parameter
- **Automatic**: Old backups are deleted during each backup run
- **Size**: Typically 5-20 MB per backup (compressed)

## Typical Backup Size

Based on current database:
- **Uncompressed**: ~50-100 MB (SQL dump)
- **Compressed**: ~5-20 MB (gzip)
- **Weekly storage**: ~35-140 MB (7 backups)

## Restore Scenarios

### 1. Local Development Reset

Reset your local database to production state:

```bash
# Option 1: Restore from MinIO backup
python scripts/restore_database_from_minio.py --latest

# Option 2: Direct sync from Synology (faster)
bash scripts/sync_local_db_from_synology.sh
```

### 2. Production Disaster Recovery

Restore production database from backup:

```bash
# SSH into Synology
ssh admin@192.168.1.201

# List available backups
python scripts/restore_database_from_minio.py --list

# Restore specific backup
python scripts/restore_database_from_minio.py --backup ahold_options_2025-12-15_120000.sql.gz
```

### 3. Point-in-Time Recovery

Restore to a specific date:

```bash
# List backups to find the date you need
python scripts/restore_database_from_minio.py --list

# Restore that specific backup
python scripts/restore_database_from_minio.py --backup ahold_options_2025-12-10_230000.sql.gz
```

## Monitoring

### Check Backup Status

```bash
# View recent backup logs
docker exec -it airflow-scheduler airflow tasks logs database_backup_pipeline create_backup latest

# List backups via CLI
python scripts/restore_database_from_minio.py --list
```

### Check MinIO Directly

Access MinIO UI:
- URL: http://192.168.1.201:9000 (Synology) or http://localhost:9000 (local)
- Navigate to `options-data` bucket → `backups/postgres/`

## Backup Process Details

### What's Backed Up

- All tables and data
- Table structures (CREATE TABLE)
- Indexes
- Sequences
- Views and materialized views
- Foreign key constraints

### What's NOT Backed Up

- User/role definitions (`--no-owner`)
- Privileges (`--no-privileges`)
- Database configuration (these are handled separately)

### Backup Command

The actual pg_dump command used:

```bash
pg_dump \
  -h localhost \
  -p 5432 \
  -U airflow \
  -d ahold_options \
  --clean \
  --if-exists \
  --no-owner \
  --no-privileges \
  | gzip > backup.sql.gz
```

## Troubleshooting

### Backup Fails

```bash
# Check PostgreSQL connection
docker exec -it ahold-options-postgres-1 psql -U airflow -d ahold_options -c "SELECT 1"

# Check MinIO connection
python -c "from src.utils.minio_client import get_minio_client; print(get_minio_client())"

# Check disk space
df -h
```

### Restore Fails

```bash
# Test decompression
gunzip -t backup.sql.gz

# Check backup file integrity
python scripts/restore_database_from_minio.py --list

# Manually download and inspect
# (MinIO UI or mc client)
```

### Old Backups Not Deleted

```bash
# Manually run cleanup
python scripts/backup_database_to_minio.py --retention-days 7

# Check MinIO bucket permissions
```

## Best Practices

1. **Regular Testing**: Periodically test restore process
2. **Monitor Backup Size**: Sudden changes may indicate issues
3. **Off-site Backups**: Consider additional backup location for disaster recovery
4. **Retention Tuning**: Adjust retention based on your recovery needs
5. **Before Major Changes**: Manually create backup before schema migrations

## Integration with Other Systems

### With Local Development

```bash
# Fresh start for local testing
python scripts/restore_database_from_minio.py --latest --no-confirm
```

### With CI/CD

```bash
# Backup before deployment
python scripts/backup_database_to_minio.py --date $(date +%Y-%m-%d)
```

### With Data Migrations

```bash
# Create backup before migration
python scripts/backup_database_to_minio.py

# Run migration
alembic upgrade head

# If migration fails, restore
python scripts/restore_database_from_minio.py --latest
```

## Future Enhancements

Potential improvements:

- [ ] Email/Slack notifications on backup completion
- [ ] Backup to multiple locations (MinIO + S3)
- [ ] Incremental backups for large databases
- [ ] Encrypted backups
- [ ] Backup validation (restore to test database)
- [ ] Metrics dashboard (backup size trends, success rate)

## Related Documentation

- [MinIO Setup](MINIO_SETUP.md)
- [Database Sync Script](../scripts/sync_local_db_from_synology.sh)
- [Disaster Recovery Plan](DISASTER_RECOVERY.md) *(to be created)*
