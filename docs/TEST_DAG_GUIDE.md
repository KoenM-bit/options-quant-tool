# Testing the Hourly Signal Generation DAG

## ✅ Pre-requisites Completed
- [x] Slack webhook configured in `.env`
- [x] `trading_signals` table created
- [x] DAG file created: `dags/hourly_signal_generation.py`
- [x] Slack environment variables added to `docker-compose.yml`
- [x] Airflow containers restarted

## 🧪 Test the DAG in Airflow

### 1. Access Airflow UI
Open your browser and go to:
```
http://localhost:8081
```

Login with:
- **Username**: `admin`
- **Password**: `admin`

### 2. Find the DAG
- Look for `hourly_signal_generation` in the DAGs list
- It should have tags: `trading`, `signals`, `production`, `automated`

### 3. Enable the DAG
- Toggle the switch on the left side of the DAG to enable it
- The DAG is scheduled to run every hour from 8 AM - 9 PM UTC (Mon-Fri)

### 4. Trigger a Test Run
- Click on the DAG name to open details
- Click the "Play" button (▶️) in the top right
- Select "Trigger DAG"
- Click "Trigger" to start the run

### 5. Monitor Execution
Watch the task execution in real-time:
- **backfill_us**: Downloads US market data (143 tickers, ~3-4 minutes)
- **backfill_nl**: Downloads NL market data (25 tickers, ~1 minute)
- **rebuild_events**: Creates consolidated parquet from database (~30 seconds)
- **generate_signals**: Applies ML model to detect breakouts (~10 seconds)
- **send_slack_notification**: Sends summary to Slack (~1 second)

### 6. Check Results

**In Airflow:**
- Click on each task to see logs
- Green = success, Red = failed
- View task logs by clicking task → "Log"

**In Slack:**
You should receive a message like:
```
🚀 3 New Trading Signals Generated

Tickers: AAPL, MSFT, GOOGL
Time: 2025-12-27 14:00

📊 Check signal_summary_2025-12-27_1400.json for details
```

Or if no signals:
```
✅ Pipeline Complete - No New Signals

Time: 2025-12-27 14:00
All markets scanned, no breakout opportunities detected.
```

**On Filesystem:**
Check `data/signals/` for:
- `breakout_signals_YYYYMMDD_HHMM.csv`
- `signal_summary_YYYYMMDD_HHMM.json`
- `latest/signals_display.txt`

### 7. Test Failure Notification
To test error alerts:
1. In Airflow UI, click on a task
2. Click "Clear" to reset it
3. Go to DAG code and introduce a bug (e.g., wrong path)
4. Re-trigger the DAG
5. You should receive a 🚨 failure alert in Slack

## 🐛 Troubleshooting

### DAG not appearing?
```bash
# Check if DAG file has syntax errors
docker exec -it ahold-options-airflow-scheduler-1 python -m py_compile /opt/airflow/dags/hourly_signal_generation.py
```

### Slack not working?
```bash
# Check environment variables in container
docker exec -it ahold-options-airflow-scheduler-1 env | grep SLACK
```

### Tasks failing?
```bash
# View scheduler logs
docker logs ahold-options-airflow-scheduler-1 --tail 100

# View task logs in Airflow UI
# Click task → Log tab
```

### Database connection issues?
```bash
# Test database connection
docker exec -it ahold-options-postgres-1 psql -U airflow -d ahold_options -c "SELECT COUNT(*) FROM bronze_ohlcv_intraday;"
```

## 📊 Expected Runtime

For a typical hourly run:
- **Total time**: ~5-7 minutes
- **Backfill (US + NL)**: ~4-5 minutes (parallel)
- **Rebuild parquet**: ~30 seconds
- **Generate signals**: ~10 seconds
- **Slack notification**: ~1 second

## 🚀 Production Ready!

Once tested successfully, the DAG will run automatically:
- **Schedule**: Every hour from 8 AM - 9 PM UTC (Monday-Friday)
- **Covers**: Both NL (9 AM - 5:30 PM CET) and US (9:30 AM - 4 PM EST) markets
- **Output**: Timestamped signal files + Slack notifications
- **Monitoring**: Automatic error alerts via Slack

🎉 Your automated trading signal pipeline is ready!
