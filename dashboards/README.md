# Data Quality Monitoring Dashboard

Interactive web dashboard for monitoring Bronze and Silver layer data quality.

## Features

### 🥉 Bronze Layer View
- **Overview Metrics**: Total records, scrape days, coverage statistics
- **Daily Scrape Trends**: Visual timeline of scraping activity
- **Latest Sample**: Table view of most recent scraped data
- **Daily Summary**: 30-day historical scrape summary

### 🥈 Silver Layer View
- **Quality Metrics**: Greeks validation rates, IV coverage
- **Daily Quality Trends**: Pass/fail rates over time
- **IV Distribution**: Histograms and box plots
- **Greeks Time Series**: Delta, Gamma trends for ATM options
- **Sample Data**: High-quality Greeks with all metrics
- **Quality Summary**: Detailed daily breakdown

### ⚖️ Comparison View
- **Bronze vs Silver**: Side-by-side comparison
- **Data Pipeline Funnel**: Visualization of data flow
- **Quality Metrics**: Deduplication ratio, end-to-end quality

## Installation

Install the required dependencies:

```bash
pip install -r dashboards/requirements.txt
```

Or install individually:

```bash
pip install streamlit pandas plotly sqlalchemy psycopg2-binary
```

## Usage

### Run Locally (Outside Docker)

```bash
streamlit run dashboards/data_quality_monitor.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Run in Docker

Add to your `docker-compose.yml`:

```yaml
  dashboard:
    build: .
    command: streamlit run dashboards/data_quality_monitor.py --server.port=8501 --server.address=0.0.0.0
    ports:
      - "8501:8501"
    environment:
      - DATABASE_URL=${DATABASE_URL}
    depends_on:
      - postgres
```

Then access at `http://localhost:8501`

## Configuration

The dashboard connects to your PostgreSQL database using the same connection settings as your Airflow DAGs.

## Navigation

Use the sidebar to:
- **Select Layer**: Choose between Bronze, Silver, or Comparison view
- **Refresh Data**: Clear cache and reload all data (TTL: 5 minutes)

## Key Metrics Explained

### Bronze Layer
- **Total Records**: All scraped option records (includes duplicates)
- **Bid/Ask Coverage**: % of records with both bid and ask prices
- **Scrape Days**: Number of distinct days with data

### Silver Layer
- **Greeks Valid**: Options that passed all quality checks
- **Quality Pass Rate**: % of options with valid Greeks
- **Average IV**: Mean implied volatility across valid options
- **Risk-Free Rate**: ECB-sourced rate used in calculations

### Quality Badges
- 🟢 **Good**: >80% coverage/pass rate
- 🟡 **Warning**: 50-80% coverage/pass rate
- 🔴 **Danger**: <50% coverage/pass rate

## Data Refresh

- Cached data refreshes every 5 minutes automatically
- Use the "🔄 Refresh Dashboard" button for manual refresh
- Database queries are optimized with date filters (last 30 days)

## Troubleshooting

### Connection Issues
Ensure your database connection is configured in `src/utils/db.py`

### Missing Dependencies
Run `pip install -r dashboards/requirements.txt`

### Slow Loading
The dashboard queries 30 days of data by default. Reduce the date range in queries if needed.

## Screenshots

[Screenshots would go here in production]

## Future Enhancements

- [ ] Export to PDF/Excel
- [ ] Custom date range filters
- [ ] Real-time refresh option
- [ ] Alert notifications for quality thresholds
- [ ] Gold layer metrics
- [ ] Historical comparison slider
