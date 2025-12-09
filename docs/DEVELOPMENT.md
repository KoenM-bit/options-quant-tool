# Development Guide

Guide for developers contributing to the Ahold Options platform.

## Development Setup

### Local Environment

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd ahold-options
   ```

2. **Set up Python Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

3. **Install Development Tools**
   ```bash
   pip install black flake8 mypy pytest pytest-cov
   ```

4. **Configure Pre-commit Hooks**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Project Structure

```
ahold-options/
├── dags/                    # Airflow DAGs
│   ├── ahold_options_daily.py
│   ├── ahold_dbt_transform.py
│   └── data_quality_checks.py
├── src/                     # Source code
│   ├── scrapers/           # Web scrapers
│   ├── models/             # SQLAlchemy models
│   ├── utils/              # Utilities
│   └── config.py           # Configuration
├── dbt/                     # DBT project
│   └── ahold_options/
│       └── models/         # DBT models
├── tests/                   # Tests
├── scripts/                 # Utility scripts
└── docs/                    # Documentation
```

## Coding Standards

### Python Style

- **PEP 8** compliance
- **Type hints** for function signatures
- **Docstrings** for all public functions/classes
- **Black** for formatting (line length: 100)

Example:
```python
from typing import Optional, Dict, Any

def scrape_data(
    ticker: str,
    date: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Scrape options data for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        date: Optional scrape date
    
    Returns:
        Dictionary with scraped data
    
    Raises:
        ValueError: If ticker is invalid
    """
    pass
```

### Git Workflow

1. **Branch Naming**
   - Feature: `feature/description`
   - Bug fix: `fix/description`
   - Hotfix: `hotfix/description`

2. **Commit Messages**
   ```
   type(scope): short description
   
   Longer description if needed.
   
   Closes #123
   ```
   
   Types: feat, fix, docs, style, refactor, test, chore

3. **Pull Requests**
   - Descriptive title
   - Reference related issues
   - Include tests
   - Update documentation

## Testing

### Unit Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_scrapers.py::TestParsers::test_parse_float_nl
```

### Integration Tests

```bash
# Test with local database
pytest tests/integration/

# Test scrapers (requires network)
pytest tests/test_scrapers.py -m "not skip"
```

### Testing DAGs

```bash
# Test DAG structure
docker-compose exec airflow-webserver \
  airflow dags test ahold_options_daily 2024-01-01

# Test specific task
docker-compose exec airflow-webserver \
  airflow tasks test ahold_options_daily scrape_overview 2024-01-01
```

## Adding New Features

### New Scraper

1. **Create scraper class** in `src/scrapers/`
   ```python
   class NewScraper:
       def __init__(self):
           pass
       
       def scrape(self) -> Dict[str, Any]:
           pass
   ```

2. **Add Bronze model** in `src/models/bronze.py`

3. **Create DAG task** in `dags/`

4. **Add tests** in `tests/`

5. **Update documentation**

### New DBT Model

1. **Create SQL file** in `dbt/ahold_options/models/`
   ```sql
   {{ config(
       materialized='incremental',
       unique_key='id'
   ) }}
   
   SELECT ...
   ```

2. **Add tests** in `schema.yml`

3. **Document** in model docstring

4. **Test locally**
   ```bash
   dbt run --models model_name
   dbt test --models model_name
   ```

## Debugging

### Debug Airflow Task

```python
# Add to DAG file
import pdb; pdb.set_trace()

# Or use logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"Debug: {variable}")
```

### Debug DBT Model

```bash
# Compile without running
dbt compile --models model_name

# Check compiled SQL
cat target/compiled/ahold_options/models/model_name.sql

# Run in database
docker-compose exec postgres psql -U airflow -d ahold_options -f /path/to/compiled.sql
```

### Debug Database

```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U airflow -d ahold_options

# Useful queries
SELECT COUNT(*) FROM bronze_fd_overview;
SELECT * FROM bronze_fd_overview ORDER BY scraped_at DESC LIMIT 5;
EXPLAIN ANALYZE SELECT ...;
```

## Performance Optimization

### Scraper Performance

- Use connection pooling
- Implement caching where appropriate
- Batch API requests
- Profile with `cProfile`

### Database Performance

- Add indexes on frequently queried columns
- Use EXPLAIN ANALYZE to check query plans
- Consider partitioning large tables
- Regular VACUUM ANALYZE

### DBT Performance

- Use incremental materialization
- Limit data with `{{ var() }}`
- Use DBT snapshots for slowly changing dimensions
- Profile with `dbt run-operation`

## Documentation

### Code Documentation

- Docstrings for all public APIs
- Type hints
- Inline comments for complex logic
- README in each major directory

### Project Documentation

- Architecture diagrams
- API documentation
- Deployment guides
- Troubleshooting guides

## Release Process

1. **Version Bump**
   - Update version in `src/__init__.py`
   - Update CHANGELOG.md

2. **Testing**
   - Run full test suite
   - Manual testing in staging

3. **Documentation**
   - Update README if needed
   - Update API docs

4. **Git Tag**
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```

5. **Deploy**
   - Build Docker images
   - Deploy to production
   - Monitor for issues

## Common Tasks

### Update Dependencies

```bash
# Update requirements.txt
pip install <package> --upgrade
pip freeze > requirements.txt

# Rebuild Docker images
docker-compose build --no-cache
```

### Add Environment Variable

1. Add to `.env.example`
2. Update `src/config.py`
3. Update documentation
4. Add to docker-compose.yml if needed

### Database Migration

```bash
# Generate migration
alembic revision --autogenerate -m "description"

# Apply migration
alembic upgrade head

# Or use init_db.py for simple changes
python scripts/init_db.py --drop
```

## Resources

- [Airflow Documentation](https://airflow.apache.org/docs/)
- [DBT Documentation](https://docs.getdbt.com/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)

## Getting Help

- 📧 Email: dev-team@example.com
- 💬 Slack: #ahold-options-dev
- 🐛 Issues: GitHub Issues
- 📖 Docs: `/docs` directory

---

**Happy Coding! 🚀**
