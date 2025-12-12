#!/usr/bin/env python3
"""
Initialize star schema tables in the database.
Creates dimension and fact tables using SQLAlchemy models.

Usage:
    python scripts/init_star_schema.py
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from src.config import settings
from src.models.base import Base
from src.models.silver_star import (
    DimUnderlying,
    DimContract,
    FactOptionTimeseries
)


def init_star_schema():
    """Initialize star schema tables in the database."""
    print("🌟 Initializing Star Schema Tables")
    print("=" * 60)
    
    # Create engine
    engine = create_engine(settings.database_url, echo=False)
    inspector = inspect(engine)
    
    # List of tables to create
    star_schema_tables = [
        ('dim_underlying', DimUnderlying),
        ('dim_option_contract', DimContract),
        ('fact_option_timeseries', FactOptionTimeseries),
    ]
    
    # Check existing tables
    existing_tables = inspector.get_table_names()
    print(f"\n📋 Existing tables in database: {len(existing_tables)}")
    
    # Create tables
    print("\n🔨 Creating star schema tables...")
    for table_name, model_class in star_schema_tables:
        if table_name in existing_tables:
            print(f"  ⏭️  {table_name} - Already exists (skipping)")
        else:
            print(f"  ✨ {table_name} - Creating...")
            model_class.__table__.create(engine)
            print(f"  ✅ {table_name} - Created successfully")
    
    # No seeding needed for simple star schema
    print("\n🔍 Verifying star schema tables...")
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    
    all_created = True
    for table_name, _ in star_schema_tables:
        if table_name in existing_tables:
            # Get row count
            from sqlalchemy import text
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                count = result.scalar()
            print(f"  ✅ {table_name} - Exists ({count} rows)")
        else:
            print(f"  ❌ {table_name} - NOT FOUND")
            all_created = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_created:
        print("✅ Star schema initialization complete!")
        print("\nNext steps:")
        print("  1. Run dbt to populate dimensions: dbt run --select tag:dimension")
        print("  2. Run dbt to populate fact: dbt run --select tag:fact")
        print("  3. Query: SELECT * FROM fact_option_timeseries LIMIT 10;")
        return 0
    else:
        print("❌ Star schema initialization failed!")
        print("Some tables were not created. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(init_star_schema())
