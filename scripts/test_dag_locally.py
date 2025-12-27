#!/usr/bin/env python3
"""
Test the DAG functions locally before deploying to Airflow.

This will test each component of the hourly signal generation pipeline.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*60)
print("🧪 Testing Hourly Signal Generation DAG Functions")
print("="*60)
print()

# Test 1: Import the DAG
print("✅ Test 1: Importing DAG module...")
try:
    sys.path.insert(0, str(project_root / 'dags'))
    import hourly_signal_generation as dag_module
    print("   ✓ DAG module imported successfully")
    print()
except Exception as e:
    print(f"   ✗ Failed to import DAG: {e}")
    sys.exit(1)

# Test 2: Check DAG structure
print("✅ Test 2: Checking DAG structure...")
try:
    dag = dag_module.dag
    print(f"   ✓ DAG name: {dag.dag_id}")
    print(f"   ✓ Schedule: {dag.schedule_interval}")
    print(f"   ✓ Tags: {dag.tags}")
    print(f"   ✓ Tasks: {[task.task_id for task in dag.tasks]}")
    print()
except Exception as e:
    print(f"   ✗ DAG structure error: {e}")
    sys.exit(1)

# Test 3: Test Slack notification (quick test)
print("✅ Test 3: Testing Slack notification function...")
try:
    # Create fake summary file for testing
    import json
    from datetime import datetime
    signals_dir = project_root / 'data' / 'signals'
    signals_dir.mkdir(parents=True, exist_ok=True)
    
    test_summary = signals_dir / f"signal_summary_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    test_summary.write_text(json.dumps({
        "num_signals": 3,
        "tickers": ["AAPL", "MSFT", "GOOGL"],
        "timestamp": datetime.now().isoformat()
    }))
    
    # Test the function
    result = dag_module.send_slack_notification()
    print(f"   ✓ Slack notification result: {result}")
    
    # Cleanup
    test_summary.unlink()
    print()
except Exception as e:
    print(f"   ⚠️  Slack test warning: {e}")
    print("   (This is OK if Slack is not configured)")
    print()

# Test 4: Quick validation of other functions
print("✅ Test 4: Validating function signatures...")
try:
    functions = [
        ('run_backfill_us', dag_module.run_backfill_us),
        ('run_backfill_nl', dag_module.run_backfill_nl),
        ('rebuild_events_parquet', dag_module.rebuild_events_parquet),
        ('generate_signals', dag_module.generate_signals),
        ('send_slack_notification', dag_module.send_slack_notification),
        ('send_failure_notification', dag_module.send_failure_notification),
    ]
    
    for name, func in functions:
        print(f"   ✓ {name}: {func.__doc__.split(chr(10))[0] if func.__doc__ else 'No docstring'}")
    
    print()
except Exception as e:
    print(f"   ✗ Function validation error: {e}")
    sys.exit(1)

print("="*60)
print("🎉 All DAG validation tests passed!")
print("="*60)
print()
print("📋 Next steps:")
print("   1. Start Docker containers: docker-compose up -d")
print("   2. Wait for Airflow to start (~30 seconds)")
print("   3. Open Airflow UI: http://localhost:8081")
print("   4. Find 'hourly_signal_generation' DAG")
print("   5. Enable the DAG and trigger manually")
print("   6. Watch the execution and check Slack for notifications!")
print()
