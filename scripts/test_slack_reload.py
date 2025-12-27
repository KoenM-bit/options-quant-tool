#!/usr/bin/env python3
"""
Test Slack with forced settings reload
"""

import sys
from pathlib import Path
import os

# Force load .env before anything else
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path, override=True)

print(f"Loading .env from: {env_path}")
print(f"SLACK_WEBHOOK_URL from env: {os.getenv('SLACK_WEBHOOK_URL', 'NOT SET')}")
print()

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import settings
from src.config import settings
from src.utils.alerts import AlertManager
from datetime import datetime

print(f"Settings loaded:")
print(f"  enable_slack_alerts: {settings.enable_slack_alerts}")
print(f"  slack_webhook_url: {settings.slack_webhook_url}")
print()

alert_manager = AlertManager()

message = "🧪 *Test Message with Forced Reload*\n\n"
message += f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

success = alert_manager.send_alert(
    message=message,
    level="info",
    context={
        "Test": "Slack Integration with .env reload",
        "Status": "✅ Testing"
    }
)

if success:
    print("✅ Slack notification sent successfully!")
else:
    print("❌ Slack notification failed!")
