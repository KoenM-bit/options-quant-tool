#!/usr/bin/env python3
"""
Test Slack notification functionality.

Quick test to verify Slack webhook is working before running full DAG.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.alerts import AlertManager
from src.config import settings
from datetime import datetime

def test_slack():
    """Test sending a Slack message."""
    
    print("🧪 Testing Slack notification...")
    print(f"   ENABLE_SLACK_ALERTS: {settings.enable_slack_alerts}")
    print(f"   SLACK_WEBHOOK_URL: {settings.slack_webhook_url[:50]}..." if settings.slack_webhook_url else "   SLACK_WEBHOOK_URL: None")
    print()
    
    if not settings.enable_slack_alerts:
        print("⚠️  Slack alerts are disabled!")
        print("   Set ENABLE_SLACK_ALERTS=true in .env")
        return 1
    
    if not settings.slack_webhook_url:
        print("❌ No webhook URL configured!")
        print("   Set SLACK_WEBHOOK_URL in .env")
        return 1
    
    alert_manager = AlertManager()
    
    # Test message
    message = "🧪 *Test Message from Options Trading Pipeline*\n\n"
    message += "This is a test notification to verify Slack integration is working.\n"
    message += f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    success = alert_manager.send_alert(
        message=message,
        level="info",
        context={
            "Test": "Slack Integration",
            "Status": "✅ Testing"
        }
    )
    
    if success:
        print("✅ Slack notification sent successfully!")
        print("   Check your Slack channel for the message!")
        return 0
    else:
        print("❌ Slack notification failed!")
        print("   Possible reasons:")
        print("   - Webhook URL is incorrect or expired")
        print("   - Slack app doesn't have permission to post")
        print("   - Network/firewall blocking the request")
        print()
        print("   To get a new webhook URL:")
        print("   1. Go to https://api.slack.com/apps")
        print("   2. Select your app (or create new)")
        print("   3. Go to 'Incoming Webhooks'")
        print("   4. Activate and add new webhook to workspace")
        return 1

if __name__ == "__main__":
    sys.exit(test_slack())
