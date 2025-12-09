"""
Alerting utilities for Slack and Email notifications.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

from src.config import settings

logger = logging.getLogger(__name__)


class AlertManager:
    """Manages alerts via Slack and Email."""
    
    def __init__(self):
        self.slack_enabled = settings.enable_slack_alerts
        self.email_enabled = settings.enable_email_alerts
    
    def send_alert(
        self,
        message: str,
        level: str = "info",
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send alert via configured channels.
        
        Args:
            message: Alert message
            level: Alert level (info, warning, error, critical)
            context: Additional context data
        
        Returns:
            True if alert sent successfully
        """
        success = True
        
        if self.slack_enabled:
            success = success and self._send_slack(message, level, context)
        
        if self.email_enabled:
            success = success and self._send_email(message, level, context)
        
        if not (self.slack_enabled or self.email_enabled):
            logger.info(f"Alerts disabled. Message: {message}")
        
        return success
    
    def _send_slack(
        self,
        message: str,
        level: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send Slack notification."""
        if not settings.slack_webhook_url:
            logger.warning("Slack webhook URL not configured")
            return False
        
        try:
            from slack_sdk.webhook import WebhookClient
            
            # Color based on level
            colors = {
                "info": "#36a64f",
                "warning": "#ff9900",
                "error": "#ff0000",
                "critical": "#8b0000",
            }
            
            # Build Slack message
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"🔔 Ahold Options Alert - {level.upper()}",
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": message
                    }
                }
            ]
            
            if context:
                context_text = "\n".join([f"*{k}*: {v}" for k, v in context.items()])
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": context_text
                    }
                })
            
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
                    }
                ]
            })
            
            webhook = WebhookClient(settings.slack_webhook_url)
            response = webhook.send(
                blocks=blocks,
                text=message,  # Fallback text
            )
            
            if response.status_code == 200:
                logger.info("Slack alert sent successfully")
                return True
            else:
                logger.error(f"Failed to send Slack alert: {response.status_code}")
                return False
        
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
            return False
    
    def _send_email(
        self,
        message: str,
        level: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send email notification."""
        if not all([
            settings.smtp_host,
            settings.smtp_user,
            settings.smtp_password,
            settings.alert_email_to,
        ]):
            logger.warning("Email configuration incomplete")
            return False
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Build email
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"Ahold Options Alert - {level.upper()}"
            msg['From'] = settings.smtp_user
            msg['To'] = settings.alert_email_to
            
            # Plain text version
            text_content = f"{message}\n\n"
            if context:
                text_content += "Context:\n"
                text_content += "\n".join([f"{k}: {v}" for k, v in context.items()])
            text_content += f"\n\nTime: {datetime.now()}"
            
            # HTML version
            html_content = f"""
            <html>
              <body>
                <h2>Ahold Options Alert - {level.upper()}</h2>
                <p>{message}</p>
            """
            if context:
                html_content += "<h3>Context:</h3><ul>"
                html_content += "".join([f"<li><b>{k}</b>: {v}</li>" for k, v in context.items()])
                html_content += "</ul>"
            html_content += f"""
                <p><i>{datetime.now()}</i></p>
              </body>
            </html>
            """
            
            part1 = MIMEText(text_content, 'plain')
            part2 = MIMEText(html_content, 'html')
            msg.attach(part1)
            msg.attach(part2)
            
            # Send email
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
                server.starttls()
                server.login(settings.smtp_user, settings.smtp_password)
                server.sendmail(
                    settings.smtp_user,
                    settings.alert_email_to,
                    msg.as_string()
                )
            
            logger.info(f"Email alert sent to {settings.alert_email_to}")
            return True
        
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
            return False


# Global instance
alert_manager = AlertManager()


def send_info_alert(message: str, context: Optional[Dict[str, Any]] = None):
    """Send info-level alert."""
    return alert_manager.send_alert(message, "info", context)


def send_warning_alert(message: str, context: Optional[Dict[str, Any]] = None):
    """Send warning-level alert."""
    return alert_manager.send_alert(message, "warning", context)


def send_error_alert(message: str, context: Optional[Dict[str, Any]] = None):
    """Send error-level alert."""
    return alert_manager.send_alert(message, "error", context)


def send_critical_alert(message: str, context: Optional[Dict[str, Any]] = None):
    """Send critical-level alert."""
    return alert_manager.send_alert(message, "critical", context)
