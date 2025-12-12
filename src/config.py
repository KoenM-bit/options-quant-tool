"""
Configuration management for Ahold Options platform.
Loads settings from environment variables with validation.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Environment
    environment: str = Field(default="local", env="ENVIRONMENT")
    
    # Database
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="ahold_options", env="POSTGRES_DB")
    postgres_user: str = Field(default="airflow", env="POSTGRES_USER")
    postgres_password: str = Field(default="airflow", env="POSTGRES_PASSWORD")
    
    # Scraper
    scraper_user_agent: str = Field(
        default="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        env="SCRAPER_USER_AGENT"
    )
    scraper_timeout: int = Field(default=30, env="SCRAPER_TIMEOUT")
    scraper_retry_attempts: int = Field(default=3, env="SCRAPER_RETRY_ATTEMPTS")
    scraper_retry_delay: int = Field(default=5, env="SCRAPER_RETRY_DELAY")
    
    # Ticker Configuration (Multi-Ticker Support)
    # Format: JSON list of dicts with ticker, symbol_code (FD), and bd_url (Beursduivel)
    # Example:
    # [
    #   {
    #     "ticker": "AD.AS",
    #     "symbol_code": "AEX.AH/O",
    #     "bd_url": "https://www.beursduivel.be/Aandeel-Koers/11755/Ahold-Delhaize-Koninklijke/opties-expiratiedatum.aspx"
    #   },
    #   {
    #     "ticker": "MT.AS", 
    #     "symbol_code": "AEX.MT/O",
    #     "bd_url": "https://www.beursduivel.be/Aandeel-Koers/11895/ArcelorMittal/opties-expiratiedatum.aspx"
    #   }
    # ]
    tickers_config: str = Field(
        default='[{"ticker": "AD.AS", "symbol_code": "AEX.AH/O", "bd_url": "https://www.beursduivel.be/Aandeel-Koers/11755/Ahold-Delhaize-Koninklijke/opties-expiratiedatum.aspx"}]',
        env="TICKERS_CONFIG"
    )
    
    # Legacy fields (backwards compatibility)
    ahold_ticker: str = Field(default="AD.AS", env="AHOLD_TICKER")
    ahold_symbol_code: str = Field(default="AEX.AH/O", env="AHOLD_SYMBOL_CODE")
    ahold_fd_base_url: str = Field(
        default="https://beurs.fd.nl/derivaten/opties/",
        env="AHOLD_FD_BASE_URL"
    )
    
    @property
    def tickers(self) -> list[dict]:
        """Parse tickers configuration from JSON string."""
        import json
        try:
            return json.loads(self.tickers_config)
        except json.JSONDecodeError:
            # Fallback to legacy single ticker
            return [{"ticker": self.ahold_ticker, "symbol_code": self.ahold_symbol_code}]
    
    # DBT
    dbt_project_dir: str = Field(default="/opt/airflow/dbt/ahold_options", env="DBT_PROJECT_DIR")
    dbt_profiles_dir: str = Field(default="/opt/airflow/dbt", env="DBT_PROFILES_DIR")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    # Monitoring
    enable_slack_alerts: bool = Field(default=False, env="ENABLE_SLACK_ALERTS")
    slack_webhook_url: Optional[str] = Field(default=None, env="SLACK_WEBHOOK_URL")
    enable_email_alerts: bool = Field(default=False, env="ENABLE_EMAIL_ALERTS")
    smtp_host: Optional[str] = Field(default=None, env="SMTP_HOST")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    smtp_user: Optional[str] = Field(default=None, env="SMTP_USER")
    smtp_password: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    alert_email_to: Optional[str] = Field(default=None, env="ALERT_EMAIL_TO")
    
    # Data Retention
    bronze_retention_days: int = Field(default=90, env="BRONZE_RETENTION_DAYS")
    silver_retention_days: int = Field(default=365, env="SILVER_RETENTION_DAYS")
    gold_retention_days: int = Field(default=-1, env="GOLD_RETENTION_DAYS")  # -1 = keep forever
    
    # Rate Limiting
    rate_limit_calls: int = Field(default=10, env="RATE_LIMIT_CALLS")
    rate_limit_period: int = Field(default=60, env="RATE_LIMIT_PERIOD")
    
    @property
    def database_url(self) -> str:
        """Construct database URL for SQLAlchemy."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @property
    def async_database_url(self) -> str:
        """Construct async database URL for SQLAlchemy."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields in .env file


# Global settings instance
settings = Settings()
