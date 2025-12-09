"""
HTTP client utilities for web scraping.
Handles retries, timeouts, and rate limiting.
"""

import time
import logging
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from ratelimit import limits, sleep_and_retry

from src.config import settings

logger = logging.getLogger(__name__)


class HTTPClient:
    """HTTP client with retry logic and rate limiting."""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = None,
        max_retries: int = None,
        backoff_factor: float = 0.3,
    ):
        self.base_url = base_url or settings.ahold_fd_base_url
        self.timeout = timeout or settings.scraper_timeout
        self.max_retries = max_retries or settings.scraper_retry_attempts
        self.backoff_factor = backoff_factor
        
        # Create session with retry logic
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create requests session with retry configuration."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers
        session.headers.update({
            "User-Agent": settings.scraper_user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "nl-NL,nl;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        })
        
        return session
    
    @sleep_and_retry
    @limits(calls=settings.rate_limit_calls, period=settings.rate_limit_period)
    def get(
        self,
        url: str,
        params: Optional[dict] = None,
        full_url: bool = False,
    ) -> requests.Response:
        """
        Make GET request with rate limiting.
        
        Args:
            url: URL path or full URL
            params: Query parameters
            full_url: If True, use url as-is; otherwise join with base_url
        
        Returns:
            Response object
        
        Raises:
            requests.exceptions.RequestException: On request failure
        """
        if not full_url and self.base_url:
            url = urljoin(self.base_url, url)
        
        logger.debug(f"GET request to {url}", extra={"params": params})
        
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            logger.debug(
                f"Successful request to {url}",
                extra={
                    "status_code": response.status_code,
                    "content_length": len(response.content),
                }
            )
            
            return response
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching {url}")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching {url}: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            raise
    
    def get_soup(
        self,
        url: str,
        params: Optional[dict] = None,
        full_url: bool = False,
        parser: str = "html.parser",
    ) -> BeautifulSoup:
        """
        Get BeautifulSoup object from URL.
        
        Args:
            url: URL path or full URL
            params: Query parameters
            full_url: If True, use url as-is
            parser: HTML parser to use
        
        Returns:
            BeautifulSoup object
        """
        response = self.get(url, params=params, full_url=full_url)
        return BeautifulSoup(response.content, parser)
    
    def close(self):
        """Close the session."""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience function for quick requests
def fetch_html(
    url: str,
    params: Optional[dict] = None,
    parser: str = "html.parser",
) -> BeautifulSoup:
    """
    Fetch HTML and return BeautifulSoup object.
    
    Args:
        url: Full URL to fetch
        params: Query parameters
        parser: HTML parser
    
    Returns:
        BeautifulSoup object
    """
    with HTTPClient() as client:
        return client.get_soup(url, params=params, full_url=True, parser=parser)
