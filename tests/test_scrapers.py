"""
Test suite for scraper utilities.
"""

import pytest
from src.utils.parsers import (
    parse_float_nl,
    parse_int_nl,
    parse_percentage,
    clean_text,
)


class TestParsers:
    """Test parsing utilities."""
    
    def test_parse_float_nl(self):
        """Test Dutch float parsing."""
        assert parse_float_nl("1.234,56") == 1234.56
        assert parse_float_nl("12,5") == 12.5
        assert parse_float_nl("-1.000,00") == -1000.0
        assert parse_float_nl("N.v.t.") is None
        assert parse_float_nl("") is None
        assert parse_float_nl(None) is None
        assert parse_float_nl(123.45) == 123.45
    
    def test_parse_int_nl(self):
        """Test Dutch integer parsing."""
        assert parse_int_nl("1.234") == 1234
        assert parse_int_nl("12.345.678") == 12345678
        assert parse_int_nl("-1.000") == -1000
        assert parse_int_nl("") is None
        assert parse_int_nl(None) is None
        assert parse_int_nl(123) == 123
    
    def test_parse_percentage(self):
        """Test percentage parsing."""
        assert parse_percentage("12,5%") == 12.5
        assert parse_percentage("-5,2%") == -5.2
        assert parse_percentage("12.5") == 12.5
        assert parse_percentage("") is None
    
    def test_clean_text(self):
        """Test text cleaning."""
        assert clean_text("  hello   world  ") == "hello world"
        assert clean_text("\n\ttest\n") == "test"
        assert clean_text("") is None
        assert clean_text(None) is None


class TestHTTPClient:
    """Test HTTP client utilities."""
    
    @pytest.mark.skip(reason="Requires network access")
    def test_fetch_html(self):
        """Test HTML fetching."""
        from src.utils.http_client import fetch_html
        soup = fetch_html("https://httpbin.org/html")
        assert soup is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
