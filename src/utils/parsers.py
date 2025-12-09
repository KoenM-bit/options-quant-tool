"""
Parsing utilities for converting Dutch number formats and extracting data.
"""

import re
from typing import Optional, Union
from decimal import Decimal, InvalidOperation


def parse_float_nl(value: Union[str, float, None]) -> Optional[float]:
    """
    Parse Dutch-formatted number string to float.
    
    Dutch format uses:
    - Period (.) as thousands separator
    - Comma (,) as decimal separator
    
    Examples:
        "1.234,56" -> 1234.56
        "12,5" -> 12.5
        "-1.000,00" -> -1000.0
        "N.v.t." -> None
        "" -> None
    
    Args:
        value: String, float, or None
    
    Returns:
        Parsed float or None if parsing fails
    """
    if value is None:
        return None
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if not isinstance(value, str):
        return None
    
    # Clean the string
    value = value.strip()
    
    # Handle empty strings
    if not value:
        return None
    
    # Handle special cases
    if value.lower() in ["n.v.t.", "nvt", "n/a", "-", ""]:
        return None
    
    try:
        # Remove thousands separator (.)
        value = value.replace(".", "")
        # Replace decimal separator (,) with (.)
        value = value.replace(",", ".")
        # Remove any remaining non-numeric characters except - and .
        value = re.sub(r"[^\d\-.]", "", value)
        
        return float(value) if value else None
    
    except (ValueError, AttributeError):
        return None


def parse_int_nl(value: Union[str, int, None]) -> Optional[int]:
    """
    Parse Dutch-formatted integer string to int.
    
    Examples:
        "1.234" -> 1234
        "12.345.678" -> 12345678
        "-1.000" -> -1000
    
    Args:
        value: String, int, or None
    
    Returns:
        Parsed int or None if parsing fails
    """
    if value is None:
        return None
    
    if isinstance(value, int):
        return value
    
    if isinstance(value, float):
        return int(value)
    
    if not isinstance(value, str):
        return None
    
    # Clean the string
    value = value.strip()
    
    # Handle empty strings
    if not value:
        return None
    
    # Handle special cases
    if value.lower() in ["n.v.t.", "nvt", "n/a", "-", ""]:
        return None
    
    try:
        # Remove thousands separator
        value = value.replace(".", "")
        # Remove any remaining non-numeric characters except -
        value = re.sub(r"[^\d\-]", "", value)
        
        return int(value) if value else None
    
    except (ValueError, AttributeError):
        return None


def parse_percentage(value: Union[str, float, None]) -> Optional[float]:
    """
    Parse percentage string to float.
    
    Examples:
        "12,5%" -> 12.5
        "12.5%" -> 12.5
        "-5,2%" -> -5.2
    
    Args:
        value: String with or without % sign
    
    Returns:
        Parsed float or None
    """
    if value is None:
        return None
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if not isinstance(value, str):
        return None
    
    # Remove % sign and whitespace
    value = value.strip().replace("%", "").strip()
    
    # Try Dutch format first, then standard format
    result = parse_float_nl(value)
    if result is None:
        try:
            result = float(value)
        except (ValueError, TypeError):
            pass
    
    return result


def parse_decimal_nl(value: Union[str, float, None], precision: int = 2) -> Optional[Decimal]:
    """
    Parse Dutch-formatted number to Decimal for precise calculations.
    
    Args:
        value: String or float
        precision: Decimal places to quantize to
    
    Returns:
        Parsed Decimal or None
    """
    parsed = parse_float_nl(value)
    if parsed is None:
        return None
    
    try:
        dec = Decimal(str(parsed))
        if precision is not None:
            quantize_str = "0." + "0" * precision
            dec = dec.quantize(Decimal(quantize_str))
        return dec
    except (InvalidOperation, ValueError):
        return None


def clean_text(text: Optional[str]) -> Optional[str]:
    """
    Clean text by removing extra whitespace and special characters.
    
    Args:
        text: Input text
    
    Returns:
        Cleaned text or None
    """
    if not text:
        return None
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    
    return text if text else None


def extract_number_from_text(text: str) -> Optional[float]:
    """
    Extract first number from text.
    
    Examples:
        "Volume: 1.234" -> 1234.0
        "Price €12,50 per share" -> 12.5
    
    Args:
        text: Text containing a number
    
    Returns:
        Extracted number or None
    """
    if not text:
        return None
    
    # Try to find Dutch-formatted number
    pattern = r"-?\d{1,3}(?:\.\d{3})*(?:,\d+)?"
    match = re.search(pattern, text)
    
    if match:
        return parse_float_nl(match.group())
    
    # Try to find standard formatted number
    pattern = r"-?\d+(?:\.\d+)?"
    match = re.search(pattern, text)
    
    if match:
        try:
            return float(match.group())
        except ValueError:
            pass
    
    return None


def parse_date_nl(date_str: Union[str, None]) -> Optional[object]:
    """
    Parse Dutch date string to date object.
    
    Supports formats:
    - DD-MM-YYYY (e.g., "31-12-2025")
    - DD/MM/YYYY (e.g., "31/12/2025")
    - DD-MM-YY (e.g., "31-12-25")
    
    Args:
        date_str: Date string in Dutch format
    
    Returns:
        datetime.date object or None
    """
    from datetime import datetime
    
    if not date_str or not isinstance(date_str, str):
        return None
    
    date_str = date_str.strip()
    
    if not date_str:
        return None
    
    # Try different date formats
    formats = [
        "%d-%m-%Y",  # 31-12-2025
        "%d/%m/%Y",  # 31/12/2025
        "%d-%m-%y",  # 31-12-25
        "%d/%m/%y",  # 31/12/25
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    
    return None
