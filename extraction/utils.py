"""
Utility functions for the extraction package.
"""

import logging
import datetime
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

def get_standard_logger(name, log_level=logging.INFO):
    """
    Create a standardized logger with a specific format.
    
    Args:
        name: Name of the logger
        log_level: Logging level (default: INFO)
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Check if logger already has handlers
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add formatter to handler
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
    
    return logger

def normalize_date(date_str: str) -> Optional[str]:
    """
    Normalize date string to YYYY-MM-DD format.
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        Normalized date string or None if invalid
    """
    # Common date formats
    formats = [
        '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y',
        '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
        '%d.%m.%Y', '%Y.%m.%d'
    ]
    
    # Clean the date string
    date_str = date_str.strip()
    
    # Try each format
    for fmt in formats:
        try:
            date_obj = datetime.datetime.strptime(date_str, fmt)
            return date_obj.strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    return None

def extract_amount(amount_str: str) -> Optional[float]:
    """
    Extract and normalize amount from string.
    
    Args:
        amount_str: String containing amount
        
    Returns:
        Float amount or None if invalid
    """
    # Remove currency symbols and whitespace
    amount_str = re.sub(r'[^\d.,]', '', amount_str)
    
    # Handle different decimal separators
    amount_str = amount_str.replace(',', '.')
    
    try:
        return float(amount_str)
    except ValueError:
        return None

def find_keyword_context(text: str, keyword: str, context_lines: int = 2) -> List[str]:
    """
    Find lines around a keyword match.
    
    Args:
        text: Text to search in
        keyword: Keyword to find
        context_lines: Number of lines before and after to include
        
    Returns:
        List of relevant lines
    """
    lines = text.split('\n')
    relevant_lines = []
    
    for i, line in enumerate(lines):
        if keyword.lower() in line.lower():
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)
            relevant_lines.extend(lines[start:end])
    
    return relevant_lines

def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing line breaks.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Replace multiple whitespace with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Replace multiple line breaks with a single one
    text = re.sub(r'\n+', '\n', text)
    
    # Strip leading and trailing whitespace
    text = text.strip()
    
    return text

def split_into_sections(text: str) -> List[Tuple[str, str]]:
    """
    Split text into sections based on common headers.
    
    Args:
        text: Input text
        
    Returns:
        List of (header, content) tuples
    """
    sections = []
    current_header = None
    current_content = []
    
    headers = [
        'invoice', 'bill to', 'ship to', 'from', 'date',
        'due date', 'payment terms', 'items', 'total'
    ]
    
    for line in text.split('\n'):
        line = line.strip().lower()
        
        # Check if line is a header
        is_header = any(header in line for header in headers)
        
        if is_header:
            # Save previous section if exists
            if current_header and current_content:
                sections.append((current_header, '\n'.join(current_content)))
            
            current_header = line
            current_content = []
        else:
            current_content.append(line)
    
    # Add last section
    if current_header and current_content:
        sections.append((current_header, '\n'.join(current_content)))
    
    return sections 