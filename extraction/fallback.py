import re
from datetime import datetime
import logging

def normalize_date(date_str):
    """
    Normalize date string to YYYY-MM-DD format.
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        Normalized date string in YYYY-MM-DD format or None if invalid
    """
    if not date_str:
        return None
        
    # Clean up the date string
    date_str = date_str.strip()
    date_str = re.sub(r'\s+', ' ', date_str)  # Normalize whitespace
    date_str = re.sub(r'[,;]', '', date_str)  # Remove commas and semicolons
    
    # Check for month name format first
    month_name_match = re.search(r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t)?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b', date_str, re.IGNORECASE)
    if month_name_match:
        return parse_month_name_date(date_str)
    
    # Try different date formats
    date_formats = [
        # MM/DD/YYYY
        (r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"),
        # DD/MM/YYYY
        (r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"),
        # YYYY/MM/DD
        (r'(\d{4})/(\d{1,2})/(\d{1,2})', lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"),
        # MM-DD-YYYY
        (r'(\d{1,2})-(\d{1,2})-(\d{4})', lambda m: f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"),
        # DD-MM-YYYY
        (r'(\d{1,2})-(\d{1,2})-(\d{4})', lambda m: f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"),
        # YYYY-MM-DD
        (r'(\d{4})-(\d{1,2})-(\d{1,2})', lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"),
        # MM.DD.YYYY
        (r'(\d{1,2})\.(\d{1,2})\.(\d{4})', lambda m: f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"),
        # DD.MM.YYYY
        (r'(\d{1,2})\.(\d{1,2})\.(\d{4})', lambda m: f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"),
        # YYYY.MM.DD
        (r'(\d{4})\.(\d{1,2})\.(\d{1,2})', lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"),
        
        # Formats with 2-digit year
        # MM/DD/YY
        (r'(\d{1,2})/(\d{1,2})/(\d{2})', lambda m: f"20{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}" if int(m.group(3)) < 50 else f"19{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"),
        # DD/MM/YY
        (r'(\d{1,2})/(\d{1,2})/(\d{2})', lambda m: f"20{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}" if int(m.group(3)) < 50 else f"19{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"),
        # YY/MM/DD
        (r'(\d{2})/(\d{1,2})/(\d{1,2})', lambda m: f"20{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if int(m.group(1)) < 50 else f"19{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"),
        # MM-DD-YY
        (r'(\d{1,2})-(\d{1,2})-(\d{2})', lambda m: f"20{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}" if int(m.group(3)) < 50 else f"19{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"),
        # DD-MM-YY
        (r'(\d{1,2})-(\d{1,2})-(\d{2})', lambda m: f"20{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}" if int(m.group(3)) < 50 else f"19{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"),
        # YY-MM-DD
        (r'(\d{2})-(\d{1,2})-(\d{1,2})', lambda m: f"20{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if int(m.group(1)) < 50 else f"19{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"),
        
        # ISO format with time
        (r'(\d{4})-(\d{1,2})-(\d{1,2})T', lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"),
        
        # Formats with spaces
        # MM DD YYYY
        (r'(\d{1,2})\s+(\d{1,2})\s+(\d{4})', lambda m: f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"),
        # DD MM YYYY
        (r'(\d{1,2})\s+(\d{1,2})\s+(\d{4})', lambda m: f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"),
        # YYYY MM DD
        (r'(\d{4})\s+(\d{1,2})\s+(\d{1,2})', lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"),
    ]
    
    for pattern, formatter in date_formats:
        match = re.search(pattern, date_str)
        if match:
            try:
                # If we match multiple patterns, check for validity
                result = formatter(match)
                # Basic validation
                year, month, day = result.split('-')
                month_int = int(month)
                day_int = int(day)
                
                # Rough validation (not perfect, but catches obvious issues)
                if month_int < 1 or month_int > 12:
                    continue
                if day_int < 1 or day_int > 31:
                    continue
                if month_int in [4, 6, 9, 11] and day_int > 30:
                    continue
                if month_int == 2 and day_int > 29:
                    continue
                
                return result
            except Exception as e:
                logging.warning(f"Error normalizing date '{date_str}': {str(e)}")
    
    # Try to infer from context as last resort
    # For MM/DD/YYYY vs DD/MM/YYYY ambiguity, we could use logic like:
    # If first number > 12, it's likely DD/MM format
    # If second number > 31, something is wrong
    # Otherwise, default to MM/DD format (or based on locale)
    
    if re.search(r'\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b', date_str):
        parts = re.split(r'[-/\.]', date_str)
        if len(parts) == 3:
            # Try to infer format
            if len(parts[2]) == 4:  # Assume year is last
                a, b, year = parts
                if int(a) > 12 and int(b) <= 12:  # Likely DD/MM/YYYY
                    return f"{year}-{int(b):02d}-{int(a):02d}"
                else:  # Assume MM/DD/YYYY
                    return f"{year}-{int(a):02d}-{int(b):02d}"
            elif len(parts[0]) == 4:  # Assume year is first
                year, month, day = parts
                return f"{year}-{int(month):02d}-{int(day):02d}"
    
    return None

def parse_month_name_date(date_str):
    """
    Parse dates with month names like 'January 15, 2023' or '15th January 2023'
    
    Args:
        date_str: Date string containing month name
        
    Returns:
        Normalized date string in YYYY-MM-DD format or None if invalid
    """
    # Map month names to numbers
    month_map = {
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 
        'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
        'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
    }
    
    # Clean the date string
    date_str = date_str.strip().lower()
    date_str = re.sub(r'\s+', ' ', date_str)  # Normalize whitespace
    date_str = re.sub(r'[,;]', '', date_str)  # Remove commas and semicolons
    
    # Try multiple formats
    
    # Format: Month DD, YYYY (e.g., "January 15, 2023", "Jan 15 2023")
    match = re.search(r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t)?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s+|-)(\d{1,2})(?:st|nd|rd|th)?(?:\s+|,\s*|-)(\d{2,4})', date_str)
    if match:
        month_str = match.group(1)[:3]  # Take first 3 chars of month
        day = int(match.group(2))
        year = match.group(3)
        
        # Handle 2-digit years
        if len(year) == 2:
            year = f"20{year}" if int(year) < 50 else f"19{year}"
            
        if month_str in month_map and 1 <= day <= 31:
            return f"{year}-{month_map[month_str]}-{day:02d}"
    
    # Format: DD Month YYYY (e.g., "15 January 2023", "15th Jan 2023")
    match = re.search(r'(\d{1,2})(?:st|nd|rd|th)?(?:\s+|-)(?:of\s+)?(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t)?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s+|,\s*|-)(\d{2,4})', date_str)
    if match:
        day = int(match.group(1))
        month_str = match.group(2)[:3]  # Take first 3 chars of month
        year = match.group(3)
        
        # Handle 2-digit years
        if len(year) == 2:
            year = f"20{year}" if int(year) < 50 else f"19{year}"
            
        if month_str in month_map and 1 <= day <= 31:
            return f"{year}-{month_map[month_str]}-{day:02d}"
    
    # Format: YYYY Month DD (e.g., "2023 January 15", "2023 Jan 15")
    match = re.search(r'(\d{4})(?:\s+|-)(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t)?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s+|-)(\d{1,2})(?:st|nd|rd|th)?', date_str)
    if match:
        year = match.group(1)
        month_str = match.group(2)[:3]  # Take first 3 chars of month
        day = int(match.group(3))
            
        if month_str in month_map and 1 <= day <= 31:
            return f"{year}-{month_map[month_str]}-{day:02d}"
    
    # Format: Month YYYY (e.g., "January 2023", "Jan 2023") - day defaults to 1
    match = re.search(r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t)?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s+|,\s*|-)(\d{2,4})(?:\s|$)', date_str)
    if match:
        month_str = match.group(1)[:3]  # Take first 3 chars of month
        year = match.group(2)
        
        # Handle 2-digit years
        if len(year) == 2:
            year = f"20{year}" if int(year) < 50 else f"19{year}"
            
        if month_str in month_map:
            return f"{year}-{month_map[month_str]}-01"  # Default to 1st of month
    
    # Format: YYYY Month (e.g., "2023 January", "2023 Jan") - day defaults to 1
    match = re.search(r'(\d{4})(?:\s+|-)(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t)?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s|$)', date_str)
    if match:
        year = match.group(1)
        month_str = match.group(2)[:3]  # Take first 3 chars of month
        
        if month_str in month_map:
            return f"{year}-{month_map[month_str]}-01"  # Default to 1st of month
    
    return None

def extract_amount(amount_str):
    """
    Extract and normalize monetary amount.
    
    Args:
        amount_str: String containing a monetary amount
        
    Returns:
        Float amount or None if invalid
    """
    if not amount_str:
        return None
    
    # Remove currency symbols and commas
    cleaned = re.sub(r'[$,]', '', amount_str.strip())
    
    # Extract the numeric value
    match = re.search(r'(\d+(?:\.\d+)?)', cleaned)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    
    return None

def extract_invoice_number(text):
    """
    Extract invoice number from text.
    
    Args:
        text: Raw text from the invoice
        
    Returns:
        Invoice number string or None if not found
    """
    patterns = [
        # Standard formats
        r'Invoice\s+(?:Number|N\.|n\.|NO|no\.)\s*[#]?([A-Za-z0-9-]+)',
        r'Invoice\s+[#]?([A-Za-z0-9-]+)',
        r'Invoice\s*#\s*([A-Za-z0-9-]+)',
        r'Invoice\s*:\s*([A-Za-z0-9-]+)',
        r'INVOICE\s*NUMBER\s*:?\s*([A-Za-z0-9-]+)',
        
        # Additional formats with different prefixes
        r'Invoice\s+(?:ID|Id|id|I\.D\.|Reference|Ref|ref)\.?\s*:?\s*([A-Za-z0-9-]+)',
        r'(?:INV|Inv|inv)\.?\s*(?:Number|No|#)?\s*:?\s*([A-Za-z0-9-/_.]+)',
        r'(?:Invoice|INV|Inv)\s*(?:Number|No)?\s*[#]?\s*:?\s*([A-Za-z0-9-/_.]+)',
        
        # Numbered formats
        r'(?:Invoice|INV|Inv)(?:oice)?(?:\s|:|#)+([A-Za-z0-9]+-?[A-Za-z0-9]+)',
        r'(?:No|Number|NUM|Num)(?:ber)?(?:\s|:|#)+([A-Za-z0-9]+-?[A-Za-z0-9]+)',
        
        # Formats with "Invoice" at the end
        r'(?:No|Number|#)\s*:?\s*([A-Za-z0-9-/_.]+)(?:\s+Invoice)',
        
        # Document number formats
        r'Document\s+(?:Number|No|#)\s*:?\s*([A-Za-z0-9-/_.]+)',
        r'Doc\s*(?:ument)?\s*(?:Number|No|#|\.)\s*:?\s*([A-Za-z0-9-/_.]+)',
        
        # Receipt number formats
        r'Receipt\s+(?:Number|No|#)\s*:?\s*([A-Za-z0-9-/_.]+)',
        r'Receipt\s*(?:Number|No|#)?\s*:?\s*([A-Za-z0-9-/_.]+)',
        
        # Bill number formats
        r'Bill\s+(?:Number|No|#)\s*:?\s*([A-Za-z0-9-/_.]+)',
        
        # Standalone numbers with specific format (like INV-12345)
        r'\b(?:INV|Inv|inv|IN)[-:]([A-Za-z0-9-]+)',
        r'\b(?:INV|Inv|inv|IN)[-:]?(\d{5,})',
        
        # Reference number
        r'(?:Reference|Ref|ref)\.?\s*(?:Number|No|#)?\s*:?\s*([A-Za-z0-9-/_.]+)',
        
        # Account number formats (often used as invoice reference)
        r'Account\s+(?:Number|No|#)\s*:?\s*([A-Za-z0-9-/_.]+)',
        
        # Order number formats
        r'Order\s+(?:Number|No|#)\s*:?\s*([A-Za-z0-9-/_.]+)',
        
        # Simple number label
        r'(?<!\w)(?:NO|No|no|#)\.?\s*:?\s*([A-Za-z0-9-/_.]+)',
        
        # Very loose pattern - use cautiously, last resort
        r'(?<!\w)([A-Z]{2,}-\d{4,}(?:-[A-Z0-9]+)?)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            # Clean the extracted invoice number
            inv_num = match.group(1).strip()
            # Remove any trailing colon, period, or comma
            inv_num = re.sub(r'[:.,-]+$', '', inv_num)
            return inv_num
    
    return None

def extract_issue_date(text):
    """
    Extract issue date from text.
    
    Args:
        text: Raw text from the invoice
        
    Returns:
        Datetime object or None if not found
    """
    patterns = [
        # Standard formats with labels
        r'(?:Issue\s+Date|Date):\s*(?:(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})|(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{2,4}))',
        r'Date\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
        r'Date\s*:?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
        r'Issue\s+Date\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
        r'Invoice\s+Date\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
        
        # Additional issue date labels
        r'(?:Date\s+(?:of|on)\s+Issue|Issue\s+Date|Date\s+Issued|Issued\s+(?:on|date)|Invoice\s+Date)\s*:?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
        r'(?:Date\s+(?:of|on)\s+Issue|Issue\s+Date|Date\s+Issued|Issued\s+(?:on|date)|Invoice\s+Date)\s*:?\s*(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})',
        
        # Date + month name patterns
        r'(?:Date|Issue\s+Date|Invoice\s+Date)\s*:?\s*(?:the)?\s*(\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*,?\s*\d{2,4})',
        r'(?:Date|Issue\s+Date|Invoice\s+Date)\s*:?\s*(?:the)?\s*(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{2,4})',
        
        # Standalone date patterns (month first)
        r'\b(\d{1,2})[-/\.](\d{1,2})[-/\.](\d{2,4})\b',
        
        # European format (day first)
        r'\b(\d{1,2})[-/\.](\d{1,2})[-/\.](\d{2,4})\b',
        
        # ISO format (year first)
        r'\b(\d{4})[-/\.](\d{1,2})[-/\.](\d{1,2})\b',
        
        # Date with month name (various formats)
        r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*,?\s*(\d{2,4})\b',
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{2,4})\b',
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{2,4})\b',
        
        # Year first with month name
        r'\b(\d{4})\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?\b',
        
        # Dates with "created" or "generated" keywords
        r'(?:Created|Generated|Prepared|Printed)\s+(?:on|at|date)?\s*:?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
        r'(?:Created|Generated|Prepared|Printed)\s+(?:on|at|date)?\s*:?\s*(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})',
        
        # Date in specific contexts
        r'Invoice\s+Date\s*:?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
        r'Bill\s+Date\s*:?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
        r'Statement\s+Date\s*:?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
        
        # Transaction date
        r'Transaction\s+Date\s*:?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
        
        # Service date
        r'Service\s+Date\s*:?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
        
        # For dates specified with only month and year
        r'(?:Date|Issue\s+Date|Invoice\s+Date)\s*:?\s*(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                # For complex patterns with two capture groups for different formats
                if match.lastindex == 2 and pattern.startswith(r'(?:Issue\s+Date|Date)'):
                    date_str = match.group(2) if match.group(2) else match.group(1)
                
                # For month and day patterns with capture groups for parts
                elif '(?:Jan' in pattern and match.lastindex >= 2:
                    # This is for patterns with month name and day, year in separate groups
                    if match.lastindex == 2:
                        # Handle "January 15, 2023" format
                        month_name = re.search(r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)', 
                                              match.group(0)).group(0)
                        day = match.group(1)
                        year = match.group(2)
                        date_str = f"{month_name} {day}, {year}"
                    else:
                        # Handle other formats with month, day, year in text
                        date_str = match.group(0)
                
                # For standard date patterns like DD/MM/YYYY
                elif match.lastindex == 3 and re.search(r'\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}', match.group(0)):
                    # This captures the three parts of a date
                    part1 = match.group(1)
                    part2 = match.group(2)
                    part3 = match.group(3)
                    
                    # Try to determine format based on values
                    if len(part1) == 4:  # YYYY-MM-DD
                        date_str = f"{part1}-{part2}-{part3}"
                    elif int(part1) > 12 and int(part1) <= 31:  # Likely DD-MM-YYYY
                        date_str = f"{part3}-{part2}-{part1}"
                    else:  # Assume MM-DD-YYYY
                        date_str = f"{part3}-{part1}-{part2}"
                
                # For simple single group patterns
                else:
                    date_str = match.group(1)
                
                normalized_date = normalize_date(date_str)
                if normalized_date:
                    return datetime.strptime(normalized_date, '%Y-%m-%d')
            except (ValueError, IndexError, AttributeError) as e:
                logging.debug(f"Error parsing date '{match.group(0)}': {str(e)}")
                continue
    
    return None

def extract_due_date(text):
    """
    Extract due date from text.
    
    Args:
        text: Raw text from the invoice
        
    Returns:
        Datetime object or None if not found
    """
    patterns = [
        # Standard formats
        r'Due\s+Date\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
        r'Due\s+Date\s*:?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
        r'Payment\s+Due\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
        r'Due\s+By\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
        
        # Additional date format variations
        r'(?:Due|Payment\s+Due|Pay\s+By|Payment\s+By)\s*(?:Date|On)?\s*:?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
        r'(?:Due|Payment\s+Due|Pay\s+By|Payment\s+By)\s*(?:Date|On)?\s*:?\s*(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})',
        
        # Payment terms with specific due dates
        r'(?:Terms|Payment\s+Terms)\s*:?\s*(?:Due|Net)?\s*(?:on|by)?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
        r'(?:Terms|Payment\s+Terms)\s*:?\s*(?:Due|Net)?\s*(?:on|by)?\s*(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})',
        
        # Due date with month names
        r'(?:Due|Payment\s+Due|Pay\s+By)\s*(?:Date|On)?\s*:?\s*(?:the)?\s*(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*,?\s*(\d{2,4})',
        r'(?:Due|Payment\s+Due|Pay\s+By)\s*(?:Date|On)?\s*:?\s*(?:the)?\s*(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{2,4})',
        
        # Payment due in X days
        r'(?:Payment\s+Due|Due|Pay)\s*(?:within|in)\s*(\d+)\s*days\s*(?:of|after|from)\s*(?:invoice|receipt)',
        
        # Net X days
        r'(?:Net|Terms|Payment\s+Terms)\s*:?\s*(?:Net)?\s*(\d+)',
        
        # Due upon receipt variations
        r'(?:Due|Payment)\s*(?:upon|on)\s*receipt',
        
        # Due date at the end of month
        r'(?:Due|Payment\s+Due)\s*(?:at|by|on)\s*(?:the)?\s*end\s*(?:of)?\s*(?:the)?\s*month',
        
        # Specific date mentioned after "payable before" or similar
        r'(?:Payable|Payment|Pay)\s*(?:before|by|prior\s+to)\s*(?:the)?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
        
        # Date to pay
        r'(?:Date|Day)\s*to\s*Pay\s*:?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
        
        # Final payment date
        r'Final\s*(?:Payment)?\s*(?:Date|Day|Due)\s*:?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
        
        # Deadline for payment
        r'(?:Deadline|Latest|Last\s+Date)\s*(?:for|of)?\s*(?:Payment|Pay)\s*:?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
        
        # Pay no later than
        r'Pay\s*(?:no|not)?\s*later\s*than\s*(?:the)?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
        
        # Expiry date (sometimes used as due date)
        r'(?:Expiry|Expire|Expiration)\s*(?:Date|On)?\s*:?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                # Check if this is a "Net X days" or "Due in X days" pattern
                if re.search(r'(?:within|in)\s*(\d+)\s*days', pattern) or re.search(r'(?:Net|Terms)\s*:?\s*(?:Net)?\s*(\d+)', pattern):
                    # These patterns require additional calculation based on invoice date
                    # For now, just logging that we found this pattern but can't process it yet
                    logging.info(f"Found payment term: {match.group(0)}")
                    continue  # Skip to next pattern
                
                # If it's a "due upon receipt" pattern
                if "upon receipt" in pattern:
                    # This requires setting due date equal to invoice date
                    logging.info("Found 'due upon receipt' term")
                    continue  # Skip to next pattern
                
                # If it's end of month
                if "end of month" in pattern:
                    # This requires calculation based on invoice date
                    logging.info("Found 'end of month' term")
                    continue  # Skip to next pattern
                
                # For due date with month name patterns with multiple groups
                if '(?:Jan' in pattern and match.lastindex >= 2:
                    # This is for patterns with month name and day, year in separate groups
                    if match.lastindex == 2:
                        # Try to extract month name from the original text
                        month_match = re.search(r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)', 
                                               match.group(0))
                        if month_match:
                            month_name = month_match.group(0)
                            day = match.group(1)
                            year = match.group(2)
                            date_str = f"{month_name} {day}, {year}"
                        else:
                            continue
                    else:
                        # Handle other formats with month, day, year in text
                        date_str = match.group(0)
                # Default case - get the first capture group as the date string
                else:
                    date_str = match.group(1)
                
                normalized_date = normalize_date(date_str)
                if normalized_date:
                    return datetime.strptime(normalized_date, '%Y-%m-%d')
            except (ValueError, IndexError, AttributeError) as e:
                logging.debug(f"Error parsing due date '{match.group(0)}': {str(e)}")
                continue
    
    return None

def extract_total_amount(text):
    """
    Extract total amount from text.
    
    Args:
        text: Raw text from the invoice
        
    Returns:
        Float amount or None if not found
    """
    patterns = [
        # Standard formats
        r'TOTAL\s+\$(\d+\.\d+)',
        r'TOTAL[\s\$]+(\d+\.\d+)',
        r'Total:\s+\$(\d+\.\d+)',
        r'Total\s+\$(\d+\.\d+)',
        r'Total\s+([A-Za-z]+)\s+\$(\d+\.\d+)',
        r'Total\s*:?\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2}))',
        r'Amount\s+Due\s*:?\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2}))',
        
        # Additional total formats (USD)
        r'(?:Total|Amount|Sum|Grand\s+Total|Invoice\s+Total)\s*:?\s*(?:USD|US\$|US)?\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        r'(?:Total|Amount|Sum|Grand\s+Total|Invoice\s+Total)\s*:?\s*\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        r'(?:Total|Amount|Sum|Grand\s+Total|Invoice\s+Total)\s*:?\s*(?:USD|US\$|US)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        
        # Euro currency formats
        r'(?:Total|Amount|Sum|Grand\s+Total|Invoice\s+Total)\s*:?\s*(?:EUR|€)\s*(\d{1,3}(?:[,.]\d{3})*(?:[,.]\d{2})?)',
        r'(?:Total|Amount|Sum|Grand\s+Total|Invoice\s+Total)\s*:?\s*(\d{1,3}(?:[,.]\d{3})*(?:[,.]\d{2})?)\s*(?:EUR|€)',
        
        # British Pound formats
        r'(?:Total|Amount|Sum|Grand\s+Total|Invoice\s+Total)\s*:?\s*(?:GBP|£)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        r'(?:Total|Amount|Sum|Grand\s+Total|Invoice\s+Total)\s*:?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:GBP|£)',
        
        # Canadian Dollar formats
        r'(?:Total|Amount|Sum|Grand\s+Total|Invoice\s+Total)\s*:?\s*(?:CAD|C\$)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        
        # Australian Dollar formats
        r'(?:Total|Amount|Sum|Grand\s+Total|Invoice\s+Total)\s*:?\s*(?:AUD|A\$)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        
        # Japanese Yen formats (typically no decimal)
        r'(?:Total|Amount|Sum|Grand\s+Total|Invoice\s+Total)\s*:?\s*(?:JPY|¥)\s*(\d{1,3}(?:,\d{3})*)',
        
        # Indian Rupee formats
        r'(?:Total|Amount|Sum|Grand\s+Total|Invoice\s+Total)\s*:?\s*(?:INR|Rs\.?)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        
        # Chinese Yuan formats
        r'(?:Total|Amount|Sum|Grand\s+Total|Invoice\s+Total)\s*:?\s*(?:CNY|RMB|¥)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        
        # No currency symbol, just numbers with total
        r'(?:Total|Amount|Sum|Balance|Grand\s+Total|Invoice\s+Total)\s*(?:Due|Payable|to\s+Pay)?\s*:?\s*(\d{1,3}(?:[,.]\d{3})*(?:[,.]\d{2})?)',
        
        # Amount due formats
        r'(?:Amount|Sum|Balance)\s*(?:Due|Payable|to\s+Pay)\s*:?\s*(?:USD|US\$|US|EUR|€|GBP|£)?\s*(\d{1,3}(?:[,.]\d{3})*(?:[,.]\d{2})?)',
        
        # Total for payment
        r'(?:Total|Amount)\s+for\s+(?:Payment|Remittance)\s*:?\s*(?:USD|US\$|US|EUR|€|GBP|£)?\s*(\d{1,3}(?:[,.]\d{3})*(?:[,.]\d{2})?)',
        
        # Total without decimal (whole numbers)
        r'(?:Total|Amount|Sum|Grand\s+Total|Invoice\s+Total)\s*:?\s*(?:USD|US\$|US|EUR|€|GBP|£)?\s*(\d{1,3}(?:,\d{3})*)',
        
        # Please pay
        r'Please\s+Pay\s*:?\s*(?:USD|US\$|US|EUR|€|GBP|£)?\s*(\d{1,3}(?:[,.]\d{3})*(?:[,.]\d{2})?)',
        
        # Payment amount
        r'Payment\s+(?:Amount|Total)\s*:?\s*(?:USD|US\$|US|EUR|€|GBP|£)?\s*(\d{1,3}(?:[,.]\d{3})*(?:[,.]\d{2})?)',
        
        # Balance due
        r'Balance\s+(?:Due|Payable|to\s+Pay)\s*:?\s*(?:USD|US\$|US|EUR|€|GBP|£)?\s*(\d{1,3}(?:[,.]\d{3})*(?:[,.]\d{2})?)',
        
        # Current balance
        r'Current\s+Balance\s*:?\s*(?:USD|US\$|US|EUR|€|GBP|£)?\s*(\d{1,3}(?:[,.]\d{3})*(?:[,.]\d{2})?)',
        
        # Outstanding balance
        r'Outstanding\s+(?:Balance|Amount)\s*:?\s*(?:USD|US\$|US|EUR|€|GBP|£)?\s*(\d{1,3}(?:[,.]\d{3})*(?:[,.]\d{2})?)',
        
        # Net amount/payable
        r'Net\s+(?:Amount|Total|Payable)\s*:?\s*(?:USD|US\$|US|EUR|€|GBP|£)?\s*(\d{1,3}(?:[,.]\d{3})*(?:[,.]\d{2})?)',
        
        # Final amount
        r'Final\s+(?:Amount|Total|Sum)\s*:?\s*(?:USD|US\$|US|EUR|€|GBP|£)?\s*(\d{1,3}(?:[,.]\d{3})*(?:[,.]\d{2})?)',
        
        # Total with tax or VAT included
        r'(?:Total|Amount)\s+(?:with|inc\.?|including)\s+(?:Tax|VAT|GST)\s*:?\s*(?:USD|US\$|US|EUR|€|GBP|£)?\s*(\d{1,3}(?:[,.]\d{3})*(?:[,.]\d{2})?)',
        
        # Total with currency code after amount
        r'(?:Total|Amount|Sum|Grand\s+Total|Invoice\s+Total)\s*:?\s*(\d{1,3}(?:[,.]\d{3})*(?:[,.]\d{2})?)\s*(?:USD|US\$|EUR|€|GBP|£)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                if pattern == r'Total\s+([A-Za-z]+)\s+\$(\d+\.\d+)':
                    amount_str = match.group(2)
                else:
                    amount_str = match.group(1)
                    
                # Replace comma as thousand separator and ensure period as decimal
                amount_str = amount_str.replace(',', '')
                
                # If there's a comma used as decimal separator (European format)
                if '.' not in amount_str and ',' in amount_str:
                    amount_str = amount_str.replace(',', '.')
                
                return extract_amount(amount_str)
            except (ValueError, IndexError) as e:
                logging.debug(f"Error parsing amount '{match.group(0)}': {str(e)}")
                continue
    
    return None

def extract_issuer_name(text):
    """
    Extract issuer name from text.
    
    Args:
        text: Raw text from the invoice
        
    Returns:
        Issuer name string or None if not found
    """
    # Try to find company name at the top of invoice
    lines = text.split('\n')
    
    # Check the first few lines for potential company names
    for line in lines[:5]:
        # Potential company name criteria
        if (len(line.strip()) > 3 and 
            not re.search(r'\d', line) and 
            line.strip().lower() not in ['invoice', 'receipt', 'statement', 'bill', 'quotation']):
            return line.strip()
    
    # Try to find with specific patterns
    patterns = [
        # From/By fields
        r'From\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        r'Issued\s+By\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        r'Company\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        r'Vendor\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        
        # Common prefixes for issuer
        r'(?:Seller|Provider|Supplier|Biller|Issuer)\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        
        # Company headers
        r'(?:Company|Business|Corporation|Inc\.?|Corp\.?|LLC|Ltd\.?)\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        r'(?:Company|Business|Corporation)\s+(?:Name|Title)\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        
        # Merchant information
        r'(?:Merchant|Store|Shop|Retailer)\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        
        # Payment to
        r'(?:Payment|Pay|Remit)\s+(?:to|To)\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        
        # Include common business entity types
        r'\b([A-Za-z0-9\s,\.&\'-]+(?:\s+(?:Inc|LLC|Ltd|Corporation|Corp|Company|Co|GmbH|S\.A\.|S\.p\.A\.|B\.V\.|Pty Ltd|Limited|L\.P\.|LLP)))\b',
        
        # Service provider
        r'(?:Service\s+Provider|Contractor)\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        
        # Invoicing entity
        r'(?:Invoicing\s+Entity|Billing\s+Entity)\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        
        # Account information with name
        r'Account\s+(?:Name|Holder|Owner)\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        
        # Tax information often includes company name
        r'(?:Tax|VAT|GST)\s+(?:Name|Entity)\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        
        # Contact information
        r'Contact\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        
        # Address block with company name first
        r'([A-Za-z0-9\s,\.&\'-]+)(?:\n|\r)(?:\d+\s+[A-Za-z0-9\s,\.&\'-]+)(?:\n|\r)(?:[A-Za-z]+(?:\s+[A-Za-z]+)*,\s*[A-Z]{2}\s+\d{5})',
        
        # Website mentions (often has company name)
        r'(?:Website|Web|Site|URL)\s*:?\s*(?:www\.|http)(?:[A-Za-z0-9\.-]+)(?=/|$)',
        
        # Email domain might indicate company name
        r'Email\s*:?\s*[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})(?:\n|$)',
        
        # Bank account details
        r'Bank\s+(?:Name|Account)\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        
        # Letterhead check (often at top of page)
        r'^([A-Za-z0-9\s,\.&\'-]{5,50})(?:\n|\r)',
        
        # Copyright notice
        r'(?:©|Copyright)\s+(?:\d{4})?\s*([A-Za-z0-9\s,\.&\'-]+)',
        
        # Registered/Trademark
        r'([A-Za-z0-9\s,\.&\'-]+)(?:®|™)',
        
        # Header section with logo
        r'Logo\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Get the matching company name
            company_name = match.group(1).strip() if match.lastindex else match.group(0).strip()
            
            # Clean the company name
            company_name = company_name.strip()
            
            # Remove common prefixes if they were included in the match
            prefixes_to_remove = [
                'company:', 'business:', 'from:', 'by:', 'vendor:', 'seller:', 'provider:',
                'supplier:', 'company name:', 'business name:', 'merchant:',
            ]
            for prefix in prefixes_to_remove:
                if company_name.lower().startswith(prefix):
                    company_name = company_name[len(prefix):].strip()
            
            # Remove trailing punctuation
            company_name = re.sub(r'[:\.,;]+$', '', company_name)
            
            # Skip if too short or contains unwanted terms
            if len(company_name) < 3:
                continue
                
            if company_name.lower() in ['invoice', 'receipt', 'statement', 'bill', 'quotation']:
                continue
                
            return company_name
    
    # Try to extract from email domain (last resort)
    email_match = re.search(r'[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+)\.[A-Za-z]{2,}', text)
    if email_match:
        domain = email_match.group(1)
        # Convert to title case and remove typical domain parts
        domain = domain.replace('.', ' ').title()
        domain = re.sub(r'\b(Mail|Email|Web|Site|Host|Domain|Server)\b', '', domain).strip()
        if len(domain) > 3:
            return domain
    
    return None

def extract_recipient_name(text):
    """
    Extract recipient name from text.
    
    Args:
        text: Raw text from the invoice
        
    Returns:
        Recipient name string or None if not found
    """
    patterns = [
        # Standard bill to formats
        r'Bill\s+To:?\s+([A-Za-z0-9\s,\.&\'-]+)',
        r'Customer:?\s+([A-Za-z0-9\s,\.&\'-]+)',
        r'Client:?\s+([A-Za-z0-9\s,\.&\'-]+)',
        r"Bill To:\s+([A-Za-z0-9\s,\.&\'-]+)",
        r"To:\s+([A-Za-z0-9\s,\.&\'-]+)",
        r'Billed\s+To\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        r'Recipient\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        
        # Additional recipient indicators
        r'(?:Sold|Ship|Deliver|Invoice)\s+To\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        r'(?:Buyer|Purchaser|Customer|Billing|Client)\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        r'(?:Attention|ATTN)\s*:?\s*(?:of|to)?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        
        # Recipient with title
        r'(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+([A-Za-z\s\'-]+)(?:\n|$)',
        r'(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+([A-Za-z\'-]+\s+[A-Za-z\'-]+)(?:\n|$)',
        
        # Addressed to patterns
        r'(?:Addressed|Sent)\s+To\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        r'(?:For|For\s+the\s+attention\s+of)\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        
        # Account name as recipient
        r'(?:Account|Client|Customer)\s+Name\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        
        # Address block with recipient name first
        r'(?:Address|Delivery\s+Address|Shipping\s+Address)\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        
        # Recipient in billing/shipping address block
        r'(?:Billing|Shipping|Delivery|Mailing)\s+(?:Address|Information|Details)\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        
        # Name field
        r'(?:Name|Full\s+Name|Contact\s+Name)\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        
        # Contact information blocks
        r'(?:Contact|Contact\s+Information|Contact\s+Details)\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        
        # Order/purchase by
        r'(?:Order|Purchase|Purchased|Ordered)\s+By\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        
        # Ship to address with name
        r'Ship\s+To\s+Address\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        
        # Address with company or person name
        r'Address\s*:?\s*(?:(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+)?([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        
        # Payee information (reverse - recipient is paying)
        r'(?:Payee|Paid\s+By)\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        
        # Customer ID with name
        r'Customer\s+(?:ID|Number|No)\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        
        # Addressee at beginning of letter format
        r'Dear\s+(?:(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+)?([A-Za-z\s\'-]+)(?:,|\n|$)',
        
        # Job site or location (often has client name)
        r'(?:Job\s+Site|Site\s+Location|Project\s+Location)\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        
        # Employer or business recipient
        r'(?:Employer|Business)\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
        
        # Check for typical billing address pattern with person/company name at start
        r'([A-Za-z\s\'-]+)(?:\n|\r)(?:\d+\s+[A-Za-z\s\'-]+)(?:\n|\r)(?:[A-Za-z]+(?:,|\s+)[A-Za-z]{2}\s+\d{5})',
        
        # Recipient tag
        r'Recipient\s*:?\s*([A-Za-z0-9\s,\.&\'-]+)(?:\n|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Get the matching recipient name
            recipient_name = match.group(1).strip()
            
            # Clean the recipient name
            recipient_name = recipient_name.strip()
            
            # Remove common prefixes if they were included in the match
            prefixes_to_remove = [
                'bill to:', 'customer:', 'client:', 'to:', 'billed to:', 'recipient:', 
                'sold to:', 'ship to:', 'deliver to:', 'invoice to:', 'buyer:', 'purchaser:',
                'billing:', 'attention:', 'attn:', 'addressed to:', 'sent to:', 'for:',
                'account name:', 'client name:', 'customer name:', 'name:', 'contact:',
                'address:', 'billing address:', 'shipping address:', 'delivery address:',
                'order by:', 'purchase by:', 'payee:', 'paid by:'
            ]
            for prefix in prefixes_to_remove:
                if recipient_name.lower().startswith(prefix):
                    recipient_name = recipient_name[len(prefix):].strip()
            
            # Remove trailing punctuation
            recipient_name = re.sub(r'[:\.,;]+$', '', recipient_name)
            
            # Skip if too short or likely not a name
            if len(recipient_name) < 3:
                continue
                
            # Skip if it's just "bill to" or similar
            if recipient_name.lower() in ['bill to', 'ship to', 'customer', 'client']:
                continue
                
            return recipient_name
    
    return None

def extract_all_fields(text):
    """
    Extract all invoice fields using rule-based methods.
    
    Args:
        text: Raw text from the invoice
        
    Returns:
        Dictionary with all extracted fields
    """
    return {
        'invoice_number': extract_invoice_number(text),
        'date': extract_issue_date(text),
        'due_date': extract_due_date(text),
        'total_amount': extract_total_amount(text),
        'issuer_name': extract_issuer_name(text),
        'recipient_name': extract_recipient_name(text)
    } 