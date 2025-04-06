import re
from datetime import datetime
from typing import Optional, Dict, Tuple, List
from .base_extractor import BaseExtractor, InvoiceData

class RuleBasedExtractor(BaseExtractor):
    """Rule-based extractor using regex patterns and keyword matching."""
    
    def __init__(self):
        # Common patterns for invoice fields
        self.patterns = {
            'invoice_number': [
                r'(?i)invoice\s*(?:number|no\.?|#)?\s*[:#]?\s*([A-Z0-9-]+)',
                r'(?i)inv\.?\s*(?:number|no\.?|#)?\s*[:#]?\s*([A-Z0-9-]+)',
            ],
            'date': [
                r'(?i)date\s*[:#]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                r'(?i)issued\s*date\s*[:#]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            ],
            'due_date': [
                r'(?i)due\s*date\s*[:#]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                r'(?i)payment\s*due\s*[:#]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            ],
            'total_amount': [
                r'(?i)total\s*(?:amount|sum)?\s*[:#]?\s*[\$€£]?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?i)amount\s*due\s*[:#]?\s*[\$€£]?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            ],
        }
        
        # Keywords to identify issuer and recipient sections
        self.issuer_keywords = ['from', 'issued by', 'sender', 'company', 'business']
        self.recipient_keywords = ['to', 'bill to', 'ship to', 'customer', 'client']
        
    def extract(self, document_text: str, **kwargs) -> InvoiceData:
        """
        Extract invoice information using rule-based patterns.
        
        Args:
            document_text: The text content of the document
            **kwargs: Additional parameters
            
        Returns:
            InvoiceData object containing the extracted information
        """
        text = self.preprocess_text(document_text)
        extracted_data = InvoiceData(raw_text=text)
        
        # Extract fields using patterns
        for field, patterns in self.patterns.items():
            value = self._extract_field(text, patterns)
            if value:
                setattr(extracted_data, field, value)
        
        # Extract issuer and recipient names
        extracted_data.issuer_name = self._extract_name(text, self.issuer_keywords)
        extracted_data.recipient_name = self._extract_name(text, self.recipient_keywords)
        
        # Calculate confidence scores
        extracted_data.confidence_scores = self._calculate_confidence_scores(extracted_data)
        
        return self.postprocess_extraction(extracted_data)
    
    def _extract_field(self, text: str, patterns: List[str]) -> Optional[str]:
        """Extract a field using multiple patterns."""
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return None
    
    def _extract_name(self, text: str, keywords: List[str]) -> Optional[str]:
        """Extract name based on keywords."""
        lines = text.split('\n')
        for i, line in enumerate(lines):
            for keyword in keywords:
                if keyword.lower() in line.lower():
                    # Try to get the next line as it often contains the name
                    if i + 1 < len(lines):
                        return lines[i + 1].strip()
                    return line.strip()
        return None
    
    def _calculate_confidence_scores(self, data: InvoiceData) -> Dict[str, float]:
        """Calculate confidence scores for extracted fields."""
        scores = {}
        for field in ['invoice_number', 'date', 'due_date', 'total_amount', 
                     'issuer_name', 'recipient_name']:
            value = getattr(data, field)
            if value is not None:
                # Simple confidence scoring based on field presence
                scores[field] = 1.0
            else:
                scores[field] = 0.0
        return scores
    
    def validate_extraction(self, extracted_data: InvoiceData) -> bool:
        """
        Validate the extracted data.
        
        Basic validation checks:
        1. At least invoice number and total amount should be present
        2. Dates should be in valid format if present
        3. Names should not be empty strings if present
        """
        if not extracted_data.invoice_number or not extracted_data.total_amount:
            return False
            
        # Validate dates if present
        for date_field in ['date', 'due_date']:
            date_value = getattr(extracted_data, date_field)
            if date_value is not None:
                try:
                    if isinstance(date_value, str):
                        datetime.strptime(date_value, '%Y-%m-%d')
                except ValueError:
                    return False
        
        # Validate names if present
        for name_field in ['issuer_name', 'recipient_name']:
            name_value = getattr(extracted_data, name_field)
            if name_value is not None and not name_value.strip():
                return False
                
        return True
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the input text.
        
        - Remove extra whitespace
        - Normalize line endings
        - Remove special characters that might interfere with pattern matching
        """
        text = super().preprocess_text(text)
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.replace('\r\n', '\n').replace('\r', '\n')  # Normalize line endings
        return text 