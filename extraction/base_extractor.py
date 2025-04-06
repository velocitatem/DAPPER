from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

@dataclass
class InvoiceData:
    """Data class to store extracted invoice information."""
    invoice_number: Optional[str] = None
    date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    issuer_name: Optional[str] = None
    recipient_name: Optional[str] = None
    total_amount: Optional[float] = None
    raw_text: Optional[str] = None
    confidence_scores: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseExtractor(ABC):
    """Abstract base class for invoice information extractors."""
    
    @abstractmethod
    def extract(self, document_text: str, **kwargs) -> InvoiceData:
        """
        Extract invoice information from document text.
        
        Args:
            document_text: The text content of the document
            **kwargs: Additional parameters specific to the extractor implementation
            
        Returns:
            InvoiceData object containing the extracted information
        """
        pass
    
    @abstractmethod
    def validate_extraction(self, extracted_data: InvoiceData) -> bool:
        """
        Validate the extracted data for completeness and correctness.
        
        Args:
            extracted_data: The InvoiceData object to validate
            
        Returns:
            bool: True if the extraction is valid, False otherwise
        """
        pass
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the input text before extraction.
        
        Args:
            text: The input text to preprocess
            
        Returns:
            str: The preprocessed text
        """
        # Basic preprocessing - can be overridden by subclasses
        return text.strip()
    
    def postprocess_extraction(self, extracted_data: InvoiceData) -> InvoiceData:
        """
        Post-process the extracted data.
        
        Args:
            extracted_data: The InvoiceData object to post-process
            
        Returns:
            InvoiceData: The post-processed InvoiceData object
        """
        return extracted_data 