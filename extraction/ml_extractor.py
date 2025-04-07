import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import pytesseract
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from .base_extractor import BaseExtractor, InvoiceData
from .utils import normalize_date, extract_amount, clean_text
from .fallback import (
    extract_invoice_number, 
    extract_issue_date, 
    extract_due_date, 
    extract_total_amount,
    extract_issuer_name,
    extract_recipient_name,
    extract_all_fields
)
import logging
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import re
from difflib import SequenceMatcher
import pandas as pd
from pathlib import Path
from tqdm import tqdm

class InvoiceDataset(Dataset):
    """Dataset for training invoice information extraction models."""
    
    def __init__(self, data_dir, annotations_file=None, transform=None, processor=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing invoice images
            annotations_file: Path to JSON file with annotations
            transform: Optional image transformations
            processor: Optional text processor for feature extraction
        """
        self.data_dir = data_dir
        self.transform = transform
        self.processor = processor
        
        # Load annotations if available
        self.annotations = {}
        if annotations_file and os.path.exists(annotations_file):
            with open(annotations_file, 'r') as f:
                self.annotations = json.load(f)
        
        # Get list of image files
        self.image_files = []
        for filename in os.listdir(data_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                self.image_files.append(filename)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        image_path = os.path.join(self.data_dir, img_name)
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        
        # Extract text using OCR
        text = pytesseract.image_to_string(image)
        
        # Get annotations if available
        target = {}
        if img_name in self.annotations:
            target = self.annotations[img_name]
        
        # Process for LayoutLMv3 if processor is provided
        if self.processor:
            # Get word-level boxes for layout analysis
            words = []
            boxes = []
            
            # Extract words and their bounding boxes
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            for i in range(len(ocr_data['text'])):
                if ocr_data['text'][i].strip():
                    words.append(ocr_data['text'][i])
                    x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                    boxes.append([x, y, x + w, y + h])
            
            # Get image dimensions for normalization
            width, height = image.size
            
            for i in range(len(ocr_data['text'])):
                if ocr_data['text'][i].strip():
                    words.append(ocr_data['text'][i])
                    
                    # Get raw bounding box coordinates
                    x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                    
                    # Normalize coordinates to 0-1000 range
                    x_norm = min(max(int((x / width) * 1000), 0), 1000)
                    y_norm = min(max(int((y / height) * 1000), 0), 1000)
                    right_norm = min(max(int(((x + w) / width) * 1000), 0), 1000)
                    bottom_norm = min(max(int(((y + h) / height) * 1000), 0), 1000)
                    
                    # Add normalized coordinates
                    boxes.append([x_norm, y_norm, right_norm, bottom_norm])
            
            # Prepare inputs for LayoutLMv3
            encoding = self.processor(
                image,
                words,
                boxes=boxes,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
            
            # Remove batch dimension
            for k, v in encoding.items():
                encoding[k] = v.squeeze()
                
            # Add target
            if target:
                # Convert string labels to token-level labels
                labels = self._align_labels(words, target)
                encoding['labels'] = torch.tensor(labels)
            
            return encoding
        
        # Return simple image, text, target format if no processor
        return {
            'image': image,
            'text': text,
            'target': target
        }
    
    def _align_labels(self, words, target):
        """Align target annotations with token-level labels."""
        # Simplified implementation - in practice, this would be more complex
        # and would align annotations with token positions
        labels = [-100] * len(words)  # -100 is a special value for ignored tokens
        
        for i, word in enumerate(words):
            # Check for invoice number
            if target.get('invoice_number') and target['invoice_number'] in word:
                labels[i] = 1  # 1 = invoice number
            # Check for date
            elif target.get('date') and target['date'] in word:
                labels[i] = 2  # 2 = date
            # Check for amount
            elif target.get('total_amount') and str(target['total_amount']) in word:
                labels[i] = 3  # 3 = amount
            # Etc. for other fields
        
        return labels


class MLExtractor(BaseExtractor):
    """Machine learning-based invoice information extractor using LayoutLM."""
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the ML extractor.
        
        Args:
            model_path: Path to a pre-trained model
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize LayoutLMv3 processor and model
        self.processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        
        if model_path and os.path.exists(model_path):
            # Load pre-trained model
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
        else:
            # Initialize with pre-trained weights
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                "microsoft/layoutlmv3-base", 
                num_labels=8  # Adjust based on your extraction needs
            )
        
        self.model.to(self.device)
        self.model.eval()
        
        # Define label mapping
        self.id2label = {
            0: "O",           # Outside (not a named entity)
            1: "invoice_num", # Invoice number
            2: "date",        # Invoice date
            3: "due_date",    # Due date
            4: "total",       # Total amount
            5: "issuer",      # Issuer name
            6: "recipient",   # Recipient name
            7: "other"        # Other relevant information
        }
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract(self, document_path, **kwargs):
        """
        Extract information from an invoice image.
        
        Args:
            document_path: Path to the invoice image
            **kwargs: Additional parameters
            
        Returns:
            InvoiceData object with extracted information
        """
        try:
            # Load and preprocess image
            image = Image.open(document_path).convert('RGB')
            
            # Extract text using OCR
            text = pytesseract.image_to_string(image)
            text = self.preprocess_text(text)
            
            # Get word-level boxes for layout analysis
            words = []
            boxes = []
            
            # Get image dimensions for normalization
            width, height = image.size
            
            # Extract words and their bounding boxes
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Filter out empty words
            valid_indices = [i for i in range(len(ocr_data['text'])) if ocr_data['text'][i].strip()]
            
            # Debug OCR output
            print(f"OCR detected {len(valid_indices)} words")
            
            for i in valid_indices:
                words.append(ocr_data['text'][i])
                
                # Get raw bounding box coordinates
                x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                
                # Normalize coordinates to 0-1000 range
                x_norm = int(np.clip((x / width) * 1000, 0, 1000))
                y_norm = int(np.clip((y / height) * 1000, 0, 1000))
                right_norm = int(np.clip(((x + w) / width) * 1000, 0, 1000))
                bottom_norm = int(np.clip(((y + h) / height) * 1000, 0, 1000))
                
                # Add normalized coordinates
                boxes.append([x_norm, y_norm, right_norm, bottom_norm])
            
            if not words:
                raise ValueError("No words detected in the image")
                
            # Prepare inputs for LayoutLMv3
            encoding = self.processor(
                image,
                words,
                boxes=boxes,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
            
            # Debug encoding shape
            print(f"Input encoding shape - input_ids: {encoding['input_ids'].shape}")
            
            # Move inputs to device
            for k, v in encoding.items():
                encoding[k] = v.to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**encoding)
                
                # Debug model output
                print(f"Model output shape: {outputs.logits.shape}")
                
                # Get predictions for actual tokens (ignore padding/special tokens)
                attention_mask = encoding['attention_mask'].cpu().numpy()[0]
                token_predictions = outputs.logits.argmax(-1).squeeze().cpu().numpy()
                
                # Map token predictions back to word predictions
                # In LayoutLMv3, not all tokens correspond to words due to tokenization
                # We'll use a simple approach mapping the first subtoken of each word
                word_predictions = []
                token_idx = 1  # Start after [CLS] token
                
                # Debug attention mask
                print(f"Attention mask shape: {attention_mask.shape}")
                print(f"Attention mask contains {attention_mask.sum()} active tokens")
                
                # Create word-level predictions
                # This is a simplified approach - in practice you'd need more sophisticated token-to-word alignment
                for word_idx in range(min(len(words), 100)):  # Limit to 100 words for safety
                    if token_idx >= len(token_predictions):
                        break
                    word_predictions.append(token_predictions[token_idx])
                    
                    # Move to next word's first token (simplified approach)
                    # In reality, you'd need to track how many tokens each word splits into
                    token_idx += 1
                    while token_idx < len(attention_mask) and attention_mask[token_idx] == 1:
                        # If we've reached a special token or end of attention mask, break
                        if token_idx >= len(token_predictions) or token_predictions[token_idx] == 0:
                            break
                        token_idx += 1
                
                # Pad predictions if needed
                if len(word_predictions) < len(words):
                    word_predictions.extend([0] * (len(words) - len(word_predictions)))
                
                predictions = np.array(word_predictions[:len(words)])
            
            # Debug predictions
            print(f"Final word predictions shape: {predictions.shape}")
            print(f"Sample predictions: {predictions[:10]}")
            
            # Extract structured information from predictions
            extracted_data = self._convert_predictions_to_data(words, predictions)
            extracted_data.raw_text = text
            
            # Apply selective rule-based extraction for missing fields
            extracted_data = self._selective_fallback_extraction(extracted_data, text)
            
            # Post-process and validate
            extracted_data = self.postprocess_extraction(extracted_data)
            
            return extracted_data
        except Exception as e:
            logging.error(f"Error extracting information: {str(e)}")
            # Fall back to rule-based extraction if ML processing fails
            return self._fallback_extraction(text if 'text' in locals() else "")
    
    def _convert_predictions_to_data(self, words, predictions):
        """
        Convert model predictions to structured data.
        
        Args:
            words: List of words from the document
            predictions: Model predictions for each word
            
        Returns:
            InvoiceData object with extracted information
        """
        data = InvoiceData()
        confidence_scores = {field: 0.0 for field in self.id2label.values() if field != "O"}
        print(confidence_scores)
        
        # Initialize fields
        fields = {
            'invoice_num': [],
            'date': [],
            'due_date': [],
            'total': [],
            'issuer': [],
            'recipient': [],
            'other': []
        }
        
        # Log predictions and words for debugging
        print(f"Words: {words[:20]}...")  # Print first 20 words 
        print(f"Predictions shape: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}")
        print(f"Unique prediction values: {np.unique(predictions)}")
        
        # Group words by their predicted labels
        prev_label = "O"
        current_text = ""
        
        for i, (word, pred_id) in enumerate(zip(words, predictions)):
            # Ensure pred_id is valid (within the range of id2label keys)
            if pred_id not in self.id2label:
                print(f"Warning: Invalid prediction ID {pred_id} at position {i}")
                pred_id = 0  # Default to "O"
                
            current_label = self.id2label[pred_id]
            
            if current_label != "O":
                if current_label != prev_label and prev_label != "O":
                    # Save the previous entity
                    fields[prev_label].append(current_text.strip())
                    current_text = ""
                
                # Add current word to the entity text
                current_text += " " + word
                prev_label = current_label
            else:
                if prev_label != "O" and current_text:
                    # Save the previous entity when we reach a non-entity token
                    fields[prev_label].append(current_text.strip())
                    current_text = ""
                prev_label = "O"
        
        # Don't forget to add the last entity if the document ends with one
        if prev_label != "O" and current_text:
            fields[prev_label].append(current_text.strip())
        
        # If no entities were found using the model, attempt to extract using regex patterns
        if all(len(field_values) == 0 for field_values in fields.values()):
            print("No entities found by model, attempting rule-based extraction")
            # Extract invoice number using regex
            invoice_match = re.search(r'Invoice\s+Number\s+[#]?([A-Za-z0-9-]+)', ' '.join(words))
            if invoice_match:
                fields['invoice_num'].append(invoice_match.group(1))
                confidence_scores['invoice_num'] = 0.6
            
            # Extract date
            date_match = re.search(r'Issue\s+Date\s+(\d{1,2}/\d{1,2}/\d{4})', ' '.join(words))
            if date_match:
                fields['date'].append(date_match.group(1))
                confidence_scores['date'] = 0.6
            
            # Extract due date
            due_date_match = re.search(r'Due\s+Date\s+(\d{1,2}/\d{1,2}/\d{4})', ' '.join(words))
            if due_date_match:
                fields['due_date'].append(due_date_match.group(1))
                confidence_scores['due_date'] = 0.6
            
            # Extract total amount
            total_match = re.search(r'TOTAL\s+\$(\d+\.\d+)', ' '.join(words))
            if total_match:
                fields['total'].append(total_match.group(1))
                confidence_scores['total'] = 0.6
            
            # Extract issuer name
            # Simplified approach - first few words might be the issuer
            for i in range(min(5, len(words))):
                if len(words[i]) > 3 and words[i].isalpha():
                    fields['issuer'].append(words[i])
                    confidence_scores['issuer'] = 0.4
                    break
            
            # Extract recipient name
            recipient_match = re.search(r'Bill\s+To:\s+([A-Za-z\s]+)', ' '.join(words))
            if recipient_match:
                fields['recipient'].append(recipient_match.group(1))
                confidence_scores['recipient'] = 0.6
        
        # Print extracted fields for debugging
        print("Extracted fields:")
        for field_name, field_values in fields.items():
            print(f"{field_name}: {field_values}")
        
        # Set InvoiceData fields
        if fields['invoice_num']:
            data.invoice_number = fields['invoice_num'][0]
            confidence_scores['invoice_num'] = max(0.8, confidence_scores['invoice_num'])  # Update confidence
        
        if fields['date']:
            date_str = fields['date'][0]
            normalized_date = normalize_date(date_str)
            if normalized_date:
                data.date = datetime.strptime(normalized_date, '%Y-%m-%d')
            confidence_scores['date'] = max(0.8, confidence_scores['date'])
        
        if fields['due_date']:
            due_date_str = fields['due_date'][0]
            normalized_due_date = normalize_date(due_date_str)
            if normalized_due_date:
                data.due_date = datetime.strptime(normalized_due_date, '%Y-%m-%d')
            confidence_scores['due_date'] = max(0.8, confidence_scores['due_date'])
        
        if fields['total']:
            total_str = fields['total'][0]
            amount = extract_amount(total_str)
            if amount is not None:
                data.total_amount = amount
            confidence_scores['total'] = max(0.8, confidence_scores['total'])
        
        if fields['issuer']:
            data.issuer_name = fields['issuer'][0]
            confidence_scores['issuer'] = max(0.8, confidence_scores['issuer'])
        
        if fields['recipient']:
            data.recipient_name = fields['recipient'][0]
            confidence_scores['recipient'] = max(0.8, confidence_scores['recipient'])
        
        # Set confidence scores
        data.confidence_scores = confidence_scores
        
        return data
    
    def _fallback_extraction(self, text):
        """
        Fallback to rule-based extraction if ML extraction fails.
        
        Args:
            text: Raw text extracted from the invoice
            
        Returns:
            InvoiceData object with extracted information
        """
        print("Using fallback extraction method")
        data = InvoiceData()
        data.raw_text = text
        
        # Initialize confidence scores
        confidence_scores = {
            'invoice_num': 0.0,
            'date': 0.0, 
            'due_date': 0.0,
            'total': 0.0,
            'issuer': 0.0,
            'recipient': 0.0,
            'other': 0.0
        }
        
        # Extract all fields using functions from fallback.py
        fields = extract_all_fields(text)
        
        # Set invoice fields and confidence scores
        if fields['invoice_number']:
            data.invoice_number = fields['invoice_number']
            confidence_scores['invoice_num'] = 0.7
        
        if fields['date']:
            data.date = fields['date']
            confidence_scores['date'] = 0.7
        
        if fields['due_date']:
            data.due_date = fields['due_date']
            confidence_scores['due_date'] = 0.7
        
        if fields['total_amount'] is not None:
            data.total_amount = fields['total_amount']
            confidence_scores['total'] = 0.7
        
        if fields['issuer_name']:
            data.issuer_name = fields['issuer_name']
            confidence_scores['issuer'] = 0.5
        
        if fields['recipient_name']:
            data.recipient_name = fields['recipient_name']
            confidence_scores['recipient'] = 0.6
        
        # Set confidence scores
        data.confidence_scores = confidence_scores
        
        # Add metadata
        data.metadata = {
            'extraction_method': 'rule_based_fallback',
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        return data
    
    def validate_extraction(self, extracted_data):
        """
        Validate the extracted data.
        
        Args:
            extracted_data: InvoiceData object to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Basic validation - require at least two fields to be present
        print(extracted_data)
        required_fields = ['invoice_number', 'total_amount', 'date', 'issuer_name', 'recipient_name']
        present_fields = sum(1 for field in required_fields if getattr(extracted_data, field) is not None)
        print(present_fields)
        
        if present_fields < 2:
            return False
        
        # Validate date formats if present
        if extracted_data.date and not isinstance(extracted_data.date, datetime):
            return False
        
        if extracted_data.due_date and not isinstance(extracted_data.due_date, datetime):
            return False
        
        # Validate amount format
        if extracted_data.total_amount is not None:
            try:
                float(extracted_data.total_amount)
            except (ValueError, TypeError):
                return False
        
        return True
    
    def train(self, train_data_dir, annotations_file, val_data_dir=None, val_annotations_file=None, 
              batch_size=8, num_epochs=5, learning_rate=5e-5, save_path='model.pth'):
        """
        Train the extraction model.
        
        Args:
            train_data_dir: Directory with training images
            annotations_file: Path to training annotations
            val_data_dir: Directory with validation images
            val_annotations_file: Path to validation annotations
            batch_size: Training batch size
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            save_path: Path to save the trained model
            
        Returns:
            dict: Training metrics
        """
        # Set model to training mode
        self.model.train()
        
        # Create datasets
        train_dataset = InvoiceDataset(
            train_data_dir, 
            annotations_file, 
            transform=self.transform,
            processor=self.processor
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Create validation dataloader if validation data is provided
        val_loader = None
        if val_data_dir and val_annotations_file:
            val_dataset = InvoiceDataset(
                val_data_dir, 
                val_annotations_file, 
                transform=self.transform,
                processor=self.processor
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Setup optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
        
        # Training loop
        best_loss = float('inf')
        metrics = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
                
                # Update statistics
                train_loss += loss.item()
            
            # Calculate average training loss
            avg_train_loss = train_loss / len(train_loader)
            metrics['train_loss'].append(avg_train_loss)
            
            # Validation phase
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch in val_loader:
                        # Move batch to device
                        batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                        
                        # Forward pass
                        outputs = self.model(**batch)
                        loss = outputs.loss
                        
                        # Update statistics
                        val_loss += loss.item()
                
                # Calculate average validation loss
                avg_val_loss = val_loss / len(val_loader)
                metrics['val_loss'].append(avg_val_loss)
                
                # Update scheduler
                scheduler.step(avg_val_loss)                


                # Save best model
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    self.model.save_pretrained(save_path)
                
                logging.info(f"Epoch {epoch+1}/{num_epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
            else:
                logging.info(f"Epoch {epoch+1}/{num_epochs}: train_loss={avg_train_loss:.4f}")
        
        # Save final model if no validation data
        if not val_loader:
            self.model.save_pretrained(save_path)
        
        # Set back to evaluation mode
        self.model.eval()
        
        return metrics
    
    def preprocess_text(self, text):
        """
        Preprocess extracted text.
        
        Args:
            text: Raw text from OCR
            
        Returns:
            Preprocessed text
        """
        # Clean text
        text = clean_text(text)
        
        return text
    
    def postprocess_extraction(self, extracted_data):
        """
        Post-process extracted data.
        
        Args:
            extracted_data: InvoiceData object with extracted information
            
        Returns:
            Post-processed InvoiceData object
        """
        # Validate extraction
        if not self.validate_extraction(extracted_data):
            logging.warning("Extracted data validation failed")
        
        # Add metadata
        if not extracted_data.metadata:
            extracted_data.metadata = {}
        extracted_data.metadata['extraction_method'] = 'ml_layoutlm'
        extracted_data.metadata['extraction_timestamp'] = datetime.now().isoformat()
        
        return extracted_data
    
    def _selective_fallback_extraction(self, extracted_data, text):
        """
        Selectively apply rule-based extraction for fields that the model failed to predict.
        
        Args:
            extracted_data: InvoiceData object with fields extracted by the model
            text: Raw text from the invoice
            
        Returns:
            InvoiceData object with missing fields filled by rule-based extraction
        """
        print("Using selective fallback extraction for missing fields")
        
        # Check and fill missing fields with rule-based methods
        confidence_scores = extracted_data.confidence_scores
        
        # Extract invoice number if missing
        if extracted_data.invoice_number is None:
            extracted_data.invoice_number = extract_invoice_number(text)
            if extracted_data.invoice_number:
                confidence_scores['invoice_num'] = 0.7
                print(f"Rule-based extraction found invoice number: {extracted_data.invoice_number}")
        
        # Extract date if missing
        if extracted_data.date is None:
            extracted_data.date = extract_issue_date(text)
            if extracted_data.date:
                confidence_scores['date'] = 0.7
                print(f"Rule-based extraction found date: {extracted_data.date}")
        
        # Extract due date if missing
        if extracted_data.due_date is None:
            extracted_data.due_date = extract_due_date(text)
            if extracted_data.due_date:
                confidence_scores['due_date'] = 0.7
                print(f"Rule-based extraction found due date: {extracted_data.due_date}")
        
        # Extract total amount if missing
        if extracted_data.total_amount is None:
            extracted_data.total_amount = extract_total_amount(text)
            if extracted_data.total_amount is not None:
                confidence_scores['total'] = 0.7
                print(f"Rule-based extraction found total amount: {extracted_data.total_amount}")
        
        # Extract issuer name if missing
        if extracted_data.issuer_name is None:
            extracted_data.issuer_name = extract_issuer_name(text)
            if extracted_data.issuer_name:
                confidence_scores['issuer'] = 0.5
                print(f"Rule-based extraction found issuer name: {extracted_data.issuer_name}")
        
        # Extract recipient name if missing
        if extracted_data.recipient_name is None:
            extracted_data.recipient_name = extract_recipient_name(text)
            if extracted_data.recipient_name:
                confidence_scores['recipient'] = 0.6
                print(f"Rule-based extraction found recipient name: {extracted_data.recipient_name}")
        
        # Update the metadata to indicate selective extraction
        if not extracted_data.metadata:
            extracted_data.metadata = {}
        extracted_data.metadata['extraction_method'] = 'hybrid_ml_rule_based'
        extracted_data.metadata['extraction_timestamp'] = datetime.now().isoformat()
        
        # Update confidence scores
        extracted_data.confidence_scores = confidence_scores
        
        return extracted_data