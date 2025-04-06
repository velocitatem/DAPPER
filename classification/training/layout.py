##
# @file layout.py
# @package classification.training.layout
# @brief LayoutLMv3-based document classifier for document understanding and classification
#
# This module implements a document classifier using LayoutLMv3, a multimodal model
# that combines text and layout information for document understanding. It provides
# state-of-the-art performance for document classification tasks.
# source: https://huggingface.co/microsoft/layoutlmv3-base
#
# @author Statistical Learning Team
# @date 2025
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import (
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3Processor,
    get_scheduler
)
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm
import os
import time
import pytesseract
    
##
# @brief LayoutLMv3-based document classifier for document understanding and classification
#
# This class implements a document classifier using LayoutLMv3, a multimodal model
# that combines text and layout information for document understanding. It provides
# state-of-the-art performance for document classification tasks.
#
class LayoutLMv3Classifier:
    """
    A document classifier using LayoutLMv3 for document understanding and classification.
    """
    
    ##
    # @brief Constructor for LayoutLMv3Classifier class
    # @param num_classes Number of document classes
    # @param model_name Name or path of the pre-trained LayoutLMv3 model
    # @param learning_rate Learning rate for the optimizer
    # @param weight_decay Weight decay for regularization
    # @param device Device to use for training ('cuda' or 'cpu')
    # @param num_epochs Number of training epochs
    # @param apply_ocr Whether to apply OCR within the processor
    # @param max_length Maximum sequence length for tokenization
    # @param processor_name Name of the processor to use (defaults to model_name if None)
    #
    def __init__(
        self, 
        num_classes: int,
        model_name: str = "microsoft/layoutlmv3-base",
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        device: Optional[str] = None,
        num_epochs: int = 10,
        apply_ocr: bool = False,
        max_length: int = 512,
        processor_name: Optional[str] = None
    ):
        """
        Initialize the LayoutLMv3 classifier.
        
        Args:
            num_classes: Number of classes to classify
            model_name: Name or path of the pre-trained LayoutLMv3 model
            learning_rate: Learning rate for the optimizer
            weight_decay: Weight decay for regularization
            device: Device to use for training ('cuda' or 'cpu')
            num_epochs: Number of training epochs
            apply_ocr: Whether to apply OCR within the processor
            max_length: Maximum sequence length for tokenization
            processor_name: Name of the processor to use (defaults to model_name if None)
        """
        self.num_classes = num_classes
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.apply_ocr = apply_ocr
        self.max_length = max_length
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Initialize processor and model
        processor_name = processor_name or model_name
        self.processor = LayoutLMv3Processor.from_pretrained(processor_name, apply_ocr=apply_ocr)
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        ).to(self.device)
        
        # Define optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Define loss function
        self.criterion = nn.CrossEntropyLoss()
        
    ##
    # @brief Preprocess an image for LayoutLMv3
    # @param image Image as PIL Image or numpy array
    # @param words List of OCR words (optional, if not using internal OCR)
    # @param boxes List of bounding boxes for words (optional, if not using internal OCR)
    # @return Preprocessed input dictionary for LayoutLMv3
    #
    def preprocess_image(self, 
                          image: Union[Image.Image, np.ndarray], 
                          words: Optional[List[str]] = None,
                          boxes: Optional[List[List[int]]] = None) -> Dict[str, torch.Tensor]:
        """
        Preprocess an image for LayoutLMv3.
        
        Args:
            image: Image as PIL Image or numpy array
            words: List of OCR words (optional, if not using internal OCR)
            boxes: List of bounding boxes for words (optional, if not using internal OCR)
            
        Returns:
            Preprocessed input dictionary for LayoutLMv3
        """
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Run OCR if needed and not using internal OCR
        if not self.apply_ocr and words is None and boxes is None:
            ocr_df = pytesseract.image_to_data(image, output_type='data.frame')
            ocr_df = ocr_df.dropna().reset_index(drop=True)
            
            # Get words and normalized bounding boxes
            words = ocr_df['text'].tolist()
            
            # Convert bounding boxes to required format (x1, y1, x2, y2) and normalize
            width, height = image.size
            boxes = []
            for _, row in ocr_df.iterrows():
                x, y, w, h = row['left'], row['top'], row['width'], row['height']
                # Normalize coordinates to 0-1000 range
                x1 = int(1000 * x / width)
                y1 = int(1000 * y / height)
                x2 = int(1000 * (x + w) / width)
                y2 = int(1000 * (y + h) / height)
                boxes.append([x1, y1, x2, y2])
                
        # Process through LayoutLMv3Processor
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        return encoding
    
    ##
    # @brief Train the model using DataLoader objects
    # @param train_loader Training DataLoader
    # @param val_loader Validation DataLoader
    # @param tb_logger Optional logger for TensorBoard logging
    # @param **kwargs Additional training parameters
    # @return Validation accuracy or training accuracy
    #
    def train_model(
        self, 
        train_loader, 
        val_loader=None, 
        tb_logger=None,
        **kwargs
    ):
        """
        Train the model using DataLoader objects.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            tb_logger: Optional logger for TensorBoard logging
            **kwargs: Additional training parameters
            
        Returns:
            Validation accuracy or training accuracy
        """
        # Enable automatic mixed precision for faster training
        try:
            # Try the newer API format
            scaler = torch.amp.GradScaler('cuda')
        except (TypeError, ValueError):
            # Fall back to the older API if needed
            scaler = torch.amp.GradScaler()
        
        # Training loop
        best_val_accuracy = 0.0
        patience = 5
        patience_counter = 0
        early_stopping = False
        
        # Calculate total training steps for scheduler
        num_training_steps = self.num_epochs * len(train_loader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps
        )
        
        for epoch in range(self.num_epochs):
            if early_stopping:
                break
                
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Process training data with progress bar
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            
            for batch in pbar:
                try:
                    # Move batch to device if needed
                    if not all(k in batch for k in ['input_ids', 'attention_mask', 'bbox', 'pixel_values']):
                        # If batch only contains images and labels, process them
                        if 'pixel_values' in batch and 'labels' in batch:
                            images = batch['pixel_values'].to(self.device)
                            labels = batch['labels'].to(self.device)
                            # Process each image and create batch
                            batch_inputs = {
                                'input_ids': [],
                                'attention_mask': [],
                                'bbox': [],
                                'pixel_values': []
                            }
                            for img in images:
                                inputs = self.preprocess_image(img)
                                for k, v in inputs.items():
                                    batch_inputs[k].append(v)
                            # Stack tensors
                            for k, v in batch_inputs.items():
                                batch_inputs[k] = torch.stack(v)
                            # Add labels
                            if isinstance(labels, torch.Tensor):
                                batch_inputs['labels'] = labels.clone().detach()
                            else:
                                batch_inputs['labels'] = torch.tensor(labels, dtype=torch.long)
                            batch = batch_inputs
                    else:
                        # If batch is already properly formatted, just move to device
                        batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    
                    # Ensure all required keys are present
                    required_keys = ['input_ids', 'attention_mask', 'bbox', 'pixel_values', 'labels']
                    if not all(k in batch for k in required_keys):
                        missing_keys = [k for k in required_keys if k not in batch]
                        raise ValueError(f"Batch is missing required keys: {missing_keys}")
                    
                    # Zero the parameter gradients
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    # Forward pass with automatic mixed precision
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = outputs.loss
                    
                    # Backward pass and optimize with gradient scaling
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Optimizer step with gradient scaling
                    scaler.step(self.optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    
                    # Statistics
                    running_loss += loss.item()
                    logits = outputs.logits
                    predicted = torch.argmax(logits, dim=1)
                    total += batch['labels'].size(0)
                    correct += (predicted == batch['labels']).sum().item()
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': loss.item(),
                        'acc': 100. * correct / total if total > 0 else 0
                    })
                    
                    # Log to TensorBoard if available
                    if tb_logger:
                        batch_acc = (predicted == batch['labels']).sum().item() / batch['labels'].size(0)
                        tb_logger.log_metrics({
                            'train/batch_loss': loss.item(),
                            'train/batch_accuracy': 100. * batch_acc,
                        }, step=epoch * len(train_loader) + pbar.n)
                except Exception as e:
                    # Log the error but continue training
                    print(f"Error processing batch: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Calculate epoch statistics
            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = 100. * (correct / total if total > 0 else 0)
            
            print(f"Epoch {epoch+1} Summary: Loss: {epoch_loss:.4f}, Acc: {epoch_accuracy:.2f}%")
            
            # Log epoch metrics to TensorBoard if available
            if tb_logger:
                tb_logger.log_metrics({
                    'train/epoch_loss': epoch_loss,
                    'train/epoch_accuracy': epoch_accuracy,
                }, step=epoch+1)
            
            # Validation
            val_accuracy = None
            if val_loader is not None:
                val_accuracy, val_loss = self.evaluate(val_loader, tb_logger, epoch + 1)
                print(f"Validation: Loss: {val_loss:.4f}, Acc: {val_accuracy:.2f}%")
                
                # Early stopping check
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                    
                    # Save best model
                    model_path = f"models/layoutlmv3_best.pth"
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'val_accuracy': val_accuracy,
                        'train_accuracy': epoch_accuracy,
                    }, model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        early_stopping = True
                        print(f"Early stopping triggered after {epoch+1} epochs")
        
        return best_val_accuracy if val_loader is not None else epoch_accuracy
    
    ##
    # @brief Run inference on a single image
    # @param image Image to classify
    # @param words List of OCR words (optional)
    # @param boxes List of bounding boxes for words (optional)
    # @return Predicted class label
    #
    def inference(self, 
                  image: Union[Image.Image, np.ndarray],
                  words: Optional[List[str]] = None,
                  boxes: Optional[List[List[int]]] = None) -> int:
        """
        Run inference on a single image.
        
        Args:
            image: Image to classify
            words: List of OCR words (optional)
            boxes: List of bounding boxes for words (optional)
            
        Returns:
            Predicted class label
        """
        # Preprocess the image
        inputs = self.preprocess_image(image, words, boxes)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted = torch.argmax(logits, dim=1)
            
        return predicted.item()
    
    ##
    # @brief Get class probabilities for an image
    # @param image Image to classify
    # @param words List of OCR words (optional)
    # @param boxes List of bounding boxes for words (optional)
    # @return Array of class probabilities
    #
    def predict_proba(self, 
                      image: Union[Image.Image, np.ndarray],
                      words: Optional[List[str]] = None,
                      boxes: Optional[List[List[int]]] = None) -> np.ndarray:
        """
        Get class probabilities for an image.
        
        Args:
            image: Image to classify
            words: List of OCR words (optional)
            boxes: List of bounding boxes for words (optional)
            
        Returns:
            Array of class probabilities
        """
        # Preprocess the image
        inputs = self.preprocess_image(image, words, boxes)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)
            
        return probabilities.cpu().numpy()[0]
    
    ##
    # @brief Evaluate the model on validation set
    # @param val_loader Validation DataLoader
    # @param tb_logger TensorBoard logger instance for metrics visualization
    # @param step Current step for logging
    # @return Validation accuracy and loss
    #
    def evaluate(self, val_loader, tb_logger=None, step=None):
        """
        Evaluate the model on validation set.
        
        Args:
            val_loader: Validation DataLoader
            tb_logger: TensorBoard logger instance for metrics visualization
            step: Current step for logging
            
        Returns:
            Validation accuracy and loss
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device if needed
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Statistics
                val_loss += loss.item()
                logits = outputs.logits
                predicted = torch.argmax(logits, dim=1)
                total += batch['labels'].size(0)
                correct += (predicted == batch['labels']).sum().item()
        
        # Calculate accuracy and loss
        avg_loss = val_loss / len(val_loader)
        accuracy = 100. * (correct / total if total > 0 else 0)
        
        # Log validation epoch metrics to TensorBoard if available
        if tb_logger and step is not None:
            tb_logger.log_metrics({
                'val/loss': avg_loss,
                'val/accuracy': accuracy
            }, step=step)
        
        return accuracy, avg_loss
    
    ##
    # @brief Save the trained model and processor to disk
    # @param path Path to save the model
    #
    def save(self, path: str) -> None:
        """
        Save the trained model and processor to disk.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        self.model.save_pretrained(path)
        
        # Save processor
        self.processor.save_pretrained(path)
        
        # Save additional information
        model_info = {
            "num_classes": self.num_classes,
            "model_name": self.model_name,
            "learning_rate": self.learning_rate,
            "apply_ocr": self.apply_ocr,
            "max_length": self.max_length
        }
        torch.save(model_info, os.path.join(path, "model_info.pt"))
    
    ##
    # @brief Load a trained model from disk
    # @param path Path to the saved model
    #
    def load(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        # Load model
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        
        # Load processor
        self.processor = LayoutLMv3Processor.from_pretrained(path)
        
        # Load additional information
        model_info = torch.load(os.path.join(path, "model_info.pt"))
        self.num_classes = model_info["num_classes"]
        self.model_name = model_info["model_name"]
        self.learning_rate = model_info["learning_rate"]
        self.apply_ocr = model_info["apply_ocr"]
        self.max_length = model_info["max_length"] 