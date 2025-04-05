import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm
import joblib
import os
import time
import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from torch.utils.data import DataLoader

# Import the custom logger from utils
from classification.utils.logger import Logger, get_standard_logger

logger = get_standard_logger("resnet_classifier")

class ResNetClassifier:
    """
    A classifier using a pre-trained ResNet model for document classification.
    """
    
    def __init__(
        self, 
        num_classes: int,
        trained_model_name: str = "resnet18",
        pretrained: bool = True,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        device: Optional[str] = None,
        num_epochs: int = 50,
        dropout_rate: float = 0.7,
        num_workers: int = 4,
        batch_size: int = 64
    ):
        """
        Initialize the ResNet classifier.
        
        Args:
            num_classes: Number of classes to classify
            trained_model_name: Name of the ResNet model to use (e.g., "resnet18", "resnet50")
            pretrained: Whether to use pre-trained weights
            learning_rate: Learning rate for the optimizer
            weight_decay: Weight decay for regularization
            device: Device to use for training ('cuda' or 'cpu')
            num_epochs: Number of training epochs
            dropout_rate: Dropout rate for regularization
            num_workers: Number of workers for parallel data processing
            batch_size: Batch size for training
        """
        self.num_classes = num_classes
        self.trained_model_name = trained_model_name
        self.pretrained = pretrained
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate
        self.num_workers = num_workers
        self.batch_size = batch_size
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Initialize model
        self.model = self._create_model()
        
        # Define optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Define scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=2, 
            verbose=True
        )
        
        # Define loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def _create_model(self):
        """
        Create and configure the ResNet model.
        
        Returns:
            Configured ResNet model
        """
        # Get the model class based on the model name
        if self.trained_model_name == "resnet18":
            model_class = models.resnet18
        elif self.trained_model_name == "resnet34":
            model_class = models.resnet34
        elif self.trained_model_name == "resnet50":
            model_class = models.resnet50
        elif self.trained_model_name == "resnet101":
            model_class = models.resnet101
        else:
            raise ValueError(f"Unsupported model: {self.trained_model_name}")
        
        # Create the model with pre-trained weights if specified
        if self.pretrained:
            model = model_class(weights=models.ResNet18_Weights.DEFAULT)
        else:
            model = model_class(weights=None)
        
        # Modify the final fully connected layer for our classification task
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )
        
        # Move model to device
        model = model.to(self.device)
        
        return model
        
    def preprocess_image(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Preprocess a single image for inference.
        
        Args:
            image: Image as PIL Image, PyTorch tensor, or numpy array
            
        Returns:
            Preprocessed image as a PyTorch tensor
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Convert to PIL Image if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:  # Batch of images
                image = image[0]
            img_np = image.permute(1, 2, 0).cpu().numpy()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
            image = Image.fromarray(img_np)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Apply transformations
        return transform(image)
    
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
            tb_logger: TensorBoard logger instance for metrics visualization
            **kwargs: Additional training parameters
            
        Returns:
            Validation accuracy or training accuracy
        """
        # Enable automatic mixed precision for faster training
        scaler = torch.cuda.amp.GradScaler()
        
        # Training loop
        best_val_accuracy = 0.0
        patience = 5
        patience_counter = 0
        early_stopping = False
        
        # Move model to GPU if available
        self.model = self.model.to(self.device)
        step = 0
        
        for epoch in range(self.num_epochs):
            if early_stopping:
                break
                
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Process training data with progress bar
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for batch_idx, (images, labels) in enumerate(pbar):
                # Move data to device
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                # Forward pass with automatic mixed precision
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize with gradient scaling
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step with gradient scaling
                scaler.step(self.optimizer)
                scaler.update()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
                tb_logger.log_metrics({
                    'train/loss': running_loss / (batch_idx + 1),
                    'train/accuracy': 100. * correct / total,
                }, step=step)
                step += 1
            # Calculate epoch statistics
            epoch_loss = running_loss / total if total > 0 else 0
            epoch_accuracy = correct / total if total > 0 else 0
            
            # Log metrics to standard logger
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
            
            # Log metrics to TensorBoard if available
            if tb_logger:
                tb_logger.log_metrics({
                    'train/epoch_loss': epoch_loss,
                    'train/epoch_accuracy': epoch_accuracy,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr']
                }, step=epoch)
            
            # Validation
            val_accuracy = None
            if val_loader is not None:
                val_accuracy = self.evaluate(val_loader, tb_logger, epoch)
                
                # Log validation metrics to standard logger
                logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
                
                # Log validation metrics to TensorBoard if available
                if tb_logger:
                    tb_logger.log_metrics({
                        'val/accuracy': val_accuracy
                    }, step=epoch)
                
                # Learning rate scheduling
                self.scheduler.step(val_accuracy)
                
                # Early stopping check
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                    
                    # Save best model
                    model_path = f"models/{self.trained_model_name}_best.pth"
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),  # Save scaler state
                        'val_accuracy': val_accuracy,
                        'train_accuracy': epoch_accuracy,
                    }, model_path)
                    logger.info(f"Saved best model with validation accuracy: {val_accuracy:.4f} to {model_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        early_stopping = True
                        logger.info(f"Early stopping triggered after {epoch+1} epochs")
        
        return best_val_accuracy if val_loader is not None else epoch_accuracy
    
    def inference(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> int:
        """
        Run inference on a single image.
        
        Args:
            image: Image to classify
            
        Returns:
            Predicted class label
        """
        self.model.eval()
        with torch.no_grad():
            # Preprocess image
            if not isinstance(image, torch.Tensor):
                image = self.preprocess_image(image)
            else:
                image = image.to(self.device)
                
            # Add batch dimension if needed
            if image.dim() == 3:
                image = image.unsqueeze(0)
                
            # Forward pass
            outputs = self.model(image)
            _, predicted = torch.max(outputs.data, 1)
            
            return predicted.item()
    
    def predict_proba(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Get class probabilities for an image.
        
        Args:
            image: Image to classify
            
        Returns:
            Class probabilities as a numpy array
        """
        self.model.eval()
        with torch.no_grad():
            # Preprocess image
            if not isinstance(image, torch.Tensor):
                image = self.preprocess_image(image)
            else:
                image = image.to(self.device)
                
            # Add batch dimension if needed
            if image.dim() == 3:
                image = image.unsqueeze(0)
                
            # Forward pass
            outputs = self.model(image)
            probabilities = F.softmax(outputs, dim=1)
            
            return probabilities.cpu().numpy()[0]
    
    def evaluate(self, val_loader, tb_logger=None, step=None):
        """
        Evaluate the model on validation set.
        
        Args:
            val_loader: Validation DataLoader
            tb_logger: TensorBoard logger instance for metrics visualization
            step: Current step for logging
            
        Returns:
            Validation accuracy
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            for images, labels in tqdm(val_loader, desc="Validation"):
                # Move data to device
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Collect predictions and labels for confusion matrix if needed
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate accuracy
        accuracy = correct / total if total > 0 else 0
        
        # Log metrics to standard logger
        logger.info(f"Validation Loss: {val_loss / total if total > 0 else 0:.4f}, Accuracy: {accuracy:.4f}")
        
        # Log metrics to TensorBoard if available
        if tb_logger:
            tb_logger.log_metrics({
                'val/loss': val_loss / total if total > 0 else 0,
                'val/accuracy': accuracy
            }, step=step)
        
        
        return accuracy
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_classes': self.num_classes,
            'trained_model_name': self.trained_model_name,
            'pretrained': self.pretrained,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'num_epochs': self.num_epochs,
            'dropout_rate': self.dropout_rate,
            'num_workers': self.num_workers,
            'batch_size': self.batch_size
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        # Load model state
        checkpoint = torch.load(path, map_location=self.device)
        
        # Update model parameters
        self.num_classes = checkpoint['num_classes']
        self.trained_model_name = checkpoint['trained_model_name']
        self.pretrained = checkpoint['pretrained']
        self.learning_rate = checkpoint['learning_rate']
        self.weight_decay = checkpoint['weight_decay']
        self.num_epochs = checkpoint['num_epochs']
        self.dropout_rate = checkpoint['dropout_rate']
        
        # Load num_workers if available, otherwise use default
        self.num_workers = checkpoint.get('num_workers', 4)
        
        # Load batch_size if available, otherwise use default
        self.batch_size = checkpoint.get('batch_size', 64)
        
        # Recreate model with updated parameters
        self.model = self._create_model()
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scaler state
        scaler = torch.cuda.amp.GradScaler()
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Model loaded from {path}") 