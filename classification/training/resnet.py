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
        num_workers: int = 4
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
        """
        self.num_classes = num_classes
        self.trained_model_name = trained_model_name
        self.pretrained = pretrained
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate
        self.num_workers = num_workers
        
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
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
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
        Preprocess an image for the ResNet model.
        
        Args:
            image: Image as PIL Image, PyTorch tensor, or numpy array
            
        Returns:
            Preprocessed image as a PyTorch tensor
        """
        # Convert to PIL Image if needed
        if isinstance(image, torch.Tensor):
            # Handle different tensor dimensions
            if image.dim() == 4:  # Batch of images: [batch, channels, height, width]
                image = image[0]  # Take first image
            
            # Convert to numpy and then to PIL Image
            img_np = image.permute(1, 2, 0).cpu().numpy()
            
            # Ensure the values are in the correct range for uint8
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
                
            image = Image.fromarray(img_np)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Apply transformations
        if not isinstance(image, torch.Tensor):
            image = self.transform(image)
            
        return image
    
    def _process_batch_parallel(self, batch_data):
        """
        Process a batch of images in parallel.
        
        Args:
            batch_data: List of (image, label) tuples
            
        Returns:
            Tuple of (processed_images, processed_labels)
        """
        processed_images = []
        processed_labels = []
        
        # Process each image in the batch
        for img, label in batch_data:
            # Apply transform if needed
            if not isinstance(img, torch.Tensor):
                img = self.preprocess_image(img)
            else:
                img = img.to(self.device)
            
            processed_images.append(img)
            
            # Handle the label
            if isinstance(label, torch.Tensor):
                if label.numel() == 1:  # Single element tensor
                    label = label.item()
                else:
                    label = label[0].item()
            
            processed_labels.append(label)
        
        return processed_images, processed_labels
    
    def train_model(
        self, 
        train_dataset, 
        val_dataset=None, 
        tb_logger=None,
        **kwargs
    ):
        """
        Train the model using dataset objects.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            logger: Standard logger instance for console and file logging
            tb_logger: TensorBoard logger instance for metrics visualization
            **kwargs: Additional training parameters
            
        Returns:
            Validation accuracy or training accuracy
        """
        # Use the global logger if none provided
            
        # Training loop
        best_val_accuracy = 0.0
        patience = 5
        patience_counter = 0
        early_stopping = False
        
        for epoch in range(self.num_epochs):
            if early_stopping:
                break
                
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Process training data
            for batch_idx, (images, labels) in enumerate(tqdm(train_dataset, desc=f"Epoch {epoch+1}/{self.num_epochs}")):
                # Process images in the batch using parallel processing
                batch_data = list(zip(images, labels))
                
                # Use ThreadPoolExecutor for parallel processing
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    # Split batch into chunks for parallel processing
                    chunk_size = max(1, len(batch_data) // self.num_workers)
                    chunks = [batch_data[i:i+chunk_size] for i in range(0, len(batch_data), chunk_size)]
                    
                    # Process chunks in parallel
                    futures = [executor.submit(self._process_batch_parallel, chunk) for chunk in chunks]
                    
                    # Collect results
                    processed_images = []
                    processed_labels = []
                    for future in futures:
                        chunk_images, chunk_labels = future.result()
                        processed_images.extend(chunk_images)
                        processed_labels.extend(chunk_labels)
                
                # Stack images into a batch tensor
                if processed_images:
                    batch_images = torch.stack(processed_images).to(self.device)
                    batch_labels = torch.tensor(processed_labels, dtype=torch.long).to(self.device)
                    
                    # Zero the parameter gradients
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(batch_images)
                    loss = self.criterion(outputs, batch_labels)
                    
                    # Backward pass and optimize
                    loss.backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    # Statistics
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            # Calculate epoch statistics
            epoch_loss = running_loss / total if total > 0 else 0
            epoch_accuracy = correct / total if total > 0 else 0
            
            # Log metrics to standard logger
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
            
            # Log metrics to TensorBoard if available
            if tb_logger:
                tb_logger.log_metrics({
                    'train/loss': epoch_loss,
                    'train/accuracy': epoch_accuracy
                }, step=epoch)
            
            # Validation
            val_accuracy = None
            if val_dataset is not None:
                val_accuracy = self.evaluate(val_dataset, logger, tb_logger, epoch)
                
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
                        'val_accuracy': val_accuracy,
                        'train_accuracy': epoch_accuracy,
                    }, model_path)
                    logger.info(f"Saved best model with validation accuracy: {val_accuracy:.4f} to {model_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        early_stopping = True
                        logger.info(f"Early stopping triggered after {epoch+1} epochs")
        
        return best_val_accuracy if val_dataset is not None else epoch_accuracy
    
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
    
    def evaluate(self, val_dataset,tb_logger=None, step=None):
        """
        Evaluate the model on validation set.
        
        Args:
            val_dataset: Validation dataset
            tb_logger: TensorBoard logger instance for metrics visualization
            step: Current step for logging
            
        Returns:
            Validation accuracy
        """
        # Use the global logger if none provided
            
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_dataset, desc="Validation"):
                # Process images in the batch using parallel processing
                batch_data = list(zip(images, labels))
                
                # Use ThreadPoolExecutor for parallel processing
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    # Split batch into chunks for parallel processing
                    chunk_size = max(1, len(batch_data) // self.num_workers)
                    chunks = [batch_data[i:i+chunk_size] for i in range(0, len(batch_data), chunk_size)]
                    
                    # Process chunks in parallel
                    futures = [executor.submit(self._process_batch_parallel, chunk) for chunk in chunks]
                    
                    # Collect results
                    processed_images = []
                    processed_labels = []
                    for future in futures:
                        chunk_images, chunk_labels = future.result()
                        processed_images.extend(chunk_images)
                        processed_labels.extend(chunk_labels)
                
                # Stack images into a batch tensor
                if processed_images:
                    batch_images = torch.stack(processed_images).to(self.device)
                    batch_labels = torch.tensor(processed_labels, dtype=torch.long).to(self.device)
                    
                    # Forward pass
                    outputs = self.model(batch_images)
                    loss = self.criterion(outputs, batch_labels)
                    
                    # Statistics
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
                    
                    # Collect predictions and labels for confusion matrix if needed
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())
        
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
            'num_workers': self.num_workers
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
        
        logger.info(f"Model loaded from {path}") 