import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm
import joblib
import os
import time

class CNNModel(nn.Module):
    """
    A simple CNN model for image classification with regularization to prevent overfitting.
    """
    
    def __init__(self, num_classes: int, input_channels: int = 3, dropout_rate: float = 0.3):
        """
        Initialize the CNN model.
        
        Args:
            num_classes: Number of classes to classify
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            dropout_rate: Dropout rate for regularization
        """
        super(CNNModel, self).__init__()
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 96, kernel_size=3, padding=1)  # Reduced from 128 to 96
        self.bn3 = nn.BatchNorm2d(96)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate the flattened feature size
        self._to_linear = 96 * 28 * 28
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 256)  # Reduced from 512 to 256
        self.fc_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        # Convolutional layers with ReLU activation, batch norm, and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, self._to_linear)
        
        # Fully connected layers with dropout and batch norm
        x = F.relu(self.fc_bn(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class CNNClassifier:
    """
    A classifier using a Convolutional Neural Network (CNN) for image classification.
    """
    
    def __init__(
        self, 
        num_classes: int,
        input_channels: int = 3,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,  # Added weight decay
        dropout_rate: float = 0.3,    # Added configurable dropout
        device: Optional[str] = None,
        num_epochs: int = 10
    ):
        """
        Initialize the CNN classifier.
        
        Args:
            num_classes: Number of classes to classify
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            learning_rate: Learning rate for the optimizer
            weight_decay: L2 regularization parameter
            dropout_rate: Dropout rate for regularization
            device: Device to use for training ('cuda' or 'cpu')
            num_epochs: Number of training epochs
        """
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.num_epochs = num_epochs
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Initialize model
        self.model = CNNModel(num_classes, input_channels, dropout_rate).to(self.device)
        
        # Define optimizer with weight decay
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Define loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Define image transformations with augmentation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Less aggressive transforms for validation/testing
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def preprocess_image(self, image: Union[Image.Image, torch.Tensor, np.ndarray], for_training: bool = False) -> torch.Tensor:
        """
        Preprocess an image for the CNN.
        
        Args:
            image: Image as PIL Image, PyTorch tensor, or numpy array
            for_training: Whether to use training augmentations
            
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
        if self.input_channels == 1:
            image = image.convert('L')
        else:
            image = image.convert('RGB')
            
        # Use appropriate transformations based on mode
        if for_training:
            return self.transform(image)
        else:
            return self.test_transform(image)
    
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
            tb_logger: Logger instance for TensorBoard logging
            **kwargs: Additional training parameters
            
        Returns:
            Validation accuracy or training accuracy
        """
        # Enable automatic mixed precision for faster training
        scaler = torch.cuda.amp.GradScaler()
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
        # Training loop
        best_val_accuracy = 0.0
        patience = 7  # Increased patience
        patience_counter = 0
        early_stopping = False
        
        # Move model to GPU if available
        self.model = self.model.to(self.device)
        
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
                
                # RE-ADD BATCH LOGGING
                if tb_logger:
                     batch_acc = (predicted == labels).sum().item() / labels.size(0)
                     tb_logger.log_metrics({
                         'train/batch_loss': loss.item(),
                         'train/batch_accuracy': 100. * batch_acc, 
                     }, step=epoch * len(train_loader) + batch_idx)
            
            # Calculate epoch statistics
            epoch_loss = running_loss / total if total > 0 else 0
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
            val_loss = None
            if val_loader is not None:
                val_accuracy, val_loss = self.evaluate(val_loader, tb_logger, epoch + 1)
                print(f"Validation: Loss: {val_loss:.4f}, Acc: {val_accuracy:.2f}%")
                
                # Update learning rate scheduler based on validation accuracy
                scheduler.step(val_accuracy)
                
                # Early stopping check
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                    
                    # Save best model
                    model_path = f"models/cnn_best.pth"
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
        
        # Load best model before returning
        if os.path.exists("models/cnn_best.pth"):
            checkpoint = torch.load("models/cnn_best.pth")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model with validation accuracy: {checkpoint['val_accuracy']:.2f}%")
            
        return best_val_accuracy if val_loader is not None else epoch_accuracy
    
    def inference(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> int:
        """
        Run inference on a single image.
        
        Args:
            image: Image to classify
            
        Returns:
            Predicted class label
        """
        # Apply transform if needed
        if not isinstance(image, torch.Tensor):
            image = self.preprocess_image(image, for_training=False)
        else:
            image = image.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(image.unsqueeze(0))
            _, predicted = torch.max(outputs.data, 1)
            
        return predicted.item()
    
    def predict_proba(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Get class probabilities for an image.
        
        Args:
            image: Image to classify
            
        Returns:
            Array of class probabilities
        """
        # Apply transform if needed
        if not isinstance(image, torch.Tensor):
            image = self.preprocess_image(image, for_training=False)
        else:
            image = image.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(image.unsqueeze(0))
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
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
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
        
        # Calculate accuracy and loss
        avg_loss = val_loss / total if total > 0 else 0
        accuracy = 100. * (correct / total if total > 0 else 0)
        
        # Log validation epoch metrics to TensorBoard if available
        if tb_logger and step is not None:
            tb_logger.log_metrics({
                'val/loss': avg_loss,
                'val/accuracy': accuracy
            }, step=step)
        
        return accuracy, avg_loss
    
    def save(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        model_info = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dropout_rate": self.dropout_rate
        }
        torch.save(model_info, path)
    
    def load(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        model_info = torch.load(path, map_location=self.device)
        
        # Handle backward compatibility
        if "dropout_rate" not in model_info:
            model_info["dropout_rate"] = 0.3
        if "weight_decay" not in model_info:
            model_info["weight_decay"] = 1e-4
            
        # Re-initialize the model with proper parameters
        self.model = CNNModel(
            model_info["num_classes"], 
            model_info["input_channels"],
            dropout_rate=model_info["dropout_rate"]
        ).to(self.device)
        
        self.model.load_state_dict(model_info["model_state_dict"])
        self.optimizer.load_state_dict(model_info["optimizer_state_dict"])
        self.num_classes = model_info["num_classes"]
        self.input_channels = model_info["input_channels"]
        self.learning_rate = model_info["learning_rate"]
        self.weight_decay = model_info["weight_decay"]
        self.dropout_rate = model_info["dropout_rate"] 