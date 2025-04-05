import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm
import os
import time
import logging

# Import the custom logger from utils
from classification.utils.logger import Logger, get_standard_logger

logger = get_standard_logger("lsnet_classifier")

class LSConvolution(nn.Module):
    """
    Large-Small (LS) Convolution module as described in the LSNet paper.
    
    This module combines large-kernel perception and small-kernel aggregation
    to efficiently capture a wide range of perceptual information and achieve
    precise feature aggregation.
    """
    
    def __init__(self, in_channels, out_channels, large_kernel_size=7, small_kernel_size=3, stride=1, padding=None):
        """
        Initialize the LS Convolution module.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            large_kernel_size: Size of the large kernel for perception (default: 7)
            small_kernel_size: Size of the small kernel for aggregation (default: 3)
            stride: Stride of the convolution (default: 1)
            padding: Padding for the convolution (default: None, will be calculated)
        """
        super(LSConvolution, self).__init__()
        
        # Calculate padding if not provided
        if padding is None:
            large_padding = large_kernel_size // 2
            small_padding = small_kernel_size // 2
        else:
            large_padding = padding
            small_padding = padding
        
        # Large kernel perception
        self.large_kernel_conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=large_kernel_size, 
            stride=stride, 
            padding=large_padding, 
            bias=False
        )
        
        # Small kernel aggregation
        self.small_kernel_conv = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=small_kernel_size, 
            stride=1, 
            padding=small_padding, 
            bias=False
        )
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Activation function
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Forward pass through the LS Convolution module.
        
        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width]
            
        Returns:
            Output tensor of shape [batch_size, out_channels, height, width]
        """
        # Large kernel perception
        x = self.large_kernel_conv(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        # Small kernel aggregation
        x = self.small_kernel_conv(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        return x

class LSBlock(nn.Module):
    """
    LS Block as described in the LSNet paper.
    
    This block consists of LS Convolution followed by a residual connection.
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Initialize the LS Block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride of the convolution (default: 1)
        """
        super(LSBlock, self).__init__()
        
        # LS Convolution
        self.conv = LSConvolution(in_channels, out_channels, stride=stride)
        
        # Residual connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        Forward pass through the LS Block.
        
        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width]
            
        Returns:
            Output tensor of shape [batch_size, out_channels, height, width]
        """
        # LS Convolution
        out = self.conv(x)
        
        # Residual connection
        out += self.shortcut(x)
        
        return out

class LSNet(nn.Module):
    """
    LSNet model as described in the paper.
    
    This model uses LS Convolution and LS Blocks to achieve efficient
    and effective feature extraction and classification.
    """
    
    def __init__(self, num_classes=1000, model_size='t'):
        """
        Initialize the LSNet model.
        
        Args:
            num_classes: Number of classes to classify (default: 1000 for ImageNet)
            model_size: Size of the model ('t' for tiny, 's' for small, 'b' for base)
        """
        super(LSNet, self).__init__()
        
        # Model configuration based on size
        if model_size == 't':
            # Tiny model configuration
            self.channels = [64, 128, 256, 512]
            self.num_blocks = [2, 2, 6, 2]
        elif model_size == 's':
            # Small model configuration
            self.channels = [64, 128, 256, 512]
            self.num_blocks = [3, 4, 8, 3]
        elif model_size == 'b':
            # Base model configuration
            self.channels = [64, 128, 256, 512]
            self.num_blocks = [4, 6, 12, 4]
        else:
            raise ValueError(f"Unsupported model size: {model_size}")
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Max pooling
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # LS Blocks
        self.layer1 = self._make_layer(self.channels[0], self.channels[0], self.num_blocks[0])
        self.layer2 = self._make_layer(self.channels[0], self.channels[1], self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.channels[1], self.channels[2], self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.channels[2], self.channels[3], self.num_blocks[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.fc = nn.Linear(self.channels[3], num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        """
        Create a layer of LS Blocks.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_blocks: Number of LS Blocks in the layer
            stride: Stride of the first LS Block (default: 1)
            
        Returns:
            Sequential module containing LS Blocks
        """
        layers = []
        # First block with specified stride
        layers.append(LSBlock(in_channels, out_channels, stride))
        # Remaining blocks with stride 1
        for _ in range(1, num_blocks):
            layers.append(LSBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the LSNet model.
        
        Args:
            x: Input tensor of shape [batch_size, 3, height, width]
            
        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        # Initial convolution
        x = self.conv1(x)
        
        # Max pooling
        x = self.maxpool(x)
        
        # LS Blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Classification head
        x = self.fc(x)
        
        return x

class LSNetClassifier:
    """
    A classifier using the LSNet model for image classification.
    """
    
    def __init__(
        self, 
        num_classes: int,
        model_size: str = 't',
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        device: Optional[str] = None,
        num_epochs: int = 100,
        batch_size: int = 64,
        num_workers: int = 4,
        pretrained: bool = False,
        freeze_backbone: bool = False
    ):
        """
        Initialize the LSNet classifier.
        
        Args:
            num_classes: Number of classes to classify
            model_size: Size of the LSNet model ('t' for tiny, 's' for small, 'b' for base)
            learning_rate: Learning rate for the optimizer
            weight_decay: Weight decay for regularization
            device: Device to use for training ('cuda' or 'cpu')
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            pretrained: Whether to use pre-trained weights
            freeze_backbone: Whether to freeze the backbone
        """
        self.num_classes = num_classes
        self.model_size = model_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Log device information
        logger.info(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
        # Initialize model
        self.model = LSNet(num_classes=num_classes, model_size=model_size)
        self.model.to(self.device)
        
        # Load pre-trained weights if requested
        if pretrained:
            self._load_pretrained_weights()
            
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
        
        # Define optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Define scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=num_epochs, 
            eta_min=learning_rate * 0.01
        )
        
        # Define loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_pretrained_weights(self):
        """
        Load pre-trained weights for the model.
        """
        try:
            # Check if pre-trained weights exist
            model_path = f"models/lsnet_{self.model_size}.pth"
            if os.path.exists(model_path):
                logger.info(f"Loading pre-trained weights from {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                logger.warning(f"Pre-trained weights not found at {model_path}")
        except Exception as e:
            logger.error(f"Error loading pre-trained weights: {e}")
    
    def _freeze_backbone(self):
        """
        Freeze the backbone of the model.
        """
        logger.info("Freezing backbone")
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze the classification head
        for param in self.model.fc.parameters():
            param.requires_grad = True
    
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
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            tb_logger: TensorBoard logger instance for metrics visualization
            **kwargs: Additional training parameters
            
        Returns:
            Validation accuracy or training accuracy
        """
        # Verify model is on the correct device
        if next(self.model.parameters()).device != self.device:
            logger.warning(f"Model is not on the expected device {self.device}. Moving it now.")
            self.model = self.model.to(self.device)
        
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
            for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")):
                # Apply transformations to images
                transformed_images = []
                for img in images:
                    transformed_images.append(self.transform(img))
                
                # Stack images into a batch
                images_tensor = torch.stack(transformed_images)
                
                # Move data to device
                images_tensor = images_tensor.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(images_tensor)
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Log batch progress
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{self.num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
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
                self.scheduler.step()
                
                # Early stopping check
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                    
                    # Save best model
                    model_path = f"models/lsnet_{self.model_size}_best.pth"
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
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                image = self.transform(image)
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
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                image = self.transform(image)
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
            val_loader: DataLoader for validation data
            tb_logger: TensorBoard logger instance for metrics visualization
            step: Current step for logging
            
        Returns:
            Validation accuracy
        """
        # Verify model is on the correct device
        if next(self.model.parameters()).device != self.device:
            logger.warning(f"Model is not on the expected device {self.device}. Moving it now.")
            self.model = self.model.to(self.device)
            
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                # Apply transformations to images
                transformed_images = []
                for img in images:
                    transformed_images.append(self.transform(img))
                
                # Stack images into a batch
                images_tensor = torch.stack(transformed_images)
                
                # Move data to device
                images_tensor = images_tensor.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = self.model(images_tensor)
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
    
    def save(self, path: str, logger=None) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
            logger: Optional logger for logging
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_classes': self.num_classes,
            'model_size': self.model_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'device': str(self.device)
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
        self.model_size = checkpoint['model_size']
        self.learning_rate = checkpoint['learning_rate']
        self.weight_decay = checkpoint['weight_decay']
        self.num_epochs = checkpoint['num_epochs']
        self.batch_size = checkpoint['batch_size']
        self.num_workers = checkpoint['num_workers']
        
        # Recreate model with updated parameters
        self.model = LSNet(num_classes=self.num_classes, model_size=self.model_size)
        
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