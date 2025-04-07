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
    A simple CNN model for image classification.
    """
    
    def __init__(self, num_classes: int, input_channels: int = 3):
        """
        Initialize the CNN model.
        
        Args:
            num_classes: Number of classes to classify
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
        """
        super(CNNModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        # Convolutional layers with ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 128 * 28 * 28)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
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
        device: Optional[str] = None,
        num_epochs: int = 10
    ):
        """
        Initialize the CNN classifier.
        
        Args:
            num_classes: Number of classes to classify
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            learning_rate: Learning rate for the optimizer
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Initialize model
        self.model = CNNModel(num_classes, input_channels).to(self.device)
        
        # Define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Define loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def preprocess_image(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Preprocess an image for the CNN.
        
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
        if self.input_channels == 1:
            image = image.convert('L')
        else:
            image = image.convert('RGB')
            
        return self.transform(image)
    
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
        
        # Training loop
        best_val_accuracy = 0.0
        patience = 5
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
        # Apply transform if needed
        if not isinstance(image, torch.Tensor):
            image = self.preprocess_image(image)
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
            image = self.preprocess_image(image)
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
            "learning_rate": self.learning_rate
        }
        torch.save(model_info, path)
    
    def load(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        model_info = torch.load(path, map_location=self.device)
        self.model.load_state_dict(model_info["model_state_dict"])
        self.optimizer.load_state_dict(model_info["optimizer_state_dict"])
        self.num_classes = model_info["num_classes"]
        self.input_channels = model_info["input_channels"]
        self.learning_rate = model_info["learning_rate"] 