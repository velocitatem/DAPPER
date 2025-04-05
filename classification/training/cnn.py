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
            num_epochs: Number of training epochs
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
        train_dataset, 
        val_dataset=None, 
        logger=None,
        **kwargs
    ):
        """
        Train the model using dataset objects.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            logger: Logger instance for TensorBoard logging
            **kwargs: Additional training parameters
            
        Returns:
            Validation accuracy or training accuracy
        """
        # Training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Process training data
            for batch_idx, (images, labels) in enumerate(tqdm(train_dataset, desc=f"Epoch {epoch+1}/{self.num_epochs}")):
                # Process images in the batch
                processed_images = []
                processed_labels = []
                
                # Handle single image case (when batch_size=1)
                if not isinstance(images, (list, tuple)) and not isinstance(images, torch.Tensor):
                    images = [images]
                    labels = [labels]
                
                # Process each image in the batch
                for img, label in zip(images, labels):
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
                    self.optimizer.step()
                    
                    # Statistics
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            # Calculate epoch statistics
            epoch_loss = running_loss / total if total > 0 else 0
            epoch_accuracy = correct / total if total > 0 else 0
            
            # Log metrics if logger provided
            if logger:
                logger.log_metrics({
                    'train/loss': epoch_loss,
                    'train/accuracy': epoch_accuracy
                }, step=epoch)
            
            # Validation
            val_accuracy = None
            if val_dataset is not None:
                val_accuracy = self.evaluate(val_dataset, logger, epoch)
                
                # Log validation metrics if logger provided
                if logger:
                    logger.log_metrics({
                        'val/accuracy': val_accuracy
                    }, step=epoch)
        
        # Return the appropriate accuracy
        return val_accuracy if val_accuracy is not None else epoch_accuracy
    
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
    
    def evaluate(self, val_dataset, logger=None, step=None):
        """
        Evaluate the model on a validation dataset.
        
        Args:
            val_dataset: Validation dataset
            logger: Logger instance for TensorBoard logging
            step: Step number for logging
            
        Returns:
            Validation accuracy
        """
        self.model.eval()
        correct = 0
        total = 0
        
        # Process validation data
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(val_dataset, desc="Evaluating")):
                # Process images in the batch
                processed_images = []
                processed_labels = []
                
                # Handle single image case (when batch_size=1)
                if not isinstance(images, (list, tuple)) and not isinstance(images, torch.Tensor):
                    images = [images]
                    labels = [labels]
                
                # Process each image in the batch
                for img, label in zip(images, labels):
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
                
                # Stack images into a batch tensor
                if processed_images:
                    batch_images = torch.stack(processed_images).to(self.device)
                    batch_labels = torch.tensor(processed_labels, dtype=torch.long).to(self.device)
                    
                    # Forward pass
                    outputs = self.model(batch_images)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    # Statistics
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
        
        # Calculate accuracy
        accuracy = correct / total if total > 0 else 0
        
        # Log metrics if logger provided
        if logger and step is not None:
            logger.log_metrics({'val/accuracy': accuracy}, step=step)
        
        return accuracy
    
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
            "num_epochs": self.num_epochs
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
        self.num_epochs = model_info.get("num_epochs", 10)  # Default to 10 if not present 