import numpy as np
from skimage.feature import hog
from skimage import color
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from PIL import Image
import joblib
from tqdm import tqdm
import logging
import torch
from typing import List, Union, Dict, Any, Optional, Tuple
import time
import os
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms
import cv2

# Import the custom logger from utils
from classification.utils.logger import Logger, get_standard_logger

logger = get_standard_logger("hog_classifier")

class HogClassifier:
    """
    A simple classifier using Histogram of Oriented Gradients (HOG) features 
    with either Logistic Regression or SVM for classification.
    """
    
    def __init__(
        self, 
        num_classes: int,
        classifier: str = "logistic_regression", 
        hog_params: Optional[Dict[str, Any]] = None,
        classifier_params: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        """
        Initialize the HOG classifier.
        
        Args:
            num_classes: Number of classes to classify
            classifier: Type of classifier to use ("logistic_regression" or "svm")
            hog_params: Parameters for HOG feature extraction
            classifier_params: Parameters for the classifier
            device: Device to use for computation ('cuda' or 'cpu')
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
        """
        self.num_classes = num_classes
        self.classifier_type = classifier
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Default HOG parameters
        self.hog_params = {
            "pixels_per_cell": (16, 16),
            "cells_per_block": (2, 2),
            "orientations": 9,
            "image_size": (128, 128)
        }
        
        # Update with user-provided params if given
        if hog_params:
            self.hog_params.update(hog_params)
            
        # Set up classifier
        if classifier_params is None:
            classifier_params = {}
            
        if classifier == "svm":
            clf = SVC(probability=True, **classifier_params)
        else:  # default to logistic regression
            clf = LogisticRegression(max_iter=1000, **classifier_params)
            
        self.model = make_pipeline(StandardScaler(), clf)
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # HOG parameters
        self.win_size = (64, 64)
        self.block_size = (16, 16)
        self.block_stride = (8, 8)
        self.cell_size = (8, 8)
        self.nbins = 9
        
        # Initialize HOG descriptor
        self.hog = cv2.HOGDescriptor(self.win_size, self.block_size, self.block_stride, self.cell_size, self.nbins)
        
        logger.info(f"Initialized HOG classifier with {num_classes} classes on device: {self.device}")
        if self.device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
    def preprocess_image(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Preprocess an image for HOG feature extraction.
        Resizes, converts to grayscale, and ensures uint8 format.
        
        Args:
            image: Image as PIL Image, PyTorch tensor, or numpy array
            
        Returns:
            Preprocessed image as a numpy array (uint8, grayscale)
        """
        # Convert to PIL Image if needed
        if isinstance(image, torch.Tensor):
            # Handle different tensor dimensions
            if image.dim() == 4:  # Batch of images: [batch, channels, height, width]
                image = image[0]  # Take first image
            
            # De-normalize if needed (assuming standard ImageNet normalization)
            # Inverse normalization: image = std * image + mean
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            
            # Check if tensor is on GPU and move mean/std there if necessary
            if image.is_cuda:
                mean = mean.to(image.device)
                std = std.to(image.device)
                
            img_tensor = image * std + mean
            img_tensor = torch.clamp(img_tensor, 0, 1) # Clamp values to [0, 1]

            # Convert to numpy and then to PIL Image
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            image = Image.fromarray(img_np)
        elif isinstance(image, np.ndarray):
            # Ensure uint8 if numpy array is input
            if image.dtype != np.uint8:
                 # Assuming float input is in [0, 1] range if not uint8
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8) # Direct cast if values are already 0-255
            image = Image.fromarray(image)
            
        # Resize the image (using PIL)
        # Use the image_size from hog_params for resizing
        resize_transform = transforms.Resize(self.hog_params["image_size"])
        image = resize_transform(image)
            
        # Convert PIL image to numpy array for cv2 operations
        img_np = np.array(image)
        
        # Convert to grayscale if needed (ensure it's uint8 before cvtColor)
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_np_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        elif len(img_np.shape) == 2: # Already grayscale
             img_np_gray = img_np
        else: # Handle unexpected shapes if necessary
             raise ValueError(f"Unexpected image shape: {img_np.shape}")
        
        # Ensure the final output is uint8
        if img_np_gray.dtype != np.uint8:
            # This case should ideally not happen if conversions above are correct
            # Add a safeguard conversion if needed
            if img_np_gray.max() <= 1.0 and img_np_gray.min() >=0.0:
                 img_np_gray = (img_np_gray * 255).astype(np.uint8)
            else:
                # Clamp and convert if values might be outside 0-255
                img_np_gray = np.clip(img_np_gray, 0, 255).astype(np.uint8)

        return img_np_gray
    
    def extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract HOG features from an image.
        
        Args:
            image: Preprocessed image as a numpy array
            
        Returns:
            HOG features as a numpy array
        """
        # Compute HOG features
        features = self.hog.compute(image)
        return features.flatten()
    
    def train_model(
        self,
        train_loader,
        val_loader=None,
        tb_logger=None,
        **kwargs
    ):
        """
        Train the HOG-based classifier using DataLoader objects.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            tb_logger: TensorBoard logger instance for metrics visualization
            **kwargs: Additional training parameters
            
        Returns:
            Validation accuracy or training accuracy
        """
        logger.info("Extracting HOG features from training data...")
        
        # Extract features and labels from training data
        X_train = []
        y_train = []
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Processing training data")):
            # Process each image in the batch
            for img, label in zip(images, labels):
                # Preprocess image and extract HOG features
                img_np = self.preprocess_image(img)
                features = self.extract_hog_features(img_np)
                
                # Store features and label
                X_train.append(features)
                y_train.append(label.item())
                
            # Log progress
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {(batch_idx + 1) * train_loader.batch_size} images")
                
            # Log to TensorBoard if available
            if tb_logger:
                tb_logger.log_metrics({
                    'train/processed_images': (batch_idx + 1) * train_loader.batch_size
                }, step=batch_idx)
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        logger.info(f"Training data shape: {X_train.shape}")
        
        # Train the model using scikit-learn's fit method
        logger.info(f"Training {self.classifier_type} classifier...")
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Calculate training accuracy
        train_predictions = self.model.predict(X_train)
        train_accuracy = np.mean(train_predictions == y_train)
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        
        # Log training metrics
        if tb_logger:
            tb_logger.log_metrics({
                'train/accuracy': train_accuracy,
                'train/time': training_time
            })
        
        # Evaluate on validation set if provided
        if val_loader is not None:
            val_accuracy = self.evaluate(val_loader, tb_logger)
            return val_accuracy
        
        return train_accuracy
    
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
        logger.info("Extracting HOG features from validation data...")
        
        # Extract features and labels from validation data
        X_val = []
        y_val = []
        
        for images, labels in tqdm(val_loader, desc="Processing validation data"):
            # Process each image in the batch
            for img, label in zip(images, labels):
                # Preprocess image and extract HOG features
                img_np = self.preprocess_image(img)
                features = self.extract_hog_features(img_np)
                
                # Store features and label
                X_val.append(features)
                y_val.append(label.item())
        
        # Convert to numpy arrays
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        
        logger.info(f"Validation data shape: {X_val.shape}")
        
        # Make predictions
        val_predictions = self.model.predict(X_val)
        val_accuracy = np.mean(val_predictions == y_val)
        
        # Calculate probabilities if the model supports it
        if hasattr(self.model, 'predict_proba'):
            val_probabilities = self.model.predict_proba(X_val)
            
            # Log detailed metrics if TensorBoard logger is available
            if tb_logger:
                # Create confusion matrix
                cm = confusion_matrix(y_val, val_predictions)
                
                # Log metrics
                tb_logger.log_metrics({
                    'val/accuracy': val_accuracy,
                    'val/confusion_matrix': cm
                }, step=step)
                
                # Log classification report
                report = classification_report(y_val, val_predictions)
                logger.info("\nClassification Report:\n" + report)
        
        logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        
        return val_accuracy
    
    def save(self, path: str, logger=None) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
            logger: Optional logger for logging
        """
        model_info = {
            "model": self.model,
            "hog_params": self.hog_params,
            "num_classes": self.num_classes,
            "classifier_type": self.classifier_type,
            "win_size": self.win_size,
            "block_size": self.block_size,
            "block_stride": self.block_stride,
            "cell_size": self.cell_size,
            "nbins": self.nbins
        }
        
        joblib.dump(model_info, path)
        
        if logger is not None:
            logger.info(f"Model saved to {path}")
    
    def load(self, path: str, logger=None) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
            logger: Optional logger for logging
        """
        model_info = joblib.load(path)
        self.model = model_info["model"]
        self.hog_params = model_info["hog_params"]
        self.num_classes = model_info["num_classes"]
        self.classifier_type = model_info["classifier_type"]
        
        # Load HOG parameters
        self.win_size = model_info["win_size"]
        self.block_size = model_info["block_size"]
        self.block_stride = model_info["block_stride"]
        self.cell_size = model_info["cell_size"]
        self.nbins = model_info["nbins"]
        
        # Reinitialize HOG descriptor
        self.hog = cv2.HOGDescriptor(self.win_size, self.block_size, self.block_stride, self.cell_size, self.nbins)
        
        if logger is not None:
            logger.info(f"Model loaded from {path}")
    