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
        device: Optional[str] = None
    ):
        """
        Initialize the HOG classifier.
        
        Args:
            num_classes: Number of classes to classify
            classifier: Type of classifier to use ("logistic_regression" or "svm")
            hog_params: Parameters for HOG feature extraction
            classifier_params: Parameters for the classifier
            device: Device to use for computation ('cuda' or 'cpu')
        """
        self.num_classes = num_classes
        self.classifier_type = classifier
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
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
        
    def preprocess_image(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Preprocess an image for HOG feature extraction.
        
        Args:
            image: Image as PIL Image, PyTorch tensor, or numpy array
            
        Returns:
            Preprocessed image as a numpy array
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
            
        # Convert to numpy array for HOG
        img_np = image.permute(1, 2, 0).cpu().numpy()
        
        # Convert to grayscale if needed
        if img_np.shape[2] == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
        # Ensure the image is in the correct format for HOG
        img_np = (img_np * 255).astype(np.uint8)
        
        return img_np
    
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
            tb_logger: Logger instance for TensorBoard logging
            **kwargs: Additional training parameters
            
        Returns:
            Validation accuracy or training accuracy
        """
        # Extract features from training data
        X_train = []
        y_train = []
        
        # Use standard logger for progress information
        logger.info("Processing training data")
        for images, labels in tqdm(train_dataset, desc="Processing training data"):
            # Process each image in the batch
            for img, label in zip(images, labels):
                # Preprocess image
                img_np = self.preprocess_image(img)
                
                # Extract HOG features
                features = self.extract_hog_features(img_np)
                
                # Store features and label
                X_train.append(features)
                
                # Handle the label
                if isinstance(label, torch.Tensor):
                    if label.numel() == 1:  # Single element tensor
                        label = label.item()
                    else:
                        label = label[0].item()
                
                y_train.append(label)
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Scale features
        X_train = self.model.steps[0][1].fit_transform(X_train)
        
        # Train classifier
        logger.info(f"Training {self.classifier_type} classifier")
        self.model.steps[1][1].fit(X_train, y_train)
        
        # Evaluate on training set
        y_pred = self.model.predict(X_train)
        train_accuracy = np.mean(y_pred == y_train)
        
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        
        # Log metrics if TensorBoard logger provided
        tb_logger.log_metrics({
            'train/accuracy': train_accuracy
        })
        
        # Evaluate on validation set if provided
        if val_dataset is not None:
            return self.evaluate(val_dataset, logger)
        
        return train_accuracy
    
    def inference(self, image: Union[Image.Image, torch.Tensor, np.ndarray], logger=None) -> int:
        """
        Run inference on a single image.
        
        Args:
            image: Image to classify
            logger: Optional logger for logging
            
        Returns:
            Predicted class label
        """
        # Preprocess image
        img_np = self.preprocess_image(image)
        
        # Extract HOG features
        features = self.extract_hog_features(img_np)
        
        # Scale features
        features = self.model.steps[0][1].transform(features.reshape(1, -1))
        
        # Predict class
        prediction = self.model.predict(features)[0]
        
        # Log prediction if logger provided
        if logger:
            logger.info(f"Predicted class: {prediction}")
            
        return prediction
    
    def predict_proba(self, image: Union[Image.Image, torch.Tensor, np.ndarray], logger=None) -> np.ndarray:
        """
        Get class probabilities for an image.
        
        Args:
            image: Image to classify
            logger: Optional logger for logging
            
        Returns:
            Array of class probabilities
        """
        # Preprocess image
        img_np = self.preprocess_image(image)
        
        # Extract HOG features
        features = self.extract_hog_features(img_np)
        
        # Scale features
        features = self.model.steps[0][1].transform(features.reshape(1, -1))
        
        # Get probabilities
        probabilities = self.model.predict_proba(features)[0]
        
        # Log probabilities if logger provided
        if logger:
            logger.info(f"Class probabilities: {probabilities}")
            
        return probabilities
    
    def evaluate(self, val_dataset, tb_logger=None, step=None):
        """
        Evaluate the model on a validation dataset.
        
        Args:
            val_dataset: Validation dataset
            logger: Logger instance for TensorBoard logging
            step: Step number for logging
            
        Returns:
            Validation accuracy
        """
        # Extract features from validation data
        X_val = []
        y_val = []
        
        for images, labels in tqdm(val_dataset, desc="Evaluating"):
            # Process each image in the batch
            for img, label in zip(images, labels):
                # Preprocess image
                img_np = self.preprocess_image(img)
                
                # Extract HOG features
                features = self.extract_hog_features(img_np)
                
                # Store features and label
                X_val.append(features)
                
                # Handle the label
                if isinstance(label, torch.Tensor):
                    if label.numel() == 1:  # Single element tensor
                        label = label.item()
                    else:
                        label = label[0].item()
                
                y_val.append(label)
        
        # Convert to numpy arrays
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        
        # Scale features
        X_val = self.model.steps[0][1].transform(X_val)
        
        # Predict classes
        y_pred = self.model.predict(X_val)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_val)
        
        # Log metrics if TensorBoard logger provided
        tb_logger.log_metrics({'val/accuracy': accuracy}, step=step)
        logger.info(f"Validation accuracy: {accuracy:.4f}")
        
        return accuracy
    
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
        
        if logger:
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
        
        if logger:
            logger.info(f"Model loaded from {path}")
