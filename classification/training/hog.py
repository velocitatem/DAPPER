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
        classifier_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the HOG classifier.
        
        Args:
            num_classes: Number of classes to classify
            classifier: Type of classifier to use ("logistic_regression" or "svm")
            hog_params: Parameters for HOG feature extraction
            classifier_params: Parameters for the classifier
        """
        self.num_classes = num_classes
        
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
        
    def preprocess_image(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Extract HOG features from an image.
        
        Args:
            image: Image as PIL Image, PyTorch tensor, or numpy array
            
        Returns:
            HOG feature vector
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
            
        # Convert to grayscale and resize
        image_size = self.hog_params["image_size"]
        image = image.convert('L').resize(image_size)
        image_np = np.array(image)
        
        # Extract HOG features
        features = hog(
            image_np, 
            orientations=self.hog_params["orientations"],
            pixels_per_cell=self.hog_params["pixels_per_cell"], 
            cells_per_block=self.hog_params["cells_per_block"], 
            feature_vector=True
        )
        
        return features
    
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
        # Extract images and labels from training dataset
        train_images = []
        train_labels = []
        
        # Process training data
        for img, label in tqdm(train_dataset, desc="Processing training data"):
            # Apply transform if needed
            if not isinstance(img, torch.Tensor):
                img = self.transform(img)
                
            train_images.append(img)
            
            # Handle the label
            if isinstance(label, torch.Tensor):
                if label.numel() == 1:  # Single element tensor
                    train_labels.append(label.item())
                else:
                    train_labels.append(label[0].item())
            else:
                train_labels.append(label)
        
        # Extract features for training
        features = np.array([self.preprocess_image(img) for img in tqdm(train_images, desc="Extracting HOG features")])
        
        # Train the model
        self.model.fit(features, train_labels)
        
        # Calculate training accuracy
        train_preds = self.model.predict(features)
        train_accuracy = (train_preds == train_labels).mean()
        
        # Log metrics if logger provided
        if logger:
            logger.log_metrics({'train/accuracy': train_accuracy}, step=0)
        
        # Process validation data if provided
        val_accuracy = None
        if val_dataset is not None:
            val_images = []
            val_labels = []
            
            for img, label in tqdm(val_dataset, desc="Processing validation data"):
                # Apply transform if needed
                if not isinstance(img, torch.Tensor):
                    img = self.transform(img)
                    
                val_images.append(img)
                
                # Handle the label
                if isinstance(label, torch.Tensor):
                    if label.numel() == 1:  # Single element tensor
                        val_labels.append(label.item())
                    else:
                        val_labels.append(label[0].item())
                else:
                    val_labels.append(label)
            
            # Extract features for validation
            val_features = np.array([self.preprocess_image(img) for img in tqdm(val_images, desc="Extracting validation features")])
            
            # Calculate validation accuracy
            val_preds = self.model.predict(val_features)
            val_accuracy = (val_preds == val_labels).mean()
            
            # Log metrics if logger provided
            if logger:
                logger.log_metrics({'val/accuracy': val_accuracy}, step=0)
        
        # Return the appropriate accuracy
        return val_accuracy if val_accuracy is not None else train_accuracy
    
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
            image = self.transform(image)
            
        features = self.preprocess_image(image).reshape(1, -1)
        return self.model.predict(features)[0]
    
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
            image = self.transform(image)
            
        features = self.preprocess_image(image).reshape(1, -1)
        return self.model.predict_proba(features)[0]
    
    def evaluate(self, val_dataset, logger=None):
        """
        Evaluate the model on a validation dataset.
        
        Args:
            val_dataset: Validation dataset
            logger: Logger instance for TensorBoard logging
            
        Returns:
            Validation accuracy
        """
        val_images = []
        val_labels = []
        
        # Process validation data
        for img, label in tqdm(val_dataset, desc="Processing validation data"):
            # Apply transform if needed
            if not isinstance(img, torch.Tensor):
                img = self.transform(img)
                
            val_images.append(img)
            
            # Handle the label
            if isinstance(label, torch.Tensor):
                if label.numel() == 1:  # Single element tensor
                    val_labels.append(label.item())
                else:
                    val_labels.append(label[0].item())
            else:
                val_labels.append(label)
        
        # Extract features for validation
        val_features = np.array([self.preprocess_image(img) for img in tqdm(val_images, desc="Extracting validation features")])
        
        # Calculate validation accuracy
        val_preds = self.model.predict(val_features)
        val_accuracy = (val_preds == val_labels).mean()
        
        # Log metrics if logger provided
        if logger:
            logger.log_metrics({'val/accuracy': val_accuracy}, step=0)
        
        return val_accuracy
    
    def save(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        model_info = {
            "model": self.model,
            "hog_params": self.hog_params,
            "num_classes": self.num_classes
        }
        joblib.dump(model_info, path)
    
    def load(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        model_info = joblib.load(path)
        self.model = model_info["model"]
        self.hog_params = model_info["hog_params"]
        self.num_classes = model_info["num_classes"]