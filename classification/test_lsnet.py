#!/usr/bin/env python3
"""
Test script for the LSNet model.

This script loads a pre-trained LSNet model and runs inference on a sample image.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

from classification.training.models import get_model
from classification.utils.logger import get_standard_logger

logger = get_standard_logger("lsnet_test")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test LSNet model')
    parser.add_argument('--model', type=str, default='lsnet_t', 
                        help='Model to use (lsnet_t, lsnet_s, lsnet_b)')
    parser.add_argument('--image', type=str, required=True, 
                        help='Path to the image to classify')
    parser.add_argument('--num_classes', type=int, default=1000, 
                        help='Number of classes (default: 1000 for ImageNet)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--show_image', action='store_true', 
                        help='Show the input image')
    return parser.parse_args()

def load_image(image_path):
    """Load and preprocess an image."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations
    image_tensor = transform(image)
    
    return image, image_tensor

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    model = get_model(
        model_name=args.model,
        num_classes=args.num_classes,
        pretrained=True
    )
    model.to(device)
    model.eval()
    
    # Load image
    logger.info(f"Loading image: {args.image}")
    image, image_tensor = load_image(args.image)
    
    # Show image if requested
    if args.show_image:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.title('Input Image')
        plt.show()
    
    # Run inference
    logger.info("Running inference")
    with torch.no_grad():
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Forward pass
        outputs = model(image_tensor)
        
        # Get probabilities
        probabilities = F.softmax(outputs, dim=1)
        
        # Get top-5 predictions
        top5_prob, top5_indices = torch.topk(probabilities, 5)
        
        # Convert to numpy
        top5_prob = top5_prob.cpu().numpy()[0]
        top5_indices = top5_indices.cpu().numpy()[0]
    
    # Print results
    logger.info("Top-5 predictions:")
    for i, (prob, idx) in enumerate(zip(top5_prob, top5_indices)):
        logger.info(f"{i+1}. Class {idx}: {prob:.4f}")
    
    # If using ImageNet classes, load class names
    if args.num_classes == 1000:
        try:
            # Try to load ImageNet class names
            import json
            with open('classification/data/imagenet_classes.json', 'r') as f:
                class_names = json.load(f)
            
            logger.info("Top-5 predictions with class names:")
            for i, (prob, idx) in enumerate(zip(top5_prob, top5_indices)):
                class_name = class_names[str(idx)]
                logger.info(f"{i+1}. {class_name}: {prob:.4f}")
        except (FileNotFoundError, ImportError, json.JSONDecodeError):
            logger.warning("Could not load ImageNet class names")

if __name__ == '__main__':
    main() 