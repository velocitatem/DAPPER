##
# @file train.py
# @package classification.training.train
# @brief Training script for document classification models
#
# This module provides a comprehensive training pipeline for document classification
# models. It supports multiple model architectures, handles dataset loading and
# preprocessing, and implements training with TensorBoard logging.
#
# @author Statistical Learning Team
# @date 2025
#

from classification.training.models import BaseModel
from classification.training.hog import HogClassifier
from classification.training.cnn import CNNClassifier
from classification.training.resnet import ResNetClassifier
from classification.training.lsnet import LSNetClassifier
from classification.training.eaml import EAMLClassifier
from classification.training.layout import LayoutLMv3Classifier
from classification.training.hybrid import HybridTrainer
from classification.training.layout import LayoutLMv3Dataset
from classification.data.loader import get_full_dataset
from classification.data.minio_handler import MinioManager
from classification.data.augmentor import Augmentor
from classification.data.minio_dataset import MinioImageDataset, MinioMultiModalDataset
from sklearn.model_selection import train_test_split
from classification.utils.logger import get_standard_logger, Logger
from classification.utils.seed import set_global_seed
import torch
from torch.utils.data import DataLoader
import argparse
import time
import os
import sys
import yaml
from torchvision import transforms
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

# Set up logging and random seed
logger = get_standard_logger("train")
logger.info("Starting training")
set_global_seed()

# Available models
models = {
    "hog": HogClassifier,
    "cnn": CNNClassifier,
    "resnet": ResNetClassifier,
    "lsnet_t": LSNetClassifier,
    "lsnet_s": LSNetClassifier,
    "lsnet_b": LSNetClassifier,
    "eaml": EAMLClassifier,
    "layoutlmv3": LayoutLMv3Classifier,
    "hybrid": HybridTrainer,
}

##
# @brief Train a model with optional TensorBoard logging
# @param model_name Name of the model to train
# @param train_loader Training DataLoader
# @param val_loader Validation DataLoader
# @param tb_logger Optional logger for TensorBoard
# @param **kwargs Additional arguments to pass to the model
# @return Trained model
#
def train_model(model_name, train_loader, val_loader, tb_logger=None, **kwargs):
    """
    Train a model with optional TensorBoard logging
    
    Args:
        model_name: Name of the model to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        tb_logger: Optional logger for TensorBoard
        **kwargs: Additional arguments to pass to the model
        
    Returns:
        Trained model
    """
    logger.info(f"Initializing {model_name} model")
    
    # Initialize model
    model = models[model_name](**kwargs)
    
    # Train model with logger
    accuracy = model.train_model(
        train_loader, 
        val_loader,
        tb_logger=tb_logger
    )
    
    logger.info(f"Training complete with accuracy {accuracy:.4f}")
    
    return model

##
# @brief Create optimized PyTorch DataLoaders from DataFrames
# @param train_df Training DataFrame with 'image' and 'label' columns
# @param val_df Validation DataFrame with 'image' and 'label' columns
# @param model_name Name of the model to create DataLoaders for
# @param batch_size Batch size for DataLoader
# @param num_workers Number of worker processes for data loading
# @param prefetch_factor Number of batches loaded in advance by each worker
# @param image_transform Image transformations to apply
# @param tokenizer Tokenizer instance for text processing (if needed)
# @param processor LayoutLMv3Processor for LayoutLMv3 model (if needed)
# @param text_params Dictionary of text parameters (e.g., max_sentences, max_sent_length)
# @param pin_memory Whether to pin memory for faster GPU access
# @param shuffle_train Whether to shuffle the training dataset
# @param drop_last Whether to drop the last incomplete batch
# @return tuple: (train_loader, val_loader)
#
def create_dataloaders(
    train_df, 
    val_df, 
    model_name,
    batch_size=64, 
    num_workers=4, 
    prefetch_factor=2,
    image_transform=None,
    tokenizer=None,
    processor=None,
    text_params=None,
    pin_memory=True,
    shuffle_train=True,
    drop_last=False
):
    """
    Create optimized PyTorch DataLoaders from DataFrames
    
    Args:
        train_df: Training DataFrame with 'image' and 'label' columns
        val_df: Validation DataFrame with 'image' and 'label' columns
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading
        prefetch_factor: Number of batches loaded in advance by each worker
        image_transform: Image transformations to apply.
        tokenizer: Tokenizer instance for text processing (if needed).
        processor: LayoutLMv3Processor for LayoutLMv3 model (if needed).
        text_params: Dictionary of text parameters (e.g., max_sentences, max_sent_length).
        pin_memory: Whether to pin memory for faster GPU access
        shuffle_train: Whether to shuffle the training dataset
        drop_last: Whether to drop the last incomplete batch
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Determine optimal number of workers if not specified
    if num_workers <= 0:
        num_workers = min(os.cpu_count(), 20)  # Use at most 8 workers
    
    # Use the provided image transform or define a default one
    if image_transform is None:
        logger.warning("No image_transform provided to create_dataloaders, using default Resize(224)/ToTensor/Normalize.")
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Create datasets based on model type
    if model_name in ['eaml', 'hybrid']:
        logger.info(f"Creating MinioMultiModalDataset for {model_name} model.")
        if tokenizer is None:
             raise ValueError(f"Tokenizer must be provided for {model_name} model dataset.")
        if text_params is None:
             text_params = {} # Use defaults if not provided
             logger.warning("Text parameters (max_sentences, max_sent_length) not provided, using defaults from dataset class.")

        train_dataset = MinioMultiModalDataset(
            train_df, 
            bucket_name="dapper", # TODO: Make bucket name configurable?
            image_transform=image_transform,
            tokenizer=tokenizer,
            max_sentences=text_params.get('max_sentences', 15), # Default values
            max_sent_length=text_params.get('max_sent_length', 50),
        )
        val_dataset = MinioMultiModalDataset(
            val_df, 
            bucket_name="dapper", 
            image_transform=image_transform,
            tokenizer=tokenizer,
            max_sentences=text_params.get('max_sentences', 15),
            max_sent_length=text_params.get('max_sent_length', 50),
        )
    elif model_name == 'layoutlmv3':
        logger.info("Creating LayoutLMv3Dataset for LayoutLMv3 model.")
        if processor is None:
            raise ValueError("Processor must be provided for LayoutLMv3 model dataset.")
            
        # For LayoutLMv3, we need to create a custom dataset class
        # Initialize datasets with processor
        train_dataset = LayoutLMv3Dataset(
            train_df,
            bucket_name="dapper",
            processor=processor
        )
        val_dataset = LayoutLMv3Dataset(
            val_df,
            bucket_name="dapper",
            processor=processor
        )
    else:
        logger.info(f"Creating MinioImageDataset for {model_name} model.")
        train_dataset = MinioImageDataset(train_df, bucket_name="dapper", transform=image_transform)
        val_dataset = MinioImageDataset(val_df, bucket_name="dapper", transform=image_transform)
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,  # Keep workers alive between epochs
        drop_last=drop_last,  # Drop last incomplete batch for better GPU utilization
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )
    
    return train_loader, val_loader

##
# @brief Load and parse configuration file
# @param config_path Path to the configuration file
# @return Dictionary containing configuration
#
def load_config(config_path):
    """
    Load and parse a YAML configuration file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    try:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
            logger.info(f"Loaded configuration from {config_path}")
            return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        sys.exit(1)

##
# @brief Main function to parse arguments and run training
# @return ArgumentParser object with parsed arguments
#
def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        ArgumentParser object with parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train a document classification model")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file")
    return parser.parse_args()

##
# @brief Main execution function
# @param args Command-line arguments
#
def main(args):
    """
    Main execution function.
    
    Args:
        args: Command-line arguments
    """
    # Load configuration
    config = load_config(args.config)
    
    # Get model name
    model_name = config["model"]["name"]
    
    # Create MinIO manager and augmentor
    minio_manager = MinioManager()
    augmentor = Augmentor()
    
    # Get dataset
    logger.info(f"Loading dataset with OCR: {config['training'].get('apply_ocr', False)}")
    df = get_full_dataset(
        config["training"].get("datasets", ["rvl_cdip"]),
        minio_manager,
        augmentor,
        apply_ocr=config["training"].get("apply_ocr", False)
    )
    
    # Split dataset
    train_df, val_df = train_test_split(
        df,
        test_size=config["training"].get("test_size", 0.2),
        random_state=42,
        stratify=df["label"]
    )
    
    logger.info(f"Train set: {len(train_df)} samples, Validation set: {len(val_df)} samples")
    
    # Set up image transform
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Set up tokenizer for EAML or Hybrid model
    tokenizer = None
    text_params = None
    
    if model_name in ["eaml", "hybrid"]:
        # Import tokenizer
        from transformers import BertTokenizer
        
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        text_params = {
            "max_sentences": config["training"].get("max_sentences", 15),
            "max_sent_length": config["training"].get("max_sent_length", 50)
        }
    
    # Set up processor for LayoutLMv3 model
    processor = None
    
    if model_name == "layoutlmv3":
        # Import processor
        from transformers import LayoutLMv3Processor
        
        processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        model_name=model_name,
        batch_size=config["training"].get("batch_size", 32),
        num_workers=config["training"].get("num_workers", 4),
        image_transform=image_transform,
        tokenizer=tokenizer,
        processor=processor,
        text_params=text_params
    )
    
    # Set up TensorBoard logger
    experiment_name = config["logging"].get("experiment_name", None)
    log_dir = config["logging"].get("log_dir", "logs")
    tb_logger = Logger(log_dir=log_dir, experiment_name=experiment_name)
    
    # Train model
    model_kwargs = {**config["model"]}
    
    # Remove name from kwargs as it's not needed for model initialization
    model_kwargs.pop("name", None)
    
    # Add common training parameters
    model_kwargs["num_classes"] = df["label"].nunique()
    
    if model_name == "hybrid":
        model_kwargs["save_dir"] = config["logging"].get("save_dir", "models/hybrid")
        model_kwargs["patience"] = config["training"].get("patience", 10)
        model_kwargs["mixed_precision"] = config["training"].get("mixed_precision", True)
        model_kwargs["weight_decay"] = model_kwargs.get("weight_decay", 0.01)
    
    # Train model
    blacklist= ["cnn", "layoutlmv3", "resnet", "eaml"]
    TRAIN_MODEL = False if model_name in blacklist else True
    TRAIN_MODEL = False
    save_dir = config["logging"].get("save_dir", f"models/{model_name}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{experiment_name}.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if TRAIN_MODEL:
        model = train_model(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            tb_logger=tb_logger,
            **model_kwargs
        )
        
        # Save model
        
        try:
            logger.info(f"Saving model to {save_path}")
            
            if hasattr(model, "save"):
                model.save(save_path)
            else:
                model_dict = {
                    "model_state_dict": model.model.state_dict() if hasattr(model, "model") else model.state_dict(),
                    "config": config
                }
                torch.save(model_dict, save_path)
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
        
        logger.info("Training complete")
        model = model.model
    else:
        model = models[model_name](**model_kwargs)
        model.load(save_path)
        model = model.model
        logger.info(f"Loaded model from {save_path}")
    # Evaluate model on validation set
    logger.info("Performing final model evaluation")
    tb_logger.log_model_graph(model)
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        # Check if model is EAML (which requires both docs and images)
        if model_name == "eaml":
            for batch_data in tqdm(val_loader, desc="Final evaluation"):
                # Handle different batch data formats
                if isinstance(batch_data, dict):
                    docs = batch_data['text'].to(device)
                    images = batch_data['image'].to(device)
                    labels = batch_data['label']
                elif isinstance(batch_data, (list, tuple)):
                    docs, images, labels = batch_data
                    docs = docs.to(device)
                    images = images.to(device)
                else:
                    raise TypeError("Unsupported batch data type from DataLoader")
                
                outputs = model(docs, images)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        else:
            # Standard case for models with single input
            for images, labels in tqdm(val_loader, desc="Final evaluation"):
                images = images.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Log metrics
    logger.info(f"Final Evaluation Metrics:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    # Log to tensorboard
    tb_logger.log_metrics({
        'final_eval/accuracy': accuracy,
        'final_eval/precision': precision, 
        'final_eval/recall': recall,
        'final_eval/f1': f1
    })
    
    # Plot and save confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save confusion matrix plot
    cm_path = os.path.join(log_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    
    # Log confusion matrix to tensorboard
    tb_logger.log_figure(f'final_eval/{model_name}_confusion_matrix', plt.gcf())
    
    # Generate classification report
    class_report = classification_report(all_labels, all_preds)
    logger.info("\nClassification Report:")
    logger.info(class_report)
    
    # Save classification report
    report_path = os.path.join(log_dir, f'{model_name}_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(class_report)

##
# @brief Main execution block
#
if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except Exception as e:
        logger.error(f"Error: {str(e)}")

