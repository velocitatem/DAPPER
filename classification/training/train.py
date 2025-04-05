from classification.training.models import BaseModel
from classification.training.hog import HogClassifier
from classification.training.cnn import CNNClassifier
from classification.training.resnet import ResNetClassifier
from classification.training.lsnet import LSNetClassifier
from classification.data.loader import get_full_dataset
from classification.data.minio_handler import MinioManager
from classification.data.augmentor import Augmentor
from classification.data.minio_dataset import MinioImageDataset
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
}

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

def create_dataloaders(train_df, val_df, batch_size=32, num_workers=4, prefetch_factor=2):
    """
    Create optimized PyTorch DataLoaders from DataFrames
    
    Args:
        train_df: Training DataFrame with 'image' and 'label' columns
        val_df: Validation DataFrame with 'image' and 'label' columns
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading
        prefetch_factor: Number of batches loaded in advance by each worker
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Determine optimal number of workers if not specified
    if num_workers <= 0:
        num_workers = min(os.cpu_count(), 8)  # Use at most 8 workers
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets with transforms
    train_dataset = MinioImageDataset(train_df, bucket_name="dapper", transform=transform)
    val_dataset = MinioImageDataset(val_df, bucket_name="dapper", transform=transform)
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=prefetch_factor,
        persistent_workers=True,  # Keep workers alive between epochs
        drop_last=True,  # Drop last incomplete batch for better GPU utilization
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )
    
    return train_loader, val_loader

def load_config(config_path):
    """
    Load configuration from a YAML file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train a model using a configuration file")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Extract model name from config
    model_name = config['model']['name']
    if model_name not in models:
        logger.error(f"Unknown model: {model_name}")
        sys.exit(1)
    
    # Create experiment name if not provided
    if config['logging']['experiment_name'] is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        config['logging']['experiment_name'] = f"{model_name}_{timestamp}"
    
    # Create TensorBoard logger
    tb_logger = Logger(
        log_dir=config['logging']['log_dir'],
        experiment_name=config['logging']['experiment_name'],
        config=config,
        enable_tensorboard=True
    )
    
    logger.info(f"Starting experiment: {config['logging']['experiment_name']}")
    
    # Load and prepare dataset
    minio_manager = MinioManager()
    augmentor = Augmentor()
    dataset = get_full_dataset(["rvl_cdip", "kaggle_invoices", "hf_invoices"], minio_manager, augmentor)
    dataset = dataset[['image', 'label']]
    logger.info(f"Loaded dataset with {len(dataset)} samples")
    
    # Get the number of unique classes
    num_classes = len(dataset['label'].unique())
    logger.info(f"Dataset contains {num_classes} classes")
    
    # Split dataset
    train_df, val_df = train_test_split(
        dataset, 
        test_size=config['training']['test_size'], 
        random_state=42
    )
    logger.info(f"Split dataset: {len(train_df)} training, {len(val_df)} validation")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_df, 
        val_df, 
        batch_size=config['data']['dataloader'].get('batch_size', 64), 
        num_workers=config['data']['dataloader'].get('num_workers', 4)
    )
    
    # Prepare model arguments based on model type
    model_kwargs = {
        "num_classes": num_classes,
    }
    
    # Add model-specific parameters
    if model_name == "hog":
        model_kwargs["classifier"] = config['model']['classifier']
    elif model_name == "cnn":
        model_kwargs["input_channels"] = config['model']['input_channels']
        model_kwargs["learning_rate"] = config['model']['learning_rate']
        model_kwargs["num_epochs"] = config['model']['num_epochs']
    elif model_name == "resnet":
        model_kwargs["trained_model_name"] = config['model'].get('resnet_model', "resnet18")
        model_kwargs["pretrained"] = config['model'].get('pretrained', True)
        model_kwargs["learning_rate"] = config['model'].get('learning_rate', 1e-4)
        model_kwargs["weight_decay"] = config['model'].get('weight_decay', 0.01)
        model_kwargs["num_epochs"] = config['model'].get('num_epochs', 50)
        model_kwargs["dropout_rate"] = config['model'].get('dropout_rate', 0.7)
    elif model_name.startswith("lsnet"):
        # Extract model size from model name (t, s, or b)
        model_size = model_name.split('_')[1]
        model_kwargs["model_size"] = model_size
        model_kwargs["pretrained"] = config['model'].get('pretrained', False)
        model_kwargs["freeze_backbone"] = config['model'].get('freeze_backbone', False)
        model_kwargs["learning_rate"] = config['training']['optimizer'].get('learning_rate', 0.001)
        model_kwargs["weight_decay"] = config['training']['optimizer'].get('weight_decay', 0.0001)
        model_kwargs["num_epochs"] = config['training'].get('num_epochs', 100)
        model_kwargs["batch_size"] = config['data']['dataloader'].get('batch_size', 64)
        model_kwargs["num_workers"] = config['data']['dataloader'].get('num_workers', 4)
        model_kwargs["device"] = config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # put to TB info about the dataset
    tb_logger.log_metrics({
        "dataset/num_classes": num_classes,
        "dataset/train_samples": len(train_df),
        "dataset/val_samples": len(val_df)
    })
    
    print(model_name,type(model_name))
    # Train model
    model = train_model(
        model_name, 
        train_loader, 
        val_loader, 
        tb_logger=tb_logger,
        **model_kwargs
    )
    
    # Save model
    model_dir = os.path.join(config['logging']['log_dir'], config['logging']['experiment_name'], "checkpoints")
    os.makedirs(model_dir, exist_ok=True)
    
    # Use appropriate extension based on model type
    extension = ".joblib" if model_name == "hog" else ".pt"
    model_path = os.path.join(model_dir, f"{model_name}{extension}")
    
    model.save(model_path, logger=tb_logger)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Experiment complete: {config['logging']['experiment_name']}")
    
    # Close logger
    tb_logger.close()
    
    sys.exit(0)

