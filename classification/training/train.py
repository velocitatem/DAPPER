from classification.training.models import BaseModel
from classification.training.hog import HogClassifier
from classification.training.cnn import CNNClassifier
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

# Set up logging and random seed
logger = get_standard_logger("train")
logger.info("Starting training")
set_global_seed()

# Available models
models = {
    "hog": HogClassifier,
    "cnn": CNNClassifier,
}

def train_model(model_name, train_dataset, val_dataset, tb_logger=None, **kwargs):
    """
    Train a model with optional TensorBoard logging
    
    Args:
        model_name: Name of the model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        logger: Optional logger for TensorBoard
        **kwargs: Additional arguments to pass to the model
        
    Returns:
        Trained model
    """
    logger.info(f"Initializing {model_name} model")
    
    # Initialize model
    model = models[model_name](**kwargs)
    
    # Train model with logger
    accuracy = model.train_model(
        train_dataset, 
        val_dataset,
        logger=tb_logger
    )
    
    logger.info(f"Training complete with accuracy {accuracy:.4f}")
    
    return model

def create_dataloaders(train_df, val_df, batch_size=32, num_workers=4):
    """
    Create PyTorch DataLoaders from DataFrames
    
    Args:
        train_df: Training DataFrame with 'image' and 'label' columns
        val_df: Validation DataFrame with 'image' and 'label' columns
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create datasets without transforms (models will handle their own transforms)
    train_dataset = MinioImageDataset(train_df, bucket_name="dapper")
    val_dataset = MinioImageDataset(val_df, bucket_name="dapper")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--model", type=str, default="hog", choices=list(models.keys()),
                        help="Model to train")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of worker processes for data loading")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proportion of the dataset to use for validation")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Name for the experiment (for TensorBoard)")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory to save logs")
    parser.add_argument("--classifier", type=str, default="logistic_regression",
                        choices=["logistic_regression", "svm"],
                        help="Classifier to use for HOG features")
    parser.add_argument("--input-channels", type=int, default=3,
                        choices=[1, 3],
                        help="Number of input channels (1 for grayscale, 3 for RGB)")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate for the optimizer")
    parser.add_argument("--num-epochs", type=int, default=10,
                        help="Number of training epochs")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create experiment name if not provided
    if args.experiment_name is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        args.experiment_name = f"{args.model}_{timestamp}"
    
    # Create TensorBoard logger
    tb_logger = Logger(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
        config={
            "model_name": args.model,
            "classifier": args.classifier if args.model == "hog" else None,
            "input_channels": args.input_channels,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "test_size": args.test_size,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        enable_tensorboard=True
    )
    
    logger.info(f"Starting experiment: {args.experiment_name}")
    
    # Load and prepare dataset
    minio_manager = MinioManager()
    augmentor = Augmentor()
    dataset = get_full_dataset(["rvl_cdip", "kaggle_invoices", "hf_invoices"], minio_manager, augmentor)
    dataset = dataset[['image', 'label']]
    logger.info(f"Loaded dataset with {len(dataset)} samples")
    
    # Split dataset
    train_df, val_df = train_test_split(dataset, test_size=args.test_size, random_state=42)
    logger.info(f"Split dataset: {len(train_df)} training, {len(val_df)} validation")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_df, 
        val_df, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    # Prepare model arguments based on model type
    model_kwargs = {
        "num_classes": len(dataset['label'].unique())
    }
    
    if args.model == "hog":
        model_kwargs["classifier"] = args.classifier
    elif args.model == "cnn":
        model_kwargs["input_channels"] = args.input_channels
        model_kwargs["learning_rate"] = args.learning_rate
        model_kwargs["num_epochs"] = args.num_epochs
    
    # Train model
    model = train_model(
        args.model, 
        train_loader, 
        val_loader, 
        tb_logger=tb_logger,
        **model_kwargs
    )
    
    # Save model
    model_dir = os.path.join(args.log_dir, args.experiment_name, "checkpoints")
    os.makedirs(model_dir, exist_ok=True)
    
    # Use appropriate extension based on model type
    extension = ".joblib" if args.model == "hog" else ".pt"
    model_path = os.path.join(model_dir, f"{args.model}{extension}")
    
    model.save(model_path)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Experiment complete: {args.experiment_name}")
    
    # Close logger
    tb_logger.close()

