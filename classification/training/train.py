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
    if model_name == 'eaml':
        logger.info("Creating MinioMultiModalDataset for EAML model.")
        if tokenizer is None:
             raise ValueError("Tokenizer must be provided for EAML model dataset.")
        if text_params is None:
             text_params = {} # Use defaults if not provided
             logger.warning("Text parameters (max_sentences, max_sent_length) not provided, using defaults from dataset class.")

        # Initialize OCR cache (can be shared between train/val datasets)
        # For simplicity here, each gets its own cache, but sharing might be more efficient.
        train_ocr_cache = {}
        val_ocr_cache = {}

        train_dataset = MinioMultiModalDataset(
            train_df, 
            bucket_name="dapper", # TODO: Make bucket name configurable?
            image_transform=image_transform,
            tokenizer=tokenizer,
            max_sentences=text_params.get('max_sentences', 15), # Default values
            max_sent_length=text_params.get('max_sent_length', 50),
            ocr_cache=train_ocr_cache
            # vocab=None, # Assuming tokenizer handles vocab
            # ocr_lang='eng' # Assuming default 'eng'
        )
        train_dataset.populate_ocr_cache(max_workers=num_workers)
        val_dataset = MinioMultiModalDataset(
            val_df, 
            bucket_name="dapper", 
            image_transform=image_transform,
            tokenizer=tokenizer,
            max_sentences=text_params.get('max_sentences', 15),
            max_sent_length=text_params.get('max_sent_length', 50),
            ocr_cache=val_ocr_cache # Use separate cache for validation
        )
        val_dataset.populate_ocr_cache(max_workers=num_workers)
    elif model_name == 'layoutlmv3':
        logger.info("Creating LayoutLMv3Dataset for LayoutLMv3 model.")
        if processor is None:
            raise ValueError("Processor must be provided for LayoutLMv3 model dataset.")
            
        # For LayoutLMv3, we need to create a custom dataset class
        class LayoutLMv3Dataset(MinioImageDataset):
            def __init__(self, dataframe, bucket_name, processor, **kwargs):
                super().__init__(dataframe, bucket_name, transform=None)
                self.processor = processor
                self.apply_ocr = kwargs.get("apply_ocr", False)
                # Store labels from dataframe
                self.labels = dataframe["label"].values
                # Use simple dictionary for OCR cache
                self.ocr_cache = {}
                # Store dataframe locally to avoid attribute errors
                self.dataframe = dataframe
                
            def get_image_path(self, idx):
                """Get the image path for the given index from the dataframe."""
                try:
                    # First try to access the dataframe stored in this class
                    if hasattr(self, 'dataframe') and self.dataframe is not None:
                        index = idx
                        if hasattr(self, 'indices') and self.indices is not None:
                            index = self.indices[idx]
                        return self.dataframe.iloc[index]['pdf_path']
                        
                    # If not available, try accessing parent class attributes
                    if hasattr(self, 'data') and self.data is not None:
                        index = idx
                        if hasattr(self, 'indices') and self.indices is not None:
                            index = self.indices[idx]
                        return self.data.iloc[index]['pdf_path']
                        
                    if hasattr(self, 'df') and self.df is not None:
                        index = idx
                        if hasattr(self, 'indices') and self.indices is not None:
                            index = self.indices[idx]
                        return self.df.iloc[index]['pdf_path']
                        
                    # Last resort: try to use the parent class's get_image_path
                    return super().get_image_path(idx)
                except Exception as e:
                    logger.error(f"Could not find image path for index {idx}: {str(e)}")
                    raise ValueError(f"Could not find image path for index {idx}: {str(e)}")
                
            def __getitem__(self, idx):
                try:
                    # Get image from Minio
                    image, label = super().__getitem__(idx)
                    
                    # Convert tensor to PIL Image for processor
                    if isinstance(image, torch.Tensor):
                        # Handle different tensor dimensions
                        if image.dim() == 4:  # Batch of images
                            image = image[0]
                        
                        # Convert tensor to PIL Image
                        image = transforms.ToPILImage()(image)
                    
                    # Use a cache key based on the index
                    cache_key = f"image_{idx}"
                    
                    # Check if OCR results are cached
                    if cache_key in self.ocr_cache:
                        words, boxes = self.ocr_cache[cache_key]
                    else:
                        # Run OCR
                        import pytesseract
                        import numpy as np
                        
                        try:
                            # Get OCR results
                            ocr_df = pytesseract.image_to_data(image, output_type='data.frame')
                            ocr_df = ocr_df.dropna().reset_index(drop=True)
                            
                            # Extract words and coordinates
                            words = []
                            boxes = []
                            
                            width, height = image.size
                            
                            for _, row in ocr_df.iterrows():
                                if str(row['text']).strip() != '':
                                    words.append(str(row['text']))
                                    
                                    # Convert coordinates to normalized format (0-1000)
                                    x = int(1000 * row['left'] / width)
                                    y = int(1000 * row['top'] / height)
                                    w = int(1000 * row['width'] / width)
                                    h = int(1000 * row['height'] / height)
                                    
                                    # Bounding box in format [x0, y0, x1, y1]
                                    box = [x, y, x + w, y + h]
                                    boxes.append(box)
                            
                            # Handle empty OCR results
                            if not words:
                                words = [""]
                                boxes = [[0, 0, 0, 0]]
                            
                            # Cache the results
                            self.ocr_cache[cache_key] = (words, boxes)
                        except Exception as e:
                            logger.error(f"OCR failed for image {idx}: {e}")
                            # Provide default values on error
                            words = [""]
                            boxes = [[0, 0, 0, 0]]
                            self.ocr_cache[cache_key] = (words, boxes)
                    
                    # Process with LayoutLMv3Processor
                    try:
                        encoding = self.processor(
                            image, 
                            words,
                            boxes=boxes,
                            return_tensors="pt",
                            truncation=True,
                            padding="max_length"
                        )
                        
                        # Remove batch dimension
                        for k, v in encoding.items():
                            encoding[k] = v.squeeze(0)
                        
                        # Add label
                        if isinstance(label, torch.Tensor):
                            encoding["labels"] = label.clone().detach()
                        else:
                            encoding["labels"] = torch.tensor(label, dtype=torch.long)
                        
                        return encoding
                    except Exception as e:
                        logger.error(f"Error processing image {idx}: {e}")
                        logger.error(f"Words: {words}")
                        logger.error(f"Boxes: {boxes}")
                        # Provide a minimal valid encoding in case of errors
                        # This helps avoid stopping the entire training process
                        dummy_encoding = {
                            "input_ids": torch.zeros((512,), dtype=torch.long),
                            "attention_mask": torch.zeros((512,), dtype=torch.long),
                            "bbox": torch.zeros((512, 4), dtype=torch.long),
                            "pixel_values": torch.zeros((3, 224, 224), dtype=torch.float),
                            "labels": torch.tensor(label, dtype=torch.long) if not isinstance(label, torch.Tensor) else label.clone().detach()
                        }
                        return dummy_encoding
                except Exception as e:
                    logger.error(f"Error in __getitem__ for index {idx}: {str(e)}")
                    # Return minimal valid encoding with default label 0
                    return {
                        "input_ids": torch.zeros((512,), dtype=torch.long),
                        "attention_mask": torch.zeros((512,), dtype=torch.long),
                        "bbox": torch.zeros((512, 4), dtype=torch.long),
                        "pixel_values": torch.zeros((3, 224, 224), dtype=torch.float),
                        "labels": torch.zeros(1, dtype=torch.long)
                    }
        
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
# @brief Load configuration from a YAML file
# @param config_path Path to the configuration file
# @return Dictionary containing the configuration
#
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

##
# @brief Parse command line arguments
# @return Parsed arguments
#
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train a model using a configuration file")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file")
    return parser.parse_args()

##
# @brief Main execution block
#
# This block handles the complete training pipeline:
# 1. Parse command line arguments
# 2. Load configuration from YAML file
# 3. Initialize tokenizer/processor if needed
# 4. Load and prepare dataset
# 5. Create DataLoaders
# 6. Train model
# 7. Save model and log results
#
if __name__ == "__main__":
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # --- Tokenizer Initialization (Placeholder) ---
    # This needs to be implemented based on the chosen approach
    # Example: Load from Hugging Face
    tokenizer = None
    processor = None
    if config['model']['name'] == 'eaml':
        try:
            from transformers import AutoTokenizer
            tokenizer_name = config['data'].get('tokenizer_name', 'bert-base-uncased') # Default tokenizer
            logger.info(f"Loading tokenizer: {tokenizer_name}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            # TODO: Potentially resize tokenizer embeddings if adding special tokens or have custom vocab
        except ImportError:
             logger.error("transformers library not installed. Cannot load tokenizer for EAML.")
             sys.exit(1)
        except Exception as e:
             logger.error(f"Failed to load tokenizer '{tokenizer_name}': {e}")
             sys.exit(1)
    elif config['model']['name'] == 'layoutlmv3':
        try:
            from transformers import LayoutLMv3Processor
            processor_name = config['data'].get('processor_name', 'microsoft/layoutlmv3-base')
            apply_ocr = config['data'].get('apply_ocr', False)
            logger.info(f"Loading LayoutLMv3 processor: {processor_name} (apply_ocr={apply_ocr})")
            processor = LayoutLMv3Processor.from_pretrained(processor_name, apply_ocr=apply_ocr)
        except ImportError:
            logger.error("transformers library not installed. Cannot load processor for LayoutLMv3.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to load processor '{processor_name}': {e}")
            sys.exit(1)
    # --- End Tokenizer Initialization ---
    
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
    
    # Ensure label column is treated correctly if it's not numeric initially
    if not pd.api.types.is_numeric_dtype(dataset['label']):
        dataset['label'], unique_labels = pd.factorize(dataset['label'])
        num_classes = len(unique_labels)
        logger.info(f"Factorized labels into {num_classes} classes.")
    else:
        num_classes = len(dataset['label'].unique())
    
    # Split dataset
    train_df, val_df = train_test_split(
        dataset, 
        test_size=config['training']['test_size'], 
        random_state=42
    )
    logger.info(f"Split dataset: {len(train_df)} training, {len(val_df)} validation")
    
    # Create dataloaders
    # Define image transform based on config or default
    # TODO: Make image size configurable?
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Prepare text parameters if using EAML
    text_params = None
    if model_name == 'eaml':
        text_params = {
            'max_sentences': config['data'].get('max_sentences', 15),
            'max_sent_length': config['data'].get('max_sent_length', 50)
        }

    # Get dataloader configuration
    dataloader_config = config.get('dataloader', {})
    
    # Get batch size from different locations in config with fallbacks
    batch_size = config.get('training', {}).get('batch_size', 
                dataloader_config.get('batch_size', 64))

    train_loader, val_loader = create_dataloaders(
        train_df, 
        val_df, 
        model_name=model_name,
        batch_size=batch_size,
        num_workers=dataloader_config.get('num_workers', 4),
        prefetch_factor=dataloader_config.get('prefetch_factor', 2),
        image_transform=image_transform,
        tokenizer=tokenizer,
        processor=processor,
        text_params=text_params,
        pin_memory=dataloader_config.get('pin_memory', True),
        shuffle_train=dataloader_config.get('shuffle', True),
        drop_last=dataloader_config.get('drop_last', False)
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
        model_kwargs["num_workers"] = config['data']['dataloader'].get('num_workers', 4)
        model_kwargs["device"] = config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
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
    elif model_name == "eaml":
        if tokenizer is None:
            # This should have been caught earlier, but double-check
            raise ValueError("Tokenizer is required for EAML model but was not loaded.")
        model_kwargs["vocab_size"] = tokenizer.vocab_size # Get vocab size from tokenizer
        model_kwargs["embedding_dim"] = config['model'].get('embedding_dim', 100)
        model_kwargs["word_hidden_dim"] = config['model'].get('word_hidden_dim', 50)
        model_kwargs["sent_hidden_dim"] = config['model'].get('sent_hidden_dim', 50)
        model_kwargs["image_channels"] = config['model'].get('image_channels', 3)
        model_kwargs["image_feature_dim"] = config['model'].get('image_feature_dim', 128)
        # image_size is needed by classifier init but handled by transform in dataset
        model_kwargs["image_size"] = (224, 224) # TODO: Link to transform size?
        model_kwargs["dropout"] = config['model'].get('dropout', 0.5)
        model_kwargs["learning_rate"] = config['training'].get('learning_rate', 0.001)
        model_kwargs["num_epochs"] = config['training'].get('num_epochs', 10)
        model_kwargs["device"] = config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    elif model_name == "layoutlmv3":
        model_kwargs["learning_rate"] = float(config['model'].get('learning_rate', 5e-5))
        model_kwargs["weight_decay"] = float(config['model'].get('weight_decay', 0.01))
        model_kwargs["num_epochs"] = int(config['model'].get('num_epochs', 10))
        model_kwargs["apply_ocr"] = bool(config['model'].get('apply_ocr', False))
        model_kwargs["max_length"] = int(config['model'].get('max_length', 512))
        model_kwargs["device"] = config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        model_kwargs["processor_name"] = config['data'].get('processor_name', config['model'].get('model_name'))

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
    extension = ".joblib" if model_name == "hog" else ".pth" # Use .pth for torch models
    model_path = os.path.join(model_dir, f"{model_name}_final{extension}") # Add _final marker
    
    # The save method in EAMLClassifier takes path and optional metadata
    try:
        if hasattr(model, 'save'):
            model.save(model_path) # Simplest call, add metadata if needed
            logger.info(f"Model saved to {model_path}")
        else:
            logger.warning(f"Model type {model_name} does not have a save method implemented.")
    except Exception as e:
        logger.error(f"Failed to save model to {model_path}: {e}")
    
    logger.info(f"Experiment complete: {config['logging']['experiment_name']}")
    
    # Close logger
    tb_logger.close()
    
    # Log OCR cache stats if applicable
    if model_name == 'eaml':
        try:
            # Assumes train_loader.dataset is the MinioMultiModalDataset instance
            train_cache_stats = train_loader.dataset.get_cache_stats()
            val_cache_stats = val_loader.dataset.get_cache_stats()
            logger.info(f"Train OCR Cache Stats: {train_cache_stats}")
            logger.info(f"Val OCR Cache Stats: {val_cache_stats}")
            tb_logger.log_metrics({
                "dataset/train_ocr_cache_hit_rate": train_cache_stats['hit_rate_percent'],
                "dataset/val_ocr_cache_hit_rate": val_cache_stats['hit_rate_percent']
            })
        except AttributeError:
            logger.warning("Could not retrieve OCR cache stats from dataset.")
        except Exception as e:
            logger.error(f"Error logging OCR cache stats: {e}")
    
    sys.exit(0)

