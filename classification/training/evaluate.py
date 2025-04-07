##
# @file evaluate.py
# @package classification.training.evaluate
# @brief Evaluation script for document classification models
#
# This module provides a comprehensive evaluation pipeline for multiple document 
# classification models. It loads trained models, evaluates them on a test set,
# and produces performance metrics and reports.
#
# @author Statistical Learning Team
# @date 2025
#

from classification.training.models import BaseModel
from classification.training.cnn import CNNClassifier
from classification.training.resnet import ResNetClassifier
from classification.training.eaml import EAMLClassifier
from classification.training.lsnet import LSNetClassifier
from classification.training.hybrid import HybridClassifier
from classification.data.loader import get_full_dataset
from classification.data.minio_handler import MinioManager
from classification.data.augmentor import Augmentor
from classification.data.minio_dataset import MinioImageDataset, MinioMultiModalDataset
from sklearn.model_selection import train_test_split
from classification.utils.logger import get_standard_logger
from classification.utils.seed import set_global_seed
import torch
from torch.utils.data import DataLoader
import argparse
import os
import yaml
from torchvision import transforms
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
import time

# Set up logging and random seed
logger = get_standard_logger("evaluate")
logger.info("Starting evaluation")
set_global_seed()

# Define model paths
models = {
    "resnet": "/home/velocitatem/Documents/University/Third Year/Statistical Learning/final_project/models/resnet/resnet_sunday_night_baseline.pth",
    "eaml": "/home/velocitatem/Documents/University/Third Year/Statistical Learning/final_project/models/eaml/eaml_sunday_night_baseline.pth",
    "cnn": "/home/velocitatem/Documents/University/Third Year/Statistical Learning/final_project/models/cnn/cnn_sunday_night_baseline.pth",
    "lsnet_t": "/home/velocitatem/Documents/University/Third Year/Statistical Learning/final_project/models/lsnet_t_best.pth",
    "hybrid": "/home/velocitatem/Documents/University/Third Year/Statistical Learning/final_project/models/hybrid/best_model.pth",

}

# Available model classes
model_classes = {
    "cnn": CNNClassifier,
    "resnet": ResNetClassifier,
    "eaml": EAMLClassifier,
    "lsnet_t": LSNetClassifier,
    "hybrid": HybridClassifier,
}

def create_dataloaders(
    test_df, 
    model_name,
    batch_size=64, 
    num_workers=4, 
    prefetch_factor=2,
    image_transform=None,
    tokenizer=None,
    text_params=None,
    pin_memory=True,
):
    """
    Create optimized PyTorch DataLoaders from DataFrames for evaluation
    
    Args:
        test_df: Test DataFrame with 'image' and 'label' columns
        model_name: Name of the model to create DataLoaders for
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading
        prefetch_factor: Number of batches loaded in advance by each worker
        image_transform: Image transformations to apply
        tokenizer: Tokenizer instance for text processing (if needed)
        text_params: Dictionary of text parameters (e.g., max_sentences, max_sent_length)
        pin_memory: Whether to pin memory for faster GPU access
        
    Returns:
        test_loader: DataLoader for test dataset
    """
    # Determine optimal number of workers if not specified
    if num_workers <= 0:
        num_workers = min(os.cpu_count(), 20)
    
    # Use the provided image transform or define a default one
    if image_transform is None:
        logger.warning("No image_transform provided, using default Resize(224)/ToTensor/Normalize.")
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Create datasets based on model type
    if model_name == 'eaml':
        logger.info(f"Creating MinioMultiModalDataset for {model_name} model.")
        if tokenizer is None:
            # Import tokenizer
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            
        if text_params is None:
            text_params = {}
            logger.warning("Text parameters not provided, using defaults.")

        test_dataset = MinioMultiModalDataset(
            test_df, 
            bucket_name="dapper",
            image_transform=image_transform,
            tokenizer=tokenizer,
            max_sentences=text_params.get('max_sentences', 15),
            max_sent_length=text_params.get('max_sent_length', 50),
        )
    else:
        logger.info(f"Creating MinioImageDataset for {model_name} model.")
        test_dataset = MinioImageDataset(test_df, bucket_name="dapper", transform=image_transform)
    
    # Create data loader with optimized settings
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )
    
    return test_loader

def evaluate_model(model, test_loader, model_name, device, output_dir="evaluation_results"):
    """
    Evaluate a model on the test set and generate performance metrics and reports
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test dataset
        model_name: Name of the model being evaluated
        device: Device to run evaluation on (cuda/cpu)
        output_dir: Directory to save evaluation results
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    logger.info(f"Evaluating {model_name} model")
    model.eval()
    all_preds = []
    all_labels = []
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    with torch.no_grad():
        # Check if model is EAML (which requires both docs and images)
        if model_name == "eaml":
            for batch_data in tqdm(test_loader, desc=f"Evaluating {model_name}"):
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
            for images, labels in tqdm(test_loader, desc=f"Evaluating {model_name}"):
                images = images.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
    
    evaluation_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print metrics
    logger.info(f"{model_name} Evaluation Metrics:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Evaluation time: {evaluation_time:.2f} seconds")
    
    # Plot and save confusion matrix
    CLASSES = ["correspondence", "forms", "invoice", "other", "personal", "promotional", "scientific"]
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save confusion matrix plot
    cm_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    
    # Generate classification report
    class_report = classification_report(all_labels, all_preds)
    logger.info(f"\n{model_name} Classification Report:")
    logger.info(class_report)
    
    # Save classification report
    report_path = os.path.join(output_dir, f'{model_name}_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(class_report)
    
    # Return metrics dictionary
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "evaluation_time": evaluation_time
    }

def parse_args():
    """
    Parse command-line arguments
    
    Returns:
        ArgumentParser object with parsed arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate document classification models")
    parser.add_argument("--config", type=str, required=False, default=None,
                        help="Path to the configuration file")
    parser.add_argument("--models", type=str, nargs='+', default=None,
                        help="Names of models to evaluate (default: evaluate all)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--apply_ocr", action="store_true",
                        help="Whether to apply OCR to images")
    return parser.parse_args()

def load_config(config_path):
    """
    Load and parse a YAML configuration file
    
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
        return None

def main(args):
    """
    Main execution function
    
    Args:
        args: Command-line arguments
    """
    # Load configuration if provided
    config = None
    if args.config:
        config = load_config(args.config)
    
    # Determine which models to evaluate
    model_names = args.models if args.models else list(models.keys())
    logger.info(f"Models to evaluate: {', '.join(model_names)}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create MinIO manager and augmentor
    minio_manager = MinioManager()
    augmentor = Augmentor()
    
    # Get dataset
    apply_ocr = args.apply_ocr if args.apply_ocr else (config and config["training"].get("apply_ocr", False))
    logger.info(f"Loading dataset with OCR: {apply_ocr}")
    df = get_full_dataset(
        ["rvl_cdip"],  # Default dataset, can be configured
        minio_manager,
        augmentor,
        apply_ocr=apply_ocr
    )
    
    # Split dataset - just use validation set for evaluation
    _, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )
    
    logger.info(f"Test set: {len(test_df)} samples")
    
    # Define default transformation
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # For storing all results
    all_results = {}
    
    # Evaluate each model
    for model_name in model_names:
        if model_name not in models:
            logger.warning(f"Model {model_name} not found in available models. Skipping.")
            continue
            
        model_path = models[model_name]
        
        # Create appropriate dataloader
        tokenizer = None
        text_params = None
        
        if model_name in ["eaml", "hybrid"]:
            # Import tokenizer
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            text_params = {
                "max_sentences": 15,
                "max_sent_length": 50
            }
        
        # Create dataloader
        test_loader = create_dataloaders(
            test_df=test_df,
            model_name=model_name,
            batch_size=args.batch_size,
            image_transform=image_transform,
            tokenizer=tokenizer,
            text_params=text_params
        )
        
        # Initialize model class
        model_class = model_classes[model_name]
        if model_name == "eaml":
            # For EAML, we need to provide vocab_size from the tokenizer
            model_instance = model_class(
                num_classes=df["label"].nunique(),
                vocab_size=len(tokenizer.vocab)
            )
        elif model_name == "hybrid":
            # For hybrid model, we need to use HybridTrainer which handles loading
            from classification.training.hybrid import HybridTrainer
            # First try to analyze the saved model to see if it's LSNet or ResNet based
            try:
                # Peek at the model content without fully loading
                temp_checkpoint = torch.load(model_path, map_location=device)
                # Check for architecture indicators
                is_resnet_based = False
                if 'model_state_dict' in temp_checkpoint:
                    key_sample = list(temp_checkpoint['model_state_dict'].keys())[:10]
                    is_resnet_based = any('resnet' in k.lower() for k in key_sample)
                    is_lsnet_based = any('lsnet' in k.lower() for k in key_sample)
                    logger.info(f"Detected hybrid model type - ResNet: {is_resnet_based}, LSNet: {is_lsnet_based}")
                else:
                    logger.warning("Could not detect hybrid model architecture from checkpoint, using ResNet as default")
                
                # Create appropriate trainer
                trainer = HybridTrainer(
                    num_classes=df["label"].nunique(),
                    vocab_size=len(tokenizer.vocab),
                    use_resnet=is_resnet_based,  # Use the detected architecture
                    device=str(device)
                )
                model_instance = trainer  # Use trainer as the model instance
            except Exception as e:
                logger.error(f"Error analyzing hybrid model file: {str(e)}")
                continue
        else:
            model_instance = model_class(num_classes=df["label"].nunique())
        
        # Load model weights
        logger.info(f"Loading {model_name} model from {model_path}")
        try:
            if model_name == "hybrid":
                # Use the trainer's load method for hybrid models
                try:
                    trainer.load(model_path)
                    model = trainer.classifier.model
                    model.to(device)
                except (ValueError, RuntimeError) as e:
                    logger.warning(f"Could not load the hybrid model: {str(e)}")
                    logger.warning(f"Skipping evaluation of {model_name} model")
                    continue
            else:
                # Standard loading for other model classes
                model_instance.load(model_path)
                model = model_instance.model
                model.to(device)
            
            # Evaluate model
            results = evaluate_model(
                model=model,
                test_loader=test_loader,
                model_name=model_name,
                device=device,
                output_dir=args.output_dir
            )
            
            all_results[model_name] = results
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name} model: {str(e)}")
    
    # Print comparison of all models
    logger.info("\nModel Comparison:")
    logger.info(f"{'Model':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Eval Time (s)':<15}")
    logger.info("-" * 65)
    
    for model_name, results in all_results.items():
        logger.info(f"{model_name:<10} {results['accuracy']:<10.4f} {results['precision']:<10.4f} "
                   f"{results['recall']:<10.4f} {results['f1']:<10.4f} {results['evaluation_time']:<15.2f}")
    
    # Save comparison to file
    comparison_path = os.path.join(args.output_dir, "model_comparison.txt")
    with open(comparison_path, 'w') as f:
        f.write("Model Comparison:\n")
        f.write(f"{'Model':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Eval Time (s)':<15}\n")
        f.write("-" * 65 + "\n")
        
        for model_name, results in all_results.items():
            f.write(f"{model_name:<10} {results['accuracy']:<10.4f} {results['precision']:<10.4f} "
                    f"{results['recall']:<10.4f} {results['f1']:<10.4f} {results['evaluation_time']:<15.2f}\n")

if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except Exception as e:
        logger.error(f"Error: {str(e)}")

