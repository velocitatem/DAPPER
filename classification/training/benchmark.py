import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from classification.training.train import train_model, create_dataloaders
from classification.data.loader import get_full_dataset
from classification.data.minio_handler import MinioManager
from classification.data.augmentor import Augmentor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from classification.utils.logger import get_standard_logger
from classification.utils.logger import Logger
import os
import yaml
from datetime import datetime

# Set up logging
logger = get_standard_logger("benchmark")

def plot_confusion_matrix(y_true, y_pred, classes, model_name, save_path):
    """
    Plot confusion matrix and save it to the specified path.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        model_name: Name of the model (for title)
        save_path: Path to save the plot
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def run_benchmark(models_to_test, dataset_size=1000, batch_size=32, num_workers=4):
    """
    Run benchmark tests on different models and plot their performance.
    
    Args:
        models_to_test: Dictionary of model configurations to test
        dataset_size: Number of samples to use for benchmarking
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
    """
    # Create experiment name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"benchmark_{timestamp}"
    
    # Create TensorBoard logger
    tb_logger = Logger(
        log_dir="logs",
        experiment_name=experiment_name,
        config={
            "models": models_to_test,
            "dataset_size": dataset_size,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "timestamp": timestamp
        },
        enable_tensorboard=True
    )
    
    logger.info(f"Starting benchmark experiment: {experiment_name}")
    
    # Load dataset
    minio_manager = MinioManager()
    augmentor = Augmentor()
    dataset = get_full_dataset(["rvl_cdip", "kaggle_invoices", "hf_invoices"], 
                             minio_manager, augmentor)
    dataset = dataset[['image', 'label']][:dataset_size]
    
    # Get number of classes and class names
    num_classes = len(dataset['label'].unique())
    class_names = sorted(dataset['label'].unique().tolist())
    logger.info(f"Dataset contains {num_classes} classes")
    
    # Log dataset info as metrics
    tb_logger.log_metrics({
        "dataset/num_classes": num_classes,
        "dataset/total_samples": len(dataset)
    })
    
    # Split dataset
    train_df, val_df = train_test_split(dataset, test_size=0.2, random_state=42)
    logger.info(f"Split dataset: {len(train_df)} training, {len(val_df)} validation")
    
    # Log split info as metrics
    tb_logger.log_metrics({
        "dataset/train_samples": len(train_df),
        "dataset/val_samples": len(val_df)
    })
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_df,
        val_df,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Results storage
    training_times = []
    accuracies = []
    model_names = []
    
    # Test each model
    for model_name, model_config in models_to_test.items():
        logger.info(f"Benchmarking {model_name}")
        
        # Prepare model arguments
        model_kwargs = {
            "num_classes": num_classes,
            **model_config
        }
        
        # Log model configuration as metrics
        for key, value in model_config.items():
            if isinstance(value, (int, float)):
                tb_logger.log_metrics({
                    f"model/{model_name}/{key}": value
                })
        
        # Time the training
        start_time = time.time()
        model = train_model(
            model_name.split('_')[0],  # Extract base model name
            train_loader,
            val_loader,
            tb_logger=tb_logger,
            **model_kwargs
        )
        training_time = time.time() - start_time
        
        # Log training time
        logger.info(f"{model_name} - Training time: {training_time:.2f}s")
        tb_logger.log_metrics({
            f"benchmark/{model_name}/training_time": training_time
        })
        
        # Get accuracy from validation set
        model.model.eval() if hasattr(model, 'model') else model.eval()
        correct = 0
        total = 0
        
        # For confusion matrix
        all_true_labels = []
        all_pred_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                if hasattr(model, 'predict'):  # HOG classifier
                    outputs = model.predict(images)
                    correct += sum(outputs == labels)
                    total += len(labels)
                    
                    # Store for confusion matrix
                    all_true_labels.extend(labels)
                    all_pred_labels.extend(outputs)
                else:  # CNN classifier
                    images = torch.stack([model.preprocess_image(img) for img in images]).to(model.device)
                    labels = torch.tensor(labels, dtype=torch.long).to(model.device)
                    outputs = model.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Store for confusion matrix
                    all_true_labels.extend(labels.cpu().numpy())
                    all_pred_labels.extend(predicted.cpu().numpy())
        
        accuracy = correct / total
        
        # Log accuracy
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}")
        tb_logger.log_metrics({
            f"benchmark/{model_name}/accuracy": accuracy
        })
        
        # Generate and log confusion matrix
        cm_path = os.path.join("logs", experiment_name, f"{model_name}_confusion_matrix.png")
        os.makedirs(os.path.dirname(cm_path), exist_ok=True)
        
        plot_confusion_matrix(
            all_true_labels, 
            all_pred_labels, 
            class_names, 
            model_name, 
            cm_path
        )
        
        # Log confusion matrix to TensorBoard
        tb_logger.log_images(f"benchmark/{model_name}/confusion_matrix", 
                            [plt.imread(cm_path)], 
                            0)
        
        # Store results
        training_times.append(training_time)
        accuracies.append(accuracy)
        model_names.append(model_name)
    
    # Log summary metrics
    tb_logger.log_metrics({
        "benchmark/summary/best_accuracy": max(accuracies),
        "benchmark/summary/best_model": model_names[accuracies.index(max(accuracies))],
        "benchmark/summary/fastest_model": model_names[training_times.index(min(training_times))],
        "benchmark/summary/fastest_time": min(training_times)
    })
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Training time plot
    ax1.bar(model_names, training_times)
    ax1.set_title('Training Time Comparison')
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Accuracy plot
    ax2.bar(model_names, accuracies)
    ax2.set_title('Accuracy Comparison')
    ax2.set_ylabel('Accuracy')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join("logs", experiment_name, "benchmark_results.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    
    # Log plot to TensorBoard
    tb_logger.log_images("benchmark/results_plot", [plt.imread(plot_path)], 0)
    
    # Save benchmark results as YAML
    results = {
        "models": model_names,
        "training_times": training_times,
        "accuracies": accuracies,
        "best_model": model_names[accuracies.index(max(accuracies))],
        "best_accuracy": max(accuracies),
        "fastest_model": model_names[training_times.index(min(training_times))],
        "fastest_time": min(training_times)
    }
    
    results_path = os.path.join("logs", experiment_name, "benchmark_results.yaml")
    with open(results_path, 'w') as f:
        yaml.dump(results, f)
    
    # Log results as metrics
    tb_logger.log_metrics({
        "benchmark/results/best_model": results['best_model'],
        "benchmark/results/best_accuracy": results['best_accuracy'],
        "benchmark/results/fastest_model": results['fastest_model'],
        "benchmark/results/fastest_time": results['fastest_time']
    })
    
    # Close logger
    tb_logger.close()
    
    logger.info(f"Benchmark complete. Results saved to {results_path}")
    logger.info(f"Best model: {results['best_model']} with accuracy {results['best_accuracy']:.4f}")
    logger.info(f"Fastest model: {results['fastest_model']} with time {results['fastest_time']:.2f}s")
    
    return results

if __name__ == "__main__":
    # Define models to benchmark
    models_to_test = {
        "hog_svm": {
            "classifier": "svm",
        },
        "hog_lr": {
            "classifier": "logistic_regression",
        },
        "cnn_basic": {
            "input_channels": 3,
            "learning_rate": 0.001,
            "num_epochs": 5
        },
        "cnn_fast": {
            "input_channels": 3,
            "learning_rate": 0.01,
            "num_epochs": 3
        }
    }
    
    run_benchmark(models_to_test)
