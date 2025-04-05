import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
from torchvision import transforms
from torch import optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm

# Import the custom logger from utils
from classification.utils.logger import Logger, get_logger

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaseModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True, device=None):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model = None
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        # Define default transforms (customize if needed)
        self.transform = transforms.Compose([
            transforms.Resize((768, 992)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def forward(self, x):
        """
        Forward pass through the model.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def preprocess_image(self, pil_image):
        """
        Converts a PIL image to a normalized tensor.
        """
        return self.transform(pil_image).unsqueeze(0).to(self.device)
    
    def transform_image(self, image: Image.Image) -> torch.Tensor:
        """Legacy method for compatibility"""
        return self.preprocess_image(image)
    
    def inference(self, pil_image):
        """
        Runs inference on a single PIL image.
        """
        self.eval()
        with torch.no_grad():
            image_tensor = self.preprocess_image(pil_image)
            output = self.forward(image_tensor)
            prediction = torch.argmax(output, dim=1).item()
        return prediction
    
    def run_inference(self, image: torch.Tensor) -> torch.Tensor:
        """Legacy method for compatibility"""
        self.eval()
        with torch.no_grad():
            output = self.forward(image)
        return output
    
    def get_model(self) -> nn.Module:
        """Legacy method for compatibility"""
        return self
    
    def train_model(
        self,
        train_dataset,
        val_dataset,
        epochs=10,
        batch_size=32,
        lr=1e-4,
        criterion=None,
        optimizer_cls=optim.Adam,
        scheduler_cls=None,
        log_interval=10,
        log_dir='logs',
        logger=None,
        experiment_name=None,
        config=None
    ):
        """
        Train the model with given datasets and hyperparameters.
        Uses Logger from classification.utils.logger for tracking metrics.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            criterion: Loss function (defaults to CrossEntropyLoss)
            optimizer_cls: Optimizer class (defaults to Adam)
            scheduler_cls: Learning rate scheduler class
            log_interval: How often to log during training
            log_dir: Directory to save logs
            logger: Optional Logger instance to use
            experiment_name: Name for the experiment if creating a new logger
            config: Configuration dictionary to log
        """
        # Create or use logger
        close_logger = False
        if logger is None:
            close_logger = True
            if experiment_name is None:
                experiment_name = f"{self.model_name}_training"
            
            # Create config if not provided
            if config is None:
                config = {
                    'model_name': self.model_name,
                    'num_classes': self.num_classes,
                    'pretrained': self.pretrained,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': lr,
                    'optimizer': optimizer_cls.__name__,
                    'scheduler': scheduler_cls.__name__ if scheduler_cls else None,
                }
            
            logger = Logger(
                log_dir=log_dir,
                experiment_name=experiment_name,
                config=config,
                enable_tensorboard=True
            )
            logging.info(f"Created new logger for experiment: {experiment_name}")
        
        criterion = criterion or nn.CrossEntropyLoss()
        optimizer = optimizer_cls(self.parameters(), lr=lr)
        scheduler = scheduler_cls(optimizer) if scheduler_cls else None

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Log model graph
        try:
            sample_input = next(iter(train_loader))[0][:1].to(self.device)
            logger.log_model_graph(self, input_size=sample_input.shape)
        except Exception as e:
            logging.warning(f"Could not log model graph: {e}")

        best_val_accuracy = 0.0
        global_step = 0

        for epoch in range(epochs):
            logging.info(f"Epoch {epoch+1}/{epochs}")
            
            # Training phase
            self.train()
            total_loss, correct, total = 0, 0, 0

            for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.forward(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
                
                global_step += 1

                if batch_idx % log_interval == 0 and batch_idx > 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    accuracy = correct / total
                    
                    # Log metrics
                    metrics = {
                        'loss': avg_loss,
                        'accuracy': accuracy
                    }
                    logger.log_metrics(metrics, step=global_step, prefix='train/')
                    
                    logging.info(f"Batch {batch_idx}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

            # Validation phase
            val_loss, val_accuracy = self.evaluate(val_loader, criterion, epoch, logger)
            
            # Log learning rate
            if scheduler:
                logger.log_metrics({'learning_rate': scheduler.get_last_lr()[0]}, step=epoch)
                scheduler.step()
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                model_path = f"{logger.log_dir}/{self.model_name}_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_accuracy,
                }, model_path)
                logging.info(f"Saved best model with validation accuracy: {val_accuracy:.4f} to {model_path}")
        
        if close_logger:
            logger.close()
        
        return best_val_accuracy

    def evaluate(self, val_loader, criterion, epoch=None, logger=None):
        """
        Evaluate the model on validation set.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            epoch: Current epoch for logging
            logger: Logger instance
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.eval()
        val_loss, correct, total = 0, 0, 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.forward(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
                
                # Collect predictions and labels for confusion matrix if needed
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = val_loss / len(val_loader)
        accuracy = correct / total
        
        logging.info(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Log to Logger
        if logger and epoch is not None:
            metrics = {
                'loss': avg_loss,
                'accuracy': accuracy
            }
            logger.log_metrics(metrics, step=epoch, prefix='val/')
            
            # Log a sample of validation predictions
            if len(val_loader) > 0:
                try:
                    images, labels = next(iter(val_loader))
                    images, labels = images[:8].to(self.device), labels[:8]  # Only log a few samples
                    outputs = self.forward(images)
                    preds = outputs.argmax(dim=1)
                    
                    # Convert to CPU for logging
                    images = images.cpu()
                    
                    # Add images with their predictions
                    img_grid = torchvision.utils.make_grid(images, normalize=True)
                    logger.log_images('val/images', img_grid, epoch, dataformats='CHW')
                except Exception as e:
                    logging.warning(f"Could not log validation images: {e}")
        
        return avg_loss, accuracy
    
