##
# @file lsnet.py
# @package classification.training.lsnet
# @brief Large-Small Network (LSNet) for document classification
#
# This module implements a document classifier using LSNet, a lightweight vision network
# that combines large-kernel perception and small-kernel aggregation for efficient
# document classification. Based on the "See Large, Focus Small" strategy inspired by
# human vision systems.
#
# @author Statistical Learning Team
# @date 2025
# @see https://arxiv.org/html/2503.23135v1
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm
import os
import time
import logging

# Import the custom logger from utils
from classification.utils.logger import Logger, get_standard_logger

logger = get_standard_logger("lsnet_classifier")

##
# @brief Large-Small (LS) Convolution module for LSNet
#
# This module implements the core LS convolution operation that combines
# large-kernel perception and small-kernel aggregation for efficient feature extraction.
# Based on the "See Large, Focus Small" strategy from the LSNet paper.
#
class LSConvolution(nn.Module):
    """
    Large-Small (LS) Convolution module as described in the LSNet paper.
    
    This module combines large-kernel perception and small-kernel aggregation
    to efficiently capture a wide range of perceptual information and achieve
    precise feature aggregation.
    """
    
    ##
    # @brief Constructor for LSConvolution module
    # @param in_channels Number of input channels
    # @param out_channels Number of output channels
    # @param large_kernel_size Size of the large kernel for perception
    # @param small_kernel_size Size of the small kernel for aggregation
    # @param stride Stride of the convolution
    # @param padding Padding for the convolution
    #
    def __init__(self, in_channels, out_channels, large_kernel_size=7, small_kernel_size=3, stride=1, padding=None):
        """
        Initialize the LS Convolution module.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            large_kernel_size: Size of the large kernel for perception (default: 7)
            small_kernel_size: Size of the small kernel for aggregation (default: 3)
            stride: Stride of the convolution (default: 1)
            padding: Padding for the convolution (default: None, will be calculated)
        """
        super(LSConvolution, self).__init__()
        
        # Calculate padding if not provided
        if padding is None:
            large_padding = large_kernel_size // 2
            small_padding = small_kernel_size // 2
        else:
            large_padding = padding
            small_padding = padding
        
        # Large kernel perception
        self.large_kernel_conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=large_kernel_size, 
            stride=stride, 
            padding=large_padding, 
            bias=False
        )
        
        # Small kernel aggregation
        self.small_kernel_conv = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=small_kernel_size, 
            stride=1, 
            padding=small_padding, 
            bias=False
        )
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Activation function
        self.relu = nn.ReLU(inplace=False)
    
    ##
    # @brief Forward pass through the LS Convolution module
    # @param x Input tensor of shape [batch_size, in_channels, height, width]
    # @return Output tensor of shape [batch_size, out_channels, height, width]
    #
    def forward(self, x):
        """
        Forward pass through the LS Convolution module.
        
        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width]
            
        Returns:
            Output tensor of shape [batch_size, out_channels, height, width]
        """
        # Large kernel perception
        x = self.large_kernel_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Small kernel aggregation
        x = self.small_kernel_conv(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x

##
# @brief LS Block for LSNet architecture
#
# This block consists of LS Convolution followed by a residual connection,
# forming a basic building block for the LSNet architecture.
#
class LSBlock(nn.Module):
    """
    LS Block as described in the LSNet paper.
    
    This block consists of LS Convolution followed by a residual connection.
    """
    
    ##
    # @brief Constructor for LSBlock
    # @param in_channels Number of input channels
    # @param out_channels Number of output channels
    # @param stride Stride of the convolution
    #
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Initialize the LS Block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride of the convolution (default: 1)
        """
        super(LSBlock, self).__init__()
        
        # LS Convolution
        self.conv = LSConvolution(in_channels, out_channels, stride=stride)
        
        # Residual connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    ##
    # @brief Forward pass through the LS Block
    # @param x Input tensor of shape [batch_size, in_channels, height, width]
    # @return Output tensor of shape [batch_size, out_channels, height, width]
    #
    def forward(self, x):
        """
        Forward pass through the LS Block.
        
        Args_:
            x: Input tensor of shape [batch_size, in_channels, height, width]
            
        Returns:
            Output tensor of shape [batch_size, out_channels, height, width]
        """
        # LS Convolution
        identity = x  # Store original input for residual connection
        out = self.conv(x)
        
        # Residual connection (use standard addition)
        out = out + self.shortcut(identity)
        
        return out

##
# @brief LSNet model for document classification
#
# This model implements the LSNet architecture for document classification,
# using LS Convolution and LS Blocks to achieve efficient and effective
# feature extraction and classification.
#
class LSNet(nn.Module):
    """
    LSNet model as described in the paper.
    
    This model uses LS Convolution and LS Blocks to achieve efficient
    and effective feature extraction and classification.
    """
    
    ##
    # @brief Constructor for LSNet model
    # @param num_classes Number of classes to classify
    # @param model_size Size of the model ('t' for tiny, 's' for small, 'b' for base)
    # @param dropout_rate Dropout probability
    #
    def __init__(self, num_classes=16, model_size='s', dropout_rate=0.5):
        """
        Initialize the LSNet model.
        
        Args:
            num_classes: Number of classes to classify (default: 1000 for ImageNet)
            model_size: Size of the model ('t' for tiny, 's' for small, 'b' for base)
            dropout_rate: Dropout probability (default: 0.5)
        """
        super(LSNet, self).__init__()
        
        # Model configuration based on size
        if model_size == 't':
            self.channels = [64, 128, 256, 512]
            self.num_blocks = [2, 2, 6, 2]
        elif model_size == 's':
            self.channels = [64, 128, 256, 512]
            self.num_blocks = [3, 4, 8, 3]
        elif model_size == 'b':
            self.channels = [64, 128, 256, 512]
            self.num_blocks = [4, 6, 12, 4]
        else:
            raise ValueError(f"Unsupported model size: {model_size}")
        
        # Initial convolution with non-inplace ReLU
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(inplace=False)
        )
        
        # Max pooling
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # LS Blocks
        self.layer1 = self._make_layer(self.channels[0], self.channels[0], self.num_blocks[0])
        self.layer2 = self._make_layer(self.channels[0], self.channels[1], self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.channels[1], self.channels[2], self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.channels[2], self.channels[3], self.num_blocks[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout layer added
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Classification head
        self.fc = nn.Linear(self.channels[3], num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    ##
    # @brief Initialize weights for the model
    #
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    ##
    # @brief Create a layer of LS Blocks
    # @param in_channels Number of input channels
    # @param out_channels Number of output channels
    # @param num_blocks Number of LS Blocks in the layer
    # @param stride Stride of the first LS Block
    # @return Sequential module containing LS Blocks
    #
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        """
        Create a layer of LS Blocks.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_blocks: Number of LS Blocks in the layer
            stride: Stride of the first LS Block (default: 1)
            
        Returns:
            Sequential module containing LS Blocks
        """
        layers = []
        # First block with specified stride
        layers.append(LSBlock(in_channels, out_channels, stride))
        # Remaining blocks with stride 1
        for _ in range(1, num_blocks):
            layers.append(LSBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    ##
    # @brief Forward pass through the LSNet model
    # @param x Input tensor of shape [batch_size, 3, height, width]
    # @return Output tensor of shape [batch_size, num_classes]
    #
    def forward(self, x):
        """
        Forward pass through the LSNet model.
        
        Args:
            x: Input tensor of shape [batch_size, 3, height, width]
            
        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        # Initial convolution
        x = self.conv1(x)
        
        # Max pooling
        x = self.maxpool(x)
        
        # LS Blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Apply dropout before classification head
        x = self.dropout(x)
        
        # Classification head
        x = self.fc(x)
        
        return x

##
# @brief LSNet-based classifier for document classification
#
# This class implements a document classifier using the LSNet model,
# providing methods for training, evaluation, and inference.
#
class LSNetClassifier:
    """
    A classifier using the LSNet model for image classification.
    """
    
    ##
    # @brief Constructor for LSNetClassifier
    # @param num_classes Number of classes to classify
    # @param model_size Size of the LSNet model ('t' for tiny, 's' for small, 'b' for base)
    # @param learning_rate Learning rate for the optimizer
    # @param weight_decay Weight decay for regularization
    # @param device Device to use for training ('cuda' or 'cpu')
    # @param num_epochs Number of training epochs
    # @param batch_size Batch size for training
    # @param num_workers Number of workers for data loading
    # @param pretrained Whether to use pre-trained weights
    # @param freeze_backbone Whether to freeze the backbone
    # @param config Configuration dictionary containing model and training parameters
    #
    def __init__(
        self, 
        num_classes: Optional[int] = None,
        model_size: str = 't',
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        device: Optional[str] = None,
        num_epochs: int = 100,
        batch_size: int = 64,
        num_workers: int = 4,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the LSNet classifier.
        
        Can be initialized either with direct parameters or with a config dictionary.
        If config is provided, it takes precedence over direct parameters.
        
        Args:
            num_classes: Number of classes to classify
            model_size: Size of the LSNet model ('t' for tiny, 's' for small, 'b' for base)
            learning_rate: Learning rate for the optimizer
            weight_decay: Weight decay for regularization
            device: Device to use for training ('cuda' or 'cpu')
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            pretrained: Whether to use pre-trained weights
            freeze_backbone: Whether to freeze the backbone
            config: Configuration dictionary containing model and training parameters
        """
        # Determine dropout_rate: precedence to config, then direct param, then default 0.5
        if config is not None:
            # Initialize from config
            self.num_classes = config['model'].get('num_classes', 1000)
            self.model_size = config['model'].get('name', 't').split('_')[1]  # Extract t/s/b from lsnet_t/s/b
            self.dropout_rate = config['model'].get('dropout_rate', 0.5) # Read dropout from config
            self.pretrained = config['model'].get('pretrained', False)
            self.freeze_backbone = config['model'].get('freeze_backbone', False)
            
            # Training settings
            self.learning_rate = config['training']['optimizer'].get('learning_rate', 0.001)
            self.weight_decay = config['training']['optimizer'].get('weight_decay', 0.0001)
            self.num_epochs = config['training'].get('num_epochs', 100)
            
            # Dataloader settings
            if 'data' in config and 'dataloader' in config['data']:
                self.batch_size = config['data']['dataloader'].get('batch_size', 64)
                self.num_workers = config['data']['dataloader'].get('num_workers', 4)
                self.pin_memory = config['data']['dataloader'].get('pin_memory', True)
                self.shuffle = config['data']['dataloader'].get('shuffle', True)
                self.drop_last = config['data']['dataloader'].get('drop_last', False)
            else:
                logger.warning("No dataloader configuration found, using default values")
                self.batch_size = 64
                self.num_workers = 4
                self.pin_memory = True
                self.shuffle = True
                self.drop_last = False
            
            # Device settings
            device = config['training'].get('device', None)
        else:
            # Initialize from direct parameters
            self.num_classes = num_classes if num_classes is not None else 1000
            self.model_size = model_size
            self.dropout_rate = 0.5 # Default dropout if not using config (can be overridden by config later if needed)
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay
            self.num_epochs = num_epochs
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.pretrained = pretrained
            self.freeze_backbone = freeze_backbone

        # Set device (common for both initialization methods)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Log device information
        logger.info(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
        # Initialize model
        self.model = LSNet(
            num_classes=self.num_classes, 
            model_size=self.model_size, 
            dropout_rate=self.dropout_rate # Pass dropout rate
        )
        self.model.to(self.device)
        
        # Load pre-trained weights if requested
        if self.pretrained:
            self._load_pretrained_weights()
            
        # Freeze backbone if requested
        if self.freeze_backbone:
            self._freeze_backbone()
        
        # Initialize optimizer, scheduler, and loss function
        if config is not None:
            # Config-based initialization
            optimizer_name = config['training']['optimizer'].get('name', 'adam').lower()
            scheduler_name = config['training']['scheduler'].get('name', 'cosine_annealing').lower()
            loss_name = config['training']['loss'].get('name', 'cross_entropy').lower()
            
            # Set optimizer
            if optimizer_name == 'adam':
                self.optimizer = optim.AdamW(
                    self.model.parameters(), 
                    lr=self.learning_rate, 
                    weight_decay=self.weight_decay
                )
            elif optimizer_name == 'sgd':
                self.optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=self.learning_rate,
                    momentum=0.9,
                    weight_decay=self.weight_decay
                )
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")
            
            # Set scheduler
            if scheduler_name == 'cosine_annealing':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, 
                    T_max=self.num_epochs,
                    eta_min=config['training']['scheduler'].get('min_lr', self.learning_rate * 0.01)
                )
            elif scheduler_name == 'reduce_on_plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='max',
                    factor=config['training']['scheduler'].get('factor', 0.5),
                    patience=config['training']['scheduler'].get('patience', 5),
                    verbose=True
                )
            elif scheduler_name == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=config['training']['scheduler'].get('step_size', 30),
                    gamma=config['training']['scheduler'].get('gamma', 0.1)
                )
            else:
                raise ValueError(f"Unsupported scheduler: {scheduler_name}")
            
            # Set loss function
            if loss_name == 'cross_entropy':
                self.criterion = nn.CrossEntropyLoss()
            elif loss_name == 'focal_loss':
                self.criterion = FocalLoss(
                    alpha=config['training']['loss'].get('alpha', 0.25),
                    gamma=config['training']['loss'].get('gamma', 2.0)
                )
            else:
                raise ValueError(f"Unsupported loss function: {loss_name}")
        else:
            # Default initialization
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.num_epochs, 
                eta_min=self.learning_rate * 0.01
            )
            self.criterion = nn.CrossEntropyLoss()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    ##
    # @brief Load pre-trained weights for the model
    #
    def _load_pretrained_weights(self):
        """
        Load pre-trained weights for the model.
        """
        try:
            # Check if pre-trained weights exist
            model_path = f"models/lsnet_{self.model_size}.pth"
            if os.path.exists(model_path):
                logger.info(f"Loading pre-trained weights from {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Check if 'model_state_dict' key exists
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info("Successfully loaded pre-trained model state dict.")
                elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
                     # Try loading directly if it looks like a raw state_dict
                     self.model.load_state_dict(checkpoint)
                     logger.info("Loaded pre-trained weights directly (assumed raw state dict).")
                else:
                    logger.error(f"Checkpoint file {model_path} does not contain 'model_state_dict' key or a valid state dict.")
                    # Decide how to proceed: maybe raise error or just warn and continue without pretraining
                    # For now, just warn and continue
                    logger.warning("Continuing without loading pre-trained weights due to missing key.")
            else:
                logger.warning(f"Pre-trained weights file not found at {model_path}. Skipping pre-trained weights loading.")
        except Exception as e:
            logger.error(f"Error loading pre-trained weights: {e}")
    
    ##
    # @brief Freeze the backbone of the model
    #
    def _freeze_backbone(self):
        """
        Freeze the backbone of the model.
        """
        logger.info("Freezing backbone")
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze the classification head
        for param in self.model.fc.parameters():
            param.requires_grad = True
    
    ##
    # @brief Train the model using DataLoader objects
    # @param train_loader Training DataLoader
    # @param val_loader Validation DataLoader
    # @param tb_logger TensorBoard logger instance for metrics visualization
    # @param **kwargs Additional training parameters
    # @return Validation accuracy or training accuracy
    #
    def train_model(
        self, 
        train_loader, 
        val_loader=None, 
        tb_logger=None,
        **kwargs
    ):
        """
        Train the model using DataLoader objects.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            tb_logger: TensorBoard logger instance for metrics visualization
            **kwargs: Additional training parameters
            
        Returns:
            Validation accuracy or training accuracy
        """
        # Enable anomaly detection in debug mode
        if os.environ.get('DEBUG', '0') == '1':
            torch.autograd.set_detect_anomaly(True)
            logger.info("Gradient anomaly detection enabled")
        
        # Initialize mixed precision training
        scaler = torch.cuda.amp.GradScaler()
        
        # Verify model is on the correct device
        if next(self.model.parameters()).device != self.device:
            logger.warning(f"Model is not on the expected device {self.device}. Moving it now.")
            self.model = self.model.to(self.device)
        
        # Training loop
        best_val_accuracy = 0.0
        patience = 10
        patience_counter = 0
        early_stopping = False
        
        for epoch in range(self.num_epochs):
                
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Process training data with progress bar
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for batch_idx, (images, labels) in enumerate(pbar):
                # Move data to device
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad(set_to_none=True)
                
                try:
                    # Forward pass with automatic mixed precision
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                    
                    # Backward pass and optimize with gradient scaling
                    scaler.scale(loss).backward()
                    
                    # Unscale gradients for any gradient clipping
                    scaler.unscale_(self.optimizer)
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Optimizer step with gradient scaling
                    scaler.step(self.optimizer)
                    scaler.update()
                    
                    # Statistics
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': running_loss / (batch_idx + 1),
                        'acc': 100. * correct / total
                    })
                    
                    # RE-ADD BATCH LOGGING
                    if tb_logger:
                        batch_acc = (predicted == labels).sum().item() / labels.size(0) # Calculate batch accuracy
                        tb_logger.log_metrics({
                            'train/batch_loss': loss.item(),
                            'train/batch_accuracy': 100. * batch_acc, # Log batch accuracy
                        }, step=epoch * len(train_loader) + batch_idx)
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        logger.error(f"GPU out of memory: {str(e)}")
                        continue
                    else:
                        raise e
            
            # Calculate epoch statistics
            epoch_loss = running_loss / total if total > 0 else 0
            epoch_accuracy = 100. * (correct / total if total > 0 else 0) # Use percentage
            
            # Log metrics
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%") # Print percentage
            
            # Log epoch metrics to TensorBoard
            if tb_logger:
                tb_logger.log_metrics({
                    'train/epoch_loss': epoch_loss,
                    'train/epoch_accuracy': epoch_accuracy,
                    # 'train/learning_rate': self.optimizer.param_groups[0]['lr'] # Optional
                }, step=epoch+1) # Use epoch+1 for step
            
            # Validation
            val_accuracy = None
            val_loss = None # Initialize val_loss
            if val_loader is not None:
                val_accuracy, val_loss = self.evaluate(val_loader, tb_logger, epoch + 1) # Pass epoch+1
                # logger.info already prints validation results in evaluate()

                # Early stopping check
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                    
                    # Save best model
                    model_path = f"models/lsnet_{self.model_size}_best.pth"
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'val_accuracy': val_accuracy,
                        'train_accuracy': epoch_accuracy,
                    }, model_path)
                    logger.info(f"Saved best model with validation accuracy: {val_accuracy:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        early_stopping = True
                        logger.info(f"Early stopping triggered after {epoch+1} epochs")
            
            # Learning rate scheduling
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_accuracy if val_loader is not None else epoch_loss)
            else:
                self.scheduler.step()
        
        return best_val_accuracy if val_loader is not None else epoch_accuracy
    
    ##
    # @brief Run inference on a single image
    # @param image Image to classify
    # @return Predicted class label
    #
    def inference(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> int:
        """
        Run inference on a single image.
        
        Args:
            image: Image to classify
            
        Returns:
            Predicted class label
        """
        self.model.eval()
        with torch.no_grad():
            # Preprocess image
            if not isinstance(image, torch.Tensor):
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                image = self.transform(image)
            else:
                image = image.to(self.device)
                
            # Add batch dimension if needed
            if image.dim() == 3:
                image = image.unsqueeze(0)
                
            # Forward pass
            outputs = self.model(image)
            _, predicted = torch.max(outputs.data, 1)
            
            return predicted.item()
    
    ##
    # @brief Get class probabilities for an image
    # @param image Image to classify
    # @return Class probabilities as a numpy array
    #
    def predict_proba(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Get class probabilities for an image.
        
        Args:
            image: Image to classify
            
        Returns:
            Class probabilities as a numpy array
        """
        self.model.eval()
        with torch.no_grad():
            # Preprocess image
            if not isinstance(image, torch.Tensor):
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                image = self.transform(image)
            else:
                image = image.to(self.device)
                
            # Add batch dimension if needed
            if image.dim() == 3:
                image = image.unsqueeze(0)
                
            # Forward pass
            outputs = self.model(image)
            probabilities = F.softmax(outputs, dim=1)
            
            return probabilities.cpu().numpy()[0]
    
    ##
    # @brief Evaluate the model on validation set
    # @param val_loader Validation DataLoader
    # @param tb_logger TensorBoard logger instance for metrics visualization
    # @param step Current step for logging
    # @return Validation accuracy and loss
    #
    def evaluate(self, val_loader, tb_logger=None, step=None):
        """
        Evaluate the model on validation set.
        
        Args:
            val_loader: Validation DataLoader
            tb_logger: TensorBoard logger instance for metrics visualization
            step: Current step for logging
            
        Returns:
            Validation accuracy
        """
        # Verify model is on the correct device *once* at the beginning
        if next(self.model.parameters()).device != self.device:
            logger.warning(f"Model is not on the expected device {self.device} during evaluation. Moving it now.")
            self.model.to(self.device)
            
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            for images, labels in tqdm(val_loader, desc="Validation"):
                # Move data to device
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Collect predictions and labels for confusion matrix if needed
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate accuracy and loss
        avg_loss = val_loss / total if total > 0 else 0
        accuracy = 100. * (correct / total if total > 0 else 0) # Use percentage
        
        # Log metrics to standard logger
        logger.info(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%") # Print percentage
        
        # Log validation epoch metrics to TensorBoard if available
        if tb_logger and step is not None:
            tb_logger.log_metrics({
                'val/loss': avg_loss,
                'val/accuracy': accuracy
            }, step=step)
        
        return accuracy, avg_loss # Return percentage accuracy and avg loss
    
    ##
    # @brief Save the model to disk
    # @param path Path to save the model to
    #
    def save(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model to
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Keep model on its original device
        # self.model.cpu()  # Removed: Keep model on its device
        
        # Prepare checkpoint dictionary
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'model_size': self.model_size,
            'dropout_rate': self.dropout_rate, # Save dropout rate
            # Training state (optional, but good for resuming)
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            # Hyperparameters (optional)
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'num_epochs': self.num_epochs, # Might be current epoch if saving mid-training
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'device': str(self.device) # Save the device info
        }
        
        # Save the checkpoint
        torch.save(checkpoint, path)
        
        # Use the module-level logger instance (defined at the top)
        logger.info(f"Model saved to {path}")
    
    ##
    # @brief Load the model from disk
    # @param path Path to load the model from
    #
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        # Load model state
        checkpoint = torch.load(path, map_location=self.device)
        
        # Update model parameters stored in checkpoint
        self.num_classes = checkpoint.get('num_classes', self.num_classes) # Use get with default
        self.model_size = checkpoint.get('model_size', self.model_size)
        self.dropout_rate = checkpoint.get('dropout_rate', 0.5) # Load dropout rate
        # Training parameters (optional to load, depends on use case)
        self.learning_rate = checkpoint.get('learning_rate', self.learning_rate)
        self.weight_decay = checkpoint.get('weight_decay', self.weight_decay)
        # self.num_epochs = checkpoint.get('num_epochs', self.num_epochs)
        # self.batch_size = checkpoint.get('batch_size', self.batch_size)
        # self.num_workers = checkpoint.get('num_workers', self.num_workers)
        
        # Recreate model with loaded parameters
        self.model = LSNet(
            num_classes=self.num_classes, 
            model_size=self.model_size,
            dropout_rate=self.dropout_rate
        )
        # Ensure model is on the correct device BEFORE loading state dict
        self.model.to(self.device) 
        
        # Load model state dict
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
             logger.warning(f"Checkpoint {path} missing 'model_state_dict'. Attempting to load checkpoint directly.")
             try:
                 self.model.load_state_dict(checkpoint)
             except Exception as e:
                 logger.error(f"Failed to load state dict directly from checkpoint {path}: {e}")
                 # Handle error appropriately, e.g., raise or return

        # Recreate and load optimizer state (ensure model parameters are passed correctly)
        # Note: Recreating optimizer might reset state if parameters changed significantly
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        if 'optimizer_state_dict' in checkpoint:
             self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
             logger.warning(f"Checkpoint {path} missing 'optimizer_state_dict'. Optimizer state not loaded.")

        # Recreate and load scheduler state (optional, depends on whether you continue training)
        # Assuming CosineAnnealingLR for simplicity, adjust if using other schedulers
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.num_epochs, # Ensure num_epochs is appropriate
            eta_min=self.learning_rate * 0.01 
        )
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
             self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
             logger.warning(f"Checkpoint {path} missing 'scheduler_state_dict' or scheduler not defined. Scheduler state not loaded.")

        # Use the module-level logger instance (defined at the top)
        logger.info(f"Model loaded from {path}") 