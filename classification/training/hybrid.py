##
# @file hybrid.py
# @package classification.training.hybrid
# @brief Hybrid model combining LSNet for image processing and EAML for text processing
#
# This module implements a hybrid architecture that combines the strengths of
# LSNet for image understanding and EAML for OCR text processing.
# It uses LSNet as the image encoder and EAML's text processing branch.
#
# @author Statistical Learning Team
# @date 2025
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
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import necessary components
from classification.training.lsnet import LSNetClassifier
from classification.utils.logger import get_standard_logger

# Get logger
logger = get_standard_logger("hybrid_classifier")

##
# @brief Hybrid model combining LSNet for image processing and EAML for text processing
#
# This class implements a hybrid architecture that combines the strengths of
# LSNet for image understanding and EAML for OCR text processing.
# It uses LSNet as the image encoder and EAML's text processing branch.
#
class HybridModel(nn.Module):
    """
    Hybrid model that combines LSNet for image processing and EAML for text processing.
    This model leverages the strengths of both architectures to create a powerful
    document classifier that uses both image and OCR text modalities.
    """
    
    ##
    # @brief Constructor for HybridModel
    # @param num_classes Number of document classes
    # @param vocab_size Size of the vocabulary for text embedding
    # @param embedding_dim Dimension of the word embeddings
    # @param word_hidden_dim Dimension of the word-level GRU hidden state
    # @param sent_hidden_dim Dimension of the sentence-level GRU hidden state
    # @param image_channels Number of input image channels (typically 3 for RGB)
    # @param lsnet_model_size LSNet model size ('t', 's', or 'b')
    # @param dropout Dropout rate for regularization
    #
    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        embedding_dim: int = 100,
        word_hidden_dim: int = 50,
        sent_hidden_dim: int = 50,
        image_channels: int = 3,
        lsnet_model_size: str = 's',
        dropout: float = 0.5
    ):
        super(HybridModel, self).__init__()
        self.num_classes = num_classes
        
        # LSNet image encoder (using the base model without classifier head)
        self.lsnet = LSNetClassifier(num_classes=num_classes, model_size=lsnet_model_size).model
        # Remove the classifier head to get only the feature extractor
        self.image_feature_dim = self.lsnet.fc.in_features
        self.lsnet.fc = nn.Identity()  # Replace classifier with identity to get features
        
        # EAML text processing branch
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Word level
        self.word_gru = nn.GRU(embedding_dim, word_hidden_dim, bidirectional=True, batch_first=True)
        self.word_attention = nn.Sequential(
            nn.Linear(word_hidden_dim * 2, word_hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(word_hidden_dim * 2, 1, bias=False)
        )
        # Sentence level
        self.sent_gru = nn.GRU(word_hidden_dim * 2, sent_hidden_dim, bidirectional=True, batch_first=True)
        self.sent_attention = nn.Sequential(
            nn.Linear(sent_hidden_dim * 2, sent_hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(sent_hidden_dim * 2, 1, bias=False)
        )
        
        # Text feature dimension after bidirectional GRU
        self.text_feature_dim = sent_hidden_dim * 2
        
        # Multimodal fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.image_feature_dim + self.text_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classifier
        self.classifier = nn.Linear(256, num_classes)
        
    ##
    # @brief Forward pass through the hybrid model
    # @param images Input images tensor of shape (batch_size, channels, height, width)
    # @param docs Input text tensor of shape (batch_size, num_sentences, max_sent_length)
    # @return Output tensor of shape (batch_size, num_classes)
    #
    def forward(self, images, docs):
        # Process images with LSNet
        image_features = self.lsnet(images)
        
        # Process text with EAML text branch
        batch_size, num_sentences, max_sent_length = docs.size()
        
        # Ensure no negative indices in docs (which would cause embedding errors)
        docs = torch.clamp(docs, min=0, max=self.embedding.num_embeddings-1)
        
        # Reshape for word-level processing
        docs_reshaped = docs.view(batch_size * num_sentences, max_sent_length)
        word_embeds = self.embedding(docs_reshaped)
        
        # Word-level GRU
        word_gru_out, _ = self.word_gru(word_embeds)
        
        # Word-level attention
        word_attn_weights = self.word_attention(word_gru_out).squeeze(-1)
        word_attn_weights = F.softmax(word_attn_weights, dim=1)
        word_contexts = torch.bmm(word_attn_weights.unsqueeze(1), word_gru_out).squeeze(1)
        
        # Reshape sentence representations for sentence-level processing
        sent_reps = word_contexts.view(batch_size, num_sentences, -1)
        
        # Sentence-level GRU
        sent_gru_out, _ = self.sent_gru(sent_reps)
        
        # Sentence-level attention
        sent_attn_weights = self.sent_attention(sent_gru_out).squeeze(-1)
        sent_attn_weights = F.softmax(sent_attn_weights, dim=1)
        text_features = torch.bmm(sent_attn_weights.unsqueeze(1), sent_gru_out).squeeze(1)
        
        # Fusion of image and text features
        combined_features = torch.cat([image_features, text_features], dim=1)
        fused_features = self.fusion_layer(combined_features)
        
        # Classification
        output = self.classifier(fused_features)
        
        return output

class HybridClassifier:
    """
    Classifier that uses the HybridModel for document classification.
    This classifier combines the strengths of LSNet for image processing
    and EAML for OCR text processing.
    """
    
    ##
    # @brief Constructor for HybridClassifier
    # @param num_classes Number of document classes
    # @param vocab_size Size of the vocabulary for text embedding
    # @param embedding_dim Dimension of the word embeddings
    # @param word_hidden_dim Dimension of the word-level GRU hidden state
    # @param sent_hidden_dim Dimension of the sentence-level GRU hidden state
    # @param lsnet_model_size LSNet model size ('t', 's', or 'b')
    # @param dropout Dropout rate for regularization
    # @param learning_rate Learning rate for the optimizer
    # @param device Device to use for training (CPU or GPU)
    # @param num_epochs Number of training epochs
    #
    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        embedding_dim: int = 100,
        word_hidden_dim: int = 50,
        sent_hidden_dim: int = 50,
        lsnet_model_size: str = 's',
        dropout: float = 0.5,
        learning_rate: float = 0.0005,
        device: Optional[str] = None,
        num_epochs: int = 50
    ):
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.word_hidden_dim = word_hidden_dim
        self.sent_hidden_dim = sent_hidden_dim
        self.lsnet_model_size = lsnet_model_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Create model
        self.model = HybridModel(
            num_classes=num_classes,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            word_hidden_dim=word_hidden_dim,
            sent_hidden_dim=sent_hidden_dim,
            lsnet_model_size=lsnet_model_size,
            dropout=dropout
        ).to(self.device)
        
        # Set optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Set loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Set learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )

class HybridTrainer:
    """
    Trainer for the HybridClassifier model.
    This trainer handles training, evaluation, and inference for the hybrid model.
    """
    
    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        embedding_dim: int = 100,
        word_hidden_dim: int = 50,
        sent_hidden_dim: int = 50,
        lsnet_model_size: str = 's',
        dropout: float = 0.5,
        learning_rate: float = 0.0005,
        weight_decay: float = 0.01,
        device: Optional[str] = None,
        num_epochs: int = 50,
        save_dir: str = "models/hybrid",
        patience: int = 10,
        mixed_precision: bool = True,
        use_ocr_text: bool = True
    ):
        """
        Initialize the HybridTrainer.
        
        Args:
            num_classes: Number of document classes
            vocab_size: Size of the vocabulary for text embedding
            embedding_dim: Dimension of the word embeddings
            word_hidden_dim: Dimension of the word-level GRU hidden state
            sent_hidden_dim: Dimension of the sentence-level GRU hidden state
            lsnet_model_size: LSNet model size ('t', 's', or 'b')
            dropout: Dropout rate for regularization
            learning_rate: Learning rate for the optimizer
            weight_decay: Weight decay for the optimizer
            device: Device to use for training (CPU or GPU)
            num_epochs: Number of training epochs
            save_dir: Directory to save model checkpoints
            patience: Number of epochs to wait for improvement before early stopping
            mixed_precision: Whether to use mixed precision training
            use_ocr_text: Whether to use OCR text from the dataframe
        """
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.word_hidden_dim = word_hidden_dim
        self.sent_hidden_dim = sent_hidden_dim
        self.lsnet_model_size = lsnet_model_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.patience = patience
        self.mixed_precision = mixed_precision
        self.use_ocr_text = use_ocr_text
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize the model
        self.classifier = HybridClassifier(
            num_classes=num_classes,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            word_hidden_dim=word_hidden_dim,
            sent_hidden_dim=sent_hidden_dim,
            lsnet_model_size=lsnet_model_size,
            dropout=dropout,
            learning_rate=learning_rate,
            device=device,
            num_epochs=num_epochs
        )
        
        # Set up scaler for mixed precision training if using GPU
        self.scaler = None
        if self.mixed_precision and self.device.type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
            logger.info("Using mixed precision training with gradient scaling")
    
    def _process_batch(self, batch, batch_idx=None):
        """
        Process batches of different formats into standardized inputs.
        
        Args:
            batch: Batch from DataLoader
            batch_idx: Optional batch index for debugging
            
        Returns:
            Tuple of (docs, images, labels) on device, or None if batch cannot be processed
        """
        # Debug batch format (only for the first few batches)
        if batch_idx is not None and batch_idx < 3:
            logger.debug(f"Batch {batch_idx} type: {type(batch)}")
            if isinstance(batch, list):
                logger.debug(f"List length: {len(batch)}")
                for i, item in enumerate(batch):
                    logger.debug(f"  Item {i} type: {type(item)}, shape: {item.shape if hasattr(item, 'shape') else 'No shape'}")
        
        # Get data and move to device
        if isinstance(batch, dict):
            # Handle dict-style batches
            images = batch['image'].to(self.device)
            docs = batch['text'].to(self.device)
            labels = batch['label'].to(self.device)
            return docs, images, labels
        elif isinstance(batch, tuple) and len(batch) == 3:
            # Handle tuple-style batches (text, image, label) from MinioMultiModalDataset
            docs, images, labels = batch
            docs = docs.to(self.device)
            images = images.to(self.device)
            labels = labels.to(self.device)
            return docs, images, labels
        elif isinstance(batch, list):
            # Handle list-style batches
            if len(batch) == 3:
                docs, images, labels = batch
                docs = docs.to(self.device)
                images = images.to(self.device)
                labels = labels.to(self.device)
                return docs, images, labels
            else:
                logger.error(f"List batch has unexpected length: {len(batch)}")
        else:
            logger.error(f"Unexpected batch format: {type(batch)}")
        
        return None

    def train_model(
        self,
        train_loader,
        val_loader=None,
        tb_logger=None,
        **kwargs
    ):
        """
        Train the hybrid model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            tb_logger: Optional TensorBoard logger
            **kwargs: Additional arguments
            
        Returns:
            Best validation accuracy
        """
        logger.info("Starting training of hybrid model")
        
        # Get model, optimizer, criterion, and scheduler from classifier
        model = self.classifier.model
        optimizer = self.classifier.optimizer
        criterion = self.classifier.criterion
        scheduler = self.classifier.scheduler
        
        # Set up TensorBoard writer if tb_logger is not provided
        if tb_logger is None:
            tb_logger = SummaryWriter(os.path.join(self.save_dir, 'logs'))
        else:
            tb_logger = tb_logger
        
        # Training loop
        best_val_accuracy = 0.0
        best_model_path = os.path.join(self.save_dir, 'best_model.pth')
        epochs_without_improvement = 0
        
        for epoch in range(self.num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            all_preds = []
            all_labels = []
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for batch_idx, batch in enumerate(progress_bar):
                # Get data and move to device
                batch_data = self._process_batch(batch, batch_idx)
                if batch_data is None:
                    continue
                    
                docs, images, labels = batch_data
                
                # Debug info for the first few batches
                if batch_idx < 2:
                    logger.debug(f"Batch {batch_idx} - docs shape: {docs.shape}, images shape: {images.shape}, labels shape: {labels.shape}")
                    logger.debug(f"Docs min/max values: {docs.min().item()}/{docs.max().item()}")
                    logger.debug(f"Labels: {labels}")
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass with mixed precision if enabled
                if self.mixed_precision and self.device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        outputs = model(images, docs)
                        loss = criterion(outputs, labels)
                    
                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # Regular forward and backward pass
                    outputs = model(images, docs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                # Update statistics
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'acc': accuracy_score(all_labels, all_preds)
                })
                
                # Log batch-level metrics to TensorBoard
                if tb_logger:
                    batch_acc = (preds == labels).sum().item() / labels.size(0)
                    tb_logger.log_metrics({
                        'train/batch_loss': loss.item(),
                        'train/batch_accuracy': 100. * batch_acc,
                        'train/learning_rate': optimizer.param_groups[0]['lr']
                    }, step=epoch * len(train_loader) + batch_idx)
            
            # Calculate training metrics
            train_accuracy = accuracy_score(all_labels, all_preds)
            train_loss = running_loss / len(train_loader)
            
            # Log training metrics
            tb_logger.log_metrics({
                'train/epoch_loss': train_loss,
                'train/epoch_accuracy': train_accuracy,
                'train/learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            
            # Validation phase
            if val_loader is not None:
                val_accuracy, val_loss = self.evaluate(val_loader, tb_logger, epoch)
                
                # Scheduler step
                scheduler.step(val_accuracy)
                
                # Log validation metrics
                logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
                
                # Save best model
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'val_accuracy': val_accuracy,
                        'train_accuracy': train_accuracy,
                        'scaler': self.scaler.state_dict() if self.scaler else None
                    }, best_model_path)
                    logger.info(f"Saved best model with validation accuracy: {val_accuracy:.4f}")
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    logger.info(f"No improvement for {epochs_without_improvement} epochs")
                
                # Early stopping
                if epochs_without_improvement >= self.patience:
                    logger.info(f"Early stopping after {epoch+1} epochs")
                    break
            else:
                # If no validation data, save model after each epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'train_accuracy': train_accuracy,
                    'scaler': self.scaler.state_dict() if self.scaler else None
                }, os.path.join(self.save_dir, f'model_epoch_{epoch+1}.pth'))
        
        # Load best model if validation data was provided
        if val_loader is not None:
            logger.info(f"Loading best model with validation accuracy: {best_val_accuracy:.4f}")
            checkpoint = torch.load(best_model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Close TensorBoard writer if we created it
        if tb_logger is None:
            tb_writer.close()
        
        return best_val_accuracy if val_loader is not None else train_accuracy
    
    def evaluate(self, data_loader, tb_logger=None, step=None):
        """
        Evaluate the hybrid model on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation data
            tb_logger: Optional TensorBoard logger
            step: Current step (epoch) for logging
            
        Returns:
            Tuple of (accuracy, loss)
        """
        logger.info("Evaluating hybrid model")
        
        # Get model and criterion from classifier
        model = self.classifier.model
        criterion = self.classifier.criterion
        
        # Evaluation mode
        model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                # Get data and move to device
                batch_data = self._process_batch(batch)
                if batch_data is None:
                    continue
                    
                docs, images, labels = batch_data
                
                # Forward pass
                outputs = model(images, docs)
                loss = criterion(outputs, labels)
                
                # Update statistics
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        loss = running_loss / len(data_loader)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        # Log metrics to TensorBoard
        if tb_logger is not None and step is not None:
            tb_logger.log_metrics({
                'val/loss': loss,
                'val/accuracy': accuracy,
                'val/precision': precision,
                'val/recall': recall,
                'val/f1': f1
            }, step=step)
        
        logger.info(f"Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return accuracy, loss
    
    def inference(self, image, text):
        """
        Perform inference with the hybrid model.
        
        Args:
            image: Input image (PIL.Image, numpy array, or tensor)
            text: Input text as processed and tokenized tensor
            
        Returns:
            Predicted class index
        """
        # Get model from classifier
        model = self.classifier.model
        
        # Evaluation mode
        model.eval()
        
        # Preprocess image if necessary
        if isinstance(image, Image.Image) or isinstance(image, np.ndarray):
            # Define image transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            image = transform(image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
        
        # Process text if necessary
        if isinstance(text, torch.Tensor):
            text = text.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = model(image, text)
            _, preds = torch.max(outputs, 1)
        
        return preds.item()
    
    def predict_proba(self, image, text):
        """
        Get class probabilities for the input.
        
        Args:
            image: Input image (PIL.Image, numpy array, or tensor)
            text: Input text as processed and tokenized tensor
            
        Returns:
            Class probabilities as numpy array
        """
        # Get model from classifier
        model = self.classifier.model
        
        # Evaluation mode
        model.eval()
        
        # Preprocess image if necessary
        if isinstance(image, Image.Image) or isinstance(image, np.ndarray):
            # Define image transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            image = transform(image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
        
        # Process text if necessary
        if isinstance(text, torch.Tensor):
            text = text.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = model(image, text)
            probs = F.softmax(outputs, dim=1)
        
        return probs.cpu().numpy()
    
    def save(self, path=None):
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model (default: best_model.pth in save_dir)
        """
        if path is None:
            path = os.path.join(self.save_dir, 'best_model.pth')
        
        # Get model and optimizer from classifier
        model = self.classifier.model
        optimizer = self.classifier.optimizer
        scheduler = self.classifier.scheduler
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'model_config': {
                'num_classes': self.num_classes,
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'word_hidden_dim': self.word_hidden_dim,
                'sent_hidden_dim': self.sent_hidden_dim,
                'lsnet_model_size': self.lsnet_model_size,
                'dropout': self.dropout,
                'use_ocr_text': self.use_ocr_text
            },
            'scaler': self.scaler.state_dict() if self.scaler else None
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path, map_location=None):
        """
        Load a model from a file.
        
        Args:
            path: Path to load the model from
            map_location: Device to map the model to (default: self.device)
        """
        if map_location is None:
            map_location = self.device
        
        # Load checkpoint
        checkpoint = torch.load(path, map_location=map_location)
        
        # Update model configuration if provided
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            self.num_classes = config.get('num_classes', self.num_classes)
            self.vocab_size = config.get('vocab_size', self.vocab_size)
            self.embedding_dim = config.get('embedding_dim', self.embedding_dim)
            self.word_hidden_dim = config.get('word_hidden_dim', self.word_hidden_dim)
            self.sent_hidden_dim = config.get('sent_hidden_dim', self.sent_hidden_dim)
            self.lsnet_model_size = config.get('lsnet_model_size', self.lsnet_model_size)
            self.dropout = config.get('dropout', self.dropout)
            self.use_ocr_text = config.get('use_ocr_text', self.use_ocr_text)
            
            # Recreate classifier with updated configuration
            self.classifier = HybridClassifier(
                num_classes=self.num_classes,
                vocab_size=self.vocab_size,
                embedding_dim=self.embedding_dim,
                word_hidden_dim=self.word_hidden_dim,
                sent_hidden_dim=self.sent_hidden_dim,
                lsnet_model_size=self.lsnet_model_size,
                dropout=self.dropout,
                learning_rate=self.learning_rate,
                device=self.device,
                num_epochs=self.num_epochs
            )
        
        # Load model state
        self.classifier.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            self.classifier.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            self.classifier.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state if available
        if 'scaler' in checkpoint and checkpoint['scaler'] is not None and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        logger.info(f"Model loaded from {path}")

    @staticmethod
    def create_dataloader(
        dataframe,
        bucket_name,
        tokenizer=None,
        vocab=None,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        max_sentences=15,
        max_sent_length=50,
        image_transform=None,
        batch_tokenize=True,
        **kwargs
    ):
        """
        Create a DataLoader for the HybridModel using MinioMultiModalDataset.
        
        Args:
            dataframe: DataFrame with 'image' URL, 'label', and 'ocr_text' columns
            bucket_name: MinIO bucket name
            tokenizer: Tokenizer for text processing
            vocab: Vocabulary for text processing
            batch_size: Batch size for the DataLoader
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes for the DataLoader
            max_sentences: Maximum number of sentences to keep from OCR text
            max_sent_length: Maximum number of tokens per sentence
            image_transform: Transformations for the images
            batch_tokenize: Whether to tokenize in batches (more efficient)
            **kwargs: Additional arguments for MinioMultiModalDataset
            
        Returns:
            DataLoader for the HybridModel
        """
        from classification.data.minio_dataset import MinioMultiModalDataset
        from torch.utils.data import DataLoader
        
        # Create dataset
        dataset = MinioMultiModalDataset(
            dataframe=dataframe,
            bucket_name=bucket_name,
            image_transform=image_transform,
            tokenizer=tokenizer,
            vocab=vocab,
            max_sentences=max_sentences,
            max_sent_length=max_sent_length,
            batch_tokenize=batch_tokenize,
            **kwargs
        )
        
        # Create data loader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        ) 