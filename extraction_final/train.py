#!/usr/bin/env python3
import os
import json
import time
import argparse
import logging
import torch
import numpy as np
import warnings
from torch.utils.data import DataLoader, Subset, Dataset
from torch.optim import AdamW
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3ImageProcessor,
    LayoutLMv3TokenizerFast,
    get_linear_schedule_with_warmup,
    AutoConfig
)
from sklearn.metrics import f1_score, precision_score, recall_score
from dataset import InvoiceDataset
from tqdm import tqdm
import random
from typing import Optional

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress FutureWarning about device argument
warnings.filterwarnings("ignore", category=FutureWarning, message="The `device` argument is deprecated and will be removed in v5 of Transformers.")

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# MPS compatibility fix for pickling issue
def _dummy_worker_init_fn(worker_id):
    """Required to work around MPS pickling issues"""
    pass

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Additionally ensure CUDA operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ModelTrainer:
    def __init__(
        self,
        data_dir: str,
        output_dir: str = "output",
        model_name: str = "microsoft/layoutlmv3-base",
        num_labels: int = 7,
        batch_size: int = 2,
        learning_rate: float = 1e-5,
        epochs: int = 10,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        image_size: int = 224,
        max_length: int = 512,
        max_grad_norm: float = 1.0,
        id2label: dict = None,
        label2id: dict = None,
        fast_mode: bool = False,
        use_mps: bool = True,
        subset_size: Optional[int] = None,
        force_cpu: bool = False,
        gradient_accumulation_steps: int = 1,
        low_memory: bool = False,
        tiny_model: bool = False,
        max_steps: int = 2000,
        seed: int = 42
    ):
        """
        Initialize the model trainer
        
        Args:
            data_dir: Directory containing the dataset
            output_dir: Directory to save the model
            model_name: Name or path of the pretrained model
            num_labels: Number of entity types to predict
            batch_size: Batch size for training
            learning_rate: Learning rate
            epochs: Number of training epochs
            warmup_steps: Number of warmup steps for learning rate scheduler
            weight_decay: Weight decay for AdamW optimizer
            image_size: Size of the input images (height and width)
            max_length: Maximum sequence length for tokenizer
            max_grad_norm: Maximum gradient norm for gradient clipping
            id2label: Mapping from label ids to label names
            label2id: Mapping from label names to label ids
            fast_mode: Enable faster training (less validation)
            use_mps: Whether to use MPS (Apple Silicon GPU) if available
            subset_size: If provided, use only a subset of the data for faster training
            force_cpu: Force using CPU even if GPU is available (sometimes faster for small models)
            gradient_accumulation_steps: Number of steps to accumulate gradients before performing an update
            low_memory: If True, use smaller image size and other memory-saving techniques
            tiny_model: If True, use a tiny custom model instead of LayoutLMv3 for testing training pipeline
            max_steps: Maximum number of training steps
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.model_name = model_name
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.max_length = max_length
        self.max_grad_norm = max_grad_norm
        self.id2label = id2label
        self.label2id = label2id
        self.fast_mode = fast_mode
        self.subset_size = subset_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.low_memory = low_memory
        self.tiny_model = tiny_model
        self.max_steps = max_steps
        self.seed = seed
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Adjust image size for low memory mode
        if low_memory:
            # Use smaller image size for faster processing and less memory
            self.image_size = 224  # Keep at 224 to match model's position embeddings
            logger.info(f"Low memory mode enabled, using standard image size: {self.image_size}")
        else:
            self.image_size = image_size
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set device - handle MPS with special care
        if force_cpu:
            self.device = torch.device("cpu")
            logger.info("Forced CPU usage as requested")
        elif use_mps and torch.backends.mps.is_available():
            # Make sure MPS is properly initialized
            torch.mps.empty_cache()
            self.device = torch.device("mps")
            logger.info("Using MPS (Apple Silicon GPU) for training")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA for training")
        else:
            self.device = torch.device("cpu")
            logger.warning("Using CPU for training. This will be slow!")
        
        # Initialize tokenizer and image processor
        self.tokenizer = LayoutLMv3TokenizerFast.from_pretrained(self.model_name)
        
        # Use a slightly smaller image size for faster processing
        self.image_processor = LayoutLMv3ImageProcessor(size={"height": self.image_size, "width": self.image_size})
        
        # Initialize dataloaders
        self._setup_dataloaders()
        
        # Prepare model
        self._setup_model()
        
    def _setup_dataloaders(self):
        """Setup train, validation, and test dataloaders"""
        # Create train dataset
        train_dataset = InvoiceDataset(
            data_dir=self.data_dir,
            split="train",
            tokenizer=self.tokenizer,
            image_processor=self.image_processor
        )
        
        # Create validation dataset
        val_dataset = InvoiceDataset(
            data_dir=self.data_dir,
            split="val",
            tokenizer=self.tokenizer,
            image_processor=self.image_processor
        )
        
        # Create test dataset
        test_dataset = InvoiceDataset(
            data_dir=self.data_dir,
            split="test",
            tokenizer=self.tokenizer,
            image_processor=self.image_processor
        )
        
        # Get the original dataset sizes before subset selection
        original_train_size = len(train_dataset)
        original_val_size = len(val_dataset)
        original_test_size = len(test_dataset)
        
        # Use subsets if requested (for faster development/testing)
        if self.subset_size is not None:
            logger.info(f"Using subset of data for faster training: {self.subset_size} samples per split")
            
            # Create subsets by sampling indices 
            random.seed(42)  # For reproducibility
            
            # Ensure subset size is not larger than dataset
            train_subset_size = min(self.subset_size, original_train_size)
            val_subset_size = min(self.subset_size // 2, original_val_size)
            test_subset_size = min(self.subset_size // 2, original_test_size)
            
            # Sample random indices
            train_indices = random.sample(range(original_train_size), train_subset_size)
            val_indices = random.sample(range(original_val_size), val_subset_size)
            test_indices = random.sample(range(original_test_size), test_subset_size)
            
            # Create custom dataset and dataloader setup instead of using Subset
            # This avoids collation issues with Subset
            
            # Function to sample from dataset based on indices
            def create_subset_dataset(dataset, indices):
                """Creates a new dataset with only the samples at the given indices"""
                subset_data = []
                for idx in indices:
                    subset_data.append(dataset[idx])
                return subset_data
            
            # Create data subsets
            train_data = create_subset_dataset(train_dataset, train_indices)
            val_data = create_subset_dataset(val_dataset, val_indices)
            test_data = create_subset_dataset(test_dataset, test_indices)
            
            # Update sizes to reflect subset sizes
            train_size = len(train_data)
            val_size = len(val_data)
            test_size = len(test_data)
        else:
            # Use full datasets
            train_data = train_dataset
            val_data = val_dataset
            test_data = test_dataset
            
            # Original sizes
            train_size = original_train_size
            val_size = original_val_size
            test_size = original_test_size
        
        # Update id2label and label2id if not provided
        if self.id2label is None or self.label2id is None:
            self.id2label = train_dataset.id2label
            self.label2id = train_dataset.label2id
            self.num_labels = len(self.id2label)
            
            # Save label map to output directory
            with open(os.path.join(self.output_dir, "label_map.json"), "w") as f:
                json.dump(self.id2label, f, indent=2)
        
        # Determine number of workers and multiprocessing settings
        # MPS doesn't work well with multiprocessing, but CPU can use it
        num_workers = 0  # Default to 0 for safety with MPS
        if self.device.type == 'cpu':
            # For CPU, using multiprocessing can help
            num_workers = min(4, os.cpu_count() or 1)
        
        # Create dataloaders - be careful with multiprocessing on MPS
        if self.subset_size is not None:
            # When using subset data, we need a custom collation approach
            # because we have a list of precomputed items
            
            # Custom dataloader for pre-computed subsets
            class PrecomputedDataset(Dataset):
                def __init__(self, data):
                    self.data = data
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    return self.data[idx]
            
            # Create dataloaders from precomputed data
            self.train_dataloader = DataLoader(
                PrecomputedDataset(train_data),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=False,  # Avoiding memory issues
                collate_fn=self._collate_fn
            )
            
            self.val_dataloader = DataLoader(
                PrecomputedDataset(val_data),
                batch_size=self.batch_size * 2,  # Larger batch for validation
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,  # Avoiding memory issues
                collate_fn=self._collate_fn
            )
            
            self.test_dataloader = DataLoader(
                PrecomputedDataset(test_data),
                batch_size=self.batch_size * 2,  # Larger batch for testing
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,  # Avoiding memory issues
                collate_fn=self._collate_fn
            )
        else:
            # Standard dataloaders for full datasets
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=False,  # Avoiding memory issues
                collate_fn=self._collate_fn
            )
            
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.batch_size * 2,  # Larger batch for validation
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,  # Avoiding memory issues
                collate_fn=self._collate_fn
            )
            
            self.test_dataloader = DataLoader(
                test_dataset,
                batch_size=self.batch_size * 2,  # Larger batch for testing
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,  # Avoiding memory issues
                collate_fn=self._collate_fn
            )
        
        logger.info(f"Train dataset size: {train_size}")
        logger.info(f"Validation dataset size: {val_size}")
        logger.info(f"Test dataset size: {test_size}")
        logger.info(f"Using {num_workers} worker threads for data loading")
    
    def _collate_fn(self, examples):
        """Batch examples for training.
        This handles moving things to the right device and 
        filtering out any non-tensor items.
        """
        # Since our examples are already processed by the dataset,
        # we just need to properly batch them
        if not examples:
            return {}
        
        # Initialize our batch as an empty dict
        batch = {}
        
        # Collect all keys from the first example
        all_keys = list(examples[0].keys())
        
        # Batch each key separately
        for key in all_keys:
            # Skip non-tensor keys like "length"
            if key == "length":
                continue
                
            # Get values for this key from all examples
            values = [example[key] for example in examples]
            
            # Stack values into a batch
            if torch.is_tensor(values[0]):
                batch[key] = torch.stack(values).to(self.device)
        
        # Record the batch length
        batch["length"] = len(examples)
        
        return batch
        
    def _setup_model(self):
        """Setup the model, optimizer, and scheduler"""
        if self.tiny_model:
            # Use a simple custom model for testing the training pipeline
            logger.info("Using tiny model for testing training pipeline")
            self._setup_tiny_model()
            return
            
        # Initialize model - skip pretrained weights (much faster)
        config = AutoConfig.from_pretrained(
            self.model_name, 
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            hidden_dropout_prob=0.1,  # Reduce for faster convergence
            attention_probs_dropout_prob=0.1  # Reduce for faster convergence
        )
        
        # For low memory mode, adjust model parameters but keep the architecture intact
        if self.low_memory:
            # Keep original architecture dimensions but use higher dropout for regularization
            # This helps with memory by reducing overfitting and allowing smaller batch sizes
            config.hidden_dropout_prob = 0.2  # Increased dropout
            config.attention_probs_dropout_prob = 0.2  # Increased dropout
            # We can reduce the number of encoding layers if needed (optional)
            if config.num_hidden_layers > 6 and self.fast_mode:
                config.num_hidden_layers = 6  # Reduce number of layers for fastest training
                logger.info(f"Low memory fast mode enabled - reducing to {config.num_hidden_layers} layers")
            logger.info(f"Low memory mode enabled - using higher dropout for memory efficiency")
        
        # Initialize with random weights (much faster than loading pretrained)
        logger.info("Initializing model with random weights (faster)")
        self.model = LayoutLMv3ForTokenClassification(config)
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer with higher learning rate for faster convergence
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # Use a higher learning rate if in fast mode
        effective_lr = self.learning_rate * 2 if self.fast_mode else self.learning_rate
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=effective_lr, eps=1e-7)
        
        # Setup learning rate scheduler
        # Calculate actual steps based on gradient accumulation
        total_steps = len(self.train_dataloader) * self.epochs // self.gradient_accumulation_steps
        
        # Use provided warmup steps or calculate it as a percentage of total steps
        if self.fast_mode:
            # Reduced warmup for fast mode
            warmup_steps = min(self.warmup_steps, int(total_steps * 0.05))
        else:
            warmup_steps = self.warmup_steps
            
        logger.info(f"Using {warmup_steps} warmup steps out of {total_steps} total training steps")
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
        
    def _setup_tiny_model(self):
        """
        Setup a tiny custom model for testing the training pipeline
        This uses a simple linear model instead of LayoutLMv3 to be memory efficient
        """
        import torch.nn as nn
        
        # Define a simple model for token classification
        class TinyTokenClassifier(nn.Module):
            def __init__(self, input_dim=768, hidden_dim=32, num_labels=7):
                super().__init__()
                self.linear1 = nn.Linear(input_dim, hidden_dim)
                self.linear2 = nn.Linear(hidden_dim, num_labels)
                self.dropout = nn.Dropout(0.1)
                self.activation = nn.ReLU()
                
            def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                      bbox=None, pixel_values=None, labels=None, **kwargs):
                # Simplify by only using input_ids
                # Just take the inputs and project them to the output space
                batch_size, seq_len = input_ids.shape
                # Create random embedding representation (simulating the real model)
                x = torch.zeros((batch_size, seq_len, 768), device=input_ids.device)
                
                # Simple feed-forward
                x = self.linear1(x)
                x = self.activation(x)
                x = self.dropout(x)
                logits = self.linear2(x)
                
                # Calculate loss if labels are provided
                loss = None
                if labels is not None:
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.linear2.out_features)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    loss = loss_fct(active_logits, active_labels)
                
                # Return output similar to HuggingFace models
                return type('ModelOutput', (), {
                    'loss': loss,
                    'logits': logits
                })
        
        # Create the tiny model
        logger.info(f"Creating tiny model with {self.num_labels} labels")
        self.model = TinyTokenClassifier(num_labels=self.num_labels)
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, eps=1e-7)
        
        # Setup learning rate scheduler
        total_steps = len(self.train_dataloader) * self.epochs // self.gradient_accumulation_steps
        warmup_steps = min(10, int(total_steps * 0.1))
        
        logger.info(f"Using {warmup_steps} warmup steps out of {total_steps} total training steps")
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
        
    def train(self):
        """Train the model"""
        logger.info("Starting training")
        
        # Track best model
        best_f1 = 0.0
        best_epoch = 0
        
        # Fast training mode - validate less frequently
        validate_every_n_epoch = 1
        if self.fast_mode:
            validate_every_n_epoch = max(1, self.epochs // 2)
            logger.info(f"Fast mode enabled. Validating every {validate_every_n_epoch} epoch(s)")
        
        # Training loop
        overall_start_time = time.time()
        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            
            # Training
            train_loss = self._train_epoch()
            logger.info(f"Train loss: {train_loss:.4f}")
            
            # Validation (conditionally)
            if (epoch + 1) % validate_every_n_epoch == 0 or epoch == self.epochs - 1:
                val_loss, val_metrics = self._evaluate(self.val_dataloader)
                logger.info(f"Validation loss: {val_loss:.4f}")
                logger.info(f"Validation F1: {val_metrics['f1']:.4f}")
                logger.info(f"Validation Precision: {val_metrics['precision']:.4f}")
                logger.info(f"Validation Recall: {val_metrics['recall']:.4f}")
                
                # Save checkpoint for this epoch
                checkpoint_path = os.path.join(self.output_dir, f"checkpoint-{epoch + 1}")
                os.makedirs(checkpoint_path, exist_ok=True)
                
                # Save model - handle different model types
                self._save_model(checkpoint_path)
                
                # Track best model
                if val_metrics["f1"] > best_f1:
                    best_f1 = val_metrics["f1"]
                    best_epoch = epoch + 1
                    
                    # Save best model
                    best_model_path = os.path.join(self.output_dir, "best_model")
                    os.makedirs(best_model_path, exist_ok=True)
                    
                    # Save model - handle different model types
                    self._save_model(best_model_path)
                    
                    # Save best metrics
                    with open(os.path.join(best_model_path, "metrics.json"), "w") as f:
                        json.dump(val_metrics, f, indent=2)
            
            # Memory cleanup between epochs
            if self.device.type == "mps":
                torch.mps.empty_cache()
            elif self.device.type == "cuda":
                torch.cuda.empty_cache()
        
        overall_training_time = time.time() - overall_start_time
        logger.info(f"Total training time: {overall_training_time/60:.2f} minutes")
        logger.info(f"Best model found at epoch {best_epoch} with F1 {best_f1:.4f}")
        
        # Test best model
        logger.info("Evaluating best model on test set")
        
        # Load best model - handle different model types
        self._load_best_model()
        
        # Evaluate on test set
        test_loss, test_metrics = self._evaluate(self.test_dataloader)
        logger.info(f"Test loss: {test_loss:.4f}")
        logger.info(f"Test F1: {test_metrics['f1']:.4f}")
        logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
        logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
        
        # Save test metrics
        best_model_path = os.path.join(self.output_dir, "best_model")
        with open(os.path.join(best_model_path, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)
        
        return best_model_path
    
    def _train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_batches = len(self.train_dataloader)
        start_time = time.time()
        
        # Create progress bar with tqdm
        progress_bar = tqdm(
            enumerate(self.train_dataloader), 
            total=total_batches,
            desc=f"Training",
            leave=True
        )
        
        # Training loop with tqdm for better visualization
        for step, batch in progress_bar:
            # Free up memory at the start of each step
            if self.device.type == "mps":
                torch.mps.empty_cache()
            elif self.device.type == "cuda":
                torch.cuda.empty_cache()
                
            # Move batch to device and filter unexpected keys
            batch = {k: v.to(self.device) for k, v in batch.items() if k not in ['length', 'image_path']}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Normalize loss for gradient accumulation
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Track loss (unnormalized for logging)
            if self.gradient_accumulation_steps > 1:
                total_loss += loss.item() * self.gradient_accumulation_steps
            else:
                total_loss += loss.item()
            
            # Only update parameters after accumulating gradients
            if (step + 1) % self.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Update parameters
                self.optimizer.step()
                
                # Step the scheduler
                self.scheduler.step()
                
                # Clear gradients
                self.optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * (self.gradient_accumulation_steps if self.gradient_accumulation_steps > 1 else 1):.4f}",
                'avg_loss': f"{total_loss / (step + 1):.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Clean up GPU memory
            del outputs
            del loss
            
            if step % 3 == 0:  # More aggressive cleanup for MPS
                batch = None
                if self.device.type == "mps":
                    torch.mps.empty_cache()
                elif self.device.type == "cuda":
                    torch.cuda.empty_cache()
        
        # Calculate average loss
        avg_loss = total_loss / total_batches
        epoch_time = time.time() - start_time
        logger.info(f"Epoch completed in {epoch_time/60:.2f} minutes")
        
        return avg_loss
    
    def _evaluate(self, dataloader):
        """Evaluate the model on the given dataloader"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        # Create progress bar
        eval_bar = tqdm(
            dataloader, 
            desc="Evaluating", 
            leave=False
        )
        
        # Evaluation loop
        for batch in eval_bar:
            # Clean up memory before each batch
            if self.device.type == "mps":
                torch.mps.empty_cache()
            elif self.device.type == "cuda":
                torch.cuda.empty_cache()
                
            # Move batch to device and filter unexpected keys
            input_batch = {k: v.to(self.device) for k, v in batch.items() if k not in ['length', 'image_path']}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**input_batch)
            
            # Track loss
            loss_value = outputs.loss.item()
            total_loss += loss_value
            
            # Get predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=2)
            
            # Get labels and attention mask
            labels = input_batch["labels"]
            attention_mask = input_batch["attention_mask"]
            
            # Extract valid predictions and labels (where attention mask is 1 and label is not -100)
            for i in range(preds.shape[0]):
                pred = preds[i].cpu().numpy()
                label = labels[i].cpu().numpy()
                mask = attention_mask[i].cpu().numpy()
                
                for j in range(len(pred)):
                    if mask[j] == 1 and label[j] != -100:
                        all_preds.append(pred[j])
                        all_labels.append(label[j])
            
            # Update progress bar
            eval_bar.set_postfix({'loss': f"{loss_value:.4f}"})
            
            # Clean up GPU memory 
            del outputs
            del logits
            del preds
            del input_batch
        
        # Calculate average loss
        avg_loss = total_loss / len(dataloader)
        
        # Calculate metrics
        metrics = {
            "f1": f1_score(all_labels, all_preds, average="weighted"),
            "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
            "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        }
        
        # Final cleanup
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()
            
        return avg_loss, metrics

    def evaluate(self, dataloader, split="val"):
        """Evaluate the model on the given dataloader and return metrics"""
        logger.info(f"Evaluating on {split} set")
        loss, metrics = self._evaluate(dataloader)
        
        # Log metrics
        logger.info(f"{split.capitalize()} loss: {loss:.4f}")
        logger.info(f"{split.capitalize()} F1: {metrics['f1']:.4f}")
        logger.info(f"{split.capitalize()} Precision: {metrics['precision']:.4f}")
        logger.info(f"{split.capitalize()} Recall: {metrics['recall']:.4f}")
        
        return metrics

    def _save_model(self, path):
        """Save model to path, handling different model types"""
        if hasattr(self.model, 'save_pretrained'):
            # HuggingFace models have save_pretrained
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        else:
            # Custom models need manual saving
            os.makedirs(path, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
            
            # Save model configuration
            with open(os.path.join(path, "config.json"), "w") as f:
                config = {
                    "model_type": "tiny_token_classifier",
                    "num_labels": self.num_labels,
                    "id2label": self.id2label,
                    "label2id": self.label2id,
                }
                json.dump(config, f, indent=2)
                
    def _load_best_model(self):
        """Load the best model, handling different model types"""
        best_model_path = os.path.join(self.output_dir, "best_model")
        
        if self.tiny_model:
            # Load tiny model
            state_dict = torch.load(os.path.join(best_model_path, "model.pt"))
            self.model.load_state_dict(state_dict)
        else:
            # Load HuggingFace model
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(best_model_path)
            
        # Move model to device
        self.model.to(self.device)

def main():
    """Main function for training the model"""
    parser = argparse.ArgumentParser(description="Train LayoutLM model for invoice extraction")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save model output")
    parser.add_argument("--model_name", type=str, default="microsoft/layoutlmv3-base", help="Model name or path")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--image_size", type=int, default=224, help="Image size (height and width)")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--subset_size", type=int, default=None, help="Use a subset of data for faster testing")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage even if GPU is available")
    parser.add_argument("--fast", action="store_true", help="Enable fast mode (less validation, higher learning rate)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                      help="Number of steps to accumulate gradients before updating weights")
    parser.add_argument("--low_memory", action="store_true", 
                      help="Enable low memory mode (smaller images, more aggressive memory management)")
    parser.add_argument("--tiny_model", action="store_true", help="Enable tiny model for testing training pipeline")
    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model trainer
    trainer = ModelTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        image_size=args.image_size,
        max_length=args.max_length,
        max_grad_norm=args.max_grad_norm,
        fast_mode=args.fast,
        subset_size=args.subset_size,
        force_cpu=args.force_cpu,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        low_memory=args.low_memory,
        tiny_model=args.tiny_model
    )

    # Train model
    trainer.train()

    # Evaluate model on test set
    metrics = trainer.evaluate(trainer.test_dataloader, "test")
    
    # Save final test results
    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Test metrics: {metrics}")
    logger.info(f"Model saved to {args.output_dir}")
    logger.info("Done!")

if __name__ == "__main__":
    main() 