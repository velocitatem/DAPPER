#!/usr/bin/env python3
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LayoutLMv3ImageProcessor, LayoutLMv3TokenizerFast, DataCollatorForTokenClassification
from PIL import Image
import pandas as pd
import logging
import warnings
from typing import Dict, List, Optional, Union, Tuple

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress FutureWarning about device argument
warnings.filterwarnings("ignore", category=FutureWarning, message="The `device` argument is deprecated and will be removed in v5 of Transformers.")

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class InvoiceDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        tokenizer=None,
        image_processor=None,
        max_length=512
    ):
        """
        Dataset for invoice information extraction using LayoutLMv3
        
        Args:
            data_dir: Directory containing the processed dataset
            split: Split to use (train, val, test)
            tokenizer: LayoutLMv3 tokenizer
            image_processor: LayoutLMv3 image processor
            max_length: Maximum sequence length for tokenizer
        """
        self.split = split
        self.data_dir = data_dir
        self.max_length = max_length
        self.image_processor = image_processor or LayoutLMv3ImageProcessor()
        self.tokenizer = tokenizer or LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
        
        # Handle different split names (val vs validation)
        if split == "val":
            csv_split = "validation"
        else:
            csv_split = split
            
        # Setup paths - adjusted for the actual file structure
        self.csv_file = os.path.join(data_dir, csv_split, f"{csv_split}_files.csv")
        self.img_dir = os.path.join(data_dir, csv_split, "images")
        self.ann_dir = os.path.join(data_dir, csv_split, "annotations")
        self.label_map_path = os.path.join(data_dir, "label_map.json")
        
        # Check if all required paths exist
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not os.path.exists(self.ann_dir):
            raise FileNotFoundError(f"Annotation directory not found: {self.ann_dir}")
        if not os.path.exists(self.label_map_path):
            raise FileNotFoundError(f"Label map file not found: {self.label_map_path}")
        
        # Load the data
        self.data = pd.read_csv(self.csv_file)
        
        # Load the label map
        with open(self.label_map_path, 'r') as f:
            self.label_map = json.load(f)
        
        # Create label mappings
        self.id2label = {int(key): value for key, value in self.label_map.items()}
        self.label2id = {value: int(key) for key, value in self.label_map.items()}
        self.num_labels = len(self.id2label)
        
        logger.info(f"Created {split} dataset with {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get file paths
        sample = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, sample['image_path'])
        ann_path = os.path.join(self.ann_dir, sample['annotation_path'])
        
        # Read the image
        image = Image.open(img_path).convert("RGB")
        
        # Read the annotations
        with open(ann_path, 'r') as f:
            ann_data = json.load(f)
        
        # Get words, bounding boxes, and NER tags
        words = ann_data["words"]
        boxes = ann_data["bboxes"]
        ner_tags = ann_data["ner_tags"]
        
        # Normalize bounding boxes to the format expected by LayoutLMv3
        normalized_boxes = []
        for box in boxes:
            # Extract coordinates
            x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
            # Ensure coordinates are in the right format
            normalized_boxes.append([x0, y0, x1, y1])
        
        # Prepare inputs for the model
        encoding = self.tokenizer(
            words,
            boxes=normalized_boxes,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Convert image to pixel_values
        pixel_values = self.image_processor(
            image, return_tensors="pt"
        ).pixel_values
        
        # Remove batch dimension
        for k, v in encoding.items():
            encoding[k] = v.squeeze()
        
        pixel_values = pixel_values.squeeze()
        
        # Create label sequence with padding
        labels = []
        
        # First token [CLS] gets label -100 (ignored in loss)
        labels.append(-100)
        
        # Assign labels to the remaining tokens
        word_idx = 0
        for i in range(1, len(encoding["input_ids"]) - 1):  # Skip [CLS] and [SEP]
            if encoding["attention_mask"][i] == 0:
                # Padding token
                labels.append(-100)
            else:
                # Regular token
                if word_idx < len(ner_tags):
                    labels.append(ner_tags[word_idx])
                    
                    # Check if we should move to the next word
                    token = self.tokenizer.decode([encoding["input_ids"][i].item()])
                    next_token = ""
                    if i < len(encoding["input_ids"]) - 2:
                        next_token = self.tokenizer.decode([encoding["input_ids"][i+1].item()])
                    
                    # Move to the next word if the next token is not a word piece
                    if next_token.strip() and not next_token.startswith("##"):
                        word_idx += 1
                else:
                    labels.append(-100)
        
        # Last token [SEP] gets label -100
        labels.append(-100)
        
        # Ensure labels list has the same length as input_ids
        if len(labels) < len(encoding["input_ids"]):
            labels.extend([-100] * (len(encoding["input_ids"]) - len(labels)))
        
        # Convert labels to tensor
        encoding["labels"] = torch.tensor(labels)
        encoding["pixel_values"] = pixel_values
        
        # Store the image path for reference
        encoding["image_path"] = img_path
        
        return encoding

def create_data_loaders(
    data_dir,
    tokenizer,
    batch_size=4,
    max_length=512
):
    """
    Create dataloaders for training, validation, and testing
    
    Args:
        data_dir: Directory containing the processed dataset
        tokenizer: LayoutLMv3 tokenizer
        batch_size: Batch size for the dataloaders
        max_length: Maximum sequence length for the tokenizer
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, id2label, label2id)
    """
    # Setup paths
    train_csv = os.path.join(data_dir, "train", "train_files.csv")
    val_csv = os.path.join(data_dir, "validation", "validation_files.csv")
    test_csv = os.path.join(data_dir, "test", "test_files.csv")
    
    img_dir_train = os.path.join(data_dir, "train", "images")
    img_dir_val = os.path.join(data_dir, "validation", "images")
    img_dir_test = os.path.join(data_dir, "test", "images")
    
    ann_dir_train = os.path.join(data_dir, "train", "annotations")
    ann_dir_val = os.path.join(data_dir, "validation", "annotations")
    ann_dir_test = os.path.join(data_dir, "test", "annotations")
    
    label_map_path = os.path.join(data_dir, "label_map.json")
    
    # Check if all required files and directories exist
    required_paths = [
        train_csv, val_csv, test_csv,
        img_dir_train, img_dir_val, img_dir_test,
        ann_dir_train, ann_dir_val, ann_dir_test,
        label_map_path
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            logger.error(f"Path does not exist: {path}")
            if path == label_map_path:
                raise FileNotFoundError(f"Label map file not found: {label_map_path}. Please run data_preparation.py first.")
            else:
                raise FileNotFoundError(f"Required path not found: {path}")
    
    # Create datasets
    train_dataset = InvoiceDataset(
        data_dir=data_dir,
        split="train",
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    val_dataset = InvoiceDataset(
        data_dir=data_dir,
        split="val",
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    test_dataset = InvoiceDataset(
        data_dir=data_dir,
        split="test",
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Create data collator for token classification
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=data_collator
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=data_collator
    )
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, train_dataset.id2label, train_dataset.label2id

if __name__ == "__main__":
    # Demonstrate usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Test dataset loading")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    args = parser.parse_args()
    
    # Initialize tokenizer
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
    image_processor = LayoutLMv3ImageProcessor()
    
    # Create dataset
    dataset = InvoiceDataset(
        data_dir=args.data_dir,
        split="train",  # Use train split for testing
        tokenizer=tokenizer,
        image_processor=image_processor
    )
    
    # Show dataset information
    print(f"Dataset size: {len(dataset)}")
    
    # Show an example
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("Input IDs shape:", sample["input_ids"].shape)
    print("Attention mask shape:", sample["attention_mask"].shape)
    print("Labels shape:", sample["labels"].shape)
    print("Pixel values shape:", sample["pixel_values"].shape) 