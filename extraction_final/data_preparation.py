#!/usr/bin/env python3
import os
import json
import argparse
import logging
import pandas as pd
import numpy as np
import shutil
import warnings
from tqdm import tqdm
from PIL import Image
from transformers import LayoutLMv3ImageProcessor, LayoutLMv3TokenizerFast
from sklearn.model_selection import train_test_split
import random
import glob
import torch

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

def prepare_dataset(
    input_dir,
    output_dir,
    test_size=0.1,
    val_size=0.15,
    random_state=42
):
    """
    Prepare dataset for training, validation, and testing
    
    Args:
        input_dir: Directory containing the raw dataset
        output_dir: Directory to save the prepared dataset
        test_size: Proportion of data to use for testing
        val_size: Proportion of data to use for validation
        random_state: Random seed for reproducibility
    """
    logger.info(f"Preparing dataset from {input_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directories for images and annotations
    for split in ["train", "val", "test"]:
        for subdir in ["images", "annotations"]:
            os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)
    
    # Initialize tokenizer and image processor
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
    image_processor = LayoutLMv3ImageProcessor()
    
    # Get all annotation files
    annotation_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                annotation_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(annotation_files)} annotation files")
    
    # Create a list of all samples
    samples = []
    label_set = set()
    
    for ann_file in tqdm(annotation_files, desc="Processing annotations"):
        # Load annotation file
        with open(ann_file, 'r') as f:
            try:
                ann_data = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Error decoding JSON file: {ann_file}")
                continue
        
        # Get corresponding image file
        img_file = os.path.splitext(ann_file)[0] + '.jpg'
        if not os.path.exists(img_file):
            img_file = os.path.splitext(ann_file)[0] + '.png'
            if not os.path.exists(img_file):
                logger.warning(f"Image file not found for annotation: {ann_file}")
                continue
        
        # Validate annotation data
        if not all(key in ann_data for key in ['words', 'bboxes', 'ner_tags']):
            logger.warning(f"Missing required keys in annotation file: {ann_file}")
            continue
        
        # Make sure all lists have the same length
        if not len(ann_data['words']) == len(ann_data['bboxes']) == len(ann_data['ner_tags']):
            logger.warning(f"Inconsistent list lengths in annotation file: {ann_file}")
            continue
        
        # Collect unique labels
        for label in ann_data['ner_tags']:
            if isinstance(label, str):
                label_set.add(label)
            else:
                label_set.add(str(label))
        
        # Add sample to list
        samples.append({
            'image_file': img_file,
            'annotation_file': ann_file
        })
    
    logger.info(f"Successfully processed {len(samples)} valid samples")
    
    # Create label maps
    sorted_labels = sorted(label_set)
    id2label = {i: label for i, label in enumerate(sorted_labels)}
    label2id = {label: i for i, label in id2label.items()}
    
    # Save label maps
    with open(os.path.join(output_dir, 'label_map.json'), 'w') as f:
        json.dump(id2label, f, indent=2)
    
    logger.info(f"Found {len(id2label)} unique labels: {', '.join(sorted_labels)}")
    
    # Split data into train, validation, and test sets
    train_val_samples, test_samples = train_test_split(
        samples, test_size=test_size, random_state=random_state
    )
    
    train_samples, val_samples = train_test_split(
        train_val_samples, test_size=val_size / (1 - test_size), random_state=random_state
    )
    
    logger.info(f"Split data into {len(train_samples)} train, {len(val_samples)} validation, and {len(test_samples)} test samples")
    
    # Process each split
    process_split(train_samples, 'train', output_dir, id2label, label2id, tokenizer, image_processor)
    process_split(val_samples, 'val', output_dir, id2label, label2id, tokenizer, image_processor)
    process_split(test_samples, 'test', output_dir, id2label, label2id, tokenizer, image_processor)
    
    logger.info(f"Dataset preparation completed. Output saved to {output_dir}")

def process_split(samples, split, output_dir, id2label, label2id, tokenizer, image_processor):
    """
    Process a data split (train, val, or test)
    
    Args:
        samples: List of samples
        split: Split name ('train', 'val', or 'test')
        output_dir: Directory to save the prepared dataset
        id2label: Mapping from label IDs to label names
        label2id: Mapping from label names to label IDs
        tokenizer: Tokenizer for text processing
        image_processor: Image processor for image processing
    """
    logger.info(f"Processing {split} split with {len(samples)} samples")
    
    # Create output directories
    img_dir = os.path.join(output_dir, split, 'images')
    ann_dir = os.path.join(output_dir, split, 'annotations')
    
    # Create CSV data
    data = []
    
    for i, sample in enumerate(tqdm(samples, desc=f"Processing {split} split")):
        # Load image
        image = Image.open(sample['image_file']).convert("RGB")
        width, height = image.size
        
        # Load annotation
        with open(sample['annotation_file'], 'r') as f:
            ann_data = json.load(f)
        
        # Get words, bounding boxes, and labels
        words = ann_data['words']
        bboxes = ann_data['bboxes']
        ner_tags = ann_data['ner_tags']
        
        # Convert string labels to IDs
        if isinstance(ner_tags[0], str):
            ner_tags = [label2id.get(label, 0) for label in ner_tags]
        
        # Save processed image
        img_filename = f"{split}_{i+1:05d}.jpg"
        img_path = os.path.join(img_dir, img_filename)
        image.save(img_path)
        
        # Normalize bounding boxes
        normalized_bboxes = []
        for bbox in bboxes:
            # Ensure box coordinates are within image bounds
            x1 = max(0, min(bbox[0], width - 1))
            y1 = max(0, min(bbox[1], height - 1))
            x2 = max(0, min(bbox[2], width))
            y2 = max(0, min(bbox[3], height))
            
            normalized_bboxes.append([x1, y1, x2, y2])
        
        # Create annotation data
        processed_ann = {
            'image_path': img_filename,
            'width': width,
            'height': height,
            'words': words,
            'bboxes': normalized_bboxes,
            'ner_tags': ner_tags
        }
        
        # Save processed annotation
        ann_filename = f"{split}_{i+1:05d}.json"
        ann_path = os.path.join(ann_dir, ann_filename)
        with open(ann_path, 'w') as f:
            json.dump(processed_ann, f, indent=2)
        
        # Add to CSV data
        data.append({
            'id': f"{split}_{i+1:05d}",
            'image_path': img_filename,
            'annotation_path': ann_filename,
            'width': width,
            'height': height,
            'num_words': len(words)
        })
    
    # Create and save CSV file
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, f"{split}.csv"), index=False)
    
    logger.info(f"Saved {len(df)} samples to {split} split")

def split_data_by_template(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split data ensuring templates are distributed across splits"""
    # Group data by template
    template_groups = {}
    for item in data:
        template = item['template']
        if template not in template_groups:
            template_groups[template] = []
        template_groups[template].append(item)
    
    # Split templates into train, val, test
    templates = list(template_groups.keys())
    random.shuffle(templates)
    
    n_templates = len(templates)
    n_train = int(n_templates * train_ratio)
    n_val = int(n_templates * val_ratio)
    
    train_templates = templates[:n_train]
    val_templates = templates[n_train:n_train+n_val]
    test_templates = templates[n_train+n_val:]
    
    # Distribute data according to template splits
    train_data = []
    val_data = []
    test_data = []
    
    for template, items in template_groups.items():
        if template in train_templates:
            train_data.extend(items)
        elif template in val_templates:
            val_data.extend(items)
        else:
            test_data.extend(items)
    
    return train_data, val_data, test_data

def main():
    parser = argparse.ArgumentParser(description='Prepare invoice dataset for LayoutLMv3')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing invoice images and JSON files')
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory for processed data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    
    # Create Data2 directory for template-based split
    data2_dir = os.path.join(os.path.dirname(args.output_dir), "Data2")
    os.makedirs(data2_dir, exist_ok=True)
    os.makedirs(os.path.join(data2_dir, "images"), exist_ok=True)
    
    # Process all JSON files
    all_data = []
    for json_file in glob.glob(os.path.join(args.input_dir, "*.json")):
        data = process_json_file(json_file, args.output_dir)
        all_data.extend(data)
    
    # Split data randomly (original method)
    random.shuffle(all_data)
    n = len(all_data)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    
    train_data = all_data[:n_train]
    val_data = all_data[n_train:n_train+n_val]
    test_data = all_data[n_train+n_val:]
    
    # Split data by template (new method)
    train_data2, val_data2, test_data2 = split_data_by_template(all_data)
    
    # Save original split
    save_split(train_data, val_data, test_data, args.output_dir)
    
    # Save template-based split
    save_split(train_data2, val_data2, test_data2, data2_dir)
    
    # Copy images to Data2
    for img_file in glob.glob(os.path.join(args.output_dir, "images", "*")):
        shutil.copy(img_file, os.path.join(data2_dir, "images"))
    
    logger.info(f"Created two dataset versions:")
    logger.info(f"1. Random split in {args.output_dir}:")
    logger.info(f"   - Train: {len(train_data)} samples")
    logger.info(f"   - Val: {len(val_data)} samples")
    logger.info(f"   - Test: {len(test_data)} samples")
    logger.info(f"2. Template-based split in {data2_dir}:")
    logger.info(f"   - Train: {len(train_data2)} samples")
    logger.info(f"   - Val: {len(val_data2)} samples")
    logger.info(f"   - Test: {len(test_data2)} samples")

if __name__ == "__main__":
    main() 