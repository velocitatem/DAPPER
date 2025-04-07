# DAPPER (Document Analysis, Processing, and Pattern Extraction Repository)

[![Build and Deploy Documentation](https://github.com/velocitatem/DAPPER/actions/workflows/build-docs.yml/badge.svg)](https://github.com/velocitatem/DAPPER/actions/workflows/build-docs.yml)

# Table of Contents
1. [Classification](#classification)
2. [Information Extraction with LayoutLMv3](#information-extraction-with-layoutlmv3)

# Classification
DAPPER is a document classification system that processes and analyzes various document types using machine learning. The system trains on the RVL-CDIP dataset supplemented with additional invoice data from various sources.

## Training Pipeline

The training pipeline consists of two main components:

1. **Data Processing** (`data.py`):
   - Loads the RVL-CDIP dataset and supplementary invoice datasets
   - Processes and resizes images to standard dimensions (768x992) (emperically averaged and rounded to a multiple of 32)
   - Uploads all images to a MinIO object storage server
   - Splits data into training and testing sets (80/20 split)
   - Saves dataset references as CSV files

2. **Model Training** (`train.py`):
   - Uses a custom `MinioImageDataset` class to fetch images from MinIO
   - Implements a ResNet18 model with transfer learning
   - Trains the model on the processed images
   - Evaluates performance on test data
   - Saves the best model based on test accuracy

## MinIO Integration

The system uses MinIO as an object storage solution for efficient handling of document images:

- **Storage**: All document images are stored in the "dapper" bucket in MinIO
- **Retrieval**: During training, images are fetched from MinIO in batches
- **Parallelization**: Both upload and retrieval operations are parallelized for efficiency
- **URLs**: Images are referenced by URLs in the format `http://localhost:9900/dapper/[filename]`

## Datasets
- [RVL-CDIP](https://huggingface.co/datasets/aharley/rvl_cdip) - Main dataset with various document types
- [Additional Invoice Dataset](https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1) - Supplementary invoice data
- [Company documents dataset from Kaggle](https://www.kaggle.com/datasets/ayoubcherguelaine/company-documents-dataset)

## Setup and Usage

1. Start the MinIO server:
   ```
   docker-compose up -d
   ```

2. Process and upload documents:
   ```
   python classification/data.py
   ```

3. Train the model:
   ```
   python classification/train.py
   ```

4. The trained model will be saved as `model.pth`

# Invoice Information Extraction with LayoutLMv3

This project uses LayoutLMv3 to extract structured information from invoices through a complete pipeline from data preparation to inference. The system automatically identifies and extracts key fields from invoice images.

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Complete Pipeline Guide](#complete-pipeline-guide)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Training](#2-training)
  - [3. Inference](#3-inference)
- [Command Reference](#command-reference)
  - [Data Preparation Options](#data-preparation-options)
  - [Training Options](#training-options)
  - [Inference Options](#inference-options)
- [Memory Optimization](#memory-optimization)
- [Troubleshooting](#troubleshooting)

## Project Overview

The system works in three main stages:
1. **Data Preparation**: Process raw invoice images and annotations into a format suitable for training
2. **Training**: Train a LayoutLMv3 model to recognize and extract key fields from invoices
3. **Inference**: Use the trained model to extract information from new, unseen invoices

## Project Structure

- `data_preparation.py`: Converts raw invoice data into the formatted dataset for training
- `dataset.py`: Handles dataset loading and processing for the model
- `train.py`: Trains and evaluates the LayoutLMv3 model
- `inference.py`: Applies the trained model to extract information from new invoices
- `requirements.txt`: Lists all dependencies for the project

## Setup

1. Clone this repository to your local machine:

```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create and activate a virtual environment (recommended):

```bash
# For Python 3.8+
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Complete Pipeline Guide

### 1. Data Preparation

The data preparation step converts your raw invoice dataset into the format needed for training.

```bash
python data_preparation.py --input_dir /absolute/path/to/raw/invoice/dataset --output_dir ./data
```

This command will:
- Process all invoices in the input directory
- Create a structured dataset in the `./data` directory
- Generate a label map file at `./data/label_map.json`
- Split the data into training, validation, and test sets
- Create a secondary template-based split in the `./Data2` directory

**Expected Output Structure:**
```
data/
├── images/         # Preprocessed invoice images
├── train.json      # Training data annotations
├── val.json        # Validation data annotations
├── test.json       # Test data annotations
└── label_map.json  # Mapping of label IDs to entity types
```

### 2. Training

Train the model using the prepared dataset:

```bash
# Standard training
python train.py --data_dir ./data --output_dir ./output --batch_size 2 --epochs 10 --learning_rate 1e-5

# For best results (using recommended hyperparameters)
python train.py --data_dir ./data --output_dir ./output --batch_size 2 --epochs 10 --learning_rate 1e-5 --max_steps 2000 --warmup_steps 200
```

The training process:
1. Loads the prepared dataset
2. Initializes the LayoutLMv3 model with pre-trained weights
3. Trains the model for the specified number of epochs
4. Evaluates on validation data after each epoch
5. Saves the best model based on F1 score

**Expected Output Structure:**
```
output/
├── best_model/           # The best performing model
│   ├── config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── metrics.json      # Performance metrics on validation set
├── checkpoint-{epoch}/   # Checkpoints saved during training
├── training_metrics.json # Complete training history
├── training_curves.png   # Plot of training and validation metrics
├── confusion_matrix.png  # Confusion matrix on test set
└── test_metrics.json     # Final performance on test set
```

### 3. Inference

Extract information from new invoices using your trained model:

```bash
# Basic inference on a single invoice
python inference.py --input /path/to/invoice.jpg --use_ocr

# Inference with visualization
python inference.py --input /path/to/invoice.jpg --use_ocr --visualize

# Save results to file
python inference.py --input /path/to/invoice.jpg --use_ocr --output ./results.json --visualize --visualization_path ./visualization.png
```

The inference process:
1. Loads the trained model (from `./output/best_model` by default)
2. Processes the input invoice image
3. Extracts text using OCR (when `--use_ocr` is specified)
4. Recognizes and extracts key entities
5. Returns structured information in JSON format
6. Creates visualization if requested

## Command Reference

### Data Preparation Options

```bash
python data_preparation.py --help
```

Key options:
- `--input_dir`: Path to the directory containing raw invoice data (REQUIRED)
- `--output_dir`: Path to save the prepared dataset (default: `data`)
- `--seed`: Random seed for reproducibility (default: 42)

### Training Options

```bash
python train.py --help
```

Key options:
- `--data_dir`: Directory containing the processed dataset (default: `data`)
- `--output_dir`: Directory to save the model and results (default: `output`)
- `--model_name`: Pre-trained model to use (default: `microsoft/layoutlmv3-base`)
- `--batch_size`: Batch size for training (default: 2)
- `--learning_rate`: Learning rate (default: 1e-5)
- `--epochs`: Number of training epochs (default: 10)
- `--warmup_steps`: Warmup steps for learning rate scheduler (default: 500)
- `--max_steps`: Maximum number of training steps (default: 2000)
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients (default: 1)
- `--seed`: Random seed for reproducibility (default: 42)
- `--subset_size`: Use only a subset of data for faster training
- `--low_memory`: Enable memory optimization for limited hardware
- `--fast_mode`: Enable faster training with less frequent validation
- `--tiny_model`: Use a tiny model for testing the training pipeline

### Inference Options

```bash
python inference.py --help
```

Key options:
- `--model_path`: Path to the trained model (default: `output/best_model`)
- `--label_map`: Path to the label map (default: `data/label_map.json`)
- `--input`: Path to the input invoice image (REQUIRED)
- `--output`: Path to save extraction results as JSON
- `--use_ocr`: Use built-in OCR from LayoutLMv3 (recommended)
- `--visualize`: Create a visualization of extraction results
- `--visualization_path`: Path to save the visualization image

## Memory Optimization

For training on devices with limited memory (like laptops or Apple Silicon Macs):

```bash
# Memory-efficient training
python train.py --data_dir ./data --batch_size 2 --epochs 10 --gradient_accumulation_steps 8 --low_memory

# For extremely limited memory
python train.py --data_dir ./data --batch_size 1 --epochs 5 --gradient_accumulation_steps 16 --low_memory --subset_size 1000 --fast_mode
```

Key memory optimization techniques:
- Reduce batch size to 1 or 2
- Use gradient accumulation to simulate larger batches
- Enable `--low_memory` for optimized model configuration
- Use `--subset_size` to train on a smaller dataset first
- Enable `--fast_mode` to reduce validation frequency

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Try reducing batch size: `--batch_size 1`
   - Increase gradient accumulation: `--gradient_accumulation_steps 16`
   - Enable low memory mode: `--low_memory`
   - Use a subset of data: `--subset_size 500`

2. **Slow Training**
   - Make sure you're using GPU acceleration if available
   - Try reducing image size: `--image_size 224`
   - Use `--fast_mode` to validate less frequently

3. **OCR Issues During Inference**
   - Make sure the invoice image is clear and readable
   - Try different OCR engines or pre-process the image
   - Use high-resolution input images when possible

4. **Warning Messages**
   - To suppress tokenizer parallelism warnings: `export TOKENIZERS_PARALLELISM=false`
   - Deprecation warnings about `LayoutLMv3FeatureExtractor` can be safely ignored

### Example Commands for Common Scenarios

**Quick Test Run:**
```bash
python train.py --data_dir ./data --epochs 1 --subset_size 100 --tiny_model --fast_mode
```

**Full Production Training:**
```bash
python train.py --data_dir ./data --batch_size 2 --epochs 10 --learning_rate 1e-5 --max_steps 2000 --warmup_steps 200
```

**Process a Batch of Invoices:**
```bash
for file in /path/to/invoices/*.jpg; do
    python inference.py --input "$file" --use_ocr --output "${file%.jpg}_result.json" --visualize --visualization_path "${file%.jpg}_visual.png"
done
``` 
