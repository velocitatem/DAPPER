"""
@file example_invoice_extraction.py
@package extraction.example_invoice_extraction
@brief Example script demonstrating how to use the ML extractor for invoice information extraction

This script shows how to use the ML extractor to extract information from invoice images using
the SROIE dataset from doctr, which contains scanned receipts with annotations.

@author Statistical Learning Team
@date 2025
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import gc
import time
import psutil
import tracemalloc
import pytesseract
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
# Import doctr for SROIE dataset
from doctr.datasets import SROIE

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import necessary components
from extraction.ml_extractor import MLExtractor

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("invoice_extraction_example")

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Example script for invoice information extraction using SROIE dataset")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a pre-trained model (optional)")
    parser.add_argument("--output", type=str, default="extracted_data.json",
                        help="Path to the output JSON file")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU for inference")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force CPU usage even if GPU is available")
    parser.add_argument("--split", choices=["train", "test"], default="test",
                        help="Dataset split to use (train or test)")
    parser.add_argument("--sample_limit", type=int, default=None,
                        help="Limit the number of samples to process")
    parser.add_argument("--use_polygons", action="store_true",
                        help="Use polygon bounding boxes instead of rectangles")
    parser.add_argument("--download", action="store_true",
                        help="Download the dataset if not already available")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker processes for parallel processing")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for processing")
    parser.add_argument("--memory_efficient", action="store_true",
                        help="Enable memory-efficient mode (slower but uses less RAM)")
    parser.add_argument("--half_precision", action="store_true",
                        help="Use half precision (fp16) for model inference")
    parser.add_argument("--sequential", action="store_true",
                        help="Process samples sequentially instead of in parallel")
    parser.add_argument("--monitor_memory", action="store_true",
                        help="Monitor memory usage during processing")
    parser.add_argument("--max_memory_gb", type=float, default=None,
                        help="Maximum memory usage in GB before forcing garbage collection")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with visualization of bounding boxes")
    parser.add_argument("--debug_output_dir", type=str, default="debug_images",
                        help="Directory to save debug images with bounding boxes")
    
    # Add training-related arguments
    parser.add_argument("--train", action="store_true",
                        help="Train the model on the dataset")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for training")
    parser.add_argument("--save_model_path", type=str, default="trained_invoice_model",
                        help="Path to save the trained model")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Validation split ratio (0.0-1.0)")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW optimizer")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for gradient clipping")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--resume_training", action="store_true",
                        help="Resume training from the model_path")
    parser.add_argument("--save_best_only", action="store_true",
                        help="Save only the best model based on validation loss")
    parser.add_argument("--log_dir", type=str, default="training_logs",
                        help="Directory to save training logs")
    parser.add_argument("--wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="invoice-extraction",
                        help="Weights & Biases project name")
    
    return parser.parse_args()

def convert_to_pil_image(image_tensor):
    """
    Convert a tensor image to PIL Image.
    
    Args:
        image_tensor: Image tensor from SROIE dataset
        
    Returns:
        PIL Image
    """
    # If the image is already a PIL Image, return it
    if isinstance(image_tensor, Image.Image):
        return image_tensor
    
    # Check tensor dimensions
    if image_tensor.ndim == 3 and image_tensor.shape[0] in (1, 3):
        # Convert from [C, H, W] to [H, W, C]
        image_np = image_tensor.permute(1, 2, 0).numpy()
        # If single channel, convert to 3 channels
        if image_np.shape[2] == 1:
            image_np = np.repeat(image_np, 3, axis=2)
    else:
        image_np = np.array(image_tensor)
    
    # Convert to uint8 if needed
    if image_np.dtype == np.float32 or image_np.dtype == np.float64:
        image_np = (image_np * 255).astype(np.uint8)
    
    # Create PIL Image
    return Image.fromarray(image_np)

def get_memory_usage():
    """Get current memory usage in GB."""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
    else:
        gpu_memory = 0
    
    # Get system memory
    process = psutil.Process(os.getpid())
    ram_memory = process.memory_info().rss / (1024**3)  # Convert to GB
    
    return {"gpu": gpu_memory, "ram": ram_memory}

def free_memory():
    """Free unused memory by garbage collection and PyTorch cache clearing."""
    # Run garbage collection
    gc.collect()
    
    # Clear PyTorch cache if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def normalize_bbox(bbox, width, height):
    """
    Normalize bounding box coordinates to be within 0-1000 range.
    
    Args:
        bbox: Bounding box coordinates [x0, y0, x1, y1]
        width: Image width
        height: Image height
        
    Returns:
        Normalized bounding box coordinates within 0-1000 range
    """
    x0, y0, x1, y1 = bbox
    
    # Ensure coordinates are within image dimensions
    x0 = max(0, min(x0, width))
    y0 = max(0, min(y0, height))
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    
    # Normalize to 0-1000 range
    x0 = int(1000 * (x0 / width))
    y0 = int(1000 * (y0 / height))
    x1 = int(1000 * (x1 / width))
    y1 = int(1000 * (y1 / height))
    
    # Ensure minimum box size (3x3 pixels)
    if x1 - x0 < 3:
        x1 = min(1000, x0 + 3)
    if y1 - y0 < 3:
        y1 = min(1000, y0 + 3)
    
    return [x0, y0, x1, y1]

def visualize_boxes(image, words, boxes, output_path):
    """
    Visualize bounding boxes on the image and save it.
    
    Args:
        image: PIL Image
        words: List of words
        boxes: List of bounding boxes in normalized format [x0, y0, x1, y1]
        output_path: Path to save the visualization
    """
    try:
        # Create a copy of the image for drawing
        from PIL import ImageDraw, ImageFont
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)
        
        # Get image dimensions
        width, height = image.size
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("Arial", 12)
        except:
            font = ImageFont.load_default()
        
        # Draw each box and word
        for word, box in zip(words, boxes):
            # Denormalize box coordinates
            x0, y0, x1, y1 = box
            x0 = int(x0 * width / 1000)
            y0 = int(y0 * height / 1000)
            x1 = int(x1 * width / 1000)
            y1 = int(y1 * height / 1000)
            
            # Draw bounding box
            draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
            
            # Draw word
            draw.text((x0, y0 - 12), word, fill="blue", font=font)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save image
        draw_image.save(output_path)
        logger.info(f"Saved visualization to {output_path}")
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")

def extract_from_pil(self, pil_image, half_precision=False, debug=False, debug_output_dir=None, sample_id=None):
    """Extract information from a PIL Image."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
        pil_image.save(tmp_path)
    
    # Extract information
    try:
        # Use half precision if requested and on GPU
        original_model = None
        if half_precision and self.device == "cuda":
            # Save original model
            original_model = self.model
            # Convert model to half precision
            self.model = self.model.half()
        
        # Get image dimensions for bbox normalization
        width, height = pil_image.size
        
        # Extract text using OCR
        text = pytesseract.image_to_string(pil_image)
        
        # Get word-level boxes for layout analysis
        words = []
        boxes = []
        
        # Extract words and their bounding boxes
        ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
        for i in range(len(ocr_data['text'])):
            if ocr_data['text'][i].strip():
                words.append(ocr_data['text'][i])
                x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                # Original bbox format: [x0, y0, x1, y1]
                bbox = [x, y, x + w, y + h]
                # Normalize bbox to 0-1000 range
                normalized_bbox = normalize_bbox(bbox, width, height)
                boxes.append(normalized_bbox)
        
        # Debug: visualize boxes
        if debug and debug_output_dir is not None and sample_id is not None:
            output_path = os.path.join(debug_output_dir, f"{sample_id}_boxes.png")
            visualize_boxes(pil_image, words, boxes, output_path)
        
        if not words:
            # No text detected, create empty InvoiceData
            logger.warning("No text detected in image")
            from extraction.base_extractor import InvoiceData
            return InvoiceData()
        
        # Prepare inputs for LayoutLMv3
        encoding = self.processor(
            pil_image,
            words,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        
        # Move inputs to device
        for k, v in encoding.items():
            encoding[k] = v.to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = outputs.logits.argmax(-1).squeeze().cpu().numpy()
        
        # Extract structured information from predictions
        extracted_data = self._convert_predictions_to_data(words, predictions)
        extracted_data.raw_text = text
        
        # Post-process and validate
        extracted_data = self.postprocess_extraction(extracted_data) if hasattr(self, 'postprocess_extraction') else extracted_data
        
        # Restore original model if needed
        if original_model is not None:
            self.model = original_model
        
        return extracted_data
    except Exception as e:
        logger.error(f"Error in extract_from_pil: {str(e)}")
        # Try CPU fallback if on CUDA
        if self.device == "cuda":
            try:
                logger.warning("Attempting CPU fallback for this sample")
                # Create a CPU-based extractor
                cpu_extractor = MLExtractor(model_path=None, device="cpu")
                return cpu_extractor.extract(tmp_path)
            except Exception as cpu_err:
                logger.error(f"CPU fallback also failed: {str(cpu_err)}")
        
        # Return empty invoice data on error
        from extraction.base_extractor import InvoiceData
        return InvoiceData()
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def _convert_predictions_to_data(self, words, predictions):
    """
    Convert model predictions to structured data.
    
    Args:
        words: List of words from the document
        predictions: Model predictions for each word
        
    Returns:
        InvoiceData object with extracted information
    """
    from extraction.base_extractor import InvoiceData
    
    data = InvoiceData()
    confidence_scores = {}
    
    # Initialize fields
    fields = {
        'invoice_num': [],
        'date': [],
        'due_date': [],
        'total': [],
        'issuer': [],
        'recipient': [],
        'other': []
    }
    
    # Simple mapping of prediction IDs to field names
    id2label = {
        0: "O",           # Outside (not a named entity)
        1: "invoice_num", # Invoice number
        2: "date",        # Invoice date
        3: "due_date",    # Due date
        4: "total",       # Total amount
        5: "issuer",      # Issuer name
        6: "recipient",   # Recipient name
        7: "other"        # Other relevant information
    }
    
    # Group words by their predicted labels
    prev_label = "O"
    current_text = ""
    
    # Handle case where predictions is a single value (not iterable)
    if not hasattr(predictions, '__iter__'):
        predictions = [predictions] * len(words)
    
    for word, pred_id in zip(words, predictions):
        # Ensure pred_id is within valid range
        if isinstance(pred_id, (int, np.integer)) and 0 <= pred_id < len(id2label):
            current_label = id2label[pred_id]
        else:
            current_label = "O"
        
        if current_label != "O":
            if current_label != prev_label and prev_label != "O":
                # Save the previous entity
                fields[prev_label].append(current_text.strip())
                current_text = ""
            
            # Add current word to the entity text
            current_text += " " + word
            prev_label = current_label
        else:
            if prev_label != "O" and current_text:
                # Save the previous entity when we reach a non-entity token
                fields[prev_label].append(current_text.strip())
                current_text = ""
            prev_label = "O"
    
    # Don't forget to add the last entity if the document ends with one
    if prev_label != "O" and current_text:
        fields[prev_label].append(current_text.strip())
    
    # Set InvoiceData fields
    if fields['invoice_num']:
        data.invoice_number = fields['invoice_num'][0]
        confidence_scores['invoice_num'] = 0.8  # Simplified confidence scoring
    
    if fields['date']:
        from datetime import datetime
        date_str = fields['date'][0]
        try:
            # Try parsing date (simple implementation)
            data.date = datetime.strptime(date_str, '%Y-%m-%d')
        except:
            data.date = date_str
        confidence_scores['date'] = 0.8
    
    if fields['due_date']:
        from datetime import datetime
        due_date_str = fields['due_date'][0]
        try:
            # Try parsing date (simple implementation)
            data.due_date = datetime.strptime(due_date_str, '%Y-%m-%d')
        except:
            data.due_date = due_date_str
        confidence_scores['due_date'] = 0.8
    
    if fields['total']:
        total_str = fields['total'][0]
        # Simple float extraction (improved version would use regex)
        try:
            # Remove non-numeric except period and comma
            cleaned = ''.join([c if c.isdigit() or c in ['.', ','] else ' ' for c in total_str])
            # Replace comma with period if it's the decimal separator
            if ',' in cleaned and '.' not in cleaned:
                cleaned = cleaned.replace(',', '.')
            # Extract the first valid float
            for part in cleaned.split():
                try:
                    data.total_amount = float(part)
                    break
                except:
                    continue
        except:
            # If extraction fails, just use the raw string
            data.total_amount = total_str
        confidence_scores['total'] = 0.8
    
    if fields['issuer']:
        data.issuer_name = fields['issuer'][0]
        confidence_scores['issuer'] = 0.8
    
    if fields['recipient']:
        data.recipient_name = fields['recipient'][0]
        confidence_scores['recipient'] = 0.8
    
    # Set confidence scores
    data.confidence_scores = confidence_scores
    
    return data

def process_sample(args):
    """
    Process a single sample from the dataset.
    
    Args:
        args: Tuple containing (extractor, sample_idx, image, target, sample_id, memory_efficient, half_precision, monitor_memory, debug, debug_output_dir)
        
    Returns:
        Dictionary with extraction results
    """
    extractor, sample_idx, image, target, sample_id, memory_efficient, half_precision, monitor_memory, debug, debug_output_dir = args
    logger.info(f"Processing sample {sample_idx}: {sample_id}")
    
    if monitor_memory:
        mem_before = get_memory_usage()
        logger.info(f"Memory before processing sample {sample_idx}: GPU={mem_before['gpu']:.2f}GB, RAM={mem_before['ram']:.2f}GB")
    
    try:
        # Convert image to PIL
        pil_image = convert_to_pil_image(image)
        
        # Extract information using the ML extractor
        extracted_data = extractor.extract_from_pil(
            pil_image, 
            half_precision=half_precision,
            debug=debug,
            debug_output_dir=debug_output_dir,
            sample_id=sample_id
        )
        
        # Convert to dictionary for JSON serialization
        data_dict = extracted_data.to_dict()
        data_dict['sample_id'] = sample_id
        data_dict['sample_idx'] = sample_idx
        data_dict['status'] = 'success'
        
        # Add ground truth if available for comparison
        if target and isinstance(target, dict):
            data_dict['ground_truth'] = target
            
            # Compare extraction with ground truth if applicable
            if 'invoice_number' in target and extracted_data.invoice_number:
                logger.debug(f"Ground truth invoice number: {target['invoice_number']}")
                logger.debug(f"Extracted invoice number: {extracted_data.invoice_number}")
            
            if 'total' in target and extracted_data.total_amount:
                logger.debug(f"Ground truth total: {target['total']}")
                logger.debug(f"Extracted total: {extracted_data.total_amount}")
        
        # Print extracted information (only for key fields)
        logger.debug("Extracted information:")
        for key, value in data_dict.items():
            if key in ['invoice_number', 'date', 'total_amount', 'issuer_name', 'recipient_name']:
                logger.debug(f"  {key}: {value}")
        
        if memory_efficient:
            # Free memory after processing
            free_memory()
            
        if monitor_memory:
            mem_after = get_memory_usage()
            logger.info(f"Memory after processing sample {sample_idx}: GPU={mem_after['gpu']:.2f}GB, RAM={mem_after['ram']:.2f}GB")
            logger.info(f"Memory change: GPU={mem_after['gpu']-mem_before['gpu']:.2f}GB, RAM={mem_after['ram']-mem_before['ram']:.2f}GB")
            
        return data_dict
    
    except Exception as e:
        logger.error(f"Error processing sample {sample_idx} ({sample_id}): {str(e)}")
        if memory_efficient:
            # Free memory after error
            free_memory()
        return {
            'sample_id': sample_id,
            'sample_idx': sample_idx,
            'error': str(e),
            'status': 'failed'
        }

def extract_sroie_data_sequential(extractor, dataset, output_file, sample_limit=None, memory_efficient=False, half_precision=False, monitor_memory=False, max_memory_gb=None, debug=False, debug_output_dir=None):
    """
    Extract information from images in the SROIE dataset sequentially.
    
    Args:
        extractor: MLExtractor instance
        dataset: SROIE dataset instance
        output_file: Path to the output JSON file
        sample_limit: Maximum number of samples to process
        memory_efficient: Whether to use memory-efficient mode
        half_precision: Whether to use half precision (fp16)
        monitor_memory: Whether to monitor memory usage
        max_memory_gb: Maximum memory usage before forcing GC
        debug: Whether to enable debug mode
        debug_output_dir: Directory to save debug images with bounding boxes
        
    Returns:
        List of extraction results
    """
    num_samples = len(dataset)
    logger.info(f"SROIE dataset loaded with {num_samples} samples")
    
    if sample_limit and sample_limit < num_samples:
        num_samples = sample_limit
        logger.info(f"Processing limited to {num_samples} samples")
    
    results = []
    successful = 0
    failed = 0
    
    # Enable memory tracking if requested
    if monitor_memory:
        tracemalloc.start()
    
    # Process samples sequentially
    for i in tqdm(range(num_samples), desc="Processing SROIE samples"):
        # Check memory usage and force GC if needed
        if monitor_memory and max_memory_gb is not None:
            mem_usage = get_memory_usage()
            if mem_usage['ram'] > max_memory_gb or (torch.cuda.is_available() and mem_usage['gpu'] > max_memory_gb):
                logger.warning(f"Memory usage high: GPU={mem_usage['gpu']:.2f}GB, RAM={mem_usage['ram']:.2f}GB. Forcing garbage collection.")
                free_memory()
                
        try:
            # Get image and target
            image, target = dataset[i]
            sample_id = f"sample_{i}"
            if hasattr(dataset, 'files') and i < len(dataset.files):
                sample_id = dataset.files[i]
            
            # Process the sample
            args = (extractor, i, image, target, sample_id, memory_efficient, half_precision, monitor_memory, debug, debug_output_dir)
            result = process_sample(args)
            
            # Add to results
            results.append(result)
            if result.get('status') == 'success':
                successful += 1
            else:
                failed += 1
                
            # Free memory after each sample in memory-efficient mode
            if memory_efficient:
                free_memory()
                # Add a small delay to allow memory to be properly released
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Unexpected error processing sample {i}: {str(e)}")
            results.append({
                'sample_id': f"sample_{i}",
                'sample_idx': i,
                'error': str(e),
                'status': 'failed'
            })
            failed += 1
            
            # Free memory after error
            if memory_efficient:
                free_memory()
    
    # Display memory usage statistics if tracking
    if monitor_memory:
        current, peak = tracemalloc.get_traced_memory()
        logger.info(f"Current memory usage: {current / 10**6:.2f}MB")
        logger.info(f"Peak memory usage: {peak / 10**6:.2f}MB")
        tracemalloc.stop()
    
    # Sort results by sample index
    results.sort(key=lambda x: x.get('sample_idx', 0))
    
    # Save all extracted data to JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Saved extracted data for {len(results)} samples to {output_file}")
    logger.info(f"Processing summary: {successful} successful, {failed} failed")
    
    return results

def extract_sroie_data_parallel(extractor, dataset, output_file, sample_limit=None, num_workers=4, memory_efficient=False, half_precision=False, monitor_memory=False, debug=False, debug_output_dir=None):
    """
    Extract information from images in the SROIE dataset using parallel processing.
    
    Args:
        extractor: MLExtractor instance
        dataset: SROIE dataset instance
        output_file: Path to the output JSON file
        sample_limit: Maximum number of samples to process
        num_workers: Number of worker processes for parallel processing
        memory_efficient: Whether to use memory-efficient mode
        half_precision: Whether to use half precision (fp16)
        monitor_memory: Whether to monitor memory usage
        debug: Whether to enable debug mode
        debug_output_dir: Directory to save debug images with bounding boxes
        
    Returns:
        List of extraction results
    """
    num_samples = len(dataset)
    logger.info(f"SROIE dataset loaded with {num_samples} samples")
    
    if sample_limit and sample_limit < num_samples:
        num_samples = sample_limit
        logger.info(f"Processing limited to {num_samples} samples")
    
    # Prepare tasks for parallel processing
    tasks = []
    for i in range(num_samples):
        image, target = dataset[i]
        sample_id = f"sample_{i}"
        if hasattr(dataset, 'files') and i < len(dataset.files):
            sample_id = dataset.files[i]
        
        tasks.append((extractor, i, image, target, sample_id, memory_efficient, half_precision, monitor_memory, debug, debug_output_dir))
    
    # Process samples in parallel
    results = []
    successful = 0
    failed = 0
    
    # Adjust number of workers based on system capabilities
    max_workers = min(num_workers, multiprocessing.cpu_count())
    
    # In memory-efficient mode, reduce the number of workers
    if memory_efficient:
        max_workers = max(1, max_workers // 2)
        
    logger.info(f"Using {max_workers} workers for parallel processing")
    
    # Enable memory tracking if requested
    if monitor_memory:
        tracemalloc.start()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_sample, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing SROIE samples"):
            try:
                result = future.result()
                results.append(result)
                if result.get('status') == 'success':
                    successful += 1
                else:
                    failed += 1
                    
                # Force garbage collection periodically in memory-efficient mode
                if memory_efficient and (successful + failed) % 10 == 0:
                    free_memory()
                    
                # Monitor memory usage
                if monitor_memory and (successful + failed) % 10 == 0:
                    mem_usage = get_memory_usage()
                    logger.info(f"Memory usage after {successful + failed} samples: GPU={mem_usage['gpu']:.2f}GB, RAM={mem_usage['ram']:.2f}GB")
                    
            except Exception as e:
                logger.error(f"Unexpected error in worker: {str(e)}")
    
                failed += 1
    # Display memory usage statistics if tracking
    if monitor_memory:
        current, peak = tracemalloc.get_traced_memory()
        logger.info(f"Current memory usage: {current / 10**6:.2f}MB")
        logger.info(f"Peak memory usage: {peak / 10**6:.2f}MB")
        tracemalloc.stop()
    
    # Sort results by sample index
    results.sort(key=lambda x: x.get('sample_idx', 0))
    
    # Save all extracted data to JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Saved extracted data for {len(results)} samples to {output_file}")
    logger.info(f"Processing summary: {successful} successful, {failed} failed")
    
    return results

class SROIEDataset(Dataset):
    """Dataset for SROIE invoice information extraction."""
    
    def __init__(self, sroie_dataset, processor, max_length=512):
        """
        Initialize the dataset.
        
        Args:
            sroie_dataset: SROIE dataset instance
            processor: LayoutLMv3 processor
            max_length: Maximum sequence length
        """
        self.sroie_dataset = sroie_dataset
        self.processor = processor
        self.max_length = max_length
        self.id2label = {
            0: "O",           # Outside (not a named entity)
            1: "invoice_num", # Invoice number
            2: "date",        # Invoice date
            3: "due_date",    # Due date
            4: "total",       # Total amount
            5: "issuer",      # Issuer name
            6: "recipient",   # Recipient name
            7: "other"        # Other relevant information
        }
        self.label2id = {v: k for k, v in self.id2label.items()}
    
    def __len__(self):
        return len(self.sroie_dataset)
    
    def __getitem__(self, idx):
        # Get image and target
        image, target = self.sroie_dataset[idx]
        
        # Convert image to PIL if needed
        pil_image = convert_to_pil_image(image)
        
        # Extract text and bboxes using OCR
        width, height = pil_image.size
        
        # Extract words and their bounding boxes
        words = []
        boxes = []
        
        ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
        for i in range(len(ocr_data['text'])):
            if ocr_data['text'][i].strip():
                words.append(ocr_data['text'][i])
                x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                # Original bbox format: [x0, y0, x1, y1]
                bbox = [x, y, x + w, y + h]
                # Normalize bbox to 0-1000 range
                normalized_bbox = normalize_bbox(bbox, width, height)
                boxes.append(normalized_bbox)
        
        if not words:
            # Handle empty words case (fallback to a placeholder)
            words = ["placeholder"]
            boxes = [[0, 0, 100, 100]]
        
        # Create token labels based on the words and target
        word_labels = self._assign_labels(words, target)
        
        # Prepare inputs for LayoutLMv3
        encoding = self.processor(
            pil_image,
            words,
            boxes=boxes,
            word_labels=word_labels,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        for k, v in encoding.items():
            encoding[k] = v.squeeze(0)
        
        return encoding
    
    def _assign_labels(self, words, target):
        """
        Assign token-level labels based on the target annotations.
        
        Args:
            words: List of words from OCR
            target: Target annotations
            
        Returns:
            List of label IDs for each word
        """
        # Initialize all words with "O" (outside) label
        labels = [self.label2id["O"]] * len(words)
        
        if not target or not isinstance(target, dict):
            return labels
        
        # Convert all words and target values to lowercase for easier matching
        words_lower = [w.lower() for w in words]
        
        # Process invoice number
        if 'invoice_number' in target and target['invoice_number']:
            inv_num = str(target['invoice_number']).lower()
            self._find_and_label(words_lower, inv_num, labels, "invoice_num")
        
        # Process dates
        if 'date' in target and target['date']:
            date_str = str(target['date']).lower()
            self._find_and_label(words_lower, date_str, labels, "date")
        
        # Process total amount
        if 'total' in target and target['total']:
            total_str = str(target['total']).lower()
            self._find_and_label(words_lower, total_str, labels, "total")
        
        # Process company/issuer name
        if 'company' in target and target['company']:
            company_str = str(target['company']).lower()
            self._find_and_label(words_lower, company_str, labels, "issuer")
        
        # Process address (if available, consider as other information)
        if 'address' in target and target['address']:
            addr_str = str(target['address']).lower()
            self._find_and_label(words_lower, addr_str, labels, "other")
        
        return labels
    
    def _find_and_label(self, words_lower, target_str, labels, label_type):
        """
        Find target string in words and assign labels.
        
        Args:
            words_lower: List of lowercase words
            target_str: Target string to find
            labels: List of label IDs to update
            label_type: Type of label to assign
        """
        target_str = target_str.lower()
        
        # Try exact word match first
        for i, word in enumerate(words_lower):
            if target_str == word:
                labels[i] = self.label2id[label_type]
                return
        
        # Try if target is contained in any word
        for i, word in enumerate(words_lower):
            if target_str in word:
                labels[i] = self.label2id[label_type]
                return
        
        # Try substring matching if target consists of multiple words
        target_words = target_str.split()
        if len(target_words) > 1:
            # Find consecutive words that match the target words
            for i in range(len(words_lower) - len(target_words) + 1):
                match = True
                for j, target_word in enumerate(target_words):
                    if target_word not in words_lower[i + j]:
                        match = False
                        break
                
                if match:
                    # Label all words in the matching sequence
                    for j in range(len(target_words)):
                        labels[i + j] = self.label2id[label_type]
                    return

def prepare_datasets(sroie_dataset, processor, val_split=0.2, max_length=512):
    """
    Prepare train and validation datasets.
    
    Args:
        sroie_dataset: SROIE dataset instance
        processor: LayoutLMv3 processor
        val_split: Validation split ratio
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    dataset = SROIEDataset(sroie_dataset, processor, max_length)
    
    # Split into train and validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    logger.info(f"Prepared dataset: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    
    return train_dataset, val_dataset

def train_model(args, model, processor, train_dataset, val_dataset=None):
    """
    Train the LayoutLMv3 model on the dataset.
    
    Args:
        args: Command-line arguments
        model: LayoutLMv3 model
        processor: LayoutLMv3 processor
        train_dataset: Training dataset
        val_dataset: Validation dataset
        
    Returns:
        Trained model
    """
    # Set up logging with wandb if enabled
    if args.wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args))
        except ImportError:
            logger.warning("wandb not installed. Run 'pip install wandb' to use Weights & Biases logging.")
            args.wandb = False
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # Set up optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Set up scheduler
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else int(total_steps * 0.1)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Set up mixed precision training if requested
    scaler = None
    if args.fp16 and torch.cuda.is_available():
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        logger.info("Using mixed precision training")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        logger.info(f"Starting epoch {epoch+1}/{args.epochs}")
        
        # Training phase
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training epoch {epoch+1}")
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision if enabled
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss = loss / args.gradient_accumulation_steps
            else:
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
            
            # Accumulate loss
            epoch_loss += loss.item() * args.gradient_accumulation_steps
            
            # Backward pass with mixed precision if enabled
            if scaler:
                scaler.scale(loss).backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item() * args.gradient_accumulation_steps})
            
            # Log with wandb if enabled
            if args.wandb and step % 10 == 0:
                wandb.log({"train_loss": loss.item() * args.gradient_accumulation_steps})
            
            # Free memory if in memory-efficient mode
            if args.memory_efficient and step % 10 == 0:
                free_memory()
        
        # Calculate average training loss for the epoch
        avg_train_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} - Average training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        if val_loader:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                progress_bar = tqdm(val_loader, desc=f"Validation epoch {epoch+1}")
                for batch in progress_bar:
                    # Move batch to device
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = model(**batch)
                    loss = outputs.loss
                    
                    # Accumulate loss
                    val_loss += loss.item()
                    
                    # Update progress bar
                    progress_bar.set_postfix({"loss": loss.item()})
                    
                    # Free memory if in memory-efficient mode
                    if args.memory_efficient:
                        free_memory()
            
            # Calculate average validation loss
            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"Epoch {epoch+1} - Validation loss: {avg_val_loss:.4f}")
            
            # Log with wandb if enabled
            if args.wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss
                })
            
            # Save model if it's the best so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                logger.info(f"New best validation loss: {best_val_loss:.4f}")
                
                # Save the model
                if args.save_best_only:
                    save_path = args.save_model_path
                else:
                    save_path = f"{args.save_model_path}_epoch_{epoch+1}"
                
                model.save_pretrained(save_path)
                processor.save_pretrained(save_path)
                logger.info(f"Model saved to {save_path}")
        else:
            # If no validation set, save after each epoch
            save_path = f"{args.save_model_path}_epoch_{epoch+1}"
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            logger.info(f"Model saved to {save_path}")
    
    # Save final model if we're not only saving the best model
    if not args.save_best_only or not val_loader:
        model.save_pretrained(args.save_model_path)
        processor.save_pretrained(args.save_model_path)
        logger.info(f"Final model saved to {args.save_model_path}")
    
    # Close wandb if enabled
    if args.wandb:
        wandb.finish()
    
    return model

def main():
    """
    Main function.
    """
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = "cpu"
    if args.use_gpu and not args.force_cpu and torch.cuda.is_available():
        try:
            # Test CUDA capability with a small tensor operation
            a = torch.tensor([1.0, 2.0], device="cuda")
            b = torch.tensor([3.0, 4.0], device="cuda")
            c = a + b
            device = "cuda"
            logger.info(f"Using device: {device}")
        except Exception as e:
            logger.warning(f"Error initializing CUDA: {str(e)}")
            logger.warning("Falling back to CPU")
    else:
        if args.force_cpu:
            logger.info("Forced CPU usage")
        elif not torch.cuda.is_available():
            logger.info("CUDA not available, using CPU")
        else:
            logger.info("Using CPU (--use_gpu not specified)")
    
    # Log memory efficiency settings
    if args.memory_efficient:
        logger.info("Memory-efficient mode enabled")
    if args.half_precision:
        logger.info("Using half precision (fp16)")
        if device == "cpu":
            logger.warning("Half precision only works on GPU. Ignoring half_precision option.")
    if args.sequential:
        logger.info("Sequential processing mode enabled")
    if args.monitor_memory:
        logger.info("Memory monitoring enabled")
        mem_usage = get_memory_usage()
        logger.info(f"Initial memory usage: GPU={mem_usage['gpu']:.2f}GB, RAM={mem_usage['ram']:.2f}GB")
    
    # Load SROIE dataset
    is_train = args.split == "train" or args.train
    logger.info(f"Loading SROIE dataset ({'train' if is_train else 'test'} split)")
    
    try:
        dataset = SROIE(
            train=is_train,
            use_polygons=args.use_polygons,
            download=args.download
        )
    except Exception as e:
        logger.error(f"Error loading SROIE dataset: {str(e)}")
        sys.exit(1)
    
    # Initialize processor
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    
    # Handle training if requested
    if args.train:
        logger.info("Training mode enabled")
        
        # Initialize model (from scratch or pre-trained)
        if args.model_path and args.resume_training:
            logger.info(f"Loading pre-trained model from {args.model_path}")
            model = LayoutLMv3ForTokenClassification.from_pretrained(args.model_path)
        else:
            logger.info("Initializing new model")
            model = LayoutLMv3ForTokenClassification.from_pretrained(
                "microsoft/layoutlmv3-base", 
                num_labels=8  # Using 8 labels as defined earlier
            )
        
        # Move model to device
        model.to(device)
        
        # Prepare datasets for training
        train_dataset, val_dataset = prepare_datasets(
            dataset,
            processor,
            val_split=args.val_split
        )
        
        # Train model
        model = train_model(args, model, processor, train_dataset, val_dataset)
        
        # Set model path for inference to the newly trained model
        args.model_path = args.save_model_path
    
    # Create extractor for inference
    extractor = MLExtractor(model_path=args.model_path, device=device)
    
    # Add methods to the extractor if not already present
    if not hasattr(extractor, 'extract_from_pil'):
        # Add the method to the extractor
        import types
        extractor.extract_from_pil = types.MethodType(extract_from_pil, extractor)
        extractor._convert_predictions_to_data = types.MethodType(_convert_predictions_to_data, extractor)
    
    if not hasattr(extractor, 'postprocess_extraction'):
        # Add a simple postprocess method
        def postprocess_extraction(self, extracted_data):
            """Simple postprocessing for extracted data."""
            return extracted_data
        extractor.postprocess_extraction = types.MethodType(postprocess_extraction, extractor)
    
    # Process SROIE dataset for inference (if not in training-only mode)
    if not args.train or args.output != "extracted_data.json":
        if args.sequential or args.memory_efficient:
            results = extract_sroie_data_sequential(
                extractor, 
                dataset, 
                args.output,
                sample_limit=args.sample_limit,
                memory_efficient=args.memory_efficient,
                half_precision=args.half_precision,
                monitor_memory=args.monitor_memory,
                max_memory_gb=args.max_memory_gb,
                debug=args.debug,
                debug_output_dir=args.debug_output_dir
            )
        else:
            results = extract_sroie_data_parallel(
                extractor, 
                dataset, 
                args.output,
                sample_limit=args.sample_limit,
                num_workers=args.num_workers,
                memory_efficient=args.memory_efficient,
                half_precision=args.half_precision,
                monitor_memory=args.monitor_memory,
                debug=args.debug, 
                debug_output_dir=args.debug_output_dir
            )
        
        # Compute statistics if we have ground truth
        if hasattr(dataset, 'labels') and dataset.labels:
            compute_statistics(results)
    
    logger.info("Done!")

def compute_statistics(results):
    """
    Compute statistics to evaluate extraction performance.
    
    Args:
        results: List of extraction results with ground truth
    """
    # Initialize counters
    stats = {
        'invoice_number': {'correct': 0, 'total': 0},
        'date': {'correct': 0, 'total': 0},
        'total_amount': {'correct': 0, 'total': 0},
        'issuer_name': {'correct': 0, 'total': 0},
    }
    
    # Compare extracted data with ground truth
    for result in results:
        # Skip failed extractions
        if result.get('status') != 'success':
            continue
        
        # Skip if no ground truth
        ground_truth = result.get('ground_truth')
        if not ground_truth:
            continue
        
        # Check invoice number
        if 'invoice_number' in ground_truth and ground_truth['invoice_number']:
            stats['invoice_number']['total'] += 1
            if result.get('invoice_number') and ground_truth['invoice_number'] in result['invoice_number']:
                stats['invoice_number']['correct'] += 1
        
        # Check date
        if 'date' in ground_truth and ground_truth['date']:
            stats['date']['total'] += 1
            if result.get('date') and str(ground_truth['date']) in str(result['date']):
                stats['date']['correct'] += 1
        
        # Check total amount
        if 'total' in ground_truth and ground_truth['total']:
            stats['total_amount']['total'] += 1
            if result.get('total_amount'):
                # Compare as strings to handle different formats
                gt_total = str(ground_truth['total']).replace(',', '').replace(' ', '')
                extracted_total = str(result['total_amount']).replace(',', '').replace(' ', '')
                if gt_total in extracted_total or extracted_total in gt_total:
                    stats['total_amount']['correct'] += 1
        
        # Check issuer name
        if 'company' in ground_truth and ground_truth['company']:
            stats['issuer_name']['total'] += 1
            if result.get('issuer_name') and ground_truth['company'] in result['issuer_name']:
                stats['issuer_name']['correct'] += 1
    
    # Calculate accuracy for each field
    for field, counts in stats.items():
        if counts['total'] > 0:
            accuracy = counts['correct'] / counts['total'] * 100
            logger.info(f"{field} accuracy: {accuracy:.2f}% ({counts['correct']}/{counts['total']})")
        else:
            logger.info(f"{field}: No ground truth available")
    
    # Calculate overall accuracy
    total_correct = sum(counts['correct'] for counts in stats.values())
    total_fields = sum(counts['total'] for counts in stats.values())
    
    if total_fields > 0:
        overall_accuracy = total_correct / total_fields * 100
        logger.info(f"Overall accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_fields})")

if __name__ == "__main__":
    main() 