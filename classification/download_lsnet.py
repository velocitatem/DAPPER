#!/usr/bin/env python3
"""
Download pre-trained LSNet models from Hugging Face.

This script downloads pre-trained LSNet models from Hugging Face and saves them locally.
"""

import argparse
import os
import torch
import requests
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download pre-trained LSNet models')
    parser.add_argument('--model', type=str, default='lsnet_t', 
                        help='Model to download (lsnet_t, lsnet_s, lsnet_b)')
    parser.add_argument('--output_dir', type=str, default='models', 
                        help='Directory to save the model')
    parser.add_argument('--distilled', action='store_true', 
                        help='Download the distilled version of the model')
    return parser.parse_args()

def download_file(url, output_path):
    """Download a file from a URL."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Download file
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=os.path.basename(output_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set model name
    model_name = args.model
    if args.distilled:
        model_name = f"{model_name}_distill"
    
    # Set URL
    url = f"https://huggingface.co/jameslahm/lsnet/resolve/main/{model_name}.pth"
    
    # Set output path
    output_path = os.path.join(args.output_dir, f"{model_name}.pth")
    
    # Download model
    logger.info(f"Downloading {model_name} from {url}")
    download_file(url, output_path)
    logger.info(f"Model saved to {output_path}")

if __name__ == '__main__':
    main() 