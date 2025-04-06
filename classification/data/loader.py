##
# @file loader.py
# @brief Data loading utility for document classification
#
# This module provides functionality for loading and combining document datasets
# from various sources. It supports parallel loading of multiple datasets and
# handles the integration of different data sources into a unified format.
#
# @author Statistical Learning Team
# @date 2025-03-20
#

import torch
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import os
import pickle
from typing import Dict, List, Tuple, Optional, Union, Callable
from classification.datasets.loader_rvl_cdip import RVLCDIPLoader
from classification.datasets.loader_kaggle_invoices import KaggleInvoicesLoader
from classification.datasets.loader_hf_invoices import HFInvoicesLoader
from classification.utils.seed import set_global_seed 
from classification.data.minio_handler import MinioManager
from classification.data.augmentor import Augmentor
from classification.utils.logger import get_standard_logger

# Use the standard logger directly
logger = get_standard_logger("data_loader")

##
# @brief Dictionary mapping dataset names to their loader classes
#
# This dictionary provides a mapping between dataset names and their corresponding
# loader classes, enabling dynamic loading of different dataset types.
#
datasets = {
    'rvl_cdip': RVLCDIPLoader,
    'kaggle_invoices': KaggleInvoicesLoader,
    'hf_invoices': HFInvoicesLoader
}

##
# @brief Loads a single dataset by name
# @param dataset_name Name of the dataset to load
# @param minio_manager MinIO manager instance for data access
# @param augmentor Augmentor instance for data augmentation
# @param apply_ocr Whether to apply OCR processing to the images
# @return DataFrame containing the loaded dataset
#
def get_dataset(dataset_name: str, minio_manager: MinioManager, augmentor: Augmentor, apply_ocr: bool = False):
    logger.info(f"Loading dataset: {dataset_name} with OCR: {apply_ocr}")
    return datasets[dataset_name](minio_manager, augmentor, apply_ocr=apply_ocr).load_dataset()

##
# @brief Loads and combines multiple datasets in parallel
# @param datasets_list List of dataset names to load
# @param minio_manager MinIO manager instance for data access
# @param augmentor Augmentor instance for data augmentation
# @param apply_ocr Whether to apply OCR processing to the images
# @param ocr_workers Number of parallel workers for OCR processing
# @param ocr_config Optional configuration for tesseract OCR
# @return Combined DataFrame containing all datasets
#
def get_full_dataset(
    datasets_list: List[str], 
    minio_manager: MinioManager, 
    augmentor: Augmentor,
    apply_ocr: bool = False,
):
    logger.info(f"Loading datasets in parallel: {datasets_list}")
    # parallel load datasets
    with ThreadPoolExecutor(max_workers=len(datasets_list)) as executor:
        futures = [executor.submit(get_dataset, dataset_name, minio_manager, augmentor, apply_ocr) for dataset_name in datasets_list]
        
        # Combine all datasets
        combined_df = pd.concat([future.result() for future in futures])
        logger.info(f"Combined dataset created with {len(combined_df)} samples")
        # remove duplicates of url-image pair
        combined_df = combined_df.drop_duplicates(subset=['label', 'image'])
        logger.info(f"Combined dataset created with {len(combined_df)} samples")
        return combined_df
    
##
# @brief Main execution block for testing dataset loading
#
# This block initializes the necessary components and demonstrates
# how to load and combine multiple datasets.
#
if __name__ == "__main__":
    # Initialize MinIO manager and augmentor
    logger.info("Initializing MinIO manager and augmentor")
    minio_manager = MinioManager()
    augmentor = Augmentor()
    
    datasets = {
        'rvl_cdip': RVLCDIPLoader,
        'kaggle_invoices': KaggleInvoicesLoader,
        'hf_invoices': HFInvoicesLoader
    }
    # Load all datasets
    df = get_full_dataset(list(datasets.keys()), minio_manager, augmentor, apply_ocr=True)
    logger.info(f"Loaded and combined {len(df)} samples")
    
    # Show sample data
    print(df.head())