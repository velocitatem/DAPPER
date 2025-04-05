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


datasets = {
    'rvl_cdip': RVLCDIPLoader,
    'kaggle_invoices': KaggleInvoicesLoader,
    'hf_invoices': HFInvoicesLoader
}

def get_dataset(dataset_name: str, minio_manager: MinioManager, augmentor: Augmentor):
    logger.info(f"Loading dataset: {dataset_name}")
    return datasets[dataset_name](minio_manager, augmentor).load_dataset()

def get_full_dataset(datasets_list: List[str], minio_manager: MinioManager, augmentor: Augmentor):
    logger.info(f"Loading datasets in parallel: {datasets_list}")
    # parallel load datasets
    with ThreadPoolExecutor(max_workers=len(datasets_list)) as executor:
        futures = [executor.submit(get_dataset, dataset_name, minio_manager, augmentor) for dataset_name in datasets_list]
        
        # Combine all datasets
        combined_df = pd.concat([future.result() for future in futures])
        logger.info(f"Combined dataset created with {len(combined_df)} samples")
        # remove duplicates of url-image pair
        combined_df = combined_df.drop_duplicates(subset=['label', 'image'])
        logger.info(f"Combined dataset created with {len(combined_df)} samples")
        return combined_df
    
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
    df = get_full_dataset(list(datasets.keys()), minio_manager, augmentor)
    logger.info(f"Loaded and combined {len(df)} samples")
    
    # Show sample data
    print(df.head())