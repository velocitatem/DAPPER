##
# @file loader_hf_invoices.py
# @package classification.datasets.loader_hf_invoices
# @brief HuggingFace Invoices dataset loader for document classification
#
# This module provides functionality for loading and processing invoice images
# from the HuggingFace datasets repository. It handles data loading, image conversion,
# augmentation, and integration with MinIO storage for document classification tasks.
# source: https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1
#
# @author Statistical Learning Team
# @date 2025
#

import pandas as pd
import io
from PIL import Image
import gc
from datasets import load_dataset
from classification.data.minio_handler import MinioManager
from classification.data.augmentor import Augmentor
from classification.utils.seed import set_global_seed
from classification.utils.logger import get_logger
from typing import Optional
from classification.datasets.cache_loader import load_dataset_from_cache, save_dataset_to_cache

# Initialize logger
logger_obj = get_logger("hf_invoices_loader")
logger = logger_obj.logger

##
# @brief Loader class for HuggingFace Invoices dataset
#
# This class provides methods for loading and processing invoice images from the
# HuggingFace datasets repository. It handles data loading, image conversion,
# augmentation, and integration with MinIO storage for document classification tasks.
#
class HFInvoicesLoader:
    ##
    # @brief Constructor for HFInvoicesLoader class
    # @param minio_manager MinIO manager instance for data storage
    # @param augmentor Augmentor instance for data augmentation
    # @param seed Random seed for reproducibility
    #
    def __init__(self, minio_manager: MinioManager, augmentor: Augmentor, seed: int = 42):
        self.minio_manager = minio_manager
        self.augmentor = augmentor
        self.seed = seed
        set_global_seed(seed)
        logger.info(f"HFInvoicesLoader initialized with seed {seed}")
        
    ##
    # @brief Loads and processes the HuggingFace Invoices dataset
    # @param chunk_size Number of samples to process in each chunk
    # @param total_samples Total number of samples to load (None for all)
    # @param apply_augmentation Whether to apply data augmentation
    # @param augmentation_factor Number of augmented versions to create per image
    # @param split Dataset split to load ('train', 'validation', or 'test')
    # @return DataFrame containing processed dataset
    #
    def load_dataset(
        self,
        chunk_size: int = 100,
        total_samples: Optional[int] = None,
        apply_augmentation: bool = True,
        augmentation_factor: int = 3,
        split: str = "train",
    ) -> pd.DataFrame:
        logger.info("Loading HuggingFace Invoices dataset")
        logger.info(f"Parameters: chunk_size={chunk_size}, total_samples={total_samples}, "
                   f"apply_augmentation={apply_augmentation}, augmentation_factor={augmentation_factor}, "
                   f"split={split}")
        
        # Try to load from cache
        cache = load_dataset_from_cache("hf_invoices", {
            "chunk_size": chunk_size,
            "total_samples": total_samples,
            "apply_augmentation": apply_augmentation,
            "augmentation_factor": augmentation_factor,
            "split": split,
        })
        if cache is not None:
            logger.info(f"Loaded dataset from cache with {len(cache)} samples")
            return cache
            
        # Define dataset splits
        splits = {
            'train': 'data/train-00000-of-00001-a5c51039eab2980a.parquet',
            'validation': 'data/validation-00000-of-00001-b8a5c4a6237baf25.parquet',
            'test': 'data/test-00000-of-00001-56af6bd5ff7eb34d.parquet'
        }
        
        if split not in splits:
            logger.error(f"Invalid split '{split}'. Available splits: {list(splits.keys())}")
            raise ValueError(f"Invalid split '{split}'. Available splits: {list(splits.keys())}")
            
        # Load dataset from HuggingFace
        logger.info(f"Loading {split} split from HuggingFace dataset")
        try:
            dataset_path = f"hf://datasets/katanaml-org/invoices-donut-data-v1/{splits[split]}"
            logger.info(f"Loading dataset from {dataset_path}")
            df = pd.read_parquet(dataset_path)
            logger.info(f"Loaded {len(df)} samples from parquet file")
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise
            
        # Limit samples if specified
        if total_samples is not None and total_samples < len(df):
            logger.info(f"Limiting to {total_samples} samples")
            df = df.sample(total_samples, random_state=self.seed)
            
        # Process dataset in chunks
        all_chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size].copy()
            logger.info(f"Processing chunk {i//chunk_size+1}/{(len(df)-1)//chunk_size+1} with {len(chunk)} samples")
            
            # Convert bytes to images
            logger.info("Converting byte data to PIL Images")
            chunk['image'] = chunk['image'].apply(lambda x: self._bytes_to_image(x))
            
            # Remove rows with failed image loading
            valid_rows = chunk.dropna(subset=['image'])
            if len(valid_rows) < len(chunk):
                logger.warning(f"Dropped {len(chunk) - len(valid_rows)} rows with failed image loading")
                chunk = valid_rows
                
            # Add invoice label
            chunk['label'] = 11  # Invoice category
            
            # Resize images
            logger.info("Resizing images")
            chunk['image'] = chunk['image'].apply(lambda img: self.augmentor.resize_image(img))
            
            # Add metadata
            chunk['source_dataset'] = 'hf_invoices'
            chunk['is_augmented'] = False
            chunk['global_id'] = [f"hf_invoices_{i + j}" for j in range(len(chunk))]
            
            # Upload to MinIO
            logger.info("Uploading images to MinIO")
            chunk = self.minio_manager.upload_dataframe(chunk)
            
            # Apply augmentation if enabled
            if apply_augmentation:
                logger.info(f"Creating {augmentation_factor}x augmentations")
                augmented_df = self.augmentor.create_augmented_rows(chunk, factor=augmentation_factor)
                augmented_df = self.minio_manager.upload_dataframe(augmented_df)
                chunk = pd.concat([chunk, augmented_df], ignore_index=True)
                logger.debug(f"Added {len(augmented_df)} augmented samples")
                
            all_chunks.append(chunk)
            logger.info(f"Chunk processed with {len(chunk)} final samples")
            
            # Clean up memory
            gc.collect()
            logger.debug("Memory cleaned up with gc.collect()")
            
        # Combine all chunks
        final_df = pd.concat(all_chunks, ignore_index=True)
        logger.info(f"HuggingFace Invoices dataset loaded with {len(final_df)} total samples")
        
        # Keep only required columns
        final_columns = ['image', 'label', 'source_dataset', 'is_augmented', 'global_id']
        final_df = final_df[final_columns]
        
        # Cache the result
        save_dataset_to_cache("hf_invoices", {
            "chunk_size": chunk_size,
            "total_samples": total_samples,
            "apply_augmentation": apply_augmentation,
            "augmentation_factor": augmentation_factor,
            "split": split,
        }, final_df)
        
        return final_df
        
    ##
    # @brief Converts bytes data to PIL Image
    # @param image_data Image data in bytes format
    # @return PIL Image or None if conversion fails
    #
    def _bytes_to_image(self, image_data):
        """Convert bytes data to PIL Image"""
        try:
            if isinstance(image_data, dict) and 'bytes' in image_data:
                return Image.open(io.BytesIO(image_data['bytes']))
            else:
                logger.warning(f"Image data not in expected format: {type(image_data)}")
                return None
        except Exception as e:
            logger.warning(f"Failed to convert bytes to image: {str(e)}")
            return None
            
##
# @brief Main execution block for testing dataset loading
#
# This block initializes the necessary components and demonstrates
# how to load and process the HuggingFace Invoices dataset.
#
if __name__ == "__main__":
    # Configure logging format for direct script execution
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize components and load dataset
    minio_manager = MinioManager()
    augmentor = Augmentor()
    loader = HFInvoicesLoader(minio_manager, augmentor)
    df = loader.load_dataset(total_samples=100)
    print(f"Loaded dataset with {len(df)} samples")
    print(df.head()) 