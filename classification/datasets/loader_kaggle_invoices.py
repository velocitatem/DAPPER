##
# @file loader_kaggle_invoices.py
# @package classification.datasets.loader_kaggle_invoices
# @brief Kaggle Invoices dataset loader for document classification
#
# This module provides functionality for loading and processing invoice images
# from the Kaggle platform. It handles data downloading, PDF conversion,
# augmentation, and integration with MinIO storage for document classification tasks.
# source: https://www.kaggle.com/datasets/ayoubcherguelaine/company-documents-dataset
#
# @author Statistical Learning Team
# @date 2025
#

import pandas as pd
import kagglehub
from pdf2image import convert_from_path
import gc
import logging
from classification.data.minio_handler import MinioManager
from classification.data.augmentor import Augmentor
from classification.utils.seed import set_global_seed
from classification.utils.logger import get_logger
from typing import Optional
from classification.datasets.cache_loader import load_dataset_from_cache, save_dataset_to_cache

# Initialize logger
logger_obj = get_logger("kaggle_invoices_loader")
logger = logger_obj.logger

##
# @brief Loader class for Kaggle Invoices dataset
#
# This class provides methods for loading and processing invoice images from the
# Kaggle platform. It handles data downloading, PDF conversion, augmentation,
# and integration with MinIO storage for document classification tasks.
#
class KaggleInvoicesLoader:
    ##
    # @brief Constructor for KaggleInvoicesLoader class
    # @param minio_manager MinIO manager instance for data storage
    # @param augmentor Augmentor instance for data augmentation
    # @param seed Random seed for reproducibility
    #
    def __init__(self, minio_manager: MinioManager, augmentor: Augmentor, seed: int = 42):
        self.minio_manager = minio_manager
        self.augmentor = augmentor
        self.seed = seed
        set_global_seed(seed)
        logger.info(f"KaggleInvoicesLoader initialized with seed {seed}")

    ##
    # @brief Loads and processes the Kaggle Invoices dataset
    # @param chunk_size Number of samples to process in each chunk
    # @param total_samples Total number of samples to load (None for all)
    # @param apply_augmentation Whether to apply data augmentation
    # @param augmentation_factor Number of augmented versions to create per image
    # @return DataFrame containing processed dataset
    #
    def load_dataset(
        self,
        chunk_size: int = 100,
        total_samples: Optional[int] = None,
        apply_augmentation: bool = True,
        augmentation_factor: int = 3,
    ) -> pd.DataFrame:
        logger.info("Loading Kaggle Invoices dataset")
        logger.info(f"Parameters: chunk_size={chunk_size}, total_samples={total_samples}, "
                   f"apply_augmentation={apply_augmentation}, augmentation_factor={augmentation_factor}")
        
        # Try to load from cache
        cache = load_dataset_from_cache("kaggle_invoices", {
            "chunk_size": chunk_size,
            "total_samples": total_samples,
            "apply_augmentation": apply_augmentation,
            "augmentation_factor": augmentation_factor,
        })
        if cache is not None:
            logger.info(f"Loaded dataset from cache with {len(cache)} samples")
            return cache

        # Download dataset from Kaggle
        logger.info("Downloading Kaggle invoices dataset")
        try:
            dataset_path = kagglehub.dataset_download("ayoubcherguelaine/company-documents-dataset")
            logger.info(f"Dataset downloaded to {dataset_path}")
        except Exception as e:
            logger.error(f"Failed to download dataset: {str(e)}")
            raise

        # Load metadata CSV
        logger.info("Loading dataset metadata CSV")
        df = pd.read_csv(f"{dataset_path}/company-document-text.csv")
        
        # Filter for invoices only
        invoices = df[df['label'] == 'invoice']
        invoices['label'] = 11
        logger.info(f"Filtered {len(invoices)} invoice samples from dataset")
        
        # Extract invoice IDs
        logger.info("Extracting invoice IDs")
        invoices['invoice_id'] = invoices['text'].apply(self._extract_invoice_id)
        
        # Limit samples if specified
        if total_samples is not None and total_samples < len(invoices):
            logger.info(f"Limiting to {total_samples} samples")
            invoices = invoices.sample(total_samples, random_state=self.seed)
        
        # Process invoices in chunks
        all_chunks = []
        directory = f"{dataset_path}/CompanyDocuments/invoices/"
        
        for i in range(0, len(invoices), chunk_size):
            chunk = invoices.iloc[i:i+chunk_size].copy()
            logger.info(f"Processing chunk {i//chunk_size+1}/{(len(invoices)-1)//chunk_size+1} with {len(chunk)} samples")
            
            # Load PDF and convert to images
            chunk['image'] = chunk['invoice_id'].apply(lambda id: self._load_pdf_as_image(directory, id))
            
            # Remove rows with failed image loading
            valid_rows = chunk.dropna(subset=['image'])
            if len(valid_rows) < len(chunk):
                logger.warning(f"Dropped {len(chunk) - len(valid_rows)} rows with failed image loading")
                chunk = valid_rows
            
            # Resize images
            logger.info("Resizing images")
            chunk['image'] = chunk['image'].apply(lambda img: self.augmentor.resize_image(img))
            
            # Add metadata
            chunk['source_dataset'] = 'kaggle_invoices'
            chunk['is_augmented'] = False
            chunk['global_id'] = [f"kaggle_invoices_{i + j}" for j in range(len(chunk))]
            
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
        logger.info(f"Kaggle Invoices dataset loaded with {len(final_df)} total samples")
        
        # Keep only required columns
        final_columns = ['image', 'label', 'source_dataset', 'is_augmented', 'global_id']
        final_df = final_df[final_columns]
        
        # Cache the result
        save_dataset_to_cache("kaggle_invoices", {
            "chunk_size": chunk_size,
            "total_samples": total_samples,
            "apply_augmentation": apply_augmentation,
            "augmentation_factor": augmentation_factor,
        }, final_df)
        
        return final_df
    
    ##
    # @brief Extracts invoice ID from text description
    # @param text Text description containing invoice ID
    # @return Invoice ID as integer or None if extraction fails
    #
    def _extract_invoice_id(self, text):
        """Extract invoice ID from text description"""
        try:
            text = text.split()
            return int(text[3])
        except (IndexError, ValueError) as e:
            logger.warning(f"Failed to extract invoice ID from '{text}': {str(e)}")
            return None
    
    ##
    # @brief Loads a PDF file and converts it to an image
    # @param directory Directory containing PDF files
    # @param invoice_id ID of the invoice to load
    # @return PIL Image or None if conversion fails
    #
    def _load_pdf_as_image(self, directory, invoice_id):
        """Load a PDF file and convert it to an image"""
        try:
            path = f"{directory}invoice_{invoice_id}.pdf"
            logger.debug(f"Converting PDF {path} to image")
            image = convert_from_path(path)
            return image[0] if len(image) > 0 else None
        except Exception as e:
            logger.warning(f"Failed to convert PDF for invoice {invoice_id}: {str(e)}")
            return None

##
# @brief Main execution block for testing dataset loading
#
# This block initializes the necessary components and demonstrates
# how to load and process the Kaggle Invoices dataset.
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
    loader = KaggleInvoicesLoader(minio_manager, augmentor)
    df = loader.load_dataset(total_samples=100)
    print(f"Loaded dataset with {len(df)} samples")
    print(df.head()) 