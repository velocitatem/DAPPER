import pandas as pd
from datasets import load_dataset
import gc
from classification.data.minio_handler import MinioManager
from classification.data.augmentor import Augmentor
from classification.utils.seed import set_global_seed
from classification.utils.logger import get_logger
from typing import Optional
from classification.datasets.cache_loader import load_dataset_from_cache, save_dataset_to_cache

# Initialize logger
logger_obj = get_logger("rvl_cdip_loader")
logger = logger_obj.logger

class RVLCDIPLoader:
    def __init__(self, minio_manager: MinioManager, augmentor: Augmentor, seed: int = 42):
        self.minio_manager = minio_manager
        self.augmentor = augmentor
        self.seed = seed
        set_global_seed(seed)
        logger.info(f"RVLCDIPLoader initialized with seed {seed}")

    def load_dataset(
        self,
        chunk_size: int = 1000,
        total_samples: Optional[int] = 10000,
        apply_augmentation: bool = True,
        augmentation_factor: int = 2,
    ) -> pd.DataFrame:
        logger.info("Loading RVL-CDIP dataset")
        logger.info(f"Parameters: chunk_size={chunk_size}, total_samples={total_samples}, "
                   f"apply_augmentation={apply_augmentation}, augmentation_factor={augmentation_factor}")
        
        # Try to load from cache
        cache = load_dataset_from_cache("rvl_cdip", {
            "chunk_size": chunk_size,
            "total_samples": total_samples,
            "apply_augmentation": apply_augmentation,
            "augmentation_factor": augmentation_factor,
        })
        if cache is not None:
            logger.info(f"Loaded dataset from cache with {len(cache)} samples")
            return cache

        # Load dataset from source
        logger.info("Loading dataset from HuggingFace")
        dataset = load_dataset("aharley/rvl_cdip", split='train', trust_remote_code=True)
        dataset = dataset.shuffle(seed=self.seed)
        logger.info(f"Dataset loaded with {len(dataset)} samples total")

        # Adjust total samples if needed
        dataset_size = len(dataset)
        if total_samples is None or total_samples > dataset_size:
            total_samples = dataset_size
            logger.info(f"Adjusted total_samples to {total_samples}")

        # Process dataset in chunks
        all_chunks = []
        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)
            logger.info(f"Processing chunk: samples {start_idx}-{end_idx} ({end_idx-start_idx} samples)")

            # Select chunk and convert to DataFrame
            chunk = dataset.select(range(start_idx, end_idx))
            df = pd.DataFrame(chunk)
            df['source_dataset'] = 'rvl_cdip'
            df['is_augmented'] = False
            df['global_id'] = [f"rvl_cdip_{start_idx + i}" for i in range(len(df))]
            logger.debug(f"Chunk DataFrame created with {len(df)} rows")

            # Process images
            logger.info("Resizing images and uploading to MinIO")
            df['image'] = df['image'].apply(lambda img: self.augmentor.resize_image(img))
            df = self.minio_manager.upload_dataframe(df)

            # Apply augmentation if enabled
            if apply_augmentation:
                logger.info(f"Creating {augmentation_factor}x augmentations")
                augmented_df = self.augmentor.create_augmented_rows(df, factor=augmentation_factor)
                augmented_df = self.minio_manager.upload_dataframe(augmented_df)
                df = pd.concat([df, augmented_df], ignore_index=True)
                logger.debug(f"Added {len(augmented_df)} augmented samples")
            else:
                logger.debug("Augmentation disabled, dropping augmented_image column")
                if 'augmented_image' in df.columns:
                    df = df.drop(columns=['augmented_image'])

            all_chunks.append(df)
            logger.info(f"Chunk {start_idx}-{end_idx} processed with {len(df)} final samples")
            
            # Clean up memory
            gc.collect()
            logger.debug("Memory cleaned up with gc.collect()")

        # Combine all chunks
        final_df = pd.concat(all_chunks, ignore_index=True)
        logger.info(f"RVL-CDIP dataset loaded with {len(final_df)} total samples")
        
        # Cache the result
        save_dataset_to_cache("rvl_cdip", {
            "chunk_size": chunk_size,
            "total_samples": total_samples,
            "apply_augmentation": apply_augmentation,
            "augmentation_factor": augmentation_factor,
        }, final_df)
        
        return final_df

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
    loader = RVLCDIPLoader(minio_manager, augmentor)
    df = loader.load_dataset(total_samples=1000)
    print(f"Loaded dataset with {len(df)} samples")
    print(df.head())