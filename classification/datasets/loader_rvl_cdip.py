import pandas as pd
from datasets import load_dataset
import logging
import gc
from classification.data.minio_handler import MinioManager
from classification.data.augmentor import Augmentor
from classification.utils.seed import set_global_seed
from typing import Optional
from classification.datasets.cache_loader import load_dataset_from_cache, save_dataset_to_cache
class RVLCDIPLoader:
    def __init__(self, minio_manager: MinioManager, augmentor: Augmentor, seed: int = 42):
        self.minio_manager = minio_manager
        self.augmentor = augmentor
        self.seed = seed
        set_global_seed(seed)

    def group_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """

        classes = ["letter", "form", "email", "handwritten", "advertisement", "scientific report",
           "scientific publication", "specification", "file folder", "news article", "budget",
           "invoice", "presentation", "questionnaire", "resume", "memo"]
        """
        mapping = {
            "invoice": "invoice",
            'letter': 'correspondence',
            "email": "correspondence",
            "memo": "correspondence",
            "form": "forms",
            "questionnaire": "forms",
            "specification": "forms",
            "scientific report": "scientific",
            "scientific publication": "scientific",
            "advertisement": "promotional",
            "presentation": "promotional",
            "resume": "personal",
            "handwritten": "personal",
            "file folder": "other",
            "news article": "other",
            "budget": "other",
        }
        df['label'] = df['class'].map(mapping)
        df = df.dropna(subset=['label'])
        df = df.reset_index(drop=True)
        return df

    def load_dataset(
        self,
        chunk_size: int = 1000,
        total_samples: Optional[int] = 10000,
        apply_augmentation: bool = True,
        augmentation_factor: int = 2,
    ) -> pd.DataFrame:
        logging.info("Loading RVL-CDIP dataset")
        dataset = load_dataset("aharley/rvl_cdip", split='train', trust_remote_code=True)
        dataset = dataset.shuffle(seed=self.seed)
        cache = load_dataset_from_cache("rvl_cdip", {
            "chunk_size": chunk_size,
            "total_samples": total_samples,
            "apply_augmentation": apply_augmentation,
            "augmentation_factor": augmentation_factor,
        })
        if cache is not None:
            return cache

        dataset_size = len(dataset)
        if total_samples is None or total_samples > dataset_size:
            total_samples = dataset_size

        all_chunks = []
        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)
            logging.info(f"Processing chunk: samples {start_idx}-{end_idx}")

            chunk = dataset.select(range(start_idx, end_idx))
            df = pd.DataFrame(chunk)
            df['source_dataset'] = 'rvl_cdip'
            df['is_augmented'] = False
            df['global_id'] = [f"rvl_cdip_{start_idx + i}" for i in range(len(df))]
            df = self.group_classes(df)

            # Resize images and upload to MinIO
            df['image'] = df['image'].apply(lambda img: self.augmentor.resize_image(img))
            df = self.minio_manager.upload_dataframe(df)

            # Apply augmentation if enabled
            if apply_augmentation:
                augmented_df = self.augmentor.create_augmented_rows(df, factor=augmentation_factor)
                augmented_df = self.minio_manager.upload_dataframe(augmented_df)
                df = pd.concat([df, augmented_df], ignore_index=True)
            else:
                df = df.drop(columns=['augmented_image'])

            all_chunks.append(df)
            gc.collect()

        final_df = pd.concat(all_chunks, ignore_index=True)
        logging.info(f"RVL-CDIP loaded with {len(final_df)} total samples")
        save_dataset_to_cache("rvl_cdip", {
            "chunk_size": chunk_size,
            "total_samples": total_samples,
            "apply_augmentation": apply_augmentation,
            "augmentation_factor": augmentation_factor,
        }, final_df)
        return final_df

if __name__ == "__main__":
    minio_manager = MinioManager()
    augmentor = Augmentor()
    loader = RVLCDIPLoader(minio_manager, augmentor)
    df = loader.load_dataset()
    print(df.head())