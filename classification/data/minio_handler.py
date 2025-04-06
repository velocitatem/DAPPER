##
# @file minio_handler.py
# @package classification.data.minio_handler
# @brief MinIO object storage manager for document classification
#
# This module provides functionality for managing document images in MinIO object storage.
# It handles uploading, organizing, and retrieving images with proper naming conventions
# and supports parallel processing for efficient data management.
#
# Files are stored in the bucket with the following naming convention:
# src_<source_dataset>_cls_<class_label>_idx_<index>_<augmentation_info>.jpg
# <source_dataset> is the name of the dataset the image belongs to.
# <class_label> is the class label of the image.
# <index> is the index of the image in the dataset.
# <augmentation_info> is either "aug" if the image is an augmented version, or "orig" if it is the original image.
#
# @author Statistical Learning Team
# @date 2025
#

import io
import re
import logging
from minio import Minio
from PIL import Image
from typing import Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

##
# @brief Manager class for MinIO object storage operations
#
# This class provides methods for interacting with MinIO object storage,
# including uploading images, managing buckets, and organizing data with
# consistent naming conventions for document classification tasks.
#
class MinioManager:
    ##
    # @brief Constructor for MinioManager class
    # @param endpoint MinIO server endpoint URL
    # @param access_key MinIO access key
    # @param secret_key MinIO secret key
    # @param bucket_name Name of the MinIO bucket to use
    # @param secure Use TLS for MinIO connection
    #
    def __init__(self,
                 endpoint = "localhost:9900",
                 access_key = "minioadmin",
                 secret_key= "minioadmin",
                 bucket_name = "dapper",
                 secure: bool = False):
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self.endpoint = endpoint
        self.bucket_name = bucket_name
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)
        self._refresh_object_list()

    ##
    # @brief Refreshes the list of objects in the bucket
    #
    # Updates the internal cache of object names in the MinIO bucket
    # to track which images have already been uploaded.
    #
    def _refresh_object_list(self):
        self.all_images = set(obj.object_name for obj in self.client.list_objects(self.bucket_name))

    ##
    # @brief Generates a standardized filename for image storage
    # @param label Class label for the image
    # @param index Index of the image in the dataset
    # @param source Source dataset name
    # @param is_augmented Whether the image is an augmented version
    # @return Standardized filename for MinIO storage
    #
    def _generate_filename(self, label: str, index: int, source: str, is_augmented: bool) -> str:
        clean_label = re.sub(r'\\W+', '', str(label))
        aug_info = "aug" if is_augmented else "orig"
        return f"src_{source}_cls_{clean_label}_idx_{index}_{aug_info}.jpg"

    ##
    # @brief Uploads an image to MinIO storage
    # @param image PIL Image to upload
    # @param filename Name to use for the uploaded file
    # @return URL of the uploaded image
    #
    def upload_image(self, image: Image.Image, filename: str) -> str:
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)
        self.client.put_object(
            self.bucket_name,
            filename,
            image_bytes,
            len(image_bytes.getvalue())
        )
        self.all_images.add(filename)
        return f"http://{self.endpoint}/{self.bucket_name}/{filename}"

    ##
    # @brief Processes a single row from a DataFrame for upload
    # @param index Index of the row in the DataFrame
    # @param row Series containing image and metadata
    # @return Tuple of (index, URL) or (index, None) if processing failed
    #
    def process_row(self, index: int, row: pd.Series) -> Optional[tuple]:
        image = row['image']
        if image is None or not isinstance(image, Image.Image):
            logging.warning(f"Row {index} has no valid image, skipping.")
            return index, None

        label = str(row['label'])
        source = row.get('source_dataset', 'unknown')
        is_augmented = row.get('is_augmented', False)

        filename = self._generate_filename(label, index, source, is_augmented)
        if filename in self.all_images:
            return index, f"http://{self.endpoint}/{self.bucket_name}/{filename}"

        image = image.convert("RGB")
        return index, self.upload_image(image, filename)

    ##
    # @brief Uploads all images from a DataFrame to MinIO
    # @param df DataFrame containing images and metadata
    # @param max_workers Maximum number of worker threads for parallel processing
    # @return DataFrame with updated image URLs
    #
    def upload_dataframe(self, df: pd.DataFrame, max_workers: int = 12) -> pd.DataFrame:
        args = [(i, row) for i, row in df.iterrows()]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(lambda args: self.process_row(*args), args)
        for i, url in results:
            if url is not None:
                df.at[i, 'image'] = url
        return df
