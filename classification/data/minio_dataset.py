##
# @file minio_dataset.py
# @package classification.data.minio_dataset
# @brief MinIO-based PyTorch datasets for document classification
#
# This module provides PyTorch Dataset implementations for loading images and text
# from MinIO object storage. It includes both image-only and multimodal datasets
# with OCR capabilities for document classification tasks.
#
# @author Statistical Learning Team
# @date 2025
#

import io
import logging
import torch
from torch.utils.data import Dataset
from PIL import Image
from minio import Minio
import pandas as pd
import pytesseract
from transformers import PreTrainedTokenizerBase
from typing import Optional, Tuple, Dict, Any
import concurrent.futures
from tqdm import tqdm
import time
from torchvision import transforms
from .augmentor import Augmentor

# Set Tesseract CMD if needed (uncomment and adjust path if not in system PATH)
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

logger = logging.getLogger(__name__)

##
# @brief PyTorch Dataset for loading images from MinIO storage
#
# This class extends PyTorch's Dataset class to load images from MinIO object storage.
# It takes a DataFrame with image URLs, connects to MinIO, fetches images by their names,
# applies transformations, and returns image-label pairs for model training.
#
class MinioImageDataset(Dataset):
    """
    PyTorch Dataset for loading images from MinIO storage
    """
    ##
    # @brief Constructor for MinioImageDataset class
    # @param dataframe DataFrame containing 'image' URLs and 'label' columns
    # @param bucket_name Name of the MinIO bucket to fetch images from
    # @param transform Optional transformations to apply to the images
    # @param label_map Optional dictionary mapping label strings to integers
    #
    def __init__(self, dataframe, bucket_name, transform=None, label_map=None):
        """
        Constructor for MinioImageDataset class
        
        Args:
            dataframe: DataFrame containing 'image' URLs and 'label' columns
            bucket_name: Name of the MinIO bucket to fetch images from
            transform: Optional transformations to apply to the images
            label_map: Optional dictionary mapping label strings to integers
        """
        self.df = dataframe.reset_index(drop=True)
        self.bucket_name = bucket_name
        self.transform = transform
        self.client = Minio(
            "localhost:9900",
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False
        )

        # Create label_map if not provided
        if label_map is None and 'label' in self.df.columns:
            unique_labels = self.df['label'].unique()
            if any(not isinstance(label, (int, float)) for label in unique_labels):
                logger.info("Creating label map from string labels to integers")
                self.label_map = {str(label): i for i, label in enumerate(sorted(unique_labels))}
                # Log the mapping for reference
                for label, idx in self.label_map.items():
                    logger.info(f"Label mapping: '{label}' -> {idx}")
            else:
                self.label_map = None
        else:
            self.label_map = label_map

    ##
    # @brief Returns the number of items in the dataset
    # @return Number of images in the dataset
    #
    def __len__(self):
        """
        Returns the number of items in the dataset
        
        Returns:
            Number of images in the dataset
        """
        return len(self.df)

    ##
    # @brief Fetches and processes a single item from the dataset
    # @param idx Index of the item to fetch
    # @return Tuple of (transformed image tensor, class label tensor)
    #
    def __getitem__(self, idx):
        """
        Fetches and processes a single item from the dataset
        
        Args:
            idx: Index of the item to fetch
            
        Returns:
            Tuple of (transformed image tensor, class label tensor)
        """
        row = self.df.iloc[idx]
        image_url = row['image']
        
        # Extract image name from URL (assuming format: http://endpoint/bucket/image_name)
        image_name = image_url.split('/')[-1]

        try:
            response = self.client.get_object(self.bucket_name, image_name)
            image_data = response.read()
            response.close()
            response.release_conn()
        except Exception as e:
            logging.error(f"Error fetching {image_name}: {e}")
            raise e

        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Process label
        label_value = row['label']
        
        # Apply label mapping if available
        if self.label_map is not None and str(label_value) in self.label_map:
            label_idx = self.label_map[str(label_value)]
            label = torch.tensor(label_idx, dtype=torch.long)
        else:
            # Try to convert directly to int
            try:
                label = torch.tensor(int(label_value), dtype=torch.long)
            except (ValueError, TypeError):
                logger.error(f"Could not convert label '{label_value}' to integer and no mapping found")
                label = torch.tensor(-1, dtype=torch.long)  # Use -1 to indicate error

        return image, label

##
# @brief PyTorch Dataset for loading images and extracting text via OCR from MinIO
#
# This class extends PyTorch's Dataset class to load images from MinIO object storage
# and utilize pre-computed OCR results from the dataframe.
#
class MinioMultiModalDataset(Dataset):
    """
    PyTorch Dataset for loading images and pre-computed OCR text from MinIO.
    """
    ##
    # @brief Constructor for MinioMultiModalDataset class
    # @param dataframe DataFrame with 'image' URL, 'label', and OCR columns ('ocr_text', 'ocr_confidence', 'ocr_boxes')
    # @param bucket_name MinIO bucket name
    # @param image_transform Transformations for the image
    # @param tokenizer Tokenizer instance to process text
    # @param max_sentences Maximum number of sentences to keep from OCR text
    # @param max_sent_length Maximum number of tokens per sentence
    # @param vocab Vocabulary mapping tokens to indices
    # @param ocr_lang Tesseract language code(s) for fallback OCR if needed
    # @param minio_endpoint MinIO server endpoint URL
    # @param minio_access_key MinIO access key
    # @param minio_secret_key MinIO secret key
    # @param minio_secure Use TLS for MinIO connection
    # @param batch_tokenize Whether to tokenize in batches (more efficient)
    # @param label_map Optional dictionary mapping label strings to integers
    #
    def __init__(
        self,
        dataframe: pd.DataFrame,
        bucket_name: str,
        image_transform: Optional[callable] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_sentences: int = 15,
        max_sent_length: int = 50,
        vocab: Optional[dict] = None,
        ocr_lang: str = 'eng',
        minio_endpoint: str = "localhost:9900",
        minio_access_key: str = "minioadmin",
        minio_secret_key: str = "minioadmin",
        minio_secure: bool = False,
        batch_tokenize: bool = False,
        label_map: Optional[dict] = None
    ):
        """
        Args:
            dataframe: DataFrame with 'image' URL, 'label', and OCR columns.
            bucket_name: MinIO bucket name.
            image_transform: Transformations for the image.
            tokenizer: Tokenizer instance (e.g., from Hugging Face) to process text.
            max_sentences: Maximum number of sentences to keep from OCR text.
            max_sent_length: Maximum number of tokens per sentence.
            vocab: Vocabulary mapping tokens to indices (if tokenizer doesn't handle it).
            ocr_lang: Tesseract language code(s) for fallback OCR if needed.
            minio_endpoint: MinIO server endpoint URL.
            minio_access_key: MinIO access key.
            minio_secret_key: MinIO secret key.
            minio_secure: Use TLS for MinIO connection.
            batch_tokenize: Whether to tokenize in batches (more efficient).
            label_map: Optional dictionary mapping label strings to integers.
        """
        self.df = dataframe.reset_index(drop=True)
        self.bucket_name = bucket_name
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.max_sentences = max_sentences
        self.max_sent_length = max_sent_length
        self.vocab = vocab
        self.ocr_lang = ocr_lang
        self.batch_tokenize = batch_tokenize
        
        # Create label_map if not provided
        if label_map is None and 'label' in self.df.columns:
            unique_labels = self.df['label'].unique()
            if any(not isinstance(label, (int, float)) for label in unique_labels):
                logger.info("Creating label map from string labels to integers")
                self.label_map = {str(label): i for i, label in enumerate(sorted(unique_labels))}
                # Log the mapping for reference
                for label, idx in self.label_map.items():
                    logger.info(f"Label mapping: '{label}' -> {idx}")
            else:
                self.label_map = None
        else:
            self.label_map = label_map
        
        # Check if the dataframe has OCR text column
        self.has_ocr_column = 'ocr_text' in self.df.columns
        if not self.has_ocr_column:
            logger.warning("The dataframe does not contain 'ocr_text' column. OCR processing may be required at runtime.")
        
        # Store MinIO config
        self.minio_endpoint = minio_endpoint
        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.minio_secure = minio_secure

        # Initialize MinIO client lazily
        self._minio_client = None
        
        # Pre-tokenize text if batch_tokenize is True and tokenizer is available
        self.tokenized_texts = None
        if self.batch_tokenize and self.tokenizer is not None and self.has_ocr_column:
            logger.info("Batch tokenizing OCR text...")
            self.tokenized_texts = self._batch_tokenize_texts()
            logger.info(f"Batch tokenization completed for {len(self.tokenized_texts)} texts")
            
        if self.tokenizer is None and self.vocab is None:
             logger.warning("Neither tokenizer nor vocab provided to MinioMultiModalDataset. Text processing might fail.")

    ##
    # @brief Tokenizes all texts in the dataframe in a batch
    # @return List of tokenized text tensors
    #
    def _batch_tokenize_texts(self):
        """Tokenizes all OCR texts in the dataframe in one batch for efficiency."""
        all_texts = self.df['ocr_text'].tolist()
        tokenized_texts = []
        
        # Process in smaller batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i+batch_size]
            batch_tokenized = [self._preprocess_text(text) for text in batch_texts]
            tokenized_texts.extend(batch_tokenized)
            
        return tokenized_texts

    ##
    # @brief Property to get MinIO client, initializing if needed
    # @return Initialized MinIO client
    #
    @property
    def client(self):
        if self._minio_client is None:
            logger.debug(f"Initializing MinIO client for {self.minio_endpoint}")
            self._minio_client = Minio(
                self.minio_endpoint,
                access_key=self.minio_access_key,
                secret_key=self.minio_secret_key,
                secure=self.minio_secure
            )
        return self._minio_client

    ##
    # @brief Returns the number of items in the dataset
    # @return Number of items in the dataset
    #
    def __len__(self):
        return len(self.df)

    ##
    # @brief Tokenizes, numericalizes, and pads text to fit model input
    # @param text Text to process
    # @return Tensor of processed text
    #
    def _preprocess_text(self, text: str) -> torch.Tensor:
        """Tokenizes, numericalizes, and pads text to fit model input."""
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
            
        # Basic sentence splitting (can be improved with NLTK, spaCy etc.)
        sentences = [s.strip() for s in text.split('\n') if s.strip()][:self.max_sentences]
        if not sentences:
             sentences = [""]

        # --- Tokenization and Numericalization ---
        if self.tokenizer:
            # Example using Hugging Face tokenizer
            tokens_list = []
            for sent in sentences:
                 # Tokenize, add special tokens (if needed), pad/truncate
                 encoded = self.tokenizer.encode_plus(
                     sent,
                     max_length=self.max_sent_length,
                     padding='max_length',
                     truncation=True,
                     return_tensors=None
                 )
                 tokens_list.append(encoded['input_ids'])

            # Pad the list of sentences if fewer than max_sentences
            while len(tokens_list) < self.max_sentences:
                pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else 0
                tokens_list.append([pad_token_id] * self.max_sent_length)

            text_tensor = torch.tensor(tokens_list, dtype=torch.long)

        elif self.vocab:
             # Manual processing using a predefined vocab dictionary
             logger.warning("Manual vocab processing not fully implemented in placeholder.")
             # Create a dummy tensor matching the expected shape
             text_tensor = torch.zeros((self.max_sentences, self.max_sent_length), dtype=torch.long)
        else:
             # No tokenizer or vocab - cannot process text
             logger.error("Cannot process text: No tokenizer or vocab available.")
             # Return zero tensor matching expected shape
             text_tensor = torch.zeros((self.max_sentences, self.max_sent_length), dtype=torch.long)

        # Ensure final shape is [max_sentences, max_sent_length]
        if text_tensor.shape != (self.max_sentences, self.max_sent_length):
             logger.warning(f"Text tensor shape mismatch: expected ({self.max_sentences}, {self.max_sent_length}), got {text_tensor.shape}. Check padding/truncation.")

        return text_tensor

    ##
    # @brief Fallback OCR if text is not in dataframe
    # @param image PIL image object
    # @return Extracted OCR text
    #
    def _fallback_ocr(self, image: Image.Image) -> str:
        """Fallback OCR method if text is not in dataframe."""
        logger.warning("Using fallback OCR - this should be rare if dataframe contains OCR data")
        try:
            text = pytesseract.image_to_string(image, lang=self.ocr_lang)
            return text
        except Exception as e:
            logger.error(f"Fallback OCR failed: {e}")
            return ""

    ##
    # @brief Fetches and processes a single item from the dataset
    # @param idx Index of the item to fetch
    # @return Tuple of (text tensor, image tensor, label tensor)
    #
    def __getitem__(self, idx):
        if idx >= len(self.df):
            raise IndexError("Index out of bounds")
        row = self.df.iloc[idx]
        image_url = row['image']
        image_name = image_url.split('/')[-1]

        # --- Get Image ---
        try:
            response = self.client.get_object(self.bucket_name, image_name)
            image_bytes = response.read()
            response.close()
            response.release_conn()
        except Exception as e:
            logger.error(f"Failed to fetch image {image_name}: {e}")
            # Return dummy data
            dummy_text = torch.zeros((self.max_sentences, self.max_sent_length), dtype=torch.long)
            dummy_image = torch.zeros((3, 224, 224), dtype=torch.float32)
            dummy_label = torch.tensor(-1, dtype=torch.long)
            return dummy_text, dummy_image, dummy_label

        # --- Get Text Tensor ---
        # If we've already batch tokenized, use the pre-computed tensor
        if self.tokenized_texts is not None:
            text_tensor = self.tokenized_texts[idx]
        else:
            # Otherwise get OCR text from dataframe or fallback
            if self.has_ocr_column:
                ocr_text = row.get('ocr_text', '')
                # Make sure text is a string
                if not isinstance(ocr_text, str):
                    ocr_text = str(ocr_text) if ocr_text is not None else ""
            else:
                # Fallback: perform OCR on the image
                image = Image.open(io.BytesIO(image_bytes))
                ocr_text = self._fallback_ocr(image)
            
            # Process text into tensor
            text_tensor = self._preprocess_text(ocr_text)

        # --- Process Image ---
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            if self.image_transform:
                image_tensor = self.image_transform(image)
            else:
                image_tensor = transforms.ToTensor()(image)
        except Exception as e:
            logger.error(f"Failed to process image {image_name}: {e}")
            # Return dummy data
            dummy_text = torch.zeros((self.max_sentences, self.max_sent_length), dtype=torch.long)
            dummy_image = torch.zeros((3, 224, 224), dtype=torch.float32)
            dummy_label = torch.tensor(-1, dtype=torch.long)
            return dummy_text, dummy_image, dummy_label

        # --- Process Label ---
        try:
            label_value = row['label']
            
            # Apply label mapping if available
            if self.label_map is not None and str(label_value) in self.label_map:
                label_idx = self.label_map[str(label_value)]
                label = torch.tensor(label_idx, dtype=torch.long)
            else:
                # Try to convert directly to int
                try:
                    label = torch.tensor(int(label_value), dtype=torch.long)
                except (ValueError, TypeError):
                    logger.error(f"Could not convert label '{label_value}' to integer and no mapping found")
                    label = torch.tensor(-1, dtype=torch.long)  # Use -1 to indicate error
        except Exception as e:
             logger.error(f"Failed to process label for index {idx}: {e}")
             label = torch.tensor(-1, dtype=torch.long)

        return text_tensor, image_tensor, label 