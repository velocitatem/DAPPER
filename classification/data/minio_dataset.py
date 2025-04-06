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

# Set Tesseract CMD if needed (uncomment and adjust path if not in system PATH)
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

logger = logging.getLogger(__name__)

class MinioImageDataset(Dataset):
    """
    PyTorch Dataset for loading images from MinIO storage
    """
    def __init__(self, dataframe, bucket_name, transform=None):
        """
        Constructor for MinioImageDataset class
        
        Args:
            dataframe: DataFrame containing 'image' URLs and 'label' columns
            bucket_name: Name of the MinIO bucket to fetch images from
            transform: Optional transformations to apply to the images
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

    def __len__(self):
        """
        Returns the number of items in the dataset
        
        Returns:
            Number of images in the dataset
        """
        return len(self.df)

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

        # Convert label to int and create tensor
        label = torch.tensor(int(row['label']), dtype=torch.long)

        return image, label 

class MinioMultiModalDataset(Dataset):
    """
    PyTorch Dataset for loading images and extracting text via OCR from MinIO.
    Includes an in-memory cache for OCR results and parallel pre-fetching.
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        bucket_name: str,
        image_transform: Optional[callable] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_sentences: int = 15,
        max_sent_length: int = 50,
        vocab: Optional[dict] = None,
        ocr_cache: Optional[dict] = None,
        ocr_lang: str = 'eng',
        minio_endpoint: str = "localhost:9900",
        minio_access_key: str = "minioadmin",
        minio_secret_key: str = "minioadmin",
        minio_secure: bool = False
    ):
        """
        Args:
            dataframe: DataFrame with 'image' URL and 'label'.
            bucket_name: MinIO bucket name.
            image_transform: Transformations for the image.
            tokenizer: Tokenizer instance (e.g., from Hugging Face) to process text.
            max_sentences: Maximum number of sentences to keep from OCR text.
            max_sent_length: Maximum number of tokens per sentence.
            vocab: Vocabulary mapping tokens to indices (if tokenizer doesn't handle it).
            ocr_cache: Optional external dictionary to use for OCR caching.
            ocr_lang: Tesseract language code(s) (e.g., 'eng', 'eng+fra').
            minio_endpoint: MinIO server endpoint URL.
            minio_access_key: MinIO access key.
            minio_secret_key: MinIO secret key.
            minio_secure: Use TLS for MinIO connection.
        """
        self.df = dataframe.reset_index(drop=True)
        self.bucket_name = bucket_name
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.max_sentences = max_sentences
        self.max_sent_length = max_sent_length
        self.vocab = vocab
        self.ocr_lang = ocr_lang

        # Initialize OCR cache (use external if provided, else create new)
        self.ocr_cache = ocr_cache if ocr_cache is not None else {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Store MinIO config
        self.minio_endpoint = minio_endpoint
        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.minio_secure = minio_secure

        # Initialize MinIO client *lazily* or per-process/thread if needed
        self._minio_client = None

        if self.tokenizer is None and self.vocab is None:
             logger.warning("Neither tokenizer nor vocab provided to MinioMultiModalDataset. Text processing might fail.")

    # --- Property to get Minio client, initializing if needed ---
    @property
    def client(self):
        # This ensures client is initialized once per instance accessing it
        # Note: For ProcessPoolExecutor, direct use of this property might not be ideal.
        # Workers might need their own client instances.
        if self._minio_client is None:
            logger.debug(f"Initializing MinIO client for {self.minio_endpoint}")
            self._minio_client = Minio(
                self.minio_endpoint,
                access_key=self.minio_access_key,
                secret_key=self.minio_secret_key,
                secure=self.minio_secure
            )
        return self._minio_client

    def __len__(self):
        return len(self.df)

    # --- Worker function for parallel OCR ---
    def _ocr_worker(self, image_name: str) -> Tuple[str, Optional[str]]:
        """Fetches image from MinIO and performs OCR. Returns (image_name, text or None on error)."""
        # Initialize MinIO client within the worker process
        try:
            worker_client = Minio(
                self.minio_endpoint,
                access_key=self.minio_access_key,
                secret_key=self.minio_secret_key,
                secure=self.minio_secure
            )
            response = worker_client.get_object(self.bucket_name, image_name)
            image_bytes = response.read()
            response.close()
            response.release_conn()

            pil_image = Image.open(io.BytesIO(image_bytes))
            # Consider adding image pre-processing for OCR if needed (e.g., grayscale)
            # text = pytesseract.image_to_string(pil_image.convert('L'), lang=self.ocr_lang)
            text = pytesseract.image_to_string(pil_image, lang=self.ocr_lang)
            return image_name, text
        except Exception as e:
            # Log error but allow skipping the item
            # Using logger here might require specific configuration if used across processes
            # print(f"OCR Worker Error for {image_name}: {e}") # Simple print for visibility
            logger.error(f"OCR Worker Error for {image_name}: {e}")
            return image_name, None # Indicate failure

    # --- New method to populate cache in parallel ---
    def populate_ocr_cache(self, max_workers: Optional[int] = None):
        """
        Pre-fetches images from MinIO and runs OCR in parallel to populate the cache.
        Args:
            max_workers: Maximum number of worker processes to use. Defaults to os.cpu_count().
        """
        logger.info(f"Starting parallel OCR cache population for {len(self.df)} items...")
        start_time = time.time() # Import time if not already done

        items_to_process = []
        for idx, row in self.df.iterrows():
            image_url = row['image']
            image_name = image_url.split('/')[-1]
            if image_name not in self.ocr_cache: # Only process if not already cached
                items_to_process.append(image_name)

        if not items_to_process:
            logger.info("OCR cache is already fully populated or dataset is empty.")
            return

        logger.info(f"Need to process OCR for {len(items_to_process)} items.")

        processed_count = 0
        # Using ProcessPoolExecutor for CPU-bound OCR tasks
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Using map preserves order and is simpler if we process all needed items
            # Use tqdm to show progress
            future_to_name = {executor.submit(self._ocr_worker, name): name for name in items_to_process}
            for future in tqdm(concurrent.futures.as_completed(future_to_name), total=len(items_to_process), desc="OCR Pre-processing"):
                try: # Add try-except block around future.result()
                    image_name, ocr_text = future.result()
                    if ocr_text is not None:
                        self.ocr_cache[image_name] = ocr_text
                        processed_count += 1
                    else:
                        # Store empty string on failure to avoid reprocessing?
                        self.ocr_cache[image_name] = ""
                        logger.warning(f"OCR failed for {image_name} (returned None), storing empty string in cache.")
                except Exception as exc:
                     # Log exception raised during the task execution itself
                     img_name_for_error = future_to_name[future] # Get image name from original mapping
                     logger.error(f"OCR task for {img_name_for_error} generated an exception: {exc}")
                     self.ocr_cache[img_name_for_error] = "" # Cache failure as empty string


        end_time = time.time()
        logger.info(f"Finished OCR cache population. Processed {processed_count}/{len(items_to_process)} items in {end_time - start_time:.2f} seconds.")
        logger.info(f"Cache size: {len(self.ocr_cache)}")


    def _get_ocr_text(self, image_name: str, image_bytes: bytes) -> str:
        """Helper to get OCR text, using cache. Should be fast after pre-population."""
        if image_name in self.ocr_cache:
            self.cache_hits += 1
            # Log cache status periodically during training if desired
            # total_access = self.cache_hits + self.cache_misses
            # if total_access > 0 and total_access % 1000 == 0:
            #    logger.info(f"OCR Cache Access: {self.cache_hits} hits ({self.cache_hits / total_access:.1%}), {self.cache_misses} misses")
            return self.ocr_cache[image_name]
        else:
            # This block should ideally only be hit if pre-population failed or was skipped
            self.cache_misses += 1
            logger.warning(f"OCR cache miss for {image_name}. Performing OCR on the fly.")
            try:
                pil_image = Image.open(io.BytesIO(image_bytes))
                text = pytesseract.image_to_string(pil_image, lang=self.ocr_lang)
                self.ocr_cache[image_name] = text
                return text
            except Exception as e:
                logger.error(f"On-the-fly OCR failed for {image_name}: {e}. Returning empty string.")
                self.ocr_cache[image_name] = "" # Cache failure as empty
                return ""

    def _preprocess_text(self, text: str) -> torch.Tensor:
        """Tokenizes, numericalizes, and pads text to fit model input."""
        # Basic sentence splitting (can be improved with NLTK, spaCy etc.)
        sentences = [s.strip() for s in text.split('\n') if s.strip()][:self.max_sentences]
        if not sentences:
             sentences = [""]

        # --- Tokenization and Numericalization ---
        # This part heavily depends on the chosen tokenizer/vocab approach
        if self.tokenizer:
            # Example using Hugging Face tokenizer (adjust based on actual tokenizer)
            # Assumes tokenizer handles padding and truncation.
            # We might need to tokenize sentence by sentence and pad/truncate each.
            # This is a simplified placeholder.
            # EAML expects (num_sentences, max_sent_length)
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
                # Add padding sentence (list of padding tokens)
                pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else 0
                tokens_list.append([pad_token_id] * self.max_sent_length)

            text_tensor = torch.tensor(tokens_list, dtype=torch.long)

        elif self.vocab:
             # Manual processing using a predefined vocab dictionary
             # Placeholder: Implement sentence tokenization, mapping to indices via vocab,
             # and padding/truncation to (max_sentences, max_sent_length).
             logger.warning("Manual vocab processing not fully implemented in placeholder.")
             # Create a dummy tensor matching the expected shape
             text_tensor = torch.zeros((self.max_sentences, self.max_sent_length), dtype=torch.long)
        else:
             # No tokenizer or vocab - cannot process text
             logger.error("Cannot process text: No tokenizer or vocab available.")
             # Return zero tensor matching expected shape
             text_tensor = torch.zeros((self.max_sentences, self.max_sent_length), dtype=torch.long)

        # Ensure final shape is [max_sentences, max_sent_length]
        # The logic above should already ensure this, but double-check.
        if text_tensor.shape != (self.max_sentences, self.max_sent_length):
             logger.warning(f"Text tensor shape mismatch: expected ({self.max_sentences}, {self.max_sent_length}), got {text_tensor.shape}. Check padding/truncation.")
             # Attempt to reshape or pad/truncate again if possible, or raise error
             # For now, just log warning.

        return text_tensor

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
            # It's important to close the connection
            response.close()
            response.release_conn()
        except Exception as e:
            logger.error(f"Failed to fetch image {image_name}: {e}")
             # Return dummy data or raise? Depends on desired robustness.
             # Returning dummy data matching expected types/shapes:
            dummy_text = torch.zeros((self.max_sentences, self.max_sent_length), dtype=torch.long)
            dummy_image = torch.zeros((3, 224, 224), dtype=torch.float32) # Assuming 3 channels, 224x224
            dummy_label = torch.tensor(-1, dtype=torch.long) # Indicate error with -1 label?
            return dummy_text, dummy_image, dummy_label

        # --- Get OCR Text (Now uses cache primarily) ---
        # This call should now be very fast if populate_ocr_cache was run
        ocr_text = self._get_ocr_text(image_name, image_bytes) # Pass image_bytes for potential on-the-fly fallback

        # --- Process Text ---
        text_tensor = self._preprocess_text(ocr_text) # This should also be faster

        # --- Process Image ---
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            if self.image_transform:
                image_tensor = self.image_transform(image)
            else:
                 # If no transform, maybe basic ToTensor? Need consistent output.
                 image_tensor = transforms.ToTensor()(image) # Basic conversion
        except Exception as e:
            logger.error(f"Failed to process image {image_name}: {e}")
            # Return dummy data or raise?
            dummy_text = torch.zeros((self.max_sentences, self.max_sent_length), dtype=torch.long)
            dummy_image = torch.zeros((3, 224, 224), dtype=torch.float32)
            dummy_label = torch.tensor(-1, dtype=torch.long)
            return dummy_text, dummy_image, dummy_label

        # --- Process Label ---
        try:
            label = torch.tensor(int(row['label']), dtype=torch.long)
        except Exception as e:
             logger.error(f"Failed to process label for index {idx}: {e}")
             label = torch.tensor(-1, dtype=torch.long) # Indicate error

        return text_tensor, image_tensor, label

    def get_cache_stats(self):
        """Returns OCR cache statistics."""
        total_access = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_access * 100) if total_access > 0 else 0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "total": total_access,
            "hit_rate_percent": hit_rate,
            "cache_size": len(self.ocr_cache)
        } 