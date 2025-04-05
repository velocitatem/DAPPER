##
# @file data.py
# @brief Data processing module for document classification.
#
# This module handles loading, preprocessing, and storage of document images
# used for training and testing a document classification model. It processes
# images from multiple sources, resizes them, and stores them in a MinIO object
# storage system for efficient access during training.
#
# @author Statistical Learning Team
# @date 2025-03-20
#

from datasets import load_dataset
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import re
import kagglehub
from pdf2image import convert_from_path
import gc
from minio import Minio
import logging
import io
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from torchvision import transforms
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Define image augmentation transformations
pil_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),  # Increased rotation
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),  # Increased color variation
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    transforms.RandomGrayscale(p=0.2),  # Increased grayscale probability
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),  # Increased perspective distortion
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Added translation
])

tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2),  # Added random erasing
    transforms.ToPILImage()
])

##
# @brief Applies data augmentation to an image
#
# This function applies various data augmentation techniques to the input image
# using torchvision transforms, including rotation, color adjustment, contrast,
# brightness, and document-specific transformations like blur and perspective changes.
#
# @param image The PIL Image object to augment
# @param augment Whether to apply augmentation (set to False to skip)
# @return The augmented PIL Image object
#
def augment_image(image, augment=True):
    if not augment:
        return image

    # Apply PIL transforms first
    augmented_image = pil_transforms(image)

    # Then apply tensor transforms
    augmented_image = tensor_transforms(augmented_image)

    return augmented_image

##
# @brief Creates additional augmented versions of training images
#
# This function takes a DataFrame of images, creates augmented copies,
# and uploads them to MinIO storage. It significantly increases the
# dataset size and diversity to improve model training.
#
# @param df DataFrame containing images to augment
# @param bucket_name Name of the MinIO bucket for storage
# @param augmentation_factor Number of augmented copies per image
# @return DataFrame containing original and augmented images
#
def create_augmented_dataset(df, bucket_name, augmentation_factor=2):
    logging.info(f"Creating {augmentation_factor} augmented versions of {len(df)} images")

    augmented_rows = []

    # Ensure source_dataset exists and is properly formatted
    if 'source_dataset' not in df.columns:
        df['source_dataset'] = 'base'

    # Generate global unique IDs for original images if not exists
    if 'global_id' not in df.columns:
        df['global_id'] = [f"orig_{i}" for i in range(len(df))]

    # Ensure is_augmented is boolean
    df['is_augmented'] = False

    # Calculate class weights for balanced augmentation
    class_counts = df['label'].value_counts()
    max_count = class_counts.max()
    class_weights = {cls: max_count/count for cls, count in class_counts.items()}

    for i, row in df.iterrows():
        if i % 100 == 0:
            logging.info(f"Processing augmentations for image {i}/{len(df)}")

        # Skip if the image is a URL (already processed)
        if isinstance(row['image'], str) and row['image'].startswith('http'):
            continue

        image = row['image']
        if image is None:
            continue

        label = row['label']
        source_dataset = row.get('source_dataset', 'base')
        global_id = row['global_id']

        # Adjust augmentation factor based on class weight
        class_weight = class_weights.get(label, 1.0)
        adjusted_factor = int(augmentation_factor * class_weight)

        # Create multiple augmented versions
        for aug_idx in range(adjusted_factor):
            # Create a deep copy of the row
            aug_row = row.copy()

            # Apply strong augmentation
            aug_image = image.copy()
            aug_image = augment_image(aug_image)
            aug_row['image'] = aug_image

            # Track augmentation version and source dataset
            aug_row['aug_version'] = aug_idx + 1
            aug_row['source_dataset'] = f"{source_dataset}_aug{aug_idx + 1}"
            aug_row['global_id'] = global_id  # Keep track of original image
            aug_row['is_augmented'] = True

            # Add to our augmented dataset
            augmented_rows.append(aug_row)

    if augmented_rows:
        augmented_df = pd.DataFrame(augmented_rows)
        # Upload augmented images to MinIO
        augmented_df = load_df_with_raw_images_into_minio(augmented_df, bucket_name)
        logging.info(f"Created {len(augmented_df)} augmented images")

        # Combine original and augmented datasets
        combined_df = pd.concat([df, augmented_df])

        # Log class distribution after augmentation
        class_distribution = combined_df['label'].value_counts()
        logging.info(f"Class distribution after augmentation: {class_distribution.to_dict()}")

        return combined_df
    else:
        logging.warning("No images were augmented")
        return df

##
# @brief Resizes input images to a standardized size
#
# This function resizes input images to a standard size of 768x992 pixels,
# which was determined by analyzing the statistics of the dataset.
#
# @param image The PIL Image object to resize
# @param augment Whether to apply data augmentation
# @return A resized PIL Image object
#
def resize_image(image, augment=False):
    width, height = 768, 992
    # Image statistics from dataset analysis:
    # Total images analyzed: 1829
    # Width: min=625, max=1654, mean=1168.64, median=787.0
    # Height: min=1000, max=2339, mean=1607.64, median=1000.0
    resized_image = image.resize((width, height))

    # Apply data augmentation if requested
    if augment:
        resized_image = augment_image(resized_image)

    return resized_image

##
# @brief MinIO client configuration
#
# Sets up the connection to the MinIO object storage server.
# MinIO is used to store and retrieve document images efficiently.
#
client = Minio(
    "localhost:9900",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

## @brief Pre-fetch list of objects in the default bucket
objects = client.list_objects("dapper")
##
# @brief Lists all images in a MinIO bucket
#
# @param bucket_name Name of the MinIO bucket to list images from
# @return List of image filenames in the bucket
#
def all_images_in_minio(bucket_name):
    logging.info(f"Connecting to Minio bucket: {bucket_name}")
    return [obj.object_name for obj in objects]

all_images = all_images_in_minio("dapper")
##
# @brief Processes a single image row and uploads it to MinIO storage
#
# This function receives an image from a dataframe row, processes it
# (converts to RGB, resizes), and uploads it to MinIO storage. If the image
# already exists in the bucket, it skips the upload and returns the URL.
#
# @param args Tuple containing (index, row, bucket_name, all_images)
# @return Tuple (index, URL) where URL points to the stored image
#
def process_image_row(args):
    i, row, bucket_name, all_images = args
    image = row['image']
    if image is None:
        logging.warning(f"Row {i} has no image, skipping.")
        return i, None

    logging.info(f"Processing image {i}")
    image = image.convert('RGB')

    # Apply data augmentation during training (50% chance)
    use_augmentation = random.random() < 0.5
    image = resize_image(image, augment=use_augmentation)

    # Get label
    label = str(row['label'])

    # Get source dataset if available (default to 'unknown')
    source_dataset = row.get('source_dataset', 'unknown')

    # Create meaningful filename with pattern: src_[source]_cls_[class]_idx_[index]_[aug_info].jpg
    aug_info = "aug" if use_augmentation else "orig"

    # Clean the label for filename use
    clean_label = re.sub(r'\W+', '', label)

    # Create meaningful filename
    filename = f"src_{source_dataset}_cls_{clean_label}_idx_{i}_{aug_info}.jpg"

    if filename in all_images:
        logging.info(f"Image {filename} already exists in bucket, skipping.")
        return i, f"http://localhost:9900/{bucket_name}/{filename}"

    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)

    client.put_object(
        bucket_name,
        filename,
        image_bytes,
        len(image_bytes.getvalue())
    )

    image.close()
    image_bytes.close()

    return i, f"http://localhost:9900/{bucket_name}/{filename}"

##
# @brief Loads images from a DataFrame into MinIO storage in parallel
#
# This function takes a DataFrame containing image data, uploads all images to
# MinIO storage in parallel using ThreadPoolExecutor, and updates the DataFrame
# with the URLs of the stored images.
#
# @param df DataFrame containing images to upload
# @param bucket_name Name of the MinIO bucket to upload to
# @param max_workers Maximum number of parallel threads to use
# @return DataFrame with image column updated to contain MinIO URLs
#
def load_df_with_raw_images_into_minio(df, bucket_name, max_workers=12):
    logging.info(f"Connecting to Minio bucket: {bucket_name} with {max_workers} parallel workers")

    # Prepare arguments for parallel processing
    args_list = [(i, row, bucket_name, all_images) for i, row in df.iterrows()]

    # Process images in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_image_row, args_list))

    # Update DataFrame with results
    for i, url in results:
        if url is not None:
            df.at[i, 'image'] = url

    logging.info(f"Finished uploading images to bucket: {bucket_name}")
    return df



##
# @brief Downloads and processes additional invoice documents from Kaggle
#
# This function downloads a company document dataset from Kaggle, extracts invoice
# documents, converts PDFs to images, and returns a DataFrame with the images and labels.
#
# @return DataFrame containing invoice images and their labels
#
def get_extra_invoices():
    logging.info("Downloading extra invoices from Kaggle dataset")
    path1 = kagglehub.dataset_download("ayoubcherguelaine/company-documents-dataset")
    df = pd.read_csv(path1+"/company-document-text.csv")

    invoices = df[df['label'] == 'invoice']

    ##
    # @brief Extracts invoice ID from text description
    # @param text Text containing invoice ID
    # @return Integer invoice ID
    def extract_invoice_id(text):
        text = text.split()
        return int(text[3])

    logging.info("Extracting invoice IDs")
    invoices['invoice_id'] = invoices['text'].apply(extract_invoice_id)

    directory = path1+"/CompanyDocuments/invoices/"

    ##
    # @brief Loads a PDF file and converts it to an image
    # @param invoice_id ID of the invoice to load
    # @return PIL Image object of the first page
    def load_pdf_as_image(invoice_id):
        path = f"{directory}invoice_{invoice_id}.pdf"
        logging.info(f"Converting PDF {path} to image")
        image = convert_from_path(path)
        return image[0] if len(image) > 0 else None

    invoices['image'] = invoices['invoice_id'].apply(load_pdf_as_image)
    logging.info("Finished converting PDFs to images")

    return invoices[['image', 'label']]


def load_base_dataset(chunk_size=1000, total_samples=10_000, seed=42, apply_augmentation=True):
    """
    Progressively loads chunks of the RVL-CDIP dataset into MinIO.

    Args:
        chunk_size (int): Number of samples per chunk
        total_samples (int): Total number of samples to load. If None, loads entire dataset
        seed (int): Random seed for shuffling
        apply_augmentation (bool): Whether to apply data augmentation to increase dataset diversity

    Returns:
        list: List of DataFrames containing the loaded chunks
    """
    logging.info("Loading RVL-CDIP dataset")
    dataset = load_dataset("aharley/rvl_cdip", trust_remote_code=True)

    # Shuffle and get total length
    shuffled_dataset = dataset['train'].shuffle(seed=seed)
    dataset_size = len(shuffled_dataset)

    if total_samples is None:
        total_samples = dataset_size
    else:
        total_samples = min(total_samples, dataset_size)

    loaded_chunks = []
    for start_idx in range(0, total_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, total_samples)

        logging.info(f"Loading chunk {start_idx//chunk_size + 1}, samples {start_idx} to {end_idx}")

        # Select chunk and convert to DataFrame
        chunk = shuffled_dataset.select(range(start_idx, end_idx))
        chunk_df = pd.DataFrame(chunk)

        # Add source dataset information
        chunk_df['source_dataset'] = 'rvl_cdip'

        # Load images into MinIO
        processed_chunk = load_df_with_raw_images_into_minio(chunk_df, "dapper")

        if apply_augmentation:
            class_counts = processed_chunk['label'].value_counts()
            median_count = class_counts.median()
            underrep_classes = class_counts[class_counts < median_count].index.tolist()

            if underrep_classes:
                logging.info(f"Augmenting underrepresented classes: {underrep_classes}")

                # Filter chunk to only include underrepresented classes
                underrep_df = processed_chunk[processed_chunk['label'].isin(underrep_classes)]

                # Apply stronger augmentation to underrepresented classes
                if not underrep_df.empty:
                    # Calculate augmentation factor based on class imbalance
                    max_count = class_counts.max()
                    aug_factors = {}

                    for cls in underrep_classes:
                        # More augmentation for more underrepresented classes
                        factor = min(3, int(max_count / class_counts[cls]))
                        aug_factors[cls] = factor

                    logging.info(f"Augmentation factors: {aug_factors}")

                    # Apply class-specific augmentation
                    augmented_dfs = []
                    for cls, factor in aug_factors.items():
                        cls_df = underrep_df[underrep_df['label'] == cls]
                        if not cls_df.empty:
                            augmented = create_augmented_dataset(cls_df, "dapper", augmentation_factor=factor)
                            augmented_dfs.append(augmented)

                    if augmented_dfs:
                        # Combine with original chunk
                        processed_chunk = pd.concat([processed_chunk] + augmented_dfs)
                        logging.info(f"Added {len(processed_chunk) - len(chunk_df)} augmented samples")

        loaded_chunks.append(processed_chunk)

        # Clean up memory
        gc.collect()

        logging.info(f"Finished loading chunk {start_idx//chunk_size + 1}")

    return pd.concat(loaded_chunks)

train = load_base_dataset(total_samples=10_000, apply_augmentation=True)


##
# @brief Loads the first additional dataset source
#
# Retrieves and processes extra invoice documents from a Kaggle dataset,
# uploads them to MinIO storage, and returns the DataFrame with URLs.
#
# @return DataFrame containing processed invoice images and their labels
#

def get_extra_invoices():
    logging.info("Downloading extra invoices from Kaggle dataset")
    path1 = kagglehub.dataset_download("ayoubcherguelaine/company-documents-dataset")
    df = pd.read_csv(path1+"/company-document-text.csv")

    invoices = df[df['label'] == 'invoice']

    ##
    # @brief Extracts invoice ID from text description
    # @param text Text containing invoice ID
    # @return Integer invoice ID
    def extract_invoice_id(text):
        text = text.split()
        return int(text[3])

    logging.info("Extracting invoice IDs")
    invoices['invoice_id'] = invoices['text'].apply(extract_invoice_id)

    directory = path1+"/CompanyDocuments/invoices/"

    ##
    # @brief Loads a PDF file and converts it to an image
    # @param invoice_id ID of the invoice to load
    # @return PIL Image object of the first page
    def load_pdf_as_image(invoice_id):
        path = f"{directory}invoice_{invoice_id}.pdf"
        logging.info(f"Converting PDF {path} to image")
        image = convert_from_path(path)
        return image[0] if len(image) > 0 else None

    invoices['image'] = invoices['invoice_id'].apply(load_pdf_as_image)
    logging.info("Finished converting PDFs to images")

    return invoices[['image', 'label']]
def load_source_1():
    logging.info("Retrieving and processing extra invoices")
    extra_invoices = get_extra_invoices()
    # Add source dataset information
    extra_invoices['source_dataset'] = 'kaggle_invoices'
    extra_invoices = load_df_with_raw_images_into_minio(extra_invoices, "dapper")
    # Create augmented versions of these invoices to increase dataset diversity
    extra_invoices = create_augmented_dataset(extra_invoices, "dapper", augmentation_factor=3)
    return extra_invoices # df with urls


##
# @brief Loads the second additional dataset source
#
# Retrieves invoice data from Hugging Face datasets,
# opens images from bytes, sets the label to 11 (invoice),
# uploads the images to MinIO storage, and returns the DataFrame with URLs.
#
# @return DataFrame containing processed invoice images
#
def load_source_2():
    splits = {'train': 'data/train-00000-of-00001-a5c51039eab2980a.parquet',
              'validation': 'data/validation-00000-of-00001-b8a5c4a6237baf25.parquet',
              'test': 'data/test-00000-of-00001-56af6bd5ff7eb34d.parquet'}
    df = pd.read_parquet("hf://datasets/katanaml-org/invoices-donut-data-v1/" + splits["train"])
    df['image'] = df['image'].apply(lambda x: Image.open(io.BytesIO(x['bytes'])))
    df['label'] = 11  # Invoice category
    # Add source dataset information
    df['source_dataset'] = 'hf_invoices'

    # Now we load the images into MinIO
    df = load_df_with_raw_images_into_minio(df, "dapper")
    df = create_augmented_dataset(df, "dapper", augmentation_factor=3)
    return df


##
# @brief Loads document data from the DocVQA dataset
#
# Retrieves document images from the lmms-lab/DocVQA dataset,
# maps question_types to document classes, and processes the images.
# This provides a diverse set of document types to improve the dataset.
#
# @return DataFrame containing processed document images with class labels
#
def load_source_3():
    logging.info("Loading documents from lmms-lab/DocVQA dataset")

    try:
        # Load the dataset from Hugging Face with the correct config
        dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
        logging.info(f"Loaded DocVQA dataset with {len(dataset)} samples")

        # Create a mapping from question_types to our document classes
        # Map the question_types to our existing document classes where possible
        type_to_label_mapping = {
            "form": 1,  # form
            "handwritten": 3,  # handwritten
            "letter": 0,  # letter
            "figure/diagram": 5,  # scientific report (contains figures/diagrams)
            "layout": 11,  # invoice (often involves layout questions)
            "table": 5,  # scientific report (contains tables)
            "list": 1,  # form (often contains lists)
            "others": 2,  # email (default for others, will filter later),
        }

        # Create a DataFrame to store our results
        data_list = []

        # Track how many items of each class we add
        class_counts = {label: 0 for label in set(type_to_label_mapping.values())}
        max_per_class = 500  # Limit samples per class

        for item in dataset:
            try:
                # Skip items without question_types
                if not item.get("question_types") or len(item["question_types"]) == 0:
                    continue

                # Get the primary question type (first in the list)
                question_type = item["question_types"][0]

                # Skip if we don't have a mapping for this type
                if question_type not in type_to_label_mapping:
                    continue

                label = type_to_label_mapping[question_type]

                # Skip if we already have enough of this class
                if class_counts[label] >= max_per_class:
                    continue

                # Get the image - handle different possible formats
                if isinstance(item["image"], dict) and "bytes" in item["image"]:
                    # If image is a dict with bytes
                    image = Image.open(io.BytesIO(item["image"]["bytes"]))
                elif isinstance(item["image"], Image.Image):
                    # If image is already a PIL Image
                    image = item["image"]
                elif hasattr(item["image"], "convert"):
                    # If image is a PIL Image but not recognized as such
                    image = item["image"]
                else:
                    # Skip if we can't handle this image format
                    logging.warning(f"Unrecognized image format: {type(item['image'])}")
                    continue

                # Add to our data list
                data_list.append({
                    "image": image,
                    "label": label,
                    "source_dataset": f"docvqa_{question_type}"
                })

                # Update our count
                class_counts[label] += 1

            except Exception as e:
                logging.warning(f"Error processing DocVQA item: {e}")
                continue

        # Create DataFrame
        df = pd.DataFrame(data_list)

        if len(df) > 0:
            logging.info(f"Created DataFrame with {len(df)} documents")
            logging.info(f"Class distribution: {class_counts}")

            # Upload to MinIO and create augmented versions
            df = load_df_with_raw_images_into_minio(df, "dapper")
            df = create_augmented_dataset(df, "dapper", augmentation_factor=3)

            return df
        else:
            logging.warning("No valid documents found in DocVQA dataset")
            return pd.DataFrame()

    except Exception as e:
        logging.error(f"Failed to load DocVQA dataset: {e}")
        return pd.DataFrame()


##
# @brief Loads additional samples for underrepresented classes
#
# Analyzes the current dataset distribution and loads additional samples
# specifically for underrepresented document classes from RVL-CDIP.
# This helps balance the dataset for better model training.
#
# @return DataFrame containing processed images for underrepresented classes
#
def load_source_4():
    logging.info("Loading additional samples for underrepresented classes")

    try:
        # First, analyze the current class distribution
        try:
            # If train.csv exists, use it to determine class distribution
            current_data = pd.read_csv("train.csv")
            current_data['label_int'] = current_data['label'].apply(
                lambda x: 11 if x == 'invoice' else int(x) if isinstance(x, (int, float)) else int(x)
            )
            class_distribution = current_data['label_int'].value_counts()

            # Calculate target count (median of current distribution)
            median_count = class_distribution.median()

            # Identify underrepresented classes (below median)
            underrep_classes = class_distribution[class_distribution < median_count].index.tolist()

            logging.info(f"Current class distribution: {class_distribution.to_dict()}")
            logging.info(f"Identified underrepresented classes: {underrep_classes}")

            # Target samples to add per class
            target_samples = {cls: int(median_count - class_distribution.get(cls, 0)) for cls in underrep_classes}

            # Ensure we don't request too many samples
            for cls in target_samples:
                target_samples[cls] = min(target_samples[cls], 300)  # Cap at 300 additional samples per class

            logging.info(f"Target samples to add: {target_samples}")

        except (FileNotFoundError, pd.errors.EmptyDataError):
            # If we can't analyze existing data, use a default approach
            logging.warning("Could not analyze existing class distribution, using default approach")
            # Focus on classes often underrepresented
            target_samples = {
                3: 200,  # handwritten
                4: 200,  # advertisement
                6: 200,  # scientific publication
                7: 200,  # specification
                8: 200,  # file folder
                10: 200,  # budget
                14: 200,  # questionnaire
                15: 200   # resume
            }

        # Load RVL-CDIP dataset with specific focus
        dataset = load_dataset("aharley/rvl_cdip", trust_remote_code=True)

        # Initialize dataframe to store results
        data_list = []

        # Track how many we've added for each class
        added_counts = {cls: 0 for cls in target_samples}

        # Process dataset to focus on underrepresented classes
        for item in dataset['train']:
            try:
                label = item['label']

                # Skip if not an underrepresented class
                if label not in target_samples:
                    continue

                # Skip if we've already added enough of this class
                if added_counts[label] >= target_samples[label]:
                    continue

                # Add to our data list
                data_list.append({
                    "image": item['image'],
                    "label": label,
                    "source_dataset": f"rvl_cdip_balancing_cls{label}"
                })

                # Update our count
                added_counts[label] += 1

                # Check if we've completed all classes
                if all(added_counts[cls] >= target_samples[cls] for cls in target_samples):
                    break

            except Exception as e:
                logging.warning(f"Error processing RVL-CDIP item: {e}")
                continue

        # Create DataFrame
        df = pd.DataFrame(data_list)

        if len(df) > 0:
            logging.info(f"Created DataFrame with {len(df)} additional samples")
            logging.info(f"Class distribution of additional samples: {df['label'].value_counts().to_dict()}")

            # Upload to MinIO and create augmented versions
            df = load_df_with_raw_images_into_minio(df, "dapper")
            df = create_augmented_dataset(df, "dapper", augmentation_factor=2)

            return df
        else:
            logging.warning("No additional samples found")
            return pd.DataFrame()

    except Exception as e:
        logging.error(f"Failed to load additional samples: {e}")
        return pd.DataFrame()


extra_sources = [
    load_source_1,
    load_source_2,
    ##load_source_3,
    #load_source_4, # alghough
]

logging.info("Loading extra sources")
sources_dfs = []
for source in extra_sources:
    extra_invoices = source()
    sources_dfs.append(extra_invoices)
sources_dfs = pd.concat(sources_dfs)

logging.info("Merging datasets")
combined_data = pd.concat([train, sources_dfs])

# Ensure all required columns exist and handle NaN values
if 'is_augmented' not in combined_data.columns:
    # First set everything to False
    combined_data['is_augmented'] = False
    # Then mark augmented images based on source_dataset
    aug_mask = combined_data['source_dataset'].str.contains('_aug', na=False)
    combined_data.loc[aug_mask, 'is_augmented'] = True

# Fill any NaN values in is_augmented with False to ensure we have only boolean values
combined_data['is_augmented'] = combined_data['is_augmented'].fillna(False).astype(bool)

# Reset global_id column to ensure uniqueness
if 'global_id' in combined_data.columns:
    combined_data = combined_data.drop('global_id', axis=1)

# Generate unique global IDs for original images
original_mask = ~combined_data['is_augmented']
combined_data.loc[original_mask, 'global_id'] = [f"orig_{i}" for i in range(sum(original_mask))]

# Propagate global IDs to augmented versions based on source_dataset
for idx, row in combined_data[original_mask].iterrows():
    source_dataset = row['source_dataset']
    if pd.isna(source_dataset):
        continue
    # Find augmentations of this original image
    aug_pattern = f"{source_dataset}_aug"
    aug_mask = combined_data['source_dataset'].str.startswith(aug_pattern, na=False)
    if aug_mask.any():
        combined_data.loc[aug_mask, 'global_id'] = row['global_id']

# Fill any missing global_ids with unique values
missing_global_id = combined_data['global_id'].isna()
combined_data.loc[missing_global_id, 'global_id'] = [f"unknown_{i}" for i in range(sum(missing_global_id))]

gc.collect()

# Split into train and test sets (80/20 split)
logging.info("Splitting into train and test sets")
from sklearn.model_selection import train_test_split

# Ensure all labels are of the same type (convert 'invoice' to 11 like in train.py)
combined_data['label_int'] = combined_data['label'].apply(lambda x: 11 if x == 'invoice' else int(x))

# Get only the original images for splitting
original_images = combined_data[~combined_data['is_augmented']]

# Verify each original image has exactly one unique global_id
unique_check = original_images.groupby('global_id').size()
if (unique_check > 1).any():
    duplicated_ids = unique_check[unique_check > 1].index.tolist()
    logging.warning(f"Found {len(duplicated_ids)} global_ids with multiple original images. Fixing...")

    # For any duplicated global_ids, reassign with new unique IDs
    for gid in duplicated_ids:
        dupe_indices = original_images[original_images['global_id'] == gid].index
        for i, idx in enumerate(dupe_indices[1:], 1):  # Skip first occurrence
            original_images.loc[idx, 'global_id'] = f"{gid}_dupe_{i}"
            # Update in combined_data as well
            combined_data.loc[idx, 'global_id'] = f"{gid}_dupe_{i}"

# Split original images with stratification
train_original, test_original = train_test_split(
    original_images,
    test_size=0.2,
    random_state=42,
    stratify=original_images['label_int']
)

# Get the global_ids for train and test sets
train_global_ids = set(train_original['global_id'])
test_global_ids = set(test_original['global_id'])

# Double-check no overlap between train and test global_ids
if train_global_ids.intersection(test_global_ids):
    raise ValueError(f"Split failed - found {len(train_global_ids.intersection(test_global_ids))} overlapping global_ids")

# Now, for each original image in train, get its augmented versions
train_augmented = combined_data[
    combined_data['is_augmented'] &
    combined_data['global_id'].isin(train_global_ids)
]

# For each original image in test, get its augmented versions
test_augmented = combined_data[
    combined_data['is_augmented'] &
    combined_data['global_id'].isin(test_global_ids)
]

# Combine original and augmented images
train_final = pd.concat([train_original, train_augmented])
test_final = pd.concat([test_original, test_augmented])

# Verify no leakage
train_global_ids = set(train_final['global_id'])
test_global_ids = set(test_final['global_id'])
leakage = train_global_ids.intersection(test_global_ids)
if leakage:
    logging.error(f"Data leakage detected! Found {len(leakage)} images with same global_id in both train and test sets")
    raise ValueError("Data leakage detected in train/test split")

# Log detailed statistics
logging.info(f"Train set size: {len(train_final)}, Test set size: {len(test_final)}")
logging.info(f"Original images in train: {len(train_original)}, augmented: {len(train_augmented)}")
logging.info(f"Original images in test: {len(test_original)}, augmented: {len(test_augmented)}")

# Log class distribution after augmentation
class_distribution = train_final['label_int'].value_counts().sort_index()
logging.info(f"Final class distribution after augmentation: {class_distribution.to_dict()}")

# Calculate and log class balance metrics
class_balance = class_distribution / class_distribution.sum()
logging.info(f"Class balance ratios: {class_balance.to_dict()}")

logging.info("Saving datasets as csv")
train_final.to_csv("train.csv", index=False)
test_final.to_csv("test.csv", index=False)
logging.info("Datasets saved successfully")
