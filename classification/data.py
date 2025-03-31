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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')



##
# @brief Applies data augmentation to an image
#
# This function applies various data augmentation techniques to the input image,
# including rotation, color adjustment, contrast, brightness, and other transformations.
#
# @param image The PIL Image object to augment
# @param augment Whether to apply augmentation (set to False to skip)
# @return The augmented PIL Image object
#
def augment_image(image, augment=True):
    if not augment:
        return image

    # Define augmentation probability
    aug_prob = 0.5

    # Randomly apply augmentations with probability
    if random.random() < aug_prob:
        # Random rotation (slight - documents shouldn't be heavily rotated)
        angle = random.uniform(-5, 5)
        image = image.rotate(angle, resample=Image.BICUBIC, expand=False)

    if random.random() < aug_prob:
        # Random brightness adjustment
        brightness_factor = random.uniform(0.8, 1.2)
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)

    if random.random() < aug_prob:
        # Random contrast adjustment
        contrast_factor = random.uniform(0.8, 1.2)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)

    if random.random() < aug_prob:
        # Random color/saturation adjustment
        color_factor = random.uniform(0.8, 1.2)
        image = ImageEnhance.Color(image).enhance(color_factor)

    if random.random() < 0.3:  # Less aggressive augmentations with lower probability
        # Random sharpness adjustment
        sharpness_factor = random.uniform(0.8, 1.5)
        image = ImageEnhance.Sharpness(image).enhance(sharpness_factor)

    if random.random() < 0.2:  # Document noise simulation
        # Add slight Gaussian noise or blur to simulate document scanning
        if random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
        else:
            # Simulate slight JPEG compression artifacts
            buffer = io.BytesIO()
            quality = random.randint(85, 95)
            image.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            image = Image.open(buffer)

    return image
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

    # Ensure source_dataset exists
    if 'source_dataset' not in df.columns:
        df['source_dataset'] = 'base'

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

        # Create multiple augmented versions
        for aug_idx in range(augmentation_factor):
            # Create a deep copy of the row
            aug_row = row.copy()

            # Apply strong augmentation
            aug_image = image.copy()
            aug_image = augment_image(aug_image)
            aug_row['image'] = aug_image

            # Track augmentation version and source dataset
            aug_row['aug_version'] = aug_idx + 1
            aug_row['source_dataset'] = f"{source_dataset}_aug{aug_idx + 1}"

            # Add to our augmented dataset
            augmented_rows.append(aug_row)

    if augmented_rows:
        augmented_df = pd.DataFrame(augmented_rows)
        # Upload augmented images to MinIO
        augmented_df = load_df_with_raw_images_into_minio(augmented_df, bucket_name)
        logging.info(f"Created {len(augmented_df)} augmented images")

        # Combine original and augmented datasets
        combined_df = pd.concat([df, augmented_df])
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


def load_base_dataset(chunk_size=1000, total_samples=None, seed=42, apply_augmentation=True):
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

train = load_base_dataset(total_samples=50_000, apply_augmentation=True)


##
# @brief Loads the first additional dataset source
#
# Retrieves and processes extra invoice documents from a Kaggle dataset,
# uploads them to MinIO storage, and returns the DataFrame with URLs.
#
# @return DataFrame containing processed invoice images and their labels
#
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
        # Load the dataset from Hugging Face
        dataset = load_dataset("lmms-lab/DocVQA", split="validation")
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

                # Get the image
                image = Image.open(io.BytesIO(item["image"]["bytes"]))

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
    load_source_3,
    load_source_4, # alghough
]

logging.info("Loading extra sources")
sources_dfs = []
for source in extra_sources:
    extra_invoices = source()
    sources_dfs.append(extra_invoices)
sources_dfs = pd.concat(sources_dfs)

logging.info("Merging datasets")
combined_data = pd.concat([train, sources_dfs])

gc.collect()

# Split into train and test sets (80/20 split)
logging.info("Splitting into train and test sets")
from sklearn.model_selection import train_test_split

# Ensure all labels are of the same type (convert 'invoice' to 11 like in train.py)
combined_data['label_int'] = combined_data['label'].apply(lambda x: 11 if x == 'invoice' else int(x))
# remove duplicates
combined_data = combined_data.drop_duplicates(subset=['image'])
train_final, test_final = train_test_split(combined_data, test_size=0.2, random_state=42, stratify=combined_data['label_int'])

logging.info(f"Train set size: {len(train_final)}, Test set size: {len(test_final)}")

# Log class distribution after augmentation
class_distribution = train_final['label_int'].value_counts().sort_index()
logging.info(f"Final class distribution after augmentation: {class_distribution.to_dict()}")

logging.info("Saving datasets as csv")
train_final.to_csv("train.csv", index=False)
test_final.to_csv("test.csv", index=False)
logging.info("Datasets saved successfully")