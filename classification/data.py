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
from PIL import Image
import re
import kagglehub
from pdf2image import convert_from_path
from PIL import Image
import gc
from minio import Minio
import logging
import io
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')



##
# @brief Resizes input images to a standardized size
#
# This function resizes input images to a standard size of 768x992 pixels,
# which was determined by analyzing the statistics of the dataset.
#
# @param image The PIL Image object to resize
# @return A resized PIL Image object
#
def resize_image(image):
    width, height = 768, 992
    # Image statistics from dataset analysis:
    # Total images analyzed: 1829
    # Width: min=625, max=1654, mean=1168.64, median=787.0
    # Height: min=1000, max=2339, mean=1607.64, median=1000.0
    return image.resize((width, height))

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
    image = resize_image(image)
    label = str(row['label'])
    random_name = re.sub(r'\W+', '', label) + str(i) + ".jpg"
    if random_name in all_images:
        logging.info(f"Image {random_name} already exists in bucket, skipping.")
        return i, f"http://localhost:9900/{bucket_name}/{random_name}"

    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)

    client.put_object(
        bucket_name,
        random_name,
        image_bytes,
        len(image_bytes.getvalue())
    )

    image.close()
    image_bytes.close()

    return i, f"http://localhost:9900/{bucket_name}/{random_name}"

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


def load_base_dataset(chunk_size=1000, total_samples=None, seed=42):
    """
    Progressively loads chunks of the RVL-CDIP dataset into MinIO.

    Args:
        chunk_size (int): Number of samples per chunk
        total_samples (int): Total number of samples to load. If None, loads entire dataset
        seed (int): Random seed for shuffling

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

        # Load images into MinIO
        processed_chunk = load_df_with_raw_images_into_minio(chunk_df, "dapper")
        loaded_chunks.append(processed_chunk)

        # Clean up memory
        gc.collect()

        logging.info(f"Finished loading chunk {start_idx//chunk_size + 1}")

    return pd.concat(loaded_chunks)

train = load_base_dataset(total_samples=120000)

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
    extra_invoices = load_df_with_raw_images_into_minio(extra_invoices, "dapper")
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

    # Now we load the images into MinIO
    df = load_df_with_raw_images_into_minio(df, "dapper")
    return df



extra_sources = [
    load_source_1,
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
train_final, test_final = train_test_split(combined_data, test_size=0.2, random_state=42, stratify=combined_data['label_int'])

logging.info(f"Train set size: {len(train_final)}, Test set size: {len(test_final)}")

logging.info("Saving datasets as csv")
train_final.to_csv("train.csv", index=False)
test_final.to_csv("test.csv", index=False)
logging.info("Datasets saved successfully")
