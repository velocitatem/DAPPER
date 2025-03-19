# DAPPER (Document Analysis, Processing, and Pattern Extraction Repository)



# Classification
DAPPER is a document classification system that processes and analyzes various document types using machine learning. The system trains on the RVL-CDIP dataset supplemented with additional invoice data from various sources.

## Training Pipeline

The training pipeline consists of two main components:

1. **Data Processing** (`data.py`):
   - Loads the RVL-CDIP dataset and supplementary invoice datasets
   - Processes and resizes images to standard dimensions (768x992) (emperically averaged and rounded to a multiple of 32)
   - Uploads all images to a MinIO object storage server
   - Splits data into training and testing sets (80/20 split)
   - Saves dataset references as CSV files

2. **Model Training** (`train.py`):
   - Uses a custom `MinioImageDataset` class to fetch images from MinIO
   - Implements a ResNet18 model with transfer learning
   - Trains the model on the processed images
   - Evaluates performance on test data
   - Saves the best model based on test accuracy

## MinIO Integration

The system uses MinIO as an object storage solution for efficient handling of document images:

- **Storage**: All document images are stored in the "dapper" bucket in MinIO
- **Retrieval**: During training, images are fetched from MinIO in batches
- **Parallelization**: Both upload and retrieval operations are parallelized for efficiency
- **URLs**: Images are referenced by URLs in the format `http://localhost:9900/dapper/[filename]`

## Datasets
- [RVL-CDIP](https://huggingface.co/datasets/aharley/rvl_cdip) - Main dataset with various document types
- [Additional Invoice Dataset](https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1) - Supplementary invoice data
- [Company documents dataset from Kaggle](https://www.kaggle.com/datasets/ayoubcherguelaine/company-documents-dataset)

## Setup and Usage

1. Start the MinIO server:
   ```
   docker-compose up -d
   ```

2. Process and upload documents:
   ```
   python classification/data.py
   ```

3. Train the model:
   ```
   python classification/train.py
   ```

4. The trained model will be saved as `model.pth`
