# DAPPER (Document Analysis, Processing, and Pattern Extraction Repository)

[![Build and Deploy Documentation](https://github.com/velocitatem/DAPPER/actions/workflows/build-docs.yml/badge.svg)](https://github.com/velocitatem/DAPPER/actions/workflows/build-docs.yml)


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

# Document Classification with OCR

This project implements a document classification system that combines image and text features using an Ensemble Self-Attention-based Mutual Learning Network (EAML). The system can process document images and extract text using OCR (Optical Character Recognition) to improve classification accuracy.

## Features

- Document image classification using a multimodal approach
- OCR text extraction from document images
- Parallel processing of OCR tasks for improved performance
- Support for multiple document datasets
- Data augmentation for improved model training

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/document-classification.git
cd document-classification
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR:
   - **Ubuntu/Debian**:
     ```bash
     sudo apt-get install tesseract-ocr
     ```
   - **macOS**:
     ```bash
     brew install tesseract
     ```
   - **Windows**:
     Download and install from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage

### Loading Datasets with OCR

To load datasets with OCR processing:

```python
from classification.data.loader import get_full_dataset
from classification.data.minio_handler import MinioManager
from classification.data.augmentor import Augmentor

# Initialize MinIO manager and augmentor
minio_manager = MinioManager()
augmentor = Augmentor()

# Load datasets with OCR
df = get_full_dataset(
    datasets_list=['rvl_cdip'],
    minio_manager=minio_manager,
    augmentor=augmentor,
    apply_ocr=True,
    ocr_workers=4
)
```

### Training the EAML Model with OCR Text

To train the EAML model using OCR text:

```python
from classification.training.eaml import EAMLClassifier
from torch.utils.data import DataLoader

# Initialize the EAML model with OCR text support
model = EAMLClassifier(
    num_classes=len(df['label'].unique()),
    vocab_size=vocab_size,
    embedding_dim=100,
    word_hidden_dim=50,
    sent_hidden_dim=50,
    image_channels=3,
    image_feature_dim=128,
    image_size=(224, 224),
    dropout=0.5,
    learning_rate=0.001,
    num_epochs=10,
    use_ocr_text=True
)

# Train the model
model.train_model(
    train_loader=data_loader,
    save_dir="models",
    patience=5
)
```

### Example Script

An example script demonstrating the OCR functionality is provided in `classification/examples/ocr_example.py`:

```bash
python -m classification.examples.ocr_example
```

## Project Structure

- `classification/data/`: Data loading and processing utilities
  - `augmentor.py`: Image augmentation and OCR processing
  - `loader.py`: Dataset loading utilities
  - `minio_handler.py`: MinIO storage integration
- `classification/datasets/`: Dataset-specific loaders
- `classification/training/`: Model training utilities
  - `eaml.py`: EAML model implementation
- `classification/examples/`: Example scripts
- `classification/utils/`: Utility functions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- RVL-CDIP dataset: [https://www.kaggle.com/datasets/ayoubcherguelaine/company-documents-dataset](https://www.kaggle.com/datasets/ayoubcherguelaine/company-documents-dataset)
- EAML architecture: [https://arxiv.org/abs/2305.06923v1](https://arxiv.org/abs/2305.06923v1)
