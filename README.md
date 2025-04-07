# DAPPER: Document Analysis, Processing, and Pattern Extraction Repository

[![Build and Deploy Documentation](https://github.com/velocitatem/DAPPER/actions/workflows/build-docs.yml/badge.svg)](https://github.com/velocitatem/DAPPER/actions/workflows/build-docs.yml)


## Transform Your Document Processing with AI

DAPPER is a powerful, end-to-end document intelligence platform that combines cutting-edge computer vision and natural language processing to automate document classification, information extraction, and analysis. Stop wasting hours manually sorting and processing documents - let DAPPER do the heavy lifting.

### Key Features

- **Multi-Model Document Classification**: Achieve up to 95% accuracy with our state-of-the-art models
- **Intelligent Information Extraction**: Automatically pull key data from invoices and forms
- **Interactive Dashboard**: User-friendly interface for document uploading and processing
- **Batch Processing**: Handle thousands of documents with a single upload
- **Scalable Architecture**: MinIO integration for efficient document storage and retrieval

## Advanced AI Models

DAPPER leverages multiple state-of-the-art models to achieve superior document classification:

### LSNet: See Large, Focus Small
Our implementation of LSNet (inspired by human vision) combines large-kernel perception and small-kernel aggregation for exceptional accuracy with minimal computational requirements:

- **LSNet-T (Tiny)**: 11.4M parameters, ideal for edge devices
- **LSNet-S (Small)**: 16.1M parameters with improved accuracy
- **LSNet-B (Base)**: 23.2M parameters for highest accuracy

### ResNet Transfer Learning
We've fine-tuned ResNet18 models specifically for document classification, achieving excellent results on diverse document types.

### Ensemble Self-Attention Mutual Learning (EAML)
Our multimodal approach combines visual features with OCR text extraction for superior accuracy on text-heavy documents.

## Data to Training Pipeline

DAPPER implements a comprehensive end-to-end pipeline from data processing to model training:

### Data Management

- **MinIO Integration**: Scalable object storage for document images with standardized naming conventions (`src_<dataset>_cls_<class>_idx_<index>_<aug-info>.jpg`)
- **Multi-Source Dataset Loading**: Parallel loading from diverse document sources including:
  - RVL-CDIP (general document types)
  - Kaggle Invoices dataset
  - Hugging Face Invoices dataset

### Data Processing Pipeline

1. **Dataset Loading**: The `loader.py` module combines multiple document datasets with `get_full_dataset()` function
2. **Data Augmentation**: The `augmentor.py` module provides:
   - Geometric transformations (rotation, perspective, affine)
   - Color adjustments (brightness, contrast, hue)
   - Noise addition and random erasing
   - Resolution adjustments for standardization
3. **OCR Processing**: Automatic text extraction using Tesseract OCR with:
   - Text confidence scoring
   - Bounding box identification
   - Parallel processing for efficiency

### Training Infrastructure

1. **Dataset Preparation**:
   - `MinioImageDataset` for image-only models
   - `MinioMultiModalDataset` for models requiring both image and OCR text
   - Automatic label mapping and standardization

2. **Model Selection**:
   - Multiple architectures available through a unified interface
   - Models: HOG, CNN, ResNet, LSNet (T/S/B), EAML, LayoutLMv3, Hybrid

3. **Training Pipeline**:
   - Configurable hyperparameters via YAML configuration files
   - TensorBoard integration for monitoring
   - Optimized DataLoaders with prefetching and parallel workers
   - Model checkpointing and early stopping
   - Comprehensive metrics tracking (accuracy, precision, recall, F1)

4. **Transfer Learning**:
   - Pre-trained foundation models fine-tuned for document understanding
   - Cross-modal knowledge transfer for multimodal approaches

### Evaluation and Analysis

- Performance metrics across different document types
- Confusion matrix visualization
- Model interpretability through attention visualization
- Comparative benchmarking between model architectures

### Model Performance Metrics

| Model    | Accuracy | Precision | Recall  | F1 Score |
|----------|----------|-----------|---------|----------|
| resnet   | 0.8295   | 0.8302    | 0.8295  | 0.8295   |
| eaml     | 0.6699   | 0.6880    | 0.6699  | 0.6674   |
| cnn      | 0.7075   | 0.7068    | 0.7075  | 0.7056   |
| lsnet_t  | 0.7432   | 0.7427    | 0.7432  | 0.7424   |

## Document Types Supported

DAPPER accurately classifies 16 document types including:
- Letters & Emails
- Invoices & Budgets
- Scientific Reports & Publications
- Handwritten Documents
- Presentations & Resumes
- And more!

## Information Extraction

Beyond classification, DAPPER intelligently extracts key information from documents:

- **Invoice Processing**: Extract invoice numbers, dates, totals, and vendor details
- **Rule-Based Extraction**: Customizable extraction rules for different document types
- **Validation**: Automatic validation of extracted information
- **Confidence Scoring**: Know how reliable the extracted information is

## Interactive Dashboard

Upload and process documents through our intuitive web interface:
- Single file or batch upload support
- PDF and image format support
- Real-time classification results with confidence scores
- Export functionality for processed documents

## Quick Setup

```bash
# Clone the repository
git clone https://github.com/velocitatem/DAPPER.git
cd DAPPER

# Install dependencies
pip install -r requirements.txt

# Start MinIO server for document storage
docker-compose up -d

# Launch the dashboard
streamlit run dashboard.py
```

## Detailed Environment Setup

### Prerequisites
- Python 3.8+
- Docker and Docker Compose
- 8GB+ RAM recommended for model training

### Step-by-Step Environment Setup

1. **Create and activate a virtual environment**:
   ```bash
   # Using venv
   python -m venv dapper-env
   
   # On Windows
   dapper-env\Scripts\activate
   
   # On Linux/macOS
   source dapper-env/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR** (required for text extraction):
   - On Ubuntu/Debian:
     ```bash
     sudo apt update
     sudo apt install tesseract-ocr
     ```
   - On macOS:
     ```bash
     brew install tesseract
     ```
   - On Windows:
     Download and install from https://github.com/UB-Mannheim/tesseract/wiki

4. **Configure MinIO** (document storage):
   ```bash
   # Start the MinIO server
   docker-compose up -d
   
   # Create required bucket: dapper (done w docker)
   ```

5. **Set environment variables** (create a .env file):
   ```
   MINIO_ENDPOINT=localhost:9000
   MINIO_ACCESS_KEY=minioadmin
   MINIO_SECRET_KEY=minioadmin
   MINIO_SECURE=False
   ```

## Complete Pipeline Execution

### 1. Data Preparation

```bash
# Download and prepare datasets
python -m classification.data.loader
```

### 2. Model Training

```bash
python classification/training/train.py --config configs/training/cnn_config.yaml
```

Now in `UI/server.py` modify the path to your desired model, I would recommender RESNET.
You can download the resnet classifier [heer](https://drive.google.com/file/d/1yygQP8i6Aw8VcpxUFdgZmMYVKHcpXU6Q/view?usp=sharing) and as for extraction, the latest weights are here in this [zip for the extractor weights](https://drive.google.com/file/d/19waYTXqFQ6Ta1RTnNlI9g7r4Fl4EpYau/view?usp=sharing) you should extract to a directory and then point to that directory.

### 3. Running the Website

```bash
python -m UI.server
```

## Dependencies

The main dependencies include:

```
torch>=1.12.0
torchvision>=0.13.0
streamlit>=1.18.0
pillow>=9.2.0
pytesseract>=0.3.10
minio>=7.1.0
numpy>=1.22.0
pandas>=1.4.0
scikit-learn>=1.0.0
opencv-python>=4.6.0
tqdm>=4.64.0
pyyaml>=6.0
matplotlib>=3.5.0
transformers>=4.21.0
python-dotenv>=0.20.0
```

Detailed dependency versions are specified in `requirements.txt`.

## Configuration

### Model Configuration


Example configuration for LSNet-T model (`classification/training/configs/cnn_config.yaml`):
```yaml
model:
  name: "cnn"
  input_channels: 3 

```

### MinIO Configuration

MinIO storage is configured through environment variables or a `.env` file:

```
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=False
```



## Example Usage

```python
# Classify a document
from classification.training.models import get_model
from PIL import Image

# Load a pre-trained model
model = get_model(model_name='lsnet_t', num_classes=16)
model.load_state_dict(torch.load('models/lsnet_t_best.pth'))

# Classify an image
image = Image.open('path/to/document.jpg')
result = model.predict(image)

# Extract information from an invoice
from extraction import RuleBasedExtractor

extractor = RuleBasedExtractor()
invoice_data = extractor.extract(document_text)
print(f"Invoice #: {invoice_data.invoice_number}, Total: {invoice_data.total_amount}")
```

## Use Cases

- **Finance Departments**: Automate invoice processing and verification
- **Legal Teams**: Sort and classify legal documents and contracts
- **Academic Research**: Organize and analyze scientific publications
- **Administrative Work**: Streamline document management and filing

## Datasets

Trained on diverse document collections including:
- [RVL-CDIP](https://huggingface.co/datasets/aharley/rvl_cdip)
- [Invoices Dataset](https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1)
- [Company Documents Dataset](https://www.kaggle.com/datasets/ayoubcherguelaine/company-documents-dataset)
