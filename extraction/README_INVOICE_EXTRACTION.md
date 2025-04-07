# Invoice Information Extraction using LayoutLM

This document explains how to use the ML extractor for invoice information extraction. The ML extractor uses the LayoutLM model to extract structured information from invoice images.

## Overview

The ML extractor is a machine learning-based solution for extracting information from invoice images. It uses the LayoutLM model, which is specifically designed for document understanding tasks. The model can extract various fields from invoices, such as invoice numbers, dates, amounts, and company names.

## Features

- Uses LayoutLMv3 for document understanding
- Extracts structured information from invoice images
- Supports pre-trained models for quick deployment
- Fallback to rule-based extraction if ML extraction fails
- Configurable for different invoice formats
- Batch processing for multiple invoices
- Integration with SROIE dataset for standardized evaluation

## Requirements

- Python 3.8+
- PyTorch 1.10+
- transformers
- torchvision
- numpy
- PIL
- pytesseract
- tqdm
- pandas
- python-doctr (for SROIE dataset support)

## Installation

1. Install the required packages:

```bash
pip install torch torchvision transformers numpy pillow pytesseract tqdm pandas python-doctr
```

2. Install Tesseract OCR:

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download and install from https://github.com/UB-Mannheim/tesseract/wiki
```

## Usage

### SROIE Dataset

The ML extractor now supports the SROIE (Scanned Receipt OCR and Information Extraction) dataset from the doctr library. This dataset contains scanned receipts with annotations for key information fields.

To use the SROIE dataset with the ML extractor:

```bash
python extraction/example_invoice_extraction.py --output extracted_data.json --use_gpu --split test --download
```

Command-line arguments:
- `--split`: Choose between "train" or "test" split (default: "test")
- `--download`: Download the dataset if not already available
- `--use_polygons`: Use polygon bounding boxes instead of rectangles
- `--sample_limit`: Limit the number of samples to process
- `--model_path`: Path to a pre-trained model (optional)
- `--output`: Path to save the extracted data
- `--use_gpu`: Use GPU for inference

The script will:
1. Load the SROIE dataset
2. Extract information from each image using the ML extractor
3. Compare the extracted information with ground truth (if available)
4. Calculate and display accuracy metrics
5. Save the results to a JSON file

Example output:
```
SROIE dataset loaded with 347 samples
Processing SROIE samples: 100%|██████████| 347/347 [05:12<00:00, 1.11it/s]
invoice_number accuracy: 82.14% (115/140)
date accuracy: 90.35% (131/145)
total_amount accuracy: 85.71% (120/140)
issuer_name accuracy: 77.69% (108/139)
Overall accuracy: 84.05% (474/564)
Saved extracted data for 347 samples to extracted_data.json
Done!
```

### Training

To train the ML extractor on your own dataset, you need to prepare:

1. A directory containing invoice images
2. A JSON file with annotations for each image

The annotations file should have the following format:

```json
{
  "invoice1.jpg": {
    "invoice_number": "INV-001",
    "date": "2023-01-01",
    "due_date": "2023-02-01",
    "total_amount": 100.00,
    "issuer_name": "Company A",
    "recipient_name": "Company B"
  },
  "invoice2.jpg": {
    "invoice_number": "INV-002",
    "date": "2023-01-15",
    "due_date": "2023-02-15",
    "total_amount": 200.00,
    "issuer_name": "Company C",
    "recipient_name": "Company D"
  }
}
```

Then, you can train the model using the following code:

```python
from extraction.ml_extractor import MLExtractor

# Create extractor
extractor = MLExtractor()

# Train model
metrics = extractor.train(
    train_data_dir="path/to/train/images",
    annotations_file="path/to/train/annotations.json",
    val_data_dir="path/to/val/images",
    val_annotations_file="path/to/val/annotations.json",
    batch_size=8,
    num_epochs=5,
    learning_rate=5e-5,
    save_path="invoice_model.pth"
)

print(f"Training metrics: {metrics}")
```

### Manual Inference

To extract information from individual invoice images, you can use the ML extractor directly in your code:

```python
from extraction.ml_extractor import MLExtractor

# Create extractor
extractor = MLExtractor(model_path="invoice_model.pth")

# Extract information from an invoice
extracted_data = extractor.extract("path/to/invoice.jpg")

# Print extracted information
print(f"Invoice Number: {extracted_data.invoice_number}")
print(f"Date: {extracted_data.date}")
print(f"Total Amount: {extracted_data.total_amount}")
print(f"Issuer: {extracted_data.issuer_name}")
print(f"Recipient: {extracted_data.recipient_name}")
```

## How It Works

The ML extractor uses the following steps to extract information from invoice images:

1. **OCR**: The image is processed using Tesseract OCR to extract text and word bounding boxes.
2. **LayoutLM Processing**: The image, text, and bounding boxes are processed by the LayoutLM model.
3. **Token Classification**: The model classifies each token (word) into one of several categories (invoice number, date, amount, etc.).
4. **Entity Extraction**: The classified tokens are grouped into entities based on their labels.
5. **Post-processing**: The extracted entities are validated and formatted into structured data.
6. **Fallback**: If ML extraction fails, the system falls back to rule-based extraction.

## SROIE Dataset Details

The SROIE dataset was created for the ICDAR 2019 Competition on Scanned Receipt OCR and Information Extraction. It contains:

- 1,000 scanned receipt images
- Annotations for key fields:
  - Company/issuer name
  - Date
  - Address
  - Total amount

The dataset is split into training (626 images) and test (347 images) sets. Each receipt has a JSON file with the ground truth labels.

Example visualization:
![SROIE Dataset Example](https://doctr-static.mindee.com/models?id=v0.5.0/sroie-grid.png&src=0)

## Customization

You can customize the ML extractor for your specific needs:

1. **Label Mapping**: Modify the `id2label` dictionary in the `MLExtractor` class to add or remove fields.
2. **Model Architecture**: Use a different pre-trained model by changing the model path in the constructor.
3. **Training Parameters**: Adjust batch size, learning rate, and number of epochs in the `train` method.
4. **Validation Rules**: Modify the `validate_extraction` method to add custom validation rules.

## Troubleshooting

- **Out of Memory Errors**: If you encounter out of memory errors, try reducing the batch size or using a smaller model.
- **Poor Extraction Quality**: If the extraction quality is poor, try:
  - Using a pre-trained model that has been fine-tuned on a similar dataset
  - Increasing the number of training epochs
  - Adjusting the learning rate
  - Adding more training data
- **OCR Issues**: If OCR is not working correctly, make sure Tesseract is installed and the image quality is good.
- **SROIE Dataset Issues**: If you encounter issues with the SROIE dataset, make sure you're using a recent version of python-doctr and that you have an internet connection for downloading the dataset.

## Conclusion

The ML extractor provides a powerful solution for extracting structured information from invoice images. By using the LayoutLM model, it can understand the layout and content of invoices, making it suitable for various invoice formats. The integration with the SROIE dataset provides a standardized benchmark for evaluating the performance of the system. 