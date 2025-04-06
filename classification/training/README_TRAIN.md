# Training Module

This module provides functionality for training different image classification models.

## Configuration-Based Training

The training script now uses YAML configuration files instead of command-line arguments. This makes it easier to manage different training configurations and experiment with various parameters.

### Available Models

- **HOG Classifier**: Uses Histogram of Oriented Gradients features with a traditional classifier
- **CNN Classifier**: Uses a Convolutional Neural Network for image classification
- **ResNet Classifier**: Uses a pre-trained ResNet model for document classification
- **EAML Classifier**: (Description for EAML model)
- **LSNet Classifier**: (Description for LSNet model)

### Configuration Files

Configuration files are stored in the `configs` directory:

- `hog_config.yaml`: Configuration for the HOG classifier
- `cnn_config.yaml`: Configuration for the CNN classifier
- `resnet_config.yaml`: Configuration for the ResNet classifier
- `eaml_config.yaml`: Configuration for the EAML classifier
- `lsnet_config.yaml`: Configuration for the LSNet classifier

### Example Configuration Files

#### HOG Classifier Configuration

```yaml
# HOG Classifier Configuration
model:
  name: "hog"
  classifier: "logistic_regression"  # Options: logistic_regression, svm

# Training parameters
training:
  batch_size: 32
  num_workers: 4
  test_size: 0.2

# Logging parameters
logging:
  log_dir: "logs"
  experiment_name: null  # Will be auto-generated if null
```

#### CNN Classifier Configuration

```yaml
# CNN Classifier Configuration
model:
  name: "cnn"
  input_channels: 3  # 1 for grayscale, 3 for RGB
  learning_rate: 0.001
  num_epochs: 10

# Training parameters
training:
  batch_size: 32
  num_workers: 4
  test_size: 0.2

# Logging parameters
logging:
  log_dir: "logs"
  experiment_name: null  # Will be auto-generated if null
```

#### ResNet Classifier Configuration

```yaml
# ResNet Classifier Configuration
model:
  name: "resnet"
  resnet_model: "resnet18"  # Options: resnet18, resnet34, resnet50, resnet101
  pretrained: true
  learning_rate: 0.0001
  weight_decay: 0.01
  num_epochs: 50
  dropout_rate: 0.7

# Training parameters
training:
  batch_size: 64
  num_workers: 4
  test_size: 0.2

# Logging parameters
logging:
  log_dir: "logs"
  experiment_name: null  # Will be auto-generated if null
```

### Running Training

To train a model using a configuration file, first define the configuration directory and file:

```bash
config_dir="classification/training/configs"
config="hog_config.yaml" # Or cnn_config.yaml, resnet_config.yaml, eaml_config.yaml, lsnet_config.yaml, or your custom config
```

Then, run the training script. The `&` runs the command in the background:

```bash
python -m classification.training.train --config "$config_dir/$config" &
```


Here's a detailed explanation of each of the architectures you've implemented, how they work with your data, and a comparison table at the end.

## 1. CNNClassifier (Convolutional Neural Network)

**Description:**

CNNClassifier is a classic convolutional neural network tailored specifically for image classification. It comprises multiple convolutional layers, each followed by pooling layers to reduce spatial dimensions, followed by dense (fully-connected) layers to learn classification features. Dropout is included for regularization to prevent overfitting.

**Architecture Highlights:**
- **Convolutional Layers:** Capture local features such as edges, textures, and patterns.
- **Pooling Layers:** Down-sample feature maps to reduce computational load.
- **Fully-connected Layers:** Interpret the abstracted features to predict the class label.

**Use with Data:**
- Inputs are images (document scans), resized to `(224, 224)`.
- Classifies documents into categories based solely on visual features.
- Does not leverage textual information.


<img src="../../model_images/cnn.png" style="max-width: 800px;">

## 2. LSNetClassifier (Local-Scale Network)

**Description:**

LSNet is an advanced convolutional architecture focused on extracting rich visual features using a custom "LSBlock" structure, involving "LSConvolutions" for better capturing of local spatial information. It offers robust performance due to deeper and more complex convolutional structures.

**Architecture Highlights:**
- **LSBlocks:** Specialized blocks designed to enhance feature extraction at local scales.
- **Convolutional Backbone:** Extensive layers that significantly abstract visual details.
- **Adaptive Average Pooling:** Generates a fixed-size representation regardless of input image size.

**Use with Data:**
- Takes images from the dataset to classify into specified categories.
- Better suited for capturing fine-grained document visual characteristics, like formats and layouts.
- Does not use textual content from OCR.

<img src="../../model_images/lsnet.png" style="max-width: 800px;">

## 3. EAMLClassifier (Enhanced Multi-level Attention Network)

**Description:**

EAML combines text and image modalities into a unified multimodal architecture. It leverages a hierarchical attention mechanism to process textual data at word and sentence levels using GRU-based recurrent layers, alongside a CNN-based image encoder to extract visual features.

**Architecture Highlights:**
- **Hierarchical GRUs:** Process OCR text, capturing sequential textual semantics.
- **Multi-level Attention (Word and Sentence level):** Automatically identifies and emphasizes important words and sentences.
- **CNN Image Encoder:** Captures visual patterns from the documents.
- **Feature Fusion:** Combines textual and visual embeddings into a joint representation.

**Use with Data:**
- Effectively leverages both visual layout and textual content (OCR data).
- Well-suited to documents like invoices where both visual and textual clues determine classification.

<img src="../../model_images/eaml.png" style="max-width: 800px;">

## 4. HybridClassifier

**Description:**

HybridClassifier integrates the strengths of LSNet (for visual feature extraction) with hierarchical attention-based textual processing (similar to EAML). This results in an advanced multimodal approach capable of deeply understanding both image structures and textual semantics.

**Architecture Highlights:**
- **LSNet backbone:** Highly capable visual feature extraction.
- **Hierarchical GRU & Attention Layers:** Robust text embedding and selection.
- **Fusion Layer:** Blends the visual and textual features into a unified representation.
- **Classification Layers:** Predicts the document class from the combined features.

**Use with Data:**
- Maximally exploits your dataset's multimodal nature (visual + textual content).
- Offers high classification accuracy due to deep multimodal fusion capabilities.

<img src="../../model_images/hybrid.png" style="max-width: 800px;">

## 5. ResNetClassifier (Residual Network)

**Description:**

ResNetClassifier leverages a pre-trained ResNet18 architecture, renowned for its residual connections allowing deeper networks with reduced vanishing gradient issues. It is fine-tuned specifically for your classification task.

**Architecture Highlights:**
- **Residual Connections:** Allow very deep networks with stable training.
- **Pre-trained on ImageNet:** Provides robust initial features for transfer learning.
- **Fine-tuned Layers:** Adapt the ResNet model specifically to your dataset and classification task.

**Use with Data:**
- Primarily uses image data.
- Powerful due to pre-training and fine-tuning but doesn't directly use textual OCR data unless separately combined.

<img src="../../model_images/resnet.png" style="max-width: 800px;">

## Comparison Table of the Models

| Feature / Model         | CNNClassifier | LSNetClassifier | EAMLClassifier | HybridClassifier | ResNetClassifier |
|-------------------------|---------------|-----------------|----------------|------------------|------------------|
| **Modality**            | Image only    | Image only      | Image + Text   | Image + Text     | Image (pretrained)|
| **Textual Handling**    | None          | None            | Attention-based hierarchical GRUs | Attention-based hierarchical GRUs | None (unless combined externally) |
| **Visual Complexity**   | Moderate      | High            | Moderate       | Very High        | High (due to residual layers) |
| **Attention Mechanism** | None          | None            | Hierarchical Text Attention | Hierarchical Text Attention | None |
| **Trainable Parameters**| 51.4M         | 47.3M           | 0.93M          | 73.5M            | 11.4M |
| **Computational Cost**  | Moderate      | High            | Moderate       | Very High        | Moderate |
| **Main Strength**       | Simple baseline, computationally efficient | Complex visual extraction | Robust multimodal understanding | Deep fusion of visual-textual features | Transfer learning capabilities |
| **Ideal Use Case**      | Quick image-based classification | Complex visual patterns | Text-rich documents | Complex multimodal documents | Good visual baseline |


## How Each Model Fits Your Data and Project Requirements:

Your project requires document classification and invoice information extraction:

- **CNNClassifier and LSNetClassifier:** Great for baseline or purely visual-based classification but do not leverage textual OCR data.
- **EAMLClassifier and HybridClassifier:** Particularly suitable for your project, effectively exploiting both image and OCR text data, improving classification accuracy for invoices, forms, and correspondence.
- **ResNetClassifier:** Provides a strong baseline leveraging transfer learning, useful for robust visual classification but might need separate OCR handling.