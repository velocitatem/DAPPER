# Training Module

This module provides functionality for training different image classification models.

## Configuration-Based Training

The training script now uses YAML configuration files instead of command-line arguments. This makes it easier to manage different training configurations and experiment with various parameters.

### Available Models

- **HOG Classifier**: Uses Histogram of Oriented Gradients features with a traditional classifier
- **CNN Classifier**: Uses a Convolutional Neural Network for image classification
- **ResNet Classifier**: Uses a pre-trained ResNet model for document classification

### Configuration Files

Configuration files are stored in the `configs` directory:

- `hog_config.yaml`: Configuration for the HOG classifier
- `cnn_config.yaml`: Configuration for the CNN classifier
- `resnet_config.yaml`: Configuration for the ResNet classifier

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

To train a model using a configuration file:

```bash
python -m classification.training.train --config classification/training/configs/hog_config.yaml
```

or

```bash
python -m classification.training.train --config classification/training/configs/cnn_config.yaml
```

or

```bash
python -m classification.training.train --config classification/training/configs/resnet_config.yaml
```

### Creating Custom Configurations

You can create custom configuration files by copying and modifying the existing ones. Make sure to:

1. Set the correct `model.name` field to match the model you want to use
2. Adjust the model-specific parameters as needed
3. Set the training parameters according to your requirements
4. Optionally, provide a custom experiment name in the logging section

## Model-Specific Parameters

### HOG Classifier

- `classifier`: The classifier to use with HOG features
  - `logistic_regression`: Logistic Regression classifier
  - `svm`: Support Vector Machine classifier

### CNN Classifier

- `input_channels`: Number of input channels (1 for grayscale, 3 for RGB)
- `learning_rate`: Learning rate for the optimizer
- `num_epochs`: Number of training epochs

### ResNet Classifier

- `resnet_model`: The ResNet model architecture to use
  - `resnet18`: ResNet-18 architecture
  - `resnet34`: ResNet-34 architecture
  - `resnet50`: ResNet-50 architecture
  - `resnet101`: ResNet-101 architecture
- `pretrained`: Whether to use pre-trained weights
- `learning_rate`: Learning rate for the optimizer
- `weight_decay`: Weight decay for regularization
- `num_epochs`: Number of training epochs
- `dropout_rate`: Dropout rate for regularization

## Training Parameters

- `batch_size`: Batch size for training
- `num_workers`: Number of worker processes for data loading
- `test_size`: Proportion of the dataset to use for validation

## Logging Parameters

- `log_dir`: Directory to save logs
- `experiment_name`: Name for the experiment (will be auto-generated if null) 