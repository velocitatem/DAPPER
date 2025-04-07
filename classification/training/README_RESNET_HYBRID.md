# ResNet Hybrid Model for Document Classification

This document explains how to use the ResNet Hybrid model for document classification. The ResNet Hybrid model combines ResNet for image processing and EAML for text processing, providing a powerful solution for document classification tasks.

## Overview

The ResNet Hybrid model is an extension of the original Hybrid model, which combines LSNet for image processing and EAML for text processing. The ResNet Hybrid model replaces LSNet with ResNet, providing better image feature extraction capabilities.

## Features

- Combines ResNet for image processing and EAML for text processing
- Supports pre-trained ResNet models (ResNet18, ResNet34, ResNet50, ResNet101)
- Configurable through the existing `hybrid_config.yaml` file
- Mixed precision training for improved performance
- Configurable learning rate schedulers
- TensorBoard integration for monitoring training progress

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- numpy
- PIL
- tqdm
- tensorboard
- optuna (for hyperparameter tuning)

## Configuration

The ResNet Hybrid model uses the existing `hybrid_config.yaml` file with additional ResNet-specific parameters:

```yaml
model:
  # Existing parameters
  vocab_size: 10000
  embedding_dim: 100
  word_hidden_dim: 64
  sent_hidden_dim: 128
  lsnet_model_size: "s"  # Ignored when use_resnet is True
  dropout: 0.6
  learning_rate: 0.0005
  weight_decay: 0.03
  num_epochs: 70
  use_ocr_text: true
  
  # ResNet-specific parameters
  use_resnet: true  # Set to true to use ResNet instead of LSNet
  resnet_model_name: "resnet50"  # Options: "resnet18", "resnet34", "resnet50", "resnet101"
  resnet_pretrained: true  # Whether to use pre-trained weights

training:
  patience: 10
  mixed_precision: true
  scheduler_type: "cosine"  # Options: "cosine", "linear", "step", "plateau"
  max_sentences: 20
  max_sent_length: 50

logging:
  log_dir: "logs"
  experiment_name: "resnet_hybrid_baseline"
  save_dir: "models/hybrid"
```

## Usage

### Training

To train the ResNet Hybrid model, use the existing training script with the `hybrid_config.yaml` file:

```bash
python classification/training/train.py --config classification/training/configs/hybrid_config.yaml --data_dir /path/to/data --output_dir /path/to/output --num_classes 10 --vocab_size 10000 --use_gpu
```

Make sure to set `use_resnet: true` in the configuration file to use ResNet instead of LSNet.

### Inference

To perform inference with the ResNet Hybrid model, use the example script:

```bash
python classification/training/example_resnet_hybrid.py --config classification/training/configs/hybrid_config.yaml --image /path/to/image.jpg --text /path/to/text.txt --vocab_path /path/to/vocabulary.txt --use_gpu
```

### Example Code

Here's a simple example of how to use the ResNet Hybrid model in your code:

```python
import torch
from classification.training.hybrid import HybridTrainer

# Create a trainer
trainer = HybridTrainer(
    num_classes=10,
    vocab_size=10000,
    embedding_dim=100,
    word_hidden_dim=64,
    sent_hidden_dim=128,
    lsnet_model_size="s",  # Ignored when use_resnet is True
    dropout=0.6,
    learning_rate=0.0005,
    weight_decay=0.03,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    num_epochs=70,
    save_dir="models/hybrid",
    patience=10,
    mixed_precision=True,
    scheduler_type="cosine",
    use_ocr_text=True,
    use_resnet=True,  # Set to true to use ResNet
    resnet_model_name="resnet50",
    resnet_pretrained=True
)

# Train the model
trainer.train(train_loader, val_loader)

# Save the model
trainer.save_model("resnet_hybrid_model.pt")

# Load the model
trainer.load_model("resnet_hybrid_model.pt")

# Perform inference
trainer.model.eval()
with torch.no_grad():
    outputs = trainer.model(image_tensor, text_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
```

## Comparison with Original Hybrid Model

The ResNet Hybrid model offers several advantages over the original Hybrid model:

1. **Better Image Feature Extraction**: ResNet models are generally better at extracting features from images compared to LSNet, especially for complex documents.
2. **Flexibility**: You can choose from different ResNet architectures (ResNet18, ResNet34, ResNet50, ResNet101) based on your needs.
3. **Pre-trained Weights**: ResNet models come with pre-trained weights on ImageNet, which can be leveraged for better performance.

## Troubleshooting

- **Out of Memory Errors**: If you encounter out of memory errors, try reducing the batch size or using a smaller ResNet model (e.g., ResNet18 instead of ResNet50).
- **Slow Training**: If training is too slow, try using mixed precision training by setting `mixed_precision: true` in the configuration file.
- **Poor Performance**: If the model's performance is not satisfactory, try using a larger ResNet model (e.g., ResNet101) or adjusting the hyperparameters.

## Conclusion

The ResNet Hybrid model provides a powerful solution for document classification tasks by combining the strengths of ResNet for image processing and EAML for text processing. By using the existing `hybrid_config.yaml` file with additional ResNet-specific parameters, you can easily switch between LSNet and ResNet for image processing. 