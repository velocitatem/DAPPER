# Configuration Files Overview

This directory contains YAML configuration files for training various document classification models. Each configuration file defines model architecture, training parameters, data loading settings, and hyperparameter tuning options.

The configuration files provide a flexible way to control model training without modifying code. They support reproducible experiments by capturing all relevant parameters in a single file.

Key configuration files:
- resnet_config.yaml: ResNet-based classifier configuration
- cnn_config.yaml: CNN-based classifier configuration  
- hog_config.yaml: HOG feature-based classifier configuration
- lsnet_config.yaml: LSNet classifier configuration
- eaml_config.yaml: EAML multimodal classifier configuration

@see classification.training.train.load_config


# ResNet Configuration
Configuration for ResNet-based document classifier

The `resnet_config.yaml` file defines parameters for training a ResNet model for document classification.

### Key Parameters:
- **model.name**: Set to "resnet" to use the ResNet classifier
- **model.resnet_model**: ResNet variant to use (resnet18, resnet34, resnet50, resnet101)
- **model.pretrained**: Whether to use pre-trained weights
- **data.dataloader.batch_size**: Batch size for training
- **data.dataloader.num_workers**: Number of worker processes for data loading
- **training.test_size**: Proportion of data to use for validation
- **training.num_epochs**: Number of training epochs
- **training.dropout_rate**: Dropout rate for regularization
- **training.optimizer.learning_rate**: Learning rate for the optimizer
- **training.optimizer.weight_decay**: Weight decay for regularization
- **tuning.n_trials**: Number of trials for hyperparameter optimization

@see classification.training.resnet.ResNetClassifier

# CNN Configuration
Configuration for CNN-based document classifier

The `cnn_config.yaml` file defines parameters for training a CNN model for document classification.

### Key Parameters:
- **model.name**: Set to "cnn" to use the CNN classifier
- **model.input_channels**: Number of input channels (typically 3 for RGB)
- **data.dataloader.batch_size**: Batch size for training
- **data.dataloader.num_workers**: Number of worker processes for data loading
- **training.test_size**: Proportion of data to use for validation
- **training.num_epochs**: Number of training epochs
- **training.optimizer.learning_rate**: Learning rate for the optimizer
- **tuning.n_trials**: Number of trials for hyperparameter optimization

@see classification.training.cnn.CNNClassifier

# HOG Configuration
Configuration for HOG-based document classifier

The `hog_config.yaml` file defines parameters for training a HOG-based model for document classification.

### Key Parameters:
- **model.name**: Set to "hog" to use the HOG classifier
- **model.classifier**: Type of classifier to use with HOG features
- **data.dataloader.batch_size**: Batch size for training
- **data.dataloader.num_workers**: Number of worker processes for data loading
- **training.test_size**: Proportion of data to use for validation
- **training.num_epochs**: Number of training epochs
- **tuning.n_trials**: Number of trials for hyperparameter optimization

@see classification.training.hog.HogClassifier

# LSNet Configuration
Configuration for LSNet-based document classifier

The `lsnet_config.yaml` file defines parameters for training a LSNet model for document classification.

### Key Parameters:
- **model.name**: Set to "lsnet_t", "lsnet_s", or "lsnet_b" to use the LSNet classifier
- **model.pretrained**: Whether to use pre-trained weights
- **model.freeze_backbone**: Whether to freeze the backbone during training
- **data.dataloader.batch_size**: Batch size for training
- **data.dataloader.num_workers**: Number of worker processes for data loading
- **training.test_size**: Proportion of data to use for validation
- **training.num_epochs**: Number of training epochs
- **training.optimizer.learning_rate**: Learning rate for the optimizer
- **training.optimizer.weight_decay**: Weight decay for regularization
- **tuning.n_trials**: Number of trials for hyperparameter optimization

@see classification.training.lsnet.LSNetClassifier

# EAML Configuration
Configuration for EAML-based document classifier

The `eaml_config.yaml` file defines parameters for training an EAML model for document classification.

### Key Parameters:
- **model.name**: Set to "eaml" to use the EAML classifier
- **model.embedding_dim**: Dimension of word embeddings
- **model.word_hidden_dim**: Hidden dimension for word-level processing
- **model.sent_hidden_dim**: Hidden dimension for sentence-level processing
- **model.image_channels**: Number of image channels
- **model.image_feature_dim**: Dimension of image features
- **model.dropout**: Dropout rate for regularization
- **data.tokenizer_name**: Name of the tokenizer to use
- **data.max_sentences**: Maximum number of sentences to process
- **data.max_sent_length**: Maximum length of each sentence
- **data.dataloader.batch_size**: Batch size for training
- **data.dataloader.num_workers**: Number of worker processes for data loading
- **training.test_size**: Proportion of data to use for validation
- **training.num_epochs**: Number of training epochs
- **training.learning_rate**: Learning rate for the optimizer
- **tuning.n_trials**: Number of trials for hyperparameter optimization

@see classification.training.eaml.EAMLClassifier

# LayoutLMv3 Configuration
Configuration for LayoutLMv3-based document classifier

The `layoutlmv3.yaml` file defines parameters for training a LayoutLMv3 model for document classification.

### Key Parameters:
- **model.name**: Set to "layoutlmv3" to use the LayoutLMv3 classifier
- **model.learning_rate**: Learning rate for the optimizer
- **model.weight_decay**: Weight decay for regularization
- **model.num_epochs**: Number of training epochs
- **model.apply_ocr**: Whether to apply OCR during preprocessing
- **model.max_length**: Maximum sequence length
- **data.processor_name**: Name of the LayoutLMv3 processor to use
- **data.apply_ocr**: Whether to apply OCR during preprocessing
- **data.dataloader.batch_size**: Batch size for training
- **data.dataloader.num_workers**: Number of worker processes for data loading
- **training.test_size**: Proportion of data to use for validation
- **training.device**: Device to use for training ('cuda' or 'cpu')
- **tuning.n_trials**: Number of trials for hyperparameter optimization

@see classification.training.layout.LayoutLMv3Classifier 