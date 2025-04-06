##
# @file docling.py
# @package classification.training.docling
# @brief Document classification using SmolDocling vision-language model
#
# This module provides functionality for document classification using the
# SmolDocling vision-language model. It implements a document classifier
# that leverages the pre-trained SmolDocling model for feature extraction.
#
# @author Statistical Learning Team
# @date 2025
#

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForVision2Seq

# Load the processor and model
processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
model = AutoModelForVision2Seq.from_pretrained("ds4sd/SmolDocling-256M-preview")

# Freeze the entire model so that only the classification head will be trained
for param in model.parameters():
    param.requires_grad = False

# Assume the encoder outputs a feature vector of dimension 'hidden_size'
hidden_size = model.config.text_config.hidden_size  # This is an example; actual value may vary

##
# @brief Document classifier using SmolDocling vision-language model
#
# This class implements a document classifier that uses the SmolDocling
# vision-language model as a feature extractor and adds a classification head
# for document type classification.
#
class DocumentClassifier(nn.Module):
    ##
    # @brief Constructor for DocumentClassifier class
    # @param encoder Pre-trained encoder from SmolDocling model
    # @param hidden_size Dimension of the encoder's hidden state
    # @param num_classes Number of document classes to classify
    #
    def __init__(self, encoder, hidden_size, num_classes):
        super().__init__()
        self.encoder = encoder  # Use the encoder portion of SmolDocling
        # A simple linear layer for classification
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    ##
    # @brief Forward pass through the network
    # @param pixel_values Input image tensor
    # @param attention_mask Attention mask for the model
    # @return Classification logits
    #
    def forward(self, pixel_values, attention_mask):
        # Forward pass through the encoder to obtain embeddings.
        # The exact call may need to be adapted based on the model's internals.
        outputs = self.encoder(pixel_values=pixel_values, attention_mask=attention_mask, output_hidden_states=True)
        
        # For demonstration, assume we use the last hidden state of the first token (like CLS token)
        cls_embedding = outputs.hidden_states[-1][:, 0, :]
        
        # Pass the embedding through the classification head
        logits = self.classifier(cls_embedding)
        return logits

# Set number of classes (for example, 4: invoice, technical report, email, receipt)
num_classes = 4
# Assume model.encoder holds the encoder we want to use.
classifier_model = DocumentClassifier(model.encoder, hidden_size, num_classes)

# Example forward pass (assuming you have a batch of images)
# Load and preprocess your image(s)
from transformers.image_utils import load_image
image = load_image("https://upload.wikimedia.org/wikipedia/commons/7/76/GazettedeFrance.jpg")
inputs = processor(images=[image], return_tensors="pt")
pixel_values = inputs["pixel_values"]
attention_mask = inputs["pixel_attention_mask"]

# Get predictions
logits = classifier_model(pixel_values, attention_mask)
predicted_class = logits.argmax(dim=-1)
print("Predicted class:", predicted_class.item())
