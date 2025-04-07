#!/usr/bin/env python3
import os
import json
import argparse
import logging
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3TokenizerFast,
    LayoutLMv3ImageProcessor
)
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class InvoiceExtractor:
    def __init__(
        self,
        model_path: str = "output/best_model",
        label_map: str = "data/label_map.json",
        use_ocr: bool = True,
    ):
        """
        Initialize the invoice extractor
        
        Args:
            model_path: Path to the trained model
            label_map: Path to the label map JSON file
            use_ocr: Whether to use OCR built into the LayoutLMv3 processor
        """
        self.model_path = model_path
        self.use_ocr = use_ocr
        
        # Load label map
        with open(label_map, 'r') as f:
            label_map_dict = json.load(f)
            
        self.id2label = {int(key): value for key, value in label_map_dict.items()}
        self.label2id = {value: int(key) for key, value in label_map_dict.items()}
        self.num_labels = len(self.id2label)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  "mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using {self.device} for inference")
        
        # Load model and processor
        self._load_model()
        
    def _load_model(self):
        """Load the model and processor with OCR support"""
        # Load processor with OCR capability
        logger.info(f"Initializing processor with OCR={self.use_ocr}")
        self.processor = LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base",
            apply_ocr=self.use_ocr
        )
        
        # Load model
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                self.model_path,
                id2label=self.id2label,
                label2id=self.label2id
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Could not load model from {self.model_path}")
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process an invoice image and extract information
        
        Args:
            image_path: Path to the invoice image
            
        Returns:
            Extracted information from the invoice
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Process the image through LayoutLMv3Processor
        # This will handle OCR if use_ocr=True
        logger.info("Processing image with OCR")
        encoding = self.processor(
            image,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        )
        
        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**encoding)
            
        # Get predicted labels
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        token_boxes = encoding.bbox.squeeze().tolist()
        words = self.processor.tokenizer.convert_ids_to_tokens(encoding.input_ids.squeeze().tolist())
        
        # Handle case where predictions is a single value (one token)
        if not isinstance(predictions, list):
            predictions = [predictions]
            token_boxes = [token_boxes]
            words = [words]
        
        # Extract predictions
        results = self._extract_entities(words, predictions, token_boxes, encoding.attention_mask.squeeze().tolist())
        
        return results
    
    def _extract_entities(self, words: List[str], predictions: List[int], 
                         boxes: List[List[int]], attention_mask: List[int]) -> Dict[str, str]:
        """
        Extract structured entities from model predictions
        
        Args:
            words: List of tokens
            predictions: List of predicted label IDs
            boxes: List of token bounding boxes
            attention_mask: Attention mask from the model
            
        Returns:
            Dictionary of extracted entities
        """
        entities = {}
        current_entity = None
        current_text = []
        
        # Process each token
        for word, pred, mask in zip(words, predictions, attention_mask):
            # Skip padding tokens
            if mask == 0 or word in ("[CLS]", "[SEP]", "[PAD]"):
                continue
                
            # Skip special tokens and subword pieces that start with ##
            if word.startswith("##"):
                # For subword pieces, append without the ##
                if current_entity:
                    current_text.append(word[2:])
                continue
            
            # Get the label for this prediction
            label = self.id2label.get(pred, "O")
            
            # If it's a B- tag, start a new entity
            if label.startswith("B-"):
                # If we had a previous entity, add it to the results
                if current_entity:
                    entity_type = current_entity.replace("B-", "").replace("I-", "")
                    entity_text = " ".join(current_text)
                    entities[entity_type] = entity_text
                
                # Start new entity
                current_entity = label
                current_text = [word]
                
            # If it's an I- tag matching the current entity, continue the entity
            elif label.startswith("I-") and current_entity and label[2:] == current_entity[2:]:
                current_text.append(word)
                
            # If O or a different entity, reset
            else:
                if current_entity:
                    entity_type = current_entity.replace("B-", "").replace("I-", "")
                    entity_text = " ".join(current_text)
                    entities[entity_type] = entity_text
                    
                current_entity = None
                current_text = []
        
        # Add the last entity if there is one
        if current_entity:
            entity_type = current_entity.replace("B-", "").replace("I-", "")
            entity_text = " ".join(current_text)
            entities[entity_type] = entity_text
            
        return entities
    
    def visualize(self, image_path: str, output_path: Optional[str] = None) -> None:
        """
        Visualize the extracted information on the image
        
        Args:
            image_path: Path to the invoice image
            output_path: Path to save the visualization
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # Process the image
        encoding = self.processor(
            image,
            return_tensors="pt",
            truncation=True,
            padding="max_length"
        )
        
        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**encoding)
            
        # Get predicted labels
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        token_boxes = encoding.bbox.squeeze().tolist()
        words = self.processor.tokenizer.convert_ids_to_tokens(encoding.input_ids.squeeze().tolist())
        
        # Handle case where predictions is a single value (one token)
        if not isinstance(predictions, list):
            predictions = [predictions]
            token_boxes = [token_boxes]
            words = [words]
        
        # Setup figure
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(image_np)
        
        # Define some colors for visualization
        colors = {
            'invoice_number': 'red',
            'date': 'blue',
            'due_date': 'green',
            'total': 'purple',
            'vendor': 'orange',
            'customer': 'cyan',
            'tax': 'magenta',
            'subtotal': 'yellow',
            'currency': 'lime',
            'O': 'white'
        }
        
        # Default colors for other labels
        default_colors = ['brown', 'olive', 'teal', 'navy', 'maroon', 'pink']
        color_idx = 0
        
        # Process each token
        height, width, _ = image_np.shape
        drawn_boxes = set()  # To avoid drawing the same box multiple times
        
        for word, pred, box, mask in zip(words, predictions, token_boxes, encoding.attention_mask.squeeze().tolist()):
            # Skip padding and special tokens
            if mask == 0 or word in ("[CLS]", "[SEP]", "[PAD]"):
                continue
                
            # Get the label for this prediction
            label = self.id2label.get(pred, "O")
            if label == "O" or word.startswith("##"):
                continue
            
            # Extract the entity type from the label
            entity_type = label.replace("B-", "").replace("I-", "")
            
            # Normalize box coordinates to image dimensions
            x1, y1, x2, y2 = box
            x1 = (x1 / 1000) * width
            y1 = (y1 / 1000) * height
            x2 = (x2 / 1000) * width
            y2 = (y2 / 1000) * height
            
            # Create a unique identifier for this box
            box_id = f"{x1}-{y1}-{x2}-{y2}"
            if box_id in drawn_boxes:
                continue
            drawn_boxes.add(box_id)
            
            # Get color for this entity type
            color = colors.get(entity_type)
            if color is None:
                color = default_colors[color_idx % len(default_colors)]
                colors[entity_type] = color
                color_idx += 1
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, 
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add text label above the box
            plt.text(
                x1, y1 - 5, 
                f"{entity_type}: {word}", 
                color=color, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8)
            )
        
        # Add a legend
        legend_elements = [
            patches.Patch(color=color, label=entity)
            for entity, color in colors.items()
            if entity != 'O'
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Remove axes
        plt.axis('off')
        
        # Extracted information
        entities = self._extract_entities(
            words, 
            predictions, 
            token_boxes, 
            encoding.attention_mask.squeeze().tolist()
        )
        
        # Print extracted information
        text = "Extracted Information:\n"
        for entity, value in entities.items():
            text += f"{entity}: {value}\n"
        
        plt.figtext(0.5, 0.01, text, ha='center', fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        # Save or show
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_path}")
        else:
            plt.tight_layout()
            plt.show()
            
        plt.close()
        
        return entities

def main():
    parser = argparse.ArgumentParser(description='Extract information from invoices')
    parser.add_argument('--model_path', type=str, default='output/best_model', help='Path to the trained model')
    parser.add_argument('--label_map', type=str, default='data/label_map.json', help='Path to the label map JSON file')
    parser.add_argument('--input', type=str, required=True, help='Path to the input invoice image')
    parser.add_argument('--output', type=str, help='Path to save the extraction results')
    parser.add_argument('--visualize', action='store_true', help='Create a visualization of extraction results')
    parser.add_argument('--visualization_path', type=str, help='Path to save the visualization')
    parser.add_argument('--use_ocr', action='store_true', help='Use built-in OCR from LayoutLMv3')
    args = parser.parse_args()
    
    # Create invoice extractor
    extractor = InvoiceExtractor(
        model_path=args.model_path,
        label_map=args.label_map,
        use_ocr=args.use_ocr
    )
    
    # Process image
    results = extractor.process_image(args.input)
    
    # Print results
    logger.info("Extracted Information:")
    for entity, value in results.items():
        logger.info(f"{entity}: {value}")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    
    # Visualize
    if args.visualize:
        extractor.visualize(args.input, args.visualization_path)

if __name__ == "__main__":
    main() 