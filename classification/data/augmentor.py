##
# @file augmentor.py
# @package classification.data.augmentor
# @brief Image augmentation utility for document classification
#
# This module provides functionality for augmenting document images to improve
# model training by creating variations of existing images. It supports various
# augmentation techniques including geometric transformations, color adjustments,
# and noise addition.
#
# @author Statistical Learning Team
# @date 2025-03
#

from PIL import Image
import logging
import torch
from torchvision import transforms
import pandas as pd
from typing import Optional, Dict, List, Any, Union
import pytesseract
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
from tqdm import tqdm
##
# @brief Image augmentation class for document classification
#
# This class provides methods to apply various image augmentation techniques
# to document images. It supports geometric transformations, color adjustments,
# and noise addition to create variations of existing images for improved model training.
#
class Augmentor:
    """
    This we can do to messa up image
    Patching
    Diffusion
    Color
    Rotation
    Jigsaw
    Resolution
    """
    ##
    # @brief Constructor for Augmentor class
    # @param width Target width for resized images
    # @param height Target height for resized images
    # @param tesseract_cmd Path to tesseract executable (if not in PATH)
    #
    def __init__(self, width: int = 768, height: int = 992, tesseract_cmd: Optional[str] = None):
        self.width = width
        self.height = height
        
        # Set tesseract command path if provided
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            
        self.pil_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])
        self.tensor_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2),
            transforms.ToPILImage()
        ])

    ##
    # @brief Resizes an image to the target dimensions
    # @param image PIL Image to resize
    # @param augment Boolean flag to apply augmentation after resizing
    # @return Resized (and optionally augmented) PIL Image
    #
    def resize_image(self, image: Image.Image, augment: bool = False) -> Image.Image:
        resized_image = image.resize((self.width, self.height))
        if augment:
            resized_image = self.augment_image(resized_image)
        return resized_image

    ##
    # @brief Applies augmentation transformations to an image
    # @param image PIL Image to augment
    # @return Augmented PIL Image
    #
    def augment_image(self, image: Image.Image) -> Image.Image:
        image = self.pil_transforms(image)
        image = self.tensor_transforms(image)
        return image

    ##
    # @brief Creates augmented versions of images in a DataFrame
    # @param df DataFrame containing image data
    # @param factor Number of augmented versions to create per image
    # @return DataFrame containing original and augmented images
    #
    def create_augmented_rows(self, df: pd.DataFrame, factor: int = 2) -> pd.DataFrame:
        logging.info(f"Creating {factor} augmentations per image for {len(df)} images")
        augmented_rows = []
        for i, row in df.iterrows():
            image = row['image']
            label = row['label']
            if not isinstance(image, Image.Image):
                continue
            for j in range(factor):
                aug_row = row.copy()
                aug_row['image'] = self.augment_image(image.copy())
                aug_row['is_augmented'] = True
                aug_row['source_dataset'] = f"{row.get('source_dataset', 'unknown')}_aug{j+1}"
                augmented_rows.append(aug_row)
        return pd.DataFrame(augmented_rows)
        
    ##
    # @brief Performs OCR on a single image
    # @param image PIL Image to process with OCR
    # @param config Optional configuration for tesseract OCR
    # @return Dictionary containing OCR results
    #
    def perform_ocr(self, image: Image.Image, config: Optional[str] = None) -> Dict[str, Any]:
        """
        Performs OCR on a single image and returns the extracted text and metadata.
        
        Args:
            image: PIL Image to process with OCR
            config: Optional configuration for tesseract OCR
            
        Returns:
            Dictionary containing OCR results including:
            - text: Extracted text
            - confidence: Confidence scores
            - boxes: Bounding boxes for detected text regions
        """
        try:
            # Convert image to grayscale for better OCR results
            if image.mode != 'L':
                image = image.convert('L')
            # Perform OCR with pytesseract
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Extract text and confidence
            text = ' '.join([word for word in ocr_data['text'] if word.strip()])
            confidences = [conf for conf in ocr_data['conf'] if conf != -1]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            # Extract bounding boxes
            boxes = []
            for i in range(len(ocr_data['text'])):
                if ocr_data['text'][i].strip():
                    box = {
                        'text': ocr_data['text'][i],
                        'conf': ocr_data['conf'][i],
                        'left': ocr_data['left'][i],
                        'top': ocr_data['top'][i],
                        'width': ocr_data['width'][i],
                        'height': ocr_data['height'][i]
                    }
                    boxes.append(box)
            
            return {
                'text': text,
                'confidence': avg_confidence,
                'boxes': boxes
            }
        except Exception as e:
            logging.error(f"OCR processing failed: {str(e)}")
            return {
                'text': '',
                'confidence': 0,
                'boxes': []
            }
    
    ##
    # @brief Process a DataFrame with OCR in parallel
    # @param df DataFrame containing image data
    # @param max_workers Number of parallel workers for OCR processing
    # @param config Optional configuration for tesseract OCR
    # @return DataFrame with added OCR results
    #
    def process_ocr(self, df: pd.DataFrame, max_workers: int = 4, config: Optional[str] = None) -> pd.DataFrame:
        """
        Process all images in a DataFrame with OCR in parallel.
        
        Args:
            df: DataFrame containing image data
            max_workers: Number of parallel workers for OCR processing
            config: Optional configuration for tesseract OCR
            
        Returns:
            DataFrame with added OCR results
        """
        logging.info(f"Processing OCR for {len(df)} images with {max_workers} workers")
        
        # Create a copy of the DataFrame to avoid modifying the original
        result_df = df.copy()
        
        # Initialize OCR results columns if they don't exist
        if 'ocr_text' not in result_df.columns:
            result_df['ocr_text'] = ''
        if 'ocr_confidence' not in result_df.columns:
            result_df['ocr_confidence'] = 0.0
        if 'ocr_boxes' not in result_df.columns:
            result_df['ocr_boxes'] = None
            
        # Function to process a single row
        def process_row(row):
            if not isinstance(row['image'], Image.Image):
                return row
                
            ocr_result = self.perform_ocr(row['image'], config)
            row['ocr_text'] = ocr_result['text']
            row['ocr_confidence'] = ocr_result['confidence']
            row['ocr_boxes'] = ocr_result['boxes']
            return row
            
        # Process rows in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            processed_rows = list(tqdm(
                executor.map(process_row, [row for _, row in result_df.iterrows()]),
                total=len(result_df),
                desc="Processing OCR"
            ))
            
        # Update the DataFrame with processed rows
        result_df = pd.DataFrame(processed_rows)
        
        logging.info(f"OCR processing completed for {len(result_df)} images")
        return result_df
