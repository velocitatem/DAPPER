from PIL import Image
import logging
import torch
from torchvision import transforms
import pandas as pd
from typing import Optional

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
    def __init__(self, width: int = 768, height: int = 992):
        self.width = width
        self.height = height
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

    def resize_image(self, image: Image.Image, augment: bool = False) -> Image.Image:
        resized_image = image.resize((self.width, self.height))
        if augment:
            resized_image = self.augment_image(resized_image)
        return resized_image

    def augment_image(self, image: Image.Image) -> Image.Image:
        image = self.pil_transforms(image)
        image = self.tensor_transforms(image)
        return image

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
