import io
import logging
import torch
from torch.utils.data import Dataset
from PIL import Image
from minio import Minio
import pandas as pd

class MinioImageDataset(Dataset):
    """
    PyTorch Dataset for loading images from MinIO storage
    """
    def __init__(self, dataframe, bucket_name, transform=None):
        """
        Constructor for MinioImageDataset class
        
        Args:
            dataframe: DataFrame containing 'image' URLs and 'label' columns
            bucket_name: Name of the MinIO bucket to fetch images from
            transform: Optional transformations to apply to the images
        """
        self.df = dataframe.reset_index(drop=True)
        self.bucket_name = bucket_name
        self.transform = transform
        self.client = Minio(
            "localhost:9900",
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False
        )

    def __len__(self):
        """
        Returns the number of items in the dataset
        
        Returns:
            Number of images in the dataset
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Fetches and processes a single item from the dataset
        
        Args:
            idx: Index of the item to fetch
            
        Returns:
            Tuple of (transformed image tensor, class label tensor)
        """
        row = self.df.iloc[idx]
        image_url = row['image']
        
        # Extract image name from URL (assuming format: http://endpoint/bucket/image_name)
        image_name = image_url.split('/')[-1]

        try:
            response = self.client.get_object(self.bucket_name, image_name)
            image_data = response.read()
            response.close()
            response.release_conn()
        except Exception as e:
            logging.error(f"Error fetching {image_name}: {e}")
            raise e

        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Convert label to int and create tensor
        label = torch.tensor(int(row['label']), dtype=torch.long)

        return image, label 