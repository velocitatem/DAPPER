##
# @file train.py
# @brief Document classification model training script
#
# This script trains a document classification model using ResNet18 on document image data
# stored in MinIO. It loads data from CSV files, processes images, and trains a deep learning
# model to classify documents into predefined categories.
#
# @author Statistical Learning Team
# @date 2025-03-20
#

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from minio import Minio
from PIL import Image
import pandas as pd
import io
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

##
# @brief Custom Dataset class for loading images from MinIO storage
#
# This class extends PyTorch's Dataset class to load images from MinIO object storage.
# It takes a DataFrame with image URLs, connects to MinIO, fetches images by their names,
# applies transformations, and returns image-label pairs for model training.
#
class MinioImageDataset(Dataset):
    ##
    # @brief Constructor for MinioImageDataset class
    # @param dataframe DataFrame containing 'image' URLs and 'label' columns
    # @param bucket_name Name of the MinIO bucket to fetch images from
    # @param transform Optional transformations to apply to the images
    #
    def __init__(self, dataframe, bucket_name, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.bucket_name = bucket_name
        self.transform = transform
        self.client = Minio(
            "localhost:9900",
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False
        )

    ##
    # @brief Returns the number of items in the dataset
    # @return Number of images in the dataset
    #
    def __len__(self):
        return len(self.df)

    ##
    # @brief Fetches and processes a single item from the dataset
    # @param idx Index of the item to fetch
    # @return Tuple of (transformed image tensor, class label tensor)
    #
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_url = row['image']
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

        # Convert 'invoice' string labels to integer class 11
        change = lambda x: 11 if x == 'invoice' else x
        label = torch.tensor(int(change(row['label'])), dtype=torch.long)

        return image, label

##
# @brief Image transformations for the model
#
# Defines the sequence of transformations to apply to input images:
# 1. Resize to 224x224 pixels (standard input size for ResNet)
# 2. Convert to PyTorch tensor
# 3. Normalize using ImageNet mean and standard deviation values
#
transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                         std=[0.229, 0.224, 0.225])   # ImageNet stds
])

##
# @brief Data loading and preparation
#
# Loads train and test datasets from CSV files, creates dataset objects,
# and prepares DataLoaders for efficient batch processing during training.
#

# Load data from CSV files
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Determine the number of unique classes in the dataset
num_classes = train_df['label'].apply(lambda x: 11 if x == 'invoice' else int(x)).nunique()

# Create dataset objects for training and testing
train_dataset = MinioImageDataset(train_df, bucket_name='dapper', transform=transformations)
test_dataset = MinioImageDataset(test_df, bucket_name='dapper', transform=transformations)

##
# @brief DataLoader for training data
#
# Creates a DataLoader for efficiently loading training data in batches.
# Enables shuffling, multi-processing, and memory pinning for faster training.
#
train_loader = DataLoader(
    train_dataset,
    batch_size=100,        # Number of samples per batch
    shuffle=True,          # Shuffle data for each epoch
    num_workers=12,        # Number of subprocesses for data loading
    pin_memory=True        # Pin memory for faster data transfer to GPU
)

##
# @brief DataLoader for test data
#
# Creates a DataLoader for efficiently loading test data in batches.
# Uses same batch size as training loader but without shuffling.
#
test_loader = DataLoader(
    test_dataset,
    batch_size=100,        # Number of samples per batch
    shuffle=False,         # No need to shuffle test data
    num_workers=12,        # Number of subprocesses for data loading
    pin_memory=True        # Pin memory for faster data transfer to GPU
)

##
# @brief Model setup and initialization
#
# Sets up the device (GPU if available, otherwise CPU), initializes the ResNet18 model
# with pre-trained weights, modifies the final fully connected layer for our classification task,
# and defines loss function and optimizer.
#

# Determine device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

##
# @brief Initialize the model architecture
#
# Uses a pre-trained ResNet18 model and adapts it for document classification
# by replacing the final fully connected layer to match our number of classes.
#
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Load pre-trained weights
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace final layer
model = model.to(device)  # Move model to GPU if available

##
# @brief Loss function and optimizer
#
# Uses CrossEntropyLoss for multi-class classification and Adam optimizer
# with a learning rate of 1e-4 for stable training.
#
criterion = nn.CrossEntropyLoss()  # Standard loss for classification tasks
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adam optimizer with learning rate 1e-4

# Training and evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples * 100
    return avg_loss, accuracy

# Training loop
num_epochs = 5
best_test_accuracy = 0

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

        if (batch_idx + 1) % 10 == 0:
            logging.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    train_loss = train_loss / train_total
    train_accuracy = train_correct / train_total * 100

    # Evaluation phase
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)


    logging.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Save best model
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        torch.save(model.state_dict(), 'model.pth')
        logging.info(f"Model saved with test accuracy: {test_accuracy:.2f}%")

logging.info("Training completed successfully.")
logging.info(f"Best test accuracy: {best_test_accuracy:.2f}%")
