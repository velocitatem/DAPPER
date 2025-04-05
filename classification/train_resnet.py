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
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

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
validation_df = train_df.sample(frac=0.2, random_state=42)

# Determine the number of unique classes in the dataset
num_classes = train_df['label'].apply(lambda x: 11 if x == 'invoice' else int(x)).nunique()

# Create dataset objects for training, validation and testing
train_dataset = MinioImageDataset(train_df, bucket_name='dapper', transform=transformations)
validation_dataset = MinioImageDataset(validation_df, bucket_name='dapper', transform=transformations)
test_dataset = MinioImageDataset(test_df, bucket_name='dapper', transform=transformations)

##
# @brief DataLoader for training data
#
# Creates a DataLoader for efficiently loading training data in batches.
# Enables shuffling, multi-processing, and memory pinning for faster training.
#
train_loader = DataLoader(
    train_dataset,
    batch_size=64,        # Reduced batch size for better generalization
    shuffle=True,
    num_workers=12,
    pin_memory=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=12,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=12,
    pin_memory=True
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

###
# @brief Initialize the model architecture
#
# Uses a pre-trained ResNet18 model and adapts it for document classification
# by replacing the final fully connected layer to match our number of classes.
#
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# Add more dropout and regularization
model.fc = nn.Sequential(
    nn.Dropout(0.7),  # Increased dropout
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)
model = model.to(device)

##
# @brief Loss function and optimizer
#
# Uses CrossEntropyLoss for multi-class classification and Adam optimizer
# with a learning rate of 1e-4 for stable training.
#
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples * 100
    return avg_loss, accuracy, all_preds, all_labels

# Training loop with early stopping
num_epochs = 50
best_validation_accuracy = 0
patience = 5
patience_counter = 0
early_stopping = False

for epoch in range(num_epochs):
    if early_stopping:
        break

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
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

        if (batch_idx + 1) % 10 == 0:
            logging.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    train_loss = train_loss / train_total
    train_accuracy = train_correct / train_total * 100

    # Validation phase
    validation_loss, validation_accuracy, _, _ = evaluate(model, validation_loader, criterion, device)

    # Learning rate scheduling
    scheduler.step(validation_accuracy)

    logging.info(f"Epoch [{epoch+1}/{num_epochs}]")
    logging.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    logging.info(f"Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.2f}%")

    # Early stopping check
    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        patience_counter = 0
        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'validation_accuracy': validation_accuracy,
            'train_accuracy': train_accuracy,
        }, 'best_model.pth')
        logging.info(f"New best model saved with validation accuracy: {validation_accuracy:.2f}%")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            early_stopping = True
            logging.info(f"Early stopping triggered after {epoch+1} epochs")

# Load best model for final evaluation
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
logging.info(f"Loaded best model from epoch {checkpoint['epoch']}")

# Final evaluation on test set
test_loss, test_accuracy, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
logging.info(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Calculate and log confusion matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.close()

# Print detailed classification report
report = classification_report(test_labels, test_preds)
logging.info("\nClassification Report:\n" + report)

logging.info("Training completed successfully.")
logging.info(f"Best validation accuracy: {best_validation_accuracy:.2f}%")
logging.info(f"Final test accuracy: {test_accuracy:.2f}%")
