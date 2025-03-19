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

# Custom Dataset class for MinIO
class MinioImageDataset(Dataset):
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

    def __len__(self):
        return len(self.df)

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

        change = lambda x: 11 if x == 'invoice' else x
        label = torch.tensor(int(change(row['label'])), dtype=torch.long)

        return image, label

# Transformations
transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
num_classes = train_df['label'].apply(lambda x: 11 if x == 'invoice' else int(x)).nunique()

# Dataset and DataLoader
train_dataset = MinioImageDataset(train_df, bucket_name='dapper', transform=transformations)
test_dataset = MinioImageDataset(test_df, bucket_name='dapper', transform=transformations)

train_loader = DataLoader(
    train_dataset,
    batch_size=100,
    shuffle=True,
    num_workers=12,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=100,
    shuffle=False,
    num_workers=12,
    pin_memory=True
)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

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
    
    logging.info(f"Epoch [{epoch+1}/{num_epochs}] completed.")
    logging.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    # Save best model
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        torch.save(model.state_dict(), 'model.pth')
        logging.info(f"Model saved with test accuracy: {test_accuracy:.2f}%")

logging.info("Training completed successfully.")
logging.info(f"Best test accuracy: {best_test_accuracy:.2f}%")
