import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm
import os
import time
# Assuming a logger utility exists, similar to cnn.py context
# import logging
# logger = logging.getLogger(__name__)

class WordAttention(nn.Module):
    """Word-level Attention layer."""
    def __init__(self, hidden_dim):
        super(WordAttention, self).__init__()
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, gru_output):
        # gru_output shape: (batch_size, seq_len, hidden_dim)
        attn_weights = torch.tanh(self.attention(gru_output))
        # attn_weights shape: (batch_size, seq_len, hidden_dim)
        attn_scores = self.context_vector(attn_weights).squeeze(-1)
        # attn_scores shape: (batch_size, seq_len)
        alpha = F.softmax(attn_scores, dim=-1)
        # alpha shape: (batch_size, seq_len)
        context = torch.bmm(alpha.unsqueeze(1), gru_output).squeeze(1)
        # context shape: (batch_size, hidden_dim)
        return context, alpha

class SentenceAttention(nn.Module):
    """Sentence-level Attention layer."""
    def __init__(self, hidden_dim):
        super(SentenceAttention, self).__init__()
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, gru_output):
        # gru_output shape: (batch_size, num_sentences, hidden_dim)
        attn_weights = torch.tanh(self.attention(gru_output))
        # attn_weights shape: (batch_size, num_sentences, hidden_dim)
        attn_scores = self.context_vector(attn_weights).squeeze(-1)
        # attn_scores shape: (batch_size, num_sentences)
        alpha = F.softmax(attn_scores, dim=-1)
        # alpha shape: (batch_size, num_sentences)
        context = torch.bmm(alpha.unsqueeze(1), gru_output).squeeze(1)
        # context shape: (batch_size, hidden_dim)
        return context, alpha


class ImageEncoder(nn.Module):
    """Simple CNN based image encoder."""
    def __init__(self, input_channels, output_dim):
        super(ImageEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Adaptive pooling allows handling variable input sizes to some extent, outputs fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, output_dim) # Adjust 128 based on final conv channels

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = F.relu(self.fc(x))
        # x shape: (batch_size, output_dim)
        return x


class EAML(nn.Module):
    """
    Enhanced Multi-Level Attention Network (EAML) for Document Classification
    with Text and Image modalities.
    Assumes text input is preprocessed into sentences and words, and embedded.
    Assumes image input is preprocessed to a fixed size.
    """
    def __init__(self, vocab_size, embedding_dim, word_hidden_dim, sent_hidden_dim,
                 image_channels, image_feature_dim, num_classes, dropout=0.5):
        super(EAML, self).__init__()
        self.embedding_dim = embedding_dim
        self.word_hidden_dim = word_hidden_dim
        self.sent_hidden_dim = sent_hidden_dim
        self.image_feature_dim = image_feature_dim

        # --- Text Branch ---
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Word level
        self.word_gru = nn.GRU(embedding_dim, word_hidden_dim, bidirectional=True, batch_first=True)
        self.word_attention = WordAttention(word_hidden_dim * 2) # *2 for bidirectional
        # Sentence level
        self.sent_gru = nn.GRU(word_hidden_dim * 2, sent_hidden_dim, bidirectional=True, batch_first=True)
        self.sent_attention = SentenceAttention(sent_hidden_dim * 2) # *2 for bidirectional

        # --- Image Branch ---
        self.image_encoder = ImageEncoder(image_channels, image_feature_dim)

        # --- Fusion and Classifier ---
        self.fc = nn.Linear(sent_hidden_dim * 2 + image_feature_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, docs, images):
        # --- Text Processing ---
        # Input `docs` shape: (batch_size, num_sentences, max_sent_length)
        batch_size, num_sentences, max_sent_length = docs.size()
        # Reshape for word-level processing
        word_input = docs.view(batch_size * num_sentences, max_sent_length)
        embedded_words = self.dropout(self.embedding(word_input))
        # Word GRU
        word_gru_out, _ = self.word_gru(embedded_words)
        # Word Attention
        sent_vectors, word_alpha = self.word_attention(word_gru_out)
        sent_vectors = self.dropout(sent_vectors)
        # Reshape back for sentence-level processing
        sent_input = sent_vectors.view(batch_size, num_sentences, self.word_hidden_dim * 2)
        # Sentence GRU
        sent_gru_out, _ = self.sent_gru(sent_input)
        # Sentence Attention
        doc_vector, sent_alpha = self.sent_attention(sent_gru_out)
        # doc_vector shape: (batch_size, sent_hidden_dim * 2)

        # --- Image Processing ---
        # Input `images` shape: (batch_size, channels, height, width)
        image_features = self.image_encoder(images)
        # image_features shape: (batch_size, image_feature_dim)
        image_features = self.dropout(image_features)

        # --- Feature Fusion ---
        combined_features = torch.cat((doc_vector, image_features), dim=1)
        # combined_features shape: (batch_size, sent_hidden_dim * 2 + image_feature_dim)
        combined_features = self.dropout(combined_features) # Apply dropout after concatenation

        # --- Classifier ---
        output = self.fc(combined_features)
        # output shape: (batch_size, num_classes)

        return output #, word_alpha, sent_alpha # Optionally return attention weights

class EAMLClassifier:
    """
    Wrapper class for training and using the EAML model.
    Mirrors the structure of CNNClassifier.
    """
    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        embedding_dim: int = 100,
        word_hidden_dim: int = 50,
        sent_hidden_dim: int = 50,
        image_channels: int = 3,
        image_feature_dim: int = 128,
        image_size: Tuple[int, int] = (224, 224),
        dropout: float = 0.5,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
        num_epochs: int = 10
    ):
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.word_hidden_dim = word_hidden_dim
        self.sent_hidden_dim = sent_hidden_dim
        self.image_channels = image_channels
        self.image_feature_dim = image_feature_dim
        self.image_size = image_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = EAML(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            word_hidden_dim=self.word_hidden_dim,
            sent_hidden_dim=self.sent_hidden_dim,
            image_channels=self.image_channels,
            image_feature_dim=self.image_feature_dim,
            num_classes=self.num_classes,
            dropout=self.dropout
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # Define image transformations (reuse from CNNClassifier)
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocesses a single image."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if self.image_channels == 1:
            image = image.convert('L')
        else:
            image = image.convert('RGB')

        return self.transform(image)

    def train_model(
        self,
        train_loader,
        val_loader=None,
        tb_logger=None,
        save_dir="models",
        patience=5,
        **kwargs
    ):
        """Train the EAML model."""
        scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == 'cuda')
        best_val_accuracy = 0.0
        patience_counter = 0
        early_stopping = False

        self.model = self.model.to(self.device)

        for epoch in range(self.num_epochs):
            if early_stopping:
                print("Early stopping.")
                break

            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for batch_idx, batch_data in enumerate(pbar):
                # Assuming loader yields dict or tuple: (docs, images, labels)
                if isinstance(batch_data, dict):
                    docs = batch_data['text'].to(self.device, non_blocking=True)
                    images = batch_data['image'].to(self.device, non_blocking=True)
                    labels = batch_data['label'].to(self.device, non_blocking=True)
                elif isinstance(batch_data, (list, tuple)):
                    docs, images, labels = batch_data
                    docs = docs.to(self.device, non_blocking=True)
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                else:
                     raise TypeError("Unsupported batch data type from DataLoader")

                self.optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                    outputs = self.model(docs, images)
                    loss = self.criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()

                running_loss += loss.item() * docs.size(0) # Use item() and scale by batch size
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                pbar.set_postfix({
                    'loss': running_loss / total if total > 0 else 0,
                    'acc': 100. * correct / total if total > 0 else 0
                })

                # RE-ADD BATCH LOGGING
                if tb_logger:
                    batch_acc = (predicted == labels).sum().item() / labels.size(0)
                    tb_logger.log_metrics({
                        'train/batch_loss': loss.item(),
                        'train/batch_accuracy': 100. * batch_acc,
                    }, step=epoch * len(train_loader) + batch_idx)

            epoch_loss = running_loss / total if total > 0 else 0
            epoch_accuracy = 100. * (correct / total if total > 0 else 0) # Use percentage

            print(f"Epoch {epoch+1} Summary: Loss: {epoch_loss:.4f}, Acc: {epoch_accuracy:.2f}%") # Print percentage

            # Log epoch training metrics
            if tb_logger:
                tb_logger.log_metrics({
                    'train/epoch_loss': epoch_loss,
                    'train/epoch_accuracy': epoch_accuracy,
                    # 'train/learning_rate': self.optimizer.param_groups[0]['lr'] # LR logging can be verbose, optional
                }, step=epoch+1) # Use epoch+1 for step

            val_accuracy = None
            val_loss = None # Initialize val_loss
            if val_loader is not None:
                val_accuracy, val_loss = self.evaluate(val_loader, tb_logger, epoch + 1) # Pass epoch+1
                print(f"Validation: Loss: {val_loss:.4f}, Acc: {val_accuracy:.2f}%") # Print percentage

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                    model_path = os.path.join(save_dir, "eaml_best.pth")
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    self.save(model_path, epoch=epoch, val_accuracy=val_accuracy, train_accuracy=epoch_accuracy, scaler=scaler)
                    print(f"Saved best model to {model_path} with accuracy {best_val_accuracy:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        early_stopping = True
                        print(f"Early stopping triggered after {epoch+1} epochs")

        print(f"Training finished. Best Validation Accuracy: {best_val_accuracy:.4f}")
        return best_val_accuracy if val_loader is not None else epoch_accuracy

    def evaluate(self, data_loader, tb_logger=None, step=None):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(data_loader, desc="Evaluation")
            for batch_data in pbar:
                 if isinstance(batch_data, dict):
                    docs = batch_data['text'].to(self.device, non_blocking=True)
                    images = batch_data['image'].to(self.device, non_blocking=True)
                    labels = batch_data['label'].to(self.device, non_blocking=True)
                 elif isinstance(batch_data, (list, tuple)):
                    docs, images, labels = batch_data
                    docs = docs.to(self.device, non_blocking=True)
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                 else:
                     raise TypeError("Unsupported batch data type from DataLoader")

                 with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                    outputs = self.model(docs, images)
                    loss = self.criterion(outputs, labels)

                 total_loss += loss.item() * docs.size(0)
                 _, predicted = torch.max(outputs.data, 1)
                 total += labels.size(0)
                 correct += (predicted == labels).sum().item()

                 all_preds.extend(predicted.cpu().numpy())
                 all_labels.extend(labels.cpu().numpy())

                 pbar.set_postfix({
                     'loss': total_loss / total if total > 0 else 0,
                     'acc': 100. * correct / total if total > 0 else 0 # Use percentage
                 })

        avg_loss = total_loss / total if total > 0 else 0
        accuracy = 100. * (correct / total if total > 0 else 0) # Use percentage

        # Log validation epoch metrics (using step passed from train_model)
        if tb_logger and step is not None:
            tb_logger.log_metrics({
                'val/loss': avg_loss,
                'val/accuracy': accuracy
            }, step=step)
            # Add confusion matrix logging here if needed using all_preds, all_labels

        return accuracy, avg_loss # Return percentage accuracy

    def inference(self, docs: torch.Tensor, image: Union[Image.Image, np.ndarray]) -> int:
        """Run inference on a single text/image pair."""
        self.model.eval()
        image_tensor = self.preprocess_image(image).unsqueeze(0).to(self.device)
        docs_tensor = docs.unsqueeze(0).to(self.device) # Assuming docs is already a numerical tensor

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
            outputs = self.model(docs_tensor, image_tensor)
            _, predicted = torch.max(outputs.data, 1)

        return predicted.item()

    def predict_proba(self, docs: torch.Tensor, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Get class probabilities for a single text/image pair."""
        self.model.eval()
        image_tensor = self.preprocess_image(image).unsqueeze(0).to(self.device)
        docs_tensor = docs.unsqueeze(0).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
            outputs = self.model(docs_tensor, image_tensor)
            probabilities = F.softmax(outputs, dim=1)

        return probabilities.cpu().numpy()[0]

    def save(self, path: str, epoch: Optional[int] = None, val_accuracy: Optional[float] = None, train_accuracy: Optional[float] = None, scaler=None) -> None:
        """Save the trained model and relevant info."""
        save_obj = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'num_classes': self.num_classes,
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'word_hidden_dim': self.word_hidden_dim,
                'sent_hidden_dim': self.sent_hidden_dim,
                'image_channels': self.image_channels,
                'image_feature_dim': self.image_feature_dim,
                'image_size': self.image_size,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs # Save the planned epochs
            },
            'epoch': epoch,
            'val_accuracy': val_accuracy,
            'train_accuracy': train_accuracy
        }
        if scaler:
           save_obj['scaler_state_dict'] = scaler.state_dict()

        torch.save(save_obj, path)

    def load(self, path: str, map_location=None) -> None:
        """Load a trained model and optimizer state."""
        if map_location is None:
            map_location = self.device

        checkpoint = torch.load(path, map_location=map_location)

        # Load config first to re-initialize model correctly
        config = checkpoint['config']
        self.num_classes = config['num_classes']
        self.vocab_size = config['vocab_size']
        self.embedding_dim = config['embedding_dim']
        self.word_hidden_dim = config['word_hidden_dim']
        self.sent_hidden_dim = config['sent_hidden_dim']
        self.image_channels = config['image_channels']
        self.image_feature_dim = config['image_feature_dim']
        self.image_size = config.get('image_size', (224, 224)) # Handle older saves
        self.dropout = config['dropout']
        self.learning_rate = config['learning_rate']
        # self.num_epochs = config['num_epochs'] # Usually not needed on load, but available

        # Re-initialize model and optimizer with loaded config before loading state dicts
        self.model = EAML(
            vocab_size=self.vocab_size, embedding_dim=self.embedding_dim,
            word_hidden_dim=self.word_hidden_dim, sent_hidden_dim=self.sent_hidden_dim,
            image_channels=self.image_channels, image_feature_dim=self.image_feature_dim,
            num_classes=self.num_classes, dropout=self.dropout
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Optionally load scaler state if saved and using cuda
        # if 'scaler_state_dict' in checkpoint and self.device.type == 'cuda':
        #     scaler = torch.cuda.amp.GradScaler()
        #     scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Loaded model from {path}. Epoch: {checkpoint.get('epoch')}, Val Acc: {checkpoint.get('val_accuracy')}")

# Example Usage (replace with actual data loading and preprocessing)
# if __name__ == '__main__':
#     # Hyperparameters (example values)
#     VOCAB_SIZE = 10000
#     NUM_CLASSES = 5
#
#     # Create classifier instance
#     classifier = EAMLClassifier(num_classes=NUM_CLASSES, vocab_size=VOCAB_SIZE)
#     print(classifier.model)
#
#     # --- Dummy Data Loading --- (Replace with actual DataLoader)
#     from torch.utils.data import Dataset, DataLoader
#     class DummyDataset(Dataset):
#         def __init__(self, num_samples=100, vocab_size=10000, num_sentences=10, max_sent_length=50, img_size=(224,224), channels=3):
#             self.num_samples = num_samples
#             self.vocab_size = vocab_size
#             self.num_sentences = num_sentences
#             self.max_sent_length = max_sent_length
#             self.img_size = img_size
#             self.channels = channels
#
#         def __len__(self):
#             return self.num_samples
#
#         def __getitem__(self, idx):
#             docs = torch.randint(0, self.vocab_size, (self.num_sentences, self.max_sent_length), dtype=torch.long)
#             images = torch.randn(self.channels, self.img_size[0], self.img_size[1])
#             labels = torch.randint(0, NUM_CLASSES, (1,), dtype=torch.long).squeeze()
#             return docs, images, labels
#
#     train_dataset = DummyDataset(num_samples=640)
#     val_dataset = DummyDataset(num_samples=128)
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
#     # --- End Dummy Data Loading ---
#
#     # Mock TensorBoard logger
#     class MockTBLogger:
#         def log_metrics(self, metrics, step):
#             # print(f"Step {step}: {metrics}")
#             pass
#     tb_logger = MockTBLogger()
#
#     # Train the model
#     # classifier.train_model(train_loader, val_loader, tb_logger)
#
#     # Example inference (using first item from val_dataset)
#     # docs_sample, image_sample, label_sample = val_dataset[0]
#     # Need to convert image tensor back to PIL/np for preprocess_image
#     # image_pil = transforms.ToPILImage()(image_sample) # Requires handling normalization if applied in dataset
#     # pred_label = classifier.inference(docs_sample, image_pil)
#     # print(f"Predicted label: {pred_label}, Actual label: {label_sample.item()}")
#
#     # Example save/load
#     # classifier.save("models/eaml_test.pth")
#     # new_classifier = EAMLClassifier(num_classes=NUM_CLASSES, vocab_size=VOCAB_SIZE)
#     # new_classifier.load("models/eaml_test.pth")
#     # print("Loaded model successfully.") 