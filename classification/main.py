import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from classification.data.loader import DataManager
from classification.training.models import get_model
from classification.utils.logger import Logger
from classification.utils.seed import set_global_seed

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train(config):
    """
    Train a model using the provided configuration
    
    Args:
        config: Dictionary containing configuration
    """
    # Set random seed
    set_global_seed(config['data']['seed'])
    
    # Create logger
    logger = Logger(
        log_dir=config['logging']['log_dir'],
        experiment_name=config['logging']['experiment_name'],
        config=config
    )
    
    # Set device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    logger.logger.info(f"Using device: {device}")
    
    
    # Create data manager and load data
    data_manager = DataManager(
        base_data_dir=config['data']['base_data_dir'],
        cache_dir=config['data']['cache_dir'],
        seed=config['data']['seed'],
        use_cache=config['data']['use_cache'],
        val_split=config['data']['val_split']
    )
    
    # Create train and validation dataloaders
    train_loader, val_loader = data_manager.create_train_val_dataloaders(
        dataset=config['data']['datasets'],
        batch_size=config['data']['dataloader']['batch_size'],
        num_workers=config['data']['dataloader']['num_workers'],
        pin_memory=config['data']['dataloader']['pin_memory']
    )
    
    logger.logger.info(f"Training dataset size: {len(train_loader.dataset)}")
    logger.logger.info(f"Validation dataset size: {len(val_loader.dataset)}")
    
    # Create model
    model = get_model(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        freeze_backbone=config['model']['freeze_backbone']
    )
    model.to(device)
    
    # Log model architecture
    if config['logging']['tensorboard']['log_model_graph']:
        logger.log_model_graph(model)
    
    # Define loss function
    if config['training']['loss']['name'] == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {config['training']['loss']['name']}")
    
    # Define optimizer
    if config['training']['optimizer']['name'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['optimizer']['learning_rate'],
            weight_decay=config['training']['optimizer']['weight_decay']
        )
    elif config['training']['optimizer']['name'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['optimizer']['learning_rate'],
            momentum=0.9,
            weight_decay=config['training']['optimizer']['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['training']['optimizer']['name']}")
    
    # Define learning rate scheduler
    if config['training']['scheduler']['name'] == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['training']['scheduler']['factor'],
            patience=config['training']['scheduler']['patience'],
            min_lr=config['training']['scheduler']['min_lr']
        )
    elif config['training']['scheduler']['name'] == 'cosine_annealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs'],
            eta_min=config['training']['scheduler']['min_lr']
        )
    elif config['training']['scheduler']['name'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=config['training']['scheduler']['factor']
        )
    else:
        raise ValueError(f"Unsupported scheduler: {config['training']['scheduler']['name']}")
    
    # Training loop
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    for epoch in range(config['training']['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct / total
        
        # Log training metrics
        logger.log_metrics(
            metrics={'loss': train_loss, 'accuracy': train_acc},
            step=epoch,
            prefix='train/'
        )
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        
        # Log validation metrics
        logger.log_metrics(
            metrics={'loss': val_loss, 'accuracy': val_acc},
            step=epoch,
            prefix='val/'
        )
        
        # Update learning rate scheduler
        if config['training']['scheduler']['name'] == 'reduce_on_plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_dir = os.path.join(logger.log_dir, 'models')
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pth'))
            logger.logger.info(f"Saved best model with validation loss: {val_loss:.6f}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        # Early stopping
        if (config['training']['early_stopping']['enabled'] and 
            early_stopping_counter >= config['training']['early_stopping']['patience']):
            logger.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Save final model
    model_dir = os.path.join(logger.log_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, 'final_model.pth'))
    
    # Close logger
    logger.close()
    
    return model

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Classification project')
    parser.add_argument('--config', type=str, default='classification/config/settings.yaml', 
                        help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Train model
    train(config)

if __name__ == '__main__':
    main() 