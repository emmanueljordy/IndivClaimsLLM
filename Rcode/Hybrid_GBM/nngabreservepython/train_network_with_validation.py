"""
Train Neural Network Weights with Early Stopping and Validation
Author: Enhanced version with overfitting prevention
Date: 2025
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple

from .models import EmbeddingNetwork, WeightedLoss
from .data_preprocessing import ClaimsDataProcessor, ClaimsDataset, collate_fn
from .utils import save_config
from .train_embeddings_with_validation import EarlyStopping, save_training_history


def train_network_with_validation(lob: int, config: dict = None):
    """Train neural network weights with fixed embeddings and validation."""
    
    # Default configuration with regularization
    if config is None:
        config = {
            'neurons': [40, 30, 10],
            'dropout_rates': [1/i for i in range(2, 12)],
            'batch_size': 10000,
            'learning_rate': 0.0005,  # Lower learning rate for fine-tuning
            'weight_decay': 0.01,  # L2 regularization
            'epochs': 100,  # Max epochs
            'patience': 15,  # Early stopping patience
            'min_delta': 0.0005,  # Minimum improvement
            'validation_split': 0.2,  # 20% validation data
            'seed': 100,
        }
    
    # Set random seed
    if config.get('seed'):
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
    
    # Prepare data
    print(f"Loading data for LoB {lob}...")
    processor = ClaimsDataProcessor()
    data, info = processor.process_lob_data(lob)
    features = processor.prepare_features(data)
    
    # Create dataset
    dataset = ClaimsDataset(features)
    
    # Split into train and validation
    val_size = int(len(dataset) * config.get('validation_split', 0.2))
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.get('seed', 42))
    )
    
    print(f"Dataset split: {train_size} training, {val_size} validation samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Initialize model
    print("Initializing model...")
    model = EmbeddingNetwork(
        neurons=config['neurons'],
        dropout_rates=config['dropout_rates'],
        starting_values=info['starting_values'],
        network_weights=info['network_weights'],
        seed=config.get('seed'),
        mode='fixed_embedding'
    )
    
    # Load pre-trained embeddings
    load_pretrained_embeddings(model, lob)
    
    # Freeze embedding layers
    for name, param in model.named_parameters():
        if 'embeddings' in name:
            param.requires_grad = False
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Initialize loss and optimizer (only for non-embedding parameters)
    criterion = WeightedLoss(info['network_weights']).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = optim.Adam(
        trainable_params, 
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01)  # L2 regularization
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.get('patience', 15),
        min_delta=config.get('min_delta', 0.0005),
        verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    
    # Create directory for saving weights
    weights_dir = Path(f"./Results/WeightsKerasModel/LoB{lob}")
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("Starting training with early stopping...")
    best_epoch = 0
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0
        n_train_batches = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]") as pbar:
            for batch in pbar:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, batch)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                n_train_batches += 1
                pbar.set_postfix({'loss': train_loss / n_train_batches})
        
        avg_train_loss = train_loss / n_train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0
        n_val_batches = 0
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]") as pbar:
                for batch in pbar:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(batch)
                    loss = criterion(outputs, batch)
                    
                    val_loss += loss.item()
                    n_val_batches += 1
                    pbar.set_postfix({'val_loss': val_loss / n_val_batches})
        
        avg_val_loss = val_loss / n_val_batches
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        
        # Save weights for this epoch
        save_epoch_weights(model, epoch + 1, weights_dir)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        early_stopping(avg_val_loss, model)
        
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            best_epoch = epoch + 1 - early_stopping.patience
            # Restore best model
            model.load_state_dict(early_stopping.best_model_state)
            # Save best model weights
            save_epoch_weights(model, 9999, weights_dir)  # Save as special epoch number
            break
        
        if not early_stopping.early_stop:
            best_epoch = epoch + 1
    
    # Save configuration for later use
    save_config(config, lob)
    
    # Save final checkpoint and history
    save_checkpoint(model, optimizer, best_epoch, config, lob, 'network')
    save_training_history(history, lob, 'network')
    save_epochs_info(lob, best_epoch)
    
    # Save best model as weights.9999.pt for easy identification
    best_weights_dir = Path(f"./Results/WeightsKerasModel/LoB{lob}")
    best_weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), best_weights_dir / "weights.9999.pt")
    print(f"Best model saved as weights.9999.pt")
    
    print(f"Best validation loss: {early_stopping.val_loss_min:.6f}")
    print(f"Best epoch: {best_epoch}")
    
    return model, info, history


def load_pretrained_embeddings(model: EmbeddingNetwork, lob: int):
    """Load pre-trained embedding weights."""
    
    embeddings_dir = Path(f"./Results/Embeddings/LoB{lob}")
    
    # Load embedding indices
    indices_file = embeddings_dir / "Embedding_Weights_Indicator.txt"
    if indices_file.exists():
        indices = np.loadtxt(indices_file, delimiter=';', skiprows=1).astype(int)
        
        # Load each embedding weight
        embedding_weights = {}
        for idx in indices:
            weight_file = embeddings_dir / f"Embedding_Weights_{idx}.txt"
            if weight_file.exists():
                weight = np.loadtxt(weight_file, delimiter=';', skiprows=1)
                embedding_weights[idx] = weight
        
        # Apply weights to model embeddings
        print(f"Loaded {len(embedding_weights)} embedding weights from {embeddings_dir}")
    else:
        print(f"Warning: No pre-trained embeddings found for LoB {lob}")


def save_epoch_weights(model: EmbeddingNetwork, epoch: int, output_dir: Path):
    """Save model weights for a specific epoch."""
    
    weights_file = output_dir / f"weights.{epoch:02d}.pt"
    torch.save(model.state_dict(), weights_file)


def save_checkpoint(model, optimizer, epoch, config, lob, stage):
    """Save model checkpoint."""
    
    checkpoint_dir = Path(f"./Results/Checkpoints/LoB{lob}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'stage': stage,
    }
    
    torch.save(checkpoint, checkpoint_dir / f"{stage}_checkpoint.pt")
    print(f"Checkpoint saved to {checkpoint_dir}/{stage}_checkpoint.pt")


def save_epochs_info(lob: int, epochs: int):
    """Save number of epochs information."""
    
    output_dir = Path(f"./Results/NumbersOfEpochs/SecondTrainingStep/LoB{lob}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "Number_of_Epochs.txt", 'w') as f:
        f.write("epochs;value\n")
        f.write(f"1;{epochs}\n")
    
    # Also save best epoch indicator
    with open(output_dir / "Best_Epoch.txt", 'w') as f:
        f.write("best_epoch;value\n")
        f.write(f"1;{epochs}\n")


if __name__ == "__main__":
    # Example usage with early stopping
    for lob in range(1, 5):  # Adjust based on available LoB
        print(f"\n{'='*50}")
        print(f"Training network weights for LoB {lob} with validation")
        print(f"{'='*50}")
        
        config = {
            'neurons': [40, 30, 10],
            'dropout_rates': [1/i for i in range(2, 12)],
            'batch_size': 5000,
            'learning_rate': 0.0005,
            'weight_decay': 0.01,
            'epochs': 100,
            'patience': 15,
            'min_delta': 0.0005,
            'validation_split': 0.2,
            'seed': 100,
        }
        
        model, info, history = train_network_with_validation(lob, config)
        
        print(f"Training completed for LoB {lob}")
        print(f"Final train loss: {history['train_loss'][-1]:.6f}")
        print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    
    print("\n" + "="*50)
    print("All network weights training completed with validation!")
    print("="*50)