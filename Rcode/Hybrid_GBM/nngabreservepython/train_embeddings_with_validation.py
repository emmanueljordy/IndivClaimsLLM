"""
Train Embedding Weights with Early Stopping and Validation
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


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.best_model_state = None
        self.best_epoch = 0
        self.current_epoch = 0
        
    def __call__(self, val_loss, model, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        self.best_model_state = model.state_dict().copy()
        self.val_loss_min = val_loss
        self.best_epoch = self.current_epoch


def train_embeddings_with_validation(lob: int, config: dict = None):
    """Train embedding weights with validation and early stopping."""
    
    # Default configuration with regularization
    if config is None:
        config = {
            'neurons': [40, 30, 10],
            'dropout_rates': [1/i for i in range(2, 12)],
            'batch_size': 10000,
            'learning_rate': 0.001,
            'weight_decay': 0.01,  # L2 regularization
            'epochs': 100,  # Max epochs (early stopping will likely stop before)
            'patience': 10,  # Early stopping patience
            'min_delta': 0.001,  # Minimum improvement
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
        mode='train_embedding'
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Initialize loss and optimizer with weight decay
    criterion = WeightedLoss(info['network_weights']).to(device)
    optimizer = optim.Adam(
        model.parameters(), 
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
        patience=config.get('patience', 10),
        min_delta=config.get('min_delta', 0.001),
        verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    
    # Training loop
    print("Starting training with early stopping...")
    
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
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
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
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        early_stopping(avg_val_loss, model, epoch+1)
        
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            # Restore best model
            model.load_state_dict(early_stopping.best_model_state)
            break
    
    # Save configuration for later use
    save_config(config, lob)
    
    # Save final model and history
    save_embeddings(model, lob, config)
    save_training_history(history, lob, 'embeddings')
    save_checkpoint(model, optimizer, epoch, config, lob, 'embeddings')
    
    # Save best model as weights.9999.pt for easy identification
    best_weights_dir = Path(f"./Results/WeightsKerasModel/LoB{lob}")
    best_weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), best_weights_dir / "weights.9999.pt")
    print(f"Best model saved as weights.9999.pt")
    
    # Record best epoch
    history['best_epoch'] = early_stopping.best_epoch if hasattr(early_stopping, 'best_epoch') else epoch + 1
    
    print(f"Best validation loss: {early_stopping.val_loss_min:.6f}")
    
    return model, info, history


def save_embeddings(model: EmbeddingNetwork, lob: int, config: dict):
    """Save embedding weights to files."""
    
    output_dir = Path(f"./Results/Embeddings/LoB{lob}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get embedding weights
    embeddings_to_save = {}
    embedding_indices = []
    
    for name, module in model.embeddings.items():
        if isinstance(module, nn.Embedding):
            weight = module.weight.detach().cpu().numpy()
            # Only save embeddings with more than 1 row and 1 column
            if weight.shape[0] > 1 and weight.shape[1] == 1:
                idx = len(embeddings_to_save)
                embeddings_to_save[idx] = weight
                embedding_indices.append(idx)
    
    # Save embedding indices
    np.savetxt(
        output_dir / "Embedding_Weights_Indicator.txt",
        embedding_indices,
        delimiter=';',
        header='embedding_index',
        comments=''
    )
    
    # Save each embedding weight matrix
    for idx, weight in embeddings_to_save.items():
        np.savetxt(
            output_dir / f"Embedding_Weights_{idx}.txt",
            weight,
            delimiter=';',
            header='weight',
            comments=''
        )
    
    # Save configuration
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Embeddings saved to {output_dir}")


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


def save_training_history(history: Dict, lob: int, stage: str):
    """Save training history for analysis."""
    
    history_dir = Path(f"./Results/TrainingHistory/LoB{lob}")
    history_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    with open(history_dir / f"{stage}_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save as CSV for easy plotting
    import pandas as pd
    df = pd.DataFrame(history)
    df.to_csv(history_dir / f"{stage}_history.csv", index=False)
    
    print(f"Training history saved to {history_dir}")


if __name__ == "__main__":
    # Example usage with early stopping
    for lob in range(1, 5):  # Adjust based on available LoB
        print(f"\n{'='*50}")
        print(f"Training embeddings for LoB {lob} with validation")
        print(f"{'='*50}")
        
        config = {
            'neurons': [40, 30, 10],
            'dropout_rates': [1/i for i in range(2, 12)],
            'batch_size': 5000,
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'epochs': 100,
            'patience': 10,
            'min_delta': 0.001,
            'validation_split': 0.2,
            'seed': 100,
        }
        
        model, info, history = train_embeddings_with_validation(lob, config)
        
        print(f"Training completed for LoB {lob}")
        print(f"Final train loss: {history['train_loss'][-1]:.6f}")
        print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    
    print("\n" + "="*50)
    print("All embeddings training completed with validation!")
    print("="*50)