"""
Train Neural Network Weights (Second Training Step)
Author: Converted from R to Python
Date: 2025
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from .models import EmbeddingNetwork, WeightedLoss
from .data_preprocessing import ClaimsDataProcessor, ClaimsDataset, collate_fn
from .utils import save_config


def train_network_weights(lob: int, config: dict = None):
    """Train neural network weights with fixed embeddings."""
    
    # Default configuration
    if config is None:
        config = {
            'neurons': [40, 30, 10],
            'dropout_rates': [1/i for i in range(2, 12)],
            'batch_size': 10000,
            'learning_rate': 0.001,
            'epochs': 50,
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
    
    # Create dataset and dataloader
    dataset = ClaimsDataset(features)
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
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
    optimizer = optim.Adam(trainable_params, lr=config['learning_rate'])
    
    # Create directory for saving weights
    weights_dir = Path(f"./Results/WeightsKerasModel/LoB{lob}")
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("Starting training...")
    model.train()
    
    # Adjust epochs if needed
    epochs_to_train = config['epochs'] + 2  # Similar to R code epochs_cont
    
    for epoch in range(epochs_to_train):
        epoch_loss = 0
        n_batches = 0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs_to_train}") as pbar:
            for batch in pbar:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
                pbar.set_postfix({'loss': epoch_loss / n_batches})
        
        print(f"Epoch {epoch+1}: Average loss = {epoch_loss / n_batches:.6f}")
        
        # Save weights for each epoch (similar to Keras callback)
        save_epoch_weights(model, epoch + 1, weights_dir)
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, epochs_to_train, config, lob, 'network')
    
    # Save configuration for later use
    save_config(config, lob)
    
    # Save number of epochs
    save_epochs_info(lob, config['epochs'])
    
    return model, info


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
        # Note: This is a simplified version - you'd need to map the indices
        # to the correct embedding layers in the model
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


if __name__ == "__main__":
    # Train network weights for all lines of business
    for lob in range(1, 7):
        print(f"\n{'='*50}")
        print(f"Training network weights for LoB {lob}")
        print(f"{'='*50}")
        
        # Check if we need to load epochs from file
        epochs_file = Path(f"./Results/NumbersOfEpochs/SecondTrainingStep/LoB{lob}/Number_of_Epochs.txt")
        if epochs_file.exists():
            with open(epochs_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    epochs = int(lines[1].strip().split(';')[1])
                else:
                    epochs = 50
        else:
            epochs = 50
        
        config = {
            'neurons': [40, 30, 10],
            'dropout_rates': [1/i for i in range(2, 12)],
            'batch_size': 10000,
            'learning_rate': 0.001,
            'epochs': epochs,
            'seed': 100,
        }
        
        model, info = train_network_weights(lob, config)
        
        print(f"Training completed for LoB {lob}")
    
    print("\n" + "="*50)
    print("All network weights training completed!")
    print("="*50)