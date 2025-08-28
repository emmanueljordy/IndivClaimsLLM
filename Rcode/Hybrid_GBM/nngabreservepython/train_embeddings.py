"""
Train Embedding Weights for Neural Network
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


def train_embeddings(lob: int, config: dict = None):
    """Train embedding weights for a specific line of business."""
    
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
        mode='train_embedding'
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Initialize loss and optimizer
    criterion = WeightedLoss(info['network_weights']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    print("Starting training...")
    model.train()
    
    for epoch in range(config['epochs']):
        epoch_loss = 0
        n_batches = 0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}") as pbar:
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
    
    # Save configuration for later use
    save_config(config, lob)
    
    # Save embeddings
    save_embeddings(model, lob, config)
    
    # Save model checkpoint
    save_checkpoint(model, optimizer, epoch, config, lob, 'embeddings')
    
    return model, info


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


def load_checkpoint(lob: int, stage: str = 'embeddings'):
    """Load model checkpoint."""
    
    checkpoint_path = Path(f"./Results/Checkpoints/LoB{lob}/{stage}_checkpoint.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    return checkpoint


if __name__ == "__main__":
    # Train embeddings for all lines of business
    for lob in range(1, 7):
        print(f"\n{'='*50}")
        print(f"Training embeddings for LoB {lob}")
        print(f"{'='*50}")
        
        # Check if we need to load epochs from file
        epochs_file = Path(f"./Results/NumbersOfEpochs/FirstTrainingStep/LoB{lob}/Number_of_Epochs_Embedding.txt")
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
        
        model, info = train_embeddings(lob, config)
        
        print(f"Training completed for LoB {lob}")
    
    print("\n" + "="*50)
    print("All embeddings training completed!")
    print("="*50)