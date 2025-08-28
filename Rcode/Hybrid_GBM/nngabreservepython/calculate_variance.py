"""
Calculate Variance Parameters for RBNS Reserve Calculation
Author: Converted from R to Python
Date: 2025
"""

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from .models import EmbeddingNetwork
from .data_preprocessing import ClaimsDataProcessor, ClaimsDataset
from torch.utils.data import DataLoader
from .utils import load_config


def calculate_variance_parameters(lob: int, config: dict = None):
    """Calculate variance parameters (sigma squared) for a specific LoB."""
    
    # Load saved configuration if not provided
    if config is None:
        config = load_config(lob)
        # Add inference-specific settings
        config['dropout_rates'] = [0.00000000001] * 10  # Very small dropout for inference
        if 'batch_size' not in config:
            config['batch_size'] = 10000
    
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
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False
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
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Load epochs information
    epochs_file = Path(f"./Results/NumbersOfEpochs/SecondTrainingStep/LoB{lob}/Number_of_Epochs.txt")
    best_epoch_file = Path(f"./Results/NumbersOfEpochs/SecondTrainingStep/LoB{lob}/Best_Epoch.txt")
    
    # Check if we have the special best model weight (epoch 9999) from early stopping
    best_weights = Path(f"./Results/WeightsKerasModel/LoB{lob}/weights.9999.pt")
    if best_weights.exists():
        print("Using best model from early stopping (epoch 9999)")
        considered_epochs = [9999]  # Special epoch number for best model
    # Check for best epoch from early stopping
    elif best_epoch_file.exists():
        with open(best_epoch_file, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                best_epoch = int(lines[1].strip().split(';')[1])
                # Use epochs around the best epoch
                considered_epochs = list(range(max(1, best_epoch - 2), min(best_epoch + 3, 100)))
                print(f"Using epochs around best epoch {best_epoch} from early stopping")
            else:
                # Fallback to Number_of_Epochs.txt
                if epochs_file.exists():
                    with open(epochs_file, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 1:
                            epochs = int(lines[1].strip().split(';')[1])
                        else:
                            epochs = 50
                else:
                    epochs = 50
                considered_epochs = list(range(max(1, epochs - 2), min(epochs + 3, 100)))
    elif epochs_file.exists():
        with open(epochs_file, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                epochs = int(lines[1].strip().split(';')[1])
            else:
                epochs = 50
        considered_epochs = list(range(max(1, epochs - 2), min(epochs + 3, 100)))
    else:
        print(f"Warning: No epoch information found for LoB {lob}, using default")
        epochs = 50
        considered_epochs = list(range(max(1, epochs - 2), min(epochs + 3, 100)))
    print(f"Averaging over epochs: {considered_epochs}")
    
    # Collect predictions
    all_predictions = []
    
    for epoch in considered_epochs:
        print(f"Loading weights for epoch {epoch}...")
        
        # Load model weights
        if epoch == 9999:
            # Special case for best model from early stopping
            weights_file = Path(f"./Results/WeightsKerasModel/LoB{lob}/weights.9999.pt")
        else:
            weights_file = Path(f"./Results/WeightsKerasModel/LoB{lob}/weights.{epoch:02d}.pt")
        
        if not weights_file.exists():
            print(f"Warning: Weights file not found for epoch {epoch}, skipping...")
            # Try to find any available weights
            weights_dir = Path(f"./Results/WeightsKerasModel/LoB{lob}")
            if weights_dir.exists():
                available_weights = sorted(weights_dir.glob("weights.*.pt"))
                if available_weights:
                    print(f"Available weight files: {[w.name for w in available_weights[:5]]}...")
            continue
        
        model.load_state_dict(torch.load(weights_file, map_location=device))
        
        # Get predictions
        model.eval()
        epoch_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Epoch {epoch} predictions"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch)
                
                # Collect predictions for each time point
                batch_pred = []
                for t in range(12):
                    ind = outputs[f'payment_indicator{t:02d}_output'].cpu().numpy()
                    mean = outputs[f'payment_mean{t:02d}_output'].cpu().numpy()
                    batch_pred.append(np.stack([ind, mean], axis=1))
                
                epoch_predictions.append(np.stack(batch_pred, axis=1))
        
        # Concatenate all batches
        epoch_predictions = np.concatenate(epoch_predictions, axis=0)
        all_predictions.append(epoch_predictions)
    
    # Average predictions
    prediction_averaged = np.mean(all_predictions, axis=0)
    
    # Calculate sigma_squared for each time point
    sigma_squared = np.zeros(12)
    
    for t in range(12):
        # Get actual payments and predictions for this time point
        actual_payments = data[f'Pay{t:02d}'].values
        pred_indicator = prediction_averaged[:, t, 0]
        pred_mean = prediction_averaged[:, t, 1]
        
        # Filter for positive predictions
        mask = pred_indicator > 0
        
        if mask.any():
            # Calculate sigma_squared using the formula from R code
            numerator = np.sum(actual_payments[mask])
            denominator = np.sum(pred_indicator[mask] * np.exp(pred_mean[mask]))
            
            if denominator > 0 and numerator > 0:
                sigma_squared[t] = 2 * np.log(numerator / denominator)
            else:
                # Use default value if calculation fails
                sigma_squared[t] = 1.0  # Default variance
        else:
            # No predictions for this time point - use default
            sigma_squared[t] = 1.0  # Default variance
        
        # Ensure valid value (no NaN or negative)
        if np.isnan(sigma_squared[t]) or np.isinf(sigma_squared[t]) or sigma_squared[t] <= 0:
            sigma_squared[t] = 1.0  # Default variance
        
        # Cap extreme values
        sigma_squared[t] = np.clip(sigma_squared[t], 0.01, 20.0)
    
    # Save results
    save_variance_parameters(sigma_squared, lob)
    
    return sigma_squared


def save_variance_parameters(sigma_squared: np.ndarray, lob: int):
    """Save variance parameters to file."""
    
    output_dir = Path(f"./Results/VarianceParameters/LoB{lob}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as text file with header
    with open(output_dir / "sigma_squared.txt", 'w') as f:
        f.write(";sigma_squared\n")
        for i, val in enumerate(sigma_squared):
            f.write(f"{i+1};{val:.10f}\n")
    
    print(f"Variance parameters saved to {output_dir}/sigma_squared.txt")


if __name__ == "__main__":
    # Calculate variance parameters for all lines of business
    for lob in range(1, 7):
        print(f"\n{'='*50}")
        print(f"Calculating variance parameters for LoB {lob}")
        print(f"{'='*50}")
        
        sigma_squared = calculate_variance_parameters(lob)
        
        print(f"Sigma squared values: {sigma_squared}")
        print(f"Calculation completed for LoB {lob}")
    
    print("\n" + "="*50)
    print("All variance parameters calculated!")
    print("="*50)