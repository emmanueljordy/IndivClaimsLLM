"""
Fixed Calculate Variance Parameters for RBNS Reserve Calculation
"""

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from .models import EmbeddingNetwork
from .data_preprocessing import ClaimsDataProcessor, ClaimsDataset
from torch.utils.data import DataLoader
from .utils import load_config


def calculate_variance_parameters_fixed(lob: int, config: dict = None):
    """Calculate variance parameters (sigma squared) for a specific LoB with fixes."""
    
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
    
    # Load best model weight
    best_weights = Path(f"./Results/WeightsKerasModel/LoB{lob}/weights.9999.pt")
    if best_weights.exists():
        print("Using best model from early stopping (epoch 9999)")
        model.load_state_dict(torch.load(best_weights, map_location=device))
    else:
        print(f"Warning: Best model not found, using latest weights")
        # Find the latest weight file
        weights_dir = Path(f"./Results/WeightsKerasModel/LoB{lob}")
        if weights_dir.exists():
            weight_files = sorted(weights_dir.glob("weights.*.pt"))
            if weight_files:
                latest_weights = weight_files[-1]
                print(f"Loading {latest_weights.name}")
                model.load_state_dict(torch.load(latest_weights, map_location=device))
    
    # Get predictions
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Getting predictions"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            
            # Collect predictions for each time point
            batch_pred = []
            for t in range(12):
                ind = outputs[f'payment_indicator{t:02d}_output'].cpu().numpy()
                mean = outputs[f'payment_mean{t:02d}_output'].cpu().numpy()
                batch_pred.append(np.stack([ind, mean], axis=1))
            
            all_predictions.append(np.stack(batch_pred, axis=1))
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)
    
    # Calculate improved sigma_squared for each time point
    sigma_squared = np.zeros(12)
    
    for t in range(12):
        # Get actual payments and predictions for this time point
        actual_payments = data[f'Pay{t:02d}'].values
        pred_indicator = predictions[:, t, 0]
        pred_mean = predictions[:, t, 1]
        
        # Method 1: Calculate variance of log-scale errors for positive predictions
        log_errors = []
        for i in range(len(actual_payments)):
            if pred_indicator[i] > 0.1 and actual_payments[i] > 0:  # Both predicted and actual payment (lower threshold)
                # Calculate log-scale error
                predicted_payment = pred_indicator[i] * np.exp(pred_mean[i])
                if predicted_payment > 0:
                    log_error = np.log(actual_payments[i]) - np.log(predicted_payment)
                    log_errors.append(log_error)
        
        if len(log_errors) > 1:
            # Use robust variance estimation (MAD-based)
            median_error = np.median(log_errors)
            mad = np.median(np.abs(log_errors - median_error))
            # Convert MAD to standard deviation estimate (assuming normal distribution)
            sigma_squared[t] = (1.4826 * mad) ** 2
        else:
            sigma_squared[t] = 0.5  # Conservative default
        
        # Cap extreme values more aggressively
        sigma_squared[t] = np.clip(sigma_squared[t], 0.01, 2.0)  # Max variance of 2.0
        
        print(f"Time {t:2d}: sigma^2 = {sigma_squared[t]:.4f} (from {len(log_errors)} samples)")
    
    # Save results
    save_variance_parameters_fixed(sigma_squared, lob)
    
    return sigma_squared


def save_variance_parameters_fixed(sigma_squared: np.ndarray, lob: int):
    """Save variance parameters to file."""
    
    output_dir = Path(f"./Results/VarianceParameters/LoB{lob}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as text file with header
    with open(output_dir / "sigma_squared_fixed.txt", 'w') as f:
        f.write(";sigma_squared\n")
        for i, val in enumerate(sigma_squared):
            f.write(f"{i+1};{val:.10f}\n")
    
    print(f"Fixed variance parameters saved to {output_dir}/sigma_squared_fixed.txt")


if __name__ == "__main__":
    # Test for LoB 1
    lob = 1
    print(f"Calculating fixed variance parameters for LoB {lob}")
    sigma_squared = calculate_variance_parameters_fixed(lob)
    print(f"Results: {sigma_squared}")