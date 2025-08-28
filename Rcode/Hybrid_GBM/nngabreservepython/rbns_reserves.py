"""
RBNS Reserves Calculation - Main Prediction Script
Author: Converted from R to Python
Date: 2025
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from .models import EmbeddingNetwork
from .data_preprocessing import ClaimsDataProcessor, ClaimsDataset
from .calculate_recoveries import load_recovery_estimates
from .utils import load_config


def calculate_claims_reserves(lob: int, config: dict = None):
    """Calculate individual RBNS claims reserves for a specific LoB."""
    
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
    
    # Load recovery estimates
    recovery_analysis = load_recovery_estimates()
    recovery_prob = recovery_analysis.loc['estimated recovery probability', f'LoB {lob}']
    recovery_mean = recovery_analysis.loc['estimated recovery mean', f'LoB {lob}']
    
    # Prepare data
    print(f"Loading data for LoB {lob}...")
    processor = ClaimsDataProcessor()
    data, info = processor.process_lob_data(lob)
    
    # Prepare prediction data
    data_pred, features_pred = processor.prepare_prediction_data(data)
    
    # Create dataset
    dataset = ClaimsDataset(features_pred)
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
                # Fallback
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
        epochs = 50
        print(f"Warning: No epoch information found for LoB {lob}, using default {epochs}")
        considered_epochs = list(range(max(1, epochs - 2), min(epochs + 3, 100)))
    
    # Load variance parameters
    sigma_squared = load_variance_parameters(lob)
    
    print(f"Averaging predictions over epochs: {considered_epochs}")
    
    # Collect predictions
    all_predictions = []
    valid_epochs = 0
    
    for epoch in considered_epochs:
        # Load model weights
        if epoch == 9999:
            # Special case for best model from early stopping
            weights_file = Path(f"./Results/WeightsKerasModel/LoB{lob}/weights.9999.pt")
        else:
            weights_file = Path(f"./Results/WeightsKerasModel/LoB{lob}/weights.{epoch:02d}.pt")
        
        if not weights_file.exists():
            print(f"Warning: Weights file not found for epoch {epoch}, skipping...")
            continue
        
        print(f"Loading weights for epoch {epoch}...")
        model.load_state_dict(torch.load(weights_file, map_location=device))
        
        # Get predictions
        model.eval()
        epoch_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Epoch {epoch}", leave=False):
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
        valid_epochs += 1
    
    if valid_epochs == 0:
        raise ValueError(f"No valid model weights found for LoB {lob}")
    
    # Average predictions
    prediction_averaged = np.mean(all_predictions, axis=0)
    
    # Calculate true reserves (from actual future payments)
    true_reserves = 0
    for t in range(12):
        pay_col = f'Pay{t:02d}'
        pred_col = f'Time_Predict_Indicator{t:02d}'
        if pay_col in data_pred.columns:
            true_reserves += (data_pred[pay_col] * features_pred[pred_col]).sum()
    
    # Calculate estimated reserves
    estimated_reserves = 0
    
    for t in range(1, 12):  # Skip time 0 as it's already known
        # Get predictions for this time point
        pred_indicator = prediction_averaged[:, t, 0]
        pred_mean = prediction_averaged[:, t, 1]
        
        # Ensure sigma_squared is valid
        sigma_t = sigma_squared[t] if not np.isnan(sigma_squared[t]) else 1.0
        
        # Calculate expected payment with NaN protection
        exp_argument = pred_mean + sigma_t / 2
        # Clip to prevent overflow
        exp_argument = np.clip(exp_argument, -20, 20)
        expected_payment = pred_indicator * np.exp(exp_argument)
        
        # Remove any NaN values
        expected_payment = np.nan_to_num(expected_payment, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Add recovery adjustment
        if not np.isnan(recovery_prob) and not np.isnan(recovery_mean):
            recovery_adjustment = (pred_indicator > 0) * recovery_prob * recovery_mean
            recovery_adjustment = np.nan_to_num(recovery_adjustment, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            recovery_adjustment = 0
        
        # Sum up reserves
        estimated_reserves += np.sum(expected_payment) + np.sum(recovery_adjustment)
    
    # Final NaN check
    if np.isnan(estimated_reserves) or np.isinf(estimated_reserves):
        print(f"Warning: Invalid reserve calculation for LoB {lob}, setting to 0")
        estimated_reserves = 0
    
    estimated_reserves = round(estimated_reserves)
    
    # Calculate bias
    bias = estimated_reserves - true_reserves
    bias_percent = round((bias / true_reserves * 100) if true_reserves != 0 else 0, 1)
    
    results = {
        'estimated_reserves': estimated_reserves,
        'true_reserves': true_reserves,
        'bias': bias,
        'bias_percent': bias_percent
    }
    
    return results


def load_variance_parameters(lob: int):
    """Load variance parameters from file."""
    
    variance_file = Path(f"./Results/VarianceParameters/LoB{lob}/sigma_squared.txt")
    
    if variance_file.exists():
        df = pd.read_csv(variance_file, sep=';', index_col=0)
        sigma_squared = df['sigma_squared'].values
        
        # Handle NaN values - replace with reasonable defaults
        for i in range(len(sigma_squared)):
            if np.isnan(sigma_squared[i]) or np.isinf(sigma_squared[i]) or sigma_squared[i] <= 0:
                # Use average of neighboring values or default
                neighbors = []
                if i > 0 and not np.isnan(sigma_squared[i-1]):
                    neighbors.append(sigma_squared[i-1])
                if i < len(sigma_squared)-1 and not np.isnan(sigma_squared[i+1]):
                    neighbors.append(sigma_squared[i+1])
                
                if neighbors:
                    sigma_squared[i] = np.mean(neighbors)
                else:
                    # Use overall mean of valid values or default
                    valid_values = sigma_squared[~np.isnan(sigma_squared) & (sigma_squared > 0)]
                    if len(valid_values) > 0:
                        sigma_squared[i] = np.mean(valid_values)
                    else:
                        sigma_squared[i] = 1.0  # Default variance
        
        print(f"Loaded variance parameters for LoB {lob} (NaN values replaced)")
        return sigma_squared
    else:
        print(f"Warning: Variance parameters not found for LoB {lob}, using defaults")
        return np.ones(12) * 1.0


def calculate_all_reserves():
    """Calculate reserves for all lines of business."""
    
    # Detect available LoB values
    from .data_preprocessing import ClaimsDataProcessor
    processor = ClaimsDataProcessor()
    available_lobs = processor.get_available_lobs()
    print(f"Available Lines of Business: {available_lobs}")
    
    results_df = pd.DataFrame(
        index=['individual RBNS claims reserves', 'true RBNS reserves', 'bias', 'bias in %'],
        columns=[f'LoB {i}' for i in available_lobs]
    )
    
    for lob in available_lobs:
        print(f"\n{'='*50}")
        print(f"Calculating reserves for LoB {lob}")
        print(f"{'='*50}")
        
        try:
            results = calculate_claims_reserves(lob)
            
            results_df.loc['individual RBNS claims reserves', f'LoB {lob}'] = results['estimated_reserves']
            results_df.loc['true RBNS reserves', f'LoB {lob}'] = results['true_reserves']
            results_df.loc['bias', f'LoB {lob}'] = results['bias']
            results_df.loc['bias in %', f'LoB {lob}'] = results['bias_percent']
            
            print(f"Results for LoB {lob}:")
            print(f"  Estimated reserves: {results['estimated_reserves']:,}")
            print(f"  True reserves: {results['true_reserves']:,}")
            print(f"  Bias: {results['bias']:,} ({results['bias_percent']}%)")
            
        except Exception as e:
            print(f"Error calculating reserves for LoB {lob}: {e}")
            continue
    
    # Save results
    save_reserves_analysis(results_df)
    
    return results_df


def save_reserves_analysis(results_df: pd.DataFrame):
    """Save reserves analysis results to file."""
    
    output_dir = Path("./Results/RBNSReserves")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "ReservesAnalysis.txt"
    results_df.to_csv(output_file, sep=';')
    
    print(f"\nReserves analysis saved to {output_file}")
    print("\nFinal Results:")
    print(results_df)


if __name__ == "__main__":
    print("="*50)
    print("RBNS Reserves Calculation")
    print("="*50)
    
    # Calculate reserves for all lines of business
    results = calculate_all_reserves()
    
    print("\n" + "="*50)
    print("All reserves calculations completed!")
    print("="*50)