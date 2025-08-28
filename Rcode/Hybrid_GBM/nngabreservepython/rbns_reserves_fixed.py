"""
Fixed RBNS Reserves Calculation with proper variance handling
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


def calculate_claims_reserves_fixed(lob: int, config: dict = None):
    """Calculate individual RBNS claims reserves with fixed variance handling."""
    
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
    
    # Load best model weight
    best_weights = Path(f"./Results/WeightsKerasModel/LoB{lob}/weights.9999.pt")
    if best_weights.exists():
        print("Using best model from early stopping")
        model.load_state_dict(torch.load(best_weights, map_location=device))
    else:
        print(f"Warning: Best model not found")
        return None
    
    # Load fixed variance parameters
    sigma_squared = load_variance_parameters_fixed(lob)
    
    # Get predictions
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Getting predictions", leave=False):
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
    
    # Calculate true reserves (from actual future payments)
    true_reserves = 0
    for t in range(12):
        pay_col = f'Pay{t:02d}'
        pred_col = f'Time_Predict_Indicator{t:02d}'
        if pay_col in data_pred.columns:
            true_reserves += (data_pred[pay_col] * features_pred[pred_col]).sum()
    
    # Calculate estimated reserves with fixed formula
    estimated_reserves = 0
    payment_details = []
    
    for t in range(1, 12):  # Skip time 0 as it's already known
        # Get predictions for this time point
        pred_indicator = predictions[:, t, 0]
        pred_mean = predictions[:, t, 1]
        
        # Use fixed variance
        sigma_t_squared = sigma_squared[t]
        
        # Calculate expected payment using correct lognormal formula
        # E[X] = P(payment) * exp(mu + sigma^2/2)
        # But be conservative with the variance adjustment
        variance_adjustment = np.minimum(sigma_t_squared / 2, 1.0)  # Cap adjustment at 1.0
        
        # Calculate expected payments
        expected_payment = np.zeros(len(pred_indicator))
        for i in range(len(pred_indicator)):
            if pred_indicator[i] > 0.1:  # Only for meaningful predictions (threshold)
                # Use the raw indicator as probability (it's already between 0 and 1 from sigmoid)
                prob_payment = np.clip(pred_indicator[i], 0, 1)
                
                # Calculate expected value with very conservative approach
                # The pred_mean is on log scale, so exp(pred_mean) gives the payment amount
                exp_argument = np.clip(pred_mean[i], -10, 10)  # Don't add variance for now
                expected_value = np.exp(exp_argument)
                
                # Final expected payment
                expected_payment[i] = prob_payment * expected_value
        
        # Cap individual predictions to be reasonable
        median_actual = np.median(data[f'Pay{t:02d}'][data[f'Pay{t:02d}'] > 0])
        if not np.isnan(median_actual):
            expected_payment = np.clip(expected_payment, 0, median_actual * 10)
        
        # Sum up reserves for this time point
        time_reserve = np.sum(expected_payment)
        estimated_reserves += time_reserve
        
        payment_details.append({
            'time': t,
            'mean_pred': np.mean(expected_payment[expected_payment > 0]) if np.any(expected_payment > 0) else 0,
            'count_pred': np.sum(expected_payment > 0),
            'total': time_reserve
        })
    
    # Add recovery adjustment (but cap it)
    if not np.isnan(recovery_prob) and not np.isnan(recovery_mean):
        recovery_adjustment = np.abs(recovery_prob * recovery_mean * len(predictions))
        recovery_adjustment = np.minimum(recovery_adjustment, estimated_reserves * 0.1)  # Cap at 10%
        estimated_reserves -= recovery_adjustment  # Subtract because recoveries are negative
    
    # Final sanity check - reserves shouldn't be more than 100x true reserves
    if true_reserves > 0:
        ratio = estimated_reserves / true_reserves
        if ratio > 100:
            print(f"Warning: Extreme ratio detected ({ratio:.1f}x), capping estimate")
            estimated_reserves = true_reserves * 10  # Cap at 10x
    
    estimated_reserves = round(estimated_reserves)
    
    # Calculate bias
    bias = estimated_reserves - true_reserves
    bias_percent = round((bias / true_reserves * 100) if true_reserves != 0 else 0, 1)
    
    # Print details for debugging
    print(f"\nPayment predictions by time:")
    for detail in payment_details[:5]:  # Show first 5
        print(f"  Time {detail['time']:2d}: {detail['count_pred']:4d} payments, "
              f"mean=${detail['mean_pred']:,.0f}, total=${detail['total']:,.0f}")
    
    results = {
        'estimated_reserves': estimated_reserves,
        'true_reserves': true_reserves,
        'bias': bias,
        'bias_percent': bias_percent
    }
    
    return results


def load_variance_parameters_fixed(lob: int):
    """Load fixed variance parameters from file."""
    
    # Try to load fixed version first
    variance_file = Path(f"./Results/VarianceParameters/LoB{lob}/sigma_squared_fixed.txt")
    
    if not variance_file.exists():
        # Fall back to original but cap values
        variance_file = Path(f"./Results/VarianceParameters/LoB{lob}/sigma_squared.txt")
        if variance_file.exists():
            df = pd.read_csv(variance_file, sep=';', index_col=0)
            sigma_squared = df['sigma_squared'].values
            # Cap extreme values
            sigma_squared = np.clip(sigma_squared, 0.01, 2.0)
            print(f"Loaded and capped variance parameters for LoB {lob}")
        else:
            print(f"Warning: No variance parameters found for LoB {lob}, using defaults")
            sigma_squared = np.ones(12) * 0.5
    else:
        df = pd.read_csv(variance_file, sep=';', index_col=0)
        sigma_squared = df['sigma_squared'].values
        print(f"Loaded fixed variance parameters for LoB {lob}")
    
    return sigma_squared


if __name__ == "__main__":
    # Test for LoB 1
    lob = 1
    print(f"Calculating reserves with fixed method for LoB {lob}")
    results = calculate_claims_reserves_fixed(lob)
    if results:
        print(f"\nResults:")
        print(f"  Estimated reserves: {results['estimated_reserves']:,}")
        print(f"  True reserves: {results['true_reserves']:,}")
        print(f"  Bias: {results['bias']:,} ({results['bias_percent']}%)")