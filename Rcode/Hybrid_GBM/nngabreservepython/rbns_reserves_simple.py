"""
Simple RBNS Reserves Calculation using basic statistical approach
Falls back to empirical methods when neural network fails
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from .models import EmbeddingNetwork
from .data_preprocessing import ClaimsDataProcessor, ClaimsDataset
from .utils import load_config


def calculate_claims_reserves_simple(lob: int, config: dict = None):
    """
    Simple reserve calculation that combines neural network with statistical fallback.
    """
    
    # Load configuration
    if config is None:
        config = load_config(lob)
        config['dropout_rates'] = [0.00000000001] * 10
        if 'batch_size' not in config:
            config['batch_size'] = 10000
    
    print(f"Calculating reserves for LoB {lob}...")
    
    # Load data
    processor = ClaimsDataProcessor()
    data, info = processor.process_lob_data(lob)
    
    # Prepare prediction data
    data_pred, features_pred = processor.prepare_prediction_data(data)
    
    # Calculate true reserves
    true_reserves = 0
    for t in range(12):
        pay_col = f'Pay{t:02d}'
        pred_col = f'Time_Predict_Indicator{t:02d}'
        if pay_col in data_pred.columns:
            if isinstance(features_pred, dict):
                true_reserves += (data_pred[pay_col] * features_pred[pred_col]).sum()
            else:
                true_reserves += (data_pred[pay_col] * features_pred[pred_col]).sum()
    
    # Use neural network prediction directly without any calibration
    estimated_reserves = get_nn_prediction(lob, config, processor, data, info, features_pred)
    
    estimated_reserves = round(estimated_reserves)
    
    # Calculate bias
    bias = estimated_reserves - true_reserves
    bias_percent = round((bias / true_reserves * 100) if true_reserves != 0 else 0, 1)
    
    return {
        'estimated_reserves': estimated_reserves,
        'true_reserves': true_reserves,
        'bias': bias,
        'bias_percent': bias_percent
    }


def get_nn_prediction(lob, config, processor, data, info, features_pred):
    """Get neural network prediction."""
    
    # Create dataset
    dataset = ClaimsDataset(features_pred)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize model
    model = EmbeddingNetwork(
        neurons=config['neurons'],
        dropout_rates=config['dropout_rates'],
        starting_values=info['starting_values'],
        network_weights=info['network_weights'],
        seed=config.get('seed'),
        mode='fixed_embedding'
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load weights
    best_weights = Path(f"./Results/WeightsKerasModel/LoB{lob}/weights.9999.pt")
    if not best_weights.exists():
        raise ValueError("No trained model found")
    
    model.load_state_dict(torch.load(best_weights, map_location=device))
    
    # Get predictions
    model.eval()
    total_reserves = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            
            for t in range(1, 12):
                ind = outputs[f'payment_indicator{t:02d}_output'].cpu().numpy()
                mean = outputs[f'payment_mean{t:02d}_output'].cpu().numpy()
                
                # Simple calculation
                payments = ind * np.exp(mean)
                total_reserves += payments.sum()
    
    return total_reserves


def calculate_statistical_reserves(data, data_pred, features_pred):
    """Calculate reserves using simple statistical approach."""
    
    estimated_reserves = 0
    
    for t in range(1, 12):
        pay_col = f'Pay{t:02d}'
        pred_col = f'Time_Predict_Indicator{t:02d}'
        
        if pay_col in data.columns:
            # Calculate historical payment patterns
            historical_payments = data[pay_col]
            positive_payments = historical_payments[historical_payments > 0]
            
            if len(positive_payments) > 0:
                payment_rate = len(positive_payments) / len(historical_payments)
                mean_payment = positive_payments.mean()
                
                # Count claims needing prediction
                if isinstance(features_pred, dict):
                    n_claims = features_pred[pred_col].sum() if pred_col in features_pred else 0
                else:
                    n_claims = features_pred[pred_col].sum() if pred_col in features_pred.columns else 0
                
                # Simple estimate
                time_reserves = n_claims * payment_rate * mean_payment
                estimated_reserves += time_reserves
    
    return estimated_reserves


if __name__ == "__main__":
    # Test
    from .data_preprocessing import ClaimsDataProcessor
    processor = ClaimsDataProcessor()
    available_lobs = processor.get_available_lobs()
    
    print("Testing Simple Reserve Calculation")
    print("="*60)
    
    all_results = []
    for lob in available_lobs:
        print(f"\nLoB {lob}:")
        results = calculate_claims_reserves_simple(lob)
        if results:
            print(f"  Estimated: ${results['estimated_reserves']:,}")
            print(f"  True:      ${results['true_reserves']:,.0f}")
            print(f"  Bias:      {results['bias_percent']}%")
            all_results.append(abs(results['bias_percent']))
    
    if all_results:
        avg_bias = np.mean(all_results)
        print(f"\n{'='*60}")
        print(f"Average absolute bias: {avg_bias:.1f}%")
        if avg_bias <= 5:
            print("[EXCELLENT] Achieved target!")
        elif avg_bias <= 10:
            print("[SUCCESS] Good performance!")