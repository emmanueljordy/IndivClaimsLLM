"""
Calculate Recovery Estimates for RBNS Reserve Calculation
Author: Converted from R to Python
Date: 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path


def calculate_recoveries():
    """Calculate recovery estimates for all lines of business."""
    
    # Load data
    print("Loading claims data...")
    data_all = pd.read_csv("./Data/data.txt", sep=';')
    
    # Get available LoB values
    available_lobs = sorted(data_all['LoB'].unique())
    print(f"Available Lines of Business: {available_lobs}")
    
    # Initialize results dataframe with only available LoB
    recovery_analysis = pd.DataFrame(
        index=['estimated recovery probability', 'estimated recovery mean', 
               'estimated recovery', 'true recovery'],
        columns=[f'LoB {i}' for i in available_lobs]
    )
    
    # Calculate for each available LoB
    for lob in available_lobs:
        print(f"Calculating recoveries for LoB {lob}...")
        
        # Filter data for current LoB
        data = data_all[data_all['LoB'] == lob].copy()
        
        if len(data) == 0:
            print(f"  No data for LoB {lob}, skipping...")
            continue
        
        # Calculate true future recovery payments
        recovery_payments = 0
        for i in range(12):
            pay_col = f'Pay{i:02d}'
            # Recoveries are negative payments
            mask = (data['AY'] + data['RepDel']) > (2006 - i)
            if mask.any():
                recovery_payments += data.loc[mask, pay_col][data.loc[mask, pay_col] < 0].sum()
        
        recovery_analysis.loc['true recovery', f'LoB {lob}'] = round(recovery_payments)
        
        # Calculate recovery amounts, numbers, and claims counts
        recovery_amounts_table = np.zeros((12, 12))
        recovery_numbers_table = np.zeros((12, 12))
        number_of_claims_table = np.zeros((12, 12))
        
        for i in range(12):
            if (13 - i) > 1:
                for j in range(1, 13 - i):
                    pay_col = f'Pay{j:02d}'
                    mask = (data['AY'] + data['RepDel']) == (1993 + i)
                    
                    if mask.any():
                        # Recovery amounts (negative payments)
                        recovery_amounts_table[i, j] = data.loc[mask, pay_col][data.loc[mask, pay_col] < 0].sum()
                        # Number of recoveries
                        recovery_numbers_table[i, j] = (data.loc[mask, pay_col] < 0).sum()
                        # Total number of claims
                        number_of_claims_table[i, j] = mask.sum()
        
        # Calculate estimated recovery probability and mean
        # Recovery probability = weighted average of recovery rates over development periods
        time_weights = np.arange(12)
        
        total_recoveries = np.nansum(recovery_numbers_table, axis=0)
        total_claims = np.nansum(number_of_claims_table, axis=0)
        
        # Calculate recovery probability with NaN protection
        denominator = np.sum(total_claims * time_weights)
        if denominator > 0:
            recovery_prob = np.sum(total_recoveries * time_weights) / denominator
        else:
            recovery_prob = 0.0
        
        # Recovery mean (average recovery amount when recovery occurs)
        total_recovery_amounts = np.nansum(recovery_amounts_table, axis=0)
        denominator = np.sum(total_recoveries * time_weights)
        if denominator > 0:
            recovery_mean = np.sum(total_recovery_amounts * time_weights) / denominator
        else:
            recovery_mean = 0.0
        
        # Handle NaN values
        if np.isnan(recovery_prob) or np.isinf(recovery_prob):
            recovery_prob = 0.0
        if np.isnan(recovery_mean) or np.isinf(recovery_mean):
            recovery_mean = 0.0
        
        # Calculate estimated recovery
        # This is based on the number of years * recovery probability * recovery mean
        year_factors = (data['AY'] - 1994).values
        estimated_recovery = round(np.sum(year_factors * recovery_prob * recovery_mean))
        
        # Handle NaN in estimated recovery
        if np.isnan(estimated_recovery) or np.isinf(estimated_recovery):
            estimated_recovery = 0
        
        # Store results
        recovery_analysis.loc['estimated recovery probability', f'LoB {lob}'] = recovery_prob
        recovery_analysis.loc['estimated recovery mean', f'LoB {lob}'] = recovery_mean
        recovery_analysis.loc['estimated recovery', f'LoB {lob}'] = estimated_recovery
    
    # Save results
    save_recovery_analysis(recovery_analysis)
    
    return recovery_analysis


def save_recovery_analysis(recovery_analysis: pd.DataFrame):
    """Save recovery analysis results to file."""
    
    output_dir = Path("./Results/Recoveries")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to text file with semicolon separator
    output_file = output_dir / "RecoveryEstimates.txt"
    recovery_analysis.to_csv(output_file, sep=';')
    
    print(f"Recovery analysis saved to {output_file}")
    print("\nRecovery Analysis Results:")
    print(recovery_analysis)


def load_recovery_estimates():
    """Load pre-calculated recovery estimates."""
    
    recovery_file = Path("./Results/Recoveries/RecoveryEstimates.txt")
    
    if recovery_file.exists():
        recovery_analysis = pd.read_csv(recovery_file, sep=';', index_col=0)
        return recovery_analysis
    else:
        print("Recovery estimates file not found. Calculating...")
        return calculate_recoveries()


if __name__ == "__main__":
    print("="*50)
    print("Calculating Recovery Estimates")
    print("="*50)
    
    recovery_analysis = calculate_recoveries()
    
    print("\n" + "="*50)
    print("Recovery calculation completed!")
    print("="*50)