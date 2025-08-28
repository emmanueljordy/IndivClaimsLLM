"""
Data Preprocessing for RBNS Reserve Calculation
Author: Converted from R to Python
Date: 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import torch
from torch.utils.data import Dataset, DataLoader


class ClaimsDataProcessor:
    """Process claims data for neural network training."""
    
    def __init__(self, data_path: str = "./Data/data.txt"):
        self.data_path = data_path
        self.data_all = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the claims data from file."""
        self.data_all = pd.read_csv(self.data_path, sep=';')
        return self.data_all
    
    def get_available_lobs(self) -> List[int]:
        """Get list of available Lines of Business in the data."""
        if self.data_all is None:
            self.load_data()
        return sorted(self.data_all['LoB'].unique().tolist())
    
    def process_lob_data(self, lob: int) -> Tuple[pd.DataFrame, Dict]:
        """Process data for a specific line of business."""
        
        if self.data_all is None:
            self.load_data()
        
        # Check if LoB exists in data
        if lob not in self.data_all['LoB'].unique():
            raise ValueError(f"LoB {lob} not found in data. Available LoB values: {sorted(self.data_all['LoB'].unique())}")
        
        # Filter data for the specified LoB
        data = self.data_all[self.data_all['LoB'] == lob].copy()
        
        if len(data) == 0:
            raise ValueError(f"No data found for LoB {lob}")
        
        # Convert to string first to avoid categorical issues
        data['LoB'] = data['LoB'].astype(str)
        data['cc'] = data['cc'].astype(str)
        data['inj_part'] = data['inj_part'].astype(str)
        
        # Cap reporting delay at 2
        data['RepDelnew'] = np.minimum(2, data['RepDel'])
        
        # Create categorical encodings
        data['LoBx'] = 0  # Since we're filtering by LoB, this is always 0
        
        # Create cc categorical encoding
        cc_values = sorted([str(v) for v in data['cc'].dropna().unique() if str(v) != 'nan'])
        cc_transform = {val: i for i, val in enumerate(cc_values)}
        # Map and handle missing values before converting to int
        data['ccx'] = data['cc'].map(cc_transform)
        data['ccx'] = data['ccx'].fillna(0).astype(int)
        
        # AY categorical (shifted to start from 0)
        data['AYx'] = data['AY'] - 1994
        
        # AQ categorical (shifted to start from 0)
        data['AQx'] = data['AQ'] - 1
        
        # Age categorical (5-year buckets)
        age_bins = [14, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
        data['agex'] = pd.cut(data['age'], bins=age_bins, right=True, labels=range(11))
        data['agex'] = data['agex'].fillna(0).astype(int)
        
        # inj_part categorical encoding
        inj_part_values = sorted([str(v) for v in data['inj_part'].dropna().unique() if str(v) != 'nan'])
        inj_part_transform = {val: i for i, val in enumerate(inj_part_values)}
        # Map and handle missing values before converting to int
        data['inj_partx'] = data['inj_part'].map(inj_part_transform)
        data['inj_partx'] = data['inj_partx'].fillna(0).astype(int)
        
        # RepDel categorical
        data['RepDelx'] = data['RepDelnew'].astype(int)
        
        # Time known at the end of 2005
        for i in range(12):
            col_name = f'Time_Known{i:02d}'
            data[col_name] = 0
            mask = (data['AY'] + data['RepDel']) <= (2006 - i)
            data.loc[mask, col_name] = 1
        
        # Past payment info (binned payment amounts)
        payment_bins = [-np.inf, -1, 0, 5000, 20000, 100000, np.inf]
        payment_labels = [1, 0, 2, 3, 4, 5]
        
        for i in range(11):
            pay_col = f'Pay{i:02d}'
            info_col = f'Pay{i:02d}Info'
            time_col = f'Time_Known{i:02d}'
            
            data[info_col] = 0
            mask = data[time_col] == 1
            if mask.any():
                data.loc[mask, info_col] = pd.cut(
                    data.loc[mask, pay_col],
                    bins=payment_bins,
                    right=True,
                    labels=payment_labels
                ).astype(int)
        
        # Payment indicators
        for i in range(12):
            pay_col = f'Pay{i:02d}'
            ind_col = f'PayInd{i:02d}'
            time_col = f'Time_Known{i:02d}'
            
            data[ind_col] = 0
            mask = data[time_col] == 1
            if mask.any():
                data.loc[mask, ind_col] = (data.loc[mask, pay_col] > 0).astype(int)
        
        # Logarithmic payments
        for i in range(12):
            pay_col = f'Pay{i:02d}'
            log_col = f'LogPay{i:02d}'
            time_col = f'Time_Known{i:02d}'
            
            data[log_col] = 0.0
            mask = (data[time_col] == 1) & (data[pay_col] > 0)
            if mask.any():
                data.loc[mask, log_col] = np.log(data.loc[mask, pay_col])
        
        # Time points for prediction (training)
        for i in range(12):
            pred_ind_col = f'Time_Predict_Indicator{i:02d}'
            pred_pay_col = f'Time_Predict_Payment{i:02d}'
            time_col = f'Time_Known{i:02d}'
            ind_col = f'PayInd{i:02d}'
            
            data[pred_ind_col] = data[time_col]
            data[pred_pay_col] = data[ind_col]
        
        # Calculate starting values
        starting_values = np.zeros((12, 2))
        for i in range(12):
            mask = (data['AY'] + data['RepDel']) <= (2006 - i)
            if mask.any():
                pay_col = f'Pay{i:02d}'
                # Payment probability
                starting_values[i, 0] = (data.loc[mask, pay_col] > 0).mean()
                # Mean log payment (for positive payments)
                positive_mask = mask & (data[pay_col] > 0)
                if positive_mask.any():
                    starting_values[i, 1] = np.log(data.loc[positive_mask, pay_col]).mean()
        
        # Calculate network weights
        network_weights = np.zeros((12, 2))
        for i in range(12):
            mask = (data['AY'] + data['RepDel']) <= (2006 - i)
            pred_col = f'Time_Predict_Indicator{i:02d}'
            mask = mask & (data[pred_col] == 1)
            
            if mask.any():
                pay_col = f'Pay{i:02d}'
                # Weight for indicator (inverse of average binary cross-entropy)
                p = (data.loc[mask, pay_col] > 0).astype(float)
                p_mean = p.mean()
                if p_mean > 0 and p_mean < 1:
                    bce = -(p * np.log(p_mean) + (1 - p) * np.log(1 - p_mean))
                    network_weights[i, 0] = 1 / bce.mean()
                
                # Weight for mean (inverse of variance)
                positive_mask = mask & (data[pay_col] > 0)
                if positive_mask.any():
                    log_payments = np.log(data.loc[positive_mask, pay_col])
                    variance = ((log_payments - log_payments.mean()) ** 2).mean()
                    if variance > 0:
                        network_weights[i, 1] = 1 / variance
        
        # Replace infinities and NaNs with 1
        network_weights = np.nan_to_num(network_weights, nan=1.0, posinf=1.0, neginf=1.0)
        network_weights[network_weights == 0] = 1.0
        
        info = {
            'starting_values': starting_values,
            'network_weights': network_weights,
            'cc_transform': cc_transform,
            'inj_part_transform': inj_part_transform,
        }
        
        return data, info
    
    def prepare_features(self, data: pd.DataFrame, for_prediction: bool = False) -> Dict[str, np.ndarray]:
        """Prepare features for model input."""
        
        features = {
            'dropout': data['LoBx'].values.astype(np.int32),
            'cc': data['ccx'].values.astype(np.int32),
            'AY': data['AYx'].values.astype(np.int32),
            'AQ': data['AQx'].values.astype(np.int32),
            'age': data['agex'].values.astype(np.int32),
            'inj_part': data['inj_partx'].values.astype(np.int32),
            'RepDel': data['RepDelx'].values.astype(np.int32),
        }
        
        # Add time known features
        for i in range(12):
            if i > 0:
                features[f'Time_Known{i:02d}'] = data[f'Time_Known{i:02d}'].values
        
        # Add payment info features
        for i in range(11):
            features[f'Pay{i:02d}Info'] = data[f'Pay{i:02d}Info'].values.astype(np.int32)
        
        # Add time prediction features
        for i in range(12):
            features[f'Time_Predict_Indicator{i:02d}'] = data[f'Time_Predict_Indicator{i:02d}'].values
            features[f'Time_Predict_Payment{i:02d}'] = data[f'Time_Predict_Payment{i:02d}'].values
        
        if not for_prediction:
            # Add target variables for training
            for i in range(12):
                features[f'PayInd{i:02d}'] = data[f'PayInd{i:02d}'].values
                features[f'LogPay{i:02d}'] = data[f'LogPay{i:02d}'].values
        
        return features
    
    def prepare_prediction_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """Prepare data for prediction (inference)."""
        
        # Filter for prediction set (AY > 1994)
        data_pred = data[data['AY'] > 1994].copy()
        
        # Set up prediction time points
        for i in range(12):
            pred_ind_col = f'Time_Predict_Indicator{i:02d}'
            pred_pay_col = f'Time_Predict_Payment{i:02d}'
            time_col = f'Time_Known{i:02d}'
            
            # For prediction, we want to predict future time points
            data_pred[pred_ind_col] = 1 - data_pred[time_col]
            
            # Adjust for reporting delay
            for r in range(1, 11):
                mask = data_pred['RepDel'] == r
                if r <= i:
                    data_pred.loc[mask, pred_ind_col] = 0
            
            data_pred[pred_pay_col] = data_pred[pred_ind_col]
        
        features = self.prepare_features(data_pred, for_prediction=True)
        
        return data_pred, features


class ClaimsDataset(Dataset):
    """PyTorch dataset for claims data."""
    
    def __init__(self, features: Dict[str, np.ndarray]):
        self.features = features
        self.n_samples = len(next(iter(features.values())))
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        sample = {}
        for key, value in self.features.items():
            if 'Time' in key or 'LogPay' in key:
                # Time and LogPay should be float
                sample[key] = torch.tensor(value[idx], dtype=torch.float32)
            elif key in ['dropout', 'cc', 'AY', 'AQ', 'age', 'inj_part', 'RepDel'] or 'Info' in key or 'PayInd' in key:
                # Categorical features and indicators should be long
                sample[key] = torch.tensor(value[idx], dtype=torch.long)
            else:
                # Default to float
                sample[key] = torch.tensor(value[idx], dtype=torch.float32)
        return sample


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching."""
    collated = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if values[0].dim() == 0:
            collated[key] = torch.stack(values)
        else:
            collated[key] = torch.cat(values, dim=0)
    return collated