"""
Neural Network Models for RBNS Reserve Calculation
Author: Converted from R to Python
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class EmbeddingNetwork(nn.Module):
    """Neural network with embedding layers for categorical features."""
    
    def __init__(
        self,
        neurons: List[int],
        dropout_rates: List[float],
        starting_values: np.ndarray,
        network_weights: np.ndarray,
        seed: Optional[int] = None,
        mode: str = 'train_embedding'
    ):
        super().__init__()
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.neurons = neurons
        self.dropout_rates = dropout_rates
        self.starting_values = starting_values
        self.network_weights = network_weights
        self.mode = mode
        
        # Embedding layers
        self.embeddings = nn.ModuleDict({
            'cc_input': nn.Embedding(51, 1),
            'cc_output': nn.Embedding(51, 1),
            'AY_input': nn.Embedding(12, 1),
            'AY_output': nn.Embedding(12, 1),
            'AQ_input': nn.Embedding(4, 1),
            'AQ_output': nn.Embedding(4, 1),
            'age_input': nn.Embedding(12, 1),
            'age_output': nn.Embedding(12, 1),
            'inj_part_input': nn.Embedding(46, 1),
            'inj_part_output': nn.Embedding(46, 1),
            'RepDel_input': nn.Embedding(3, 1),
            'RepDel_output': nn.Embedding(3, 1),
        })
        
        # Add payment info embeddings
        for i in range(11):
            self.embeddings[f'Pay{i:02d}Info_input'] = nn.Embedding(6, 1)
            self.embeddings[f'Pay{i:02d}Info_output'] = nn.Embedding(6, 1)
        
        # Dropout layers
        self.dropouts = nn.ModuleDict()
        for i in range(10):
            for j in range(10):
                self.dropouts[f'dropout_{i+1}_{j+1}'] = nn.Dropout(dropout_rates[i])
        
        # Dense layers for each time point
        self.time_networks = nn.ModuleList()
        for t in range(12):
            time_net = nn.ModuleDict()
            
            # Input feature size varies by time point
            input_size = 5 + min(t, 11)  # base features + payment history
            
            # Hidden layers
            time_net['hidden_pre'] = nn.Linear(input_size, neurons[0])
            time_net['AY_hidden'] = nn.Linear(1, neurons[0])
            time_net['hidden1'] = nn.Linear(neurons[0], neurons[0])
            time_net['hidden2'] = nn.Linear(neurons[0], neurons[1])
            
            # Output layers for indicator
            time_net['indicator_hidden'] = nn.Linear(neurons[1], neurons[2])
            output_size = neurons[2] + 5 + min(t, 11)
            time_net['indicator_pre'] = nn.Linear(output_size, 1)
            
            # Output layers for mean
            time_net['mean_hidden'] = nn.Linear(neurons[1], neurons[2])
            time_net['mean_pre'] = nn.Linear(output_size, 1)
            
            # AY adjustments
            time_net['AY_indicator'] = nn.Linear(1, 1, bias=True)
            time_net['AY_mean'] = nn.Linear(1, 1, bias=True)
            
            # Initialize with starting values
            with torch.no_grad():
                if starting_values[t, 0] > 0 and starting_values[t, 0] < 1:
                    logit = np.log(starting_values[t, 0] / (1 - starting_values[t, 0]))
                    time_net['indicator_pre'].bias.fill_(logit)
                time_net['mean_pre'].bias.fill_(starting_values[t, 1])
                
                # Initialize AY adjustment layers
                time_net['AY_indicator'].weight.fill_(0)
                time_net['AY_indicator'].bias.fill_(0)
                time_net['AY_mean'].weight.fill_(0)
                time_net['AY_mean'].bias.fill_(0)
            
            self.time_networks.append(time_net)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the network."""
        
        # Get embeddings
        embedded = {}
        for key in ['cc', 'AY', 'AQ', 'age', 'inj_part', 'RepDel']:
            embedded[f'{key}_input'] = self.embeddings[f'{key}_input'](features[key]).squeeze(-1)
            embedded[f'{key}_output'] = self.embeddings[f'{key}_output'](features[key]).squeeze(-1)
        
        # Get payment info embeddings
        for i in range(11):
            key = f'Pay{i:02d}Info'
            embedded[f'{key}_input'] = self.embeddings[f'{key}_input'](features[key]).squeeze(-1)
            embedded[f'{key}_output'] = self.embeddings[f'{key}_output'](features[key]).squeeze(-1)
        
        outputs = {}
        
        # Process each time point
        for t in range(12):
            # Prepare input features
            input_features = [
                embedded['cc_input'],
                embedded['AQ_input'],
                embedded['age_input'],
                embedded['inj_part_input'],
                embedded['RepDel_input']
            ]
            
            output_features = [
                embedded['cc_output'],
                embedded['AQ_output'],
                embedded['age_output'],
                embedded['inj_part_output'],
                embedded['RepDel_output']
            ]
            
            # Add payment history with dropout
            for i in range(min(t, 11)):
                pay_key = f'Pay{i:02d}Info'
                time_known_key = f'Time_Known{i+1:02d}'
                
                if self.training and self.mode == 'train_embedding':
                    # Apply dropout during training
                    dropout_mask = torch.rand_like(features[time_known_key]) > self.dropout_rates[min(i, 9)]
                    dropout_mask = dropout_mask.float()
                    
                    input_pay = embedded[f'{pay_key}_input'] * features[time_known_key] * dropout_mask
                    output_pay = embedded[f'{pay_key}_output'] * features[time_known_key] * dropout_mask
                else:
                    input_pay = embedded[f'{pay_key}_input'] * features[time_known_key]
                    output_pay = embedded[f'{pay_key}_output'] * features[time_known_key]
                
                input_features.append(input_pay)
                output_features.append(output_pay)
            
            # Concatenate features
            # Ensure all features are 2D tensors
            input_features_2d = []
            for feat in input_features:
                if feat.dim() == 1:
                    input_features_2d.append(feat.unsqueeze(-1))
                else:
                    input_features_2d.append(feat)
            
            output_features_2d = []
            for feat in output_features:
                if feat.dim() == 1:
                    output_features_2d.append(feat.unsqueeze(-1))
                else:
                    output_features_2d.append(feat)
            
            x_input = torch.cat(input_features_2d, dim=1)
            x_output = torch.cat(output_features_2d, dim=1)
            
            # Process through network
            time_net = self.time_networks[t]
            
            # Hidden layers
            h_pre = time_net['hidden_pre'](x_input)
            h_ay = time_net['AY_hidden'](embedded['AY_input'].unsqueeze(-1))
            h = torch.tanh(time_net['hidden1'](h_pre + h_ay.squeeze(-1)))
            h = torch.tanh(time_net['hidden2'](h))
            
            # Payment indicator
            ind_hidden = torch.tanh(time_net['indicator_hidden'](h))
            ind_features = torch.cat([ind_hidden, x_output], dim=1)
            ind_pre = time_net['indicator_pre'](ind_features)
            ind_ay = time_net['AY_indicator'](embedded['AY_output'].unsqueeze(-1))
            indicator = torch.sigmoid(ind_pre + ind_ay).squeeze(-1)
            
            # Payment mean
            mean_hidden = torch.tanh(time_net['mean_hidden'](h))
            mean_features = torch.cat([mean_hidden, x_output], dim=1)
            mean_pre = time_net['mean_pre'](mean_features)
            mean_ay = time_net['AY_mean'](embedded['AY_output'].unsqueeze(-1))
            mean = (mean_pre + mean_ay).squeeze(-1)
            
            # Apply time prediction masks
            time_pred_ind = features[f'Time_Predict_Indicator{t:02d}']
            time_pred_pay = features[f'Time_Predict_Payment{t:02d}']
            
            outputs[f'payment_indicator{t:02d}_output'] = indicator * time_pred_ind
            outputs[f'payment_mean{t:02d}_output'] = mean * time_pred_pay
        
        return outputs
    
    def load_embeddings(self, embedding_weights: Dict[int, np.ndarray]):
        """Load pre-trained embedding weights."""
        # This would be used to load embeddings from the first training step
        pass


class WeightedLoss(nn.Module):
    """Custom weighted loss function for the model."""
    
    def __init__(self, network_weights: np.ndarray):
        super().__init__()
        self.network_weights = torch.tensor(network_weights, dtype=torch.float32)
    
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate the weighted loss."""
        total_loss = 0
        
        for t in range(12):
            # Binary cross-entropy for payment indicator
            ind_pred = predictions[f'payment_indicator{t:02d}_output']
            ind_target = targets[f'PayInd{t:02d}']
            
            # Only calculate loss where we have data
            mask = targets[f'Time_Predict_Indicator{t:02d}'] > 0
            if mask.any():
                ind_loss = F.binary_cross_entropy(
                    ind_pred[mask],
                    ind_target[mask].float(),
                    reduction='mean'
                )
                total_loss += self.network_weights[t, 0] * ind_loss
            
            # MSE for payment mean (log scale)
            mean_pred = predictions[f'payment_mean{t:02d}_output']
            mean_target = targets[f'LogPay{t:02d}']
            
            mask = targets[f'Time_Predict_Payment{t:02d}'] > 0
            if mask.any():
                mean_loss = F.mse_loss(
                    mean_pred[mask],
                    mean_target[mask],
                    reduction='mean'
                )
                total_loss += self.network_weights[t, 1] * mean_loss
        
        return total_loss