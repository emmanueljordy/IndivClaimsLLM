"""
Improved Neural Network Models with Better Regularization and Adaptive Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class ImprovedEmbeddingNetwork(nn.Module):
    """
    Improved neural network with better regularization and adaptive mechanisms.
    """
    
    def __init__(self, 
                 neurons: list,
                 dropout_rates: list,
                 starting_values: np.ndarray,
                 network_weights: np.ndarray,
                 seed: Optional[int] = None,
                 mode: str = 'train_embedding',
                 use_batch_norm: bool = True,
                 use_residual: bool = True,
                 activation: str = 'leakyrelu'):
        """
        Initialize the improved embedding network.
        
        Args:
            neurons: List of neurons in hidden layers
            dropout_rates: Dropout rates for regularization
            starting_values: Starting values for output layers
            network_weights: Weights for loss calculation
            seed: Random seed for reproducibility
            mode: Operating mode ('train_embedding', 'train_network', 'fixed_embedding')
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
            activation: Activation function ('relu', 'leakyrelu', 'elu', 'selu')
        """
        super().__init__()
        
        if seed is not None:
            torch.manual_seed(seed)
            
        self.mode = mode
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        self.neurons = neurons
        self.network_weights = torch.tensor(network_weights, dtype=torch.float32)
        
        # Choose activation function
        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        else:
            self.activation = nn.ReLU()
        
        # Embedding layers with adaptive sizes
        self.embedding_dropout = nn.Embedding(1, 64)
        self.embedding_cc = nn.Embedding(100, 32)
        self.embedding_AY = nn.Embedding(20, 16)
        self.embedding_AQ = nn.Embedding(5, 8)
        self.embedding_age = nn.Embedding(20, 16)
        self.embedding_inj_part = nn.Embedding(30, 16)
        self.embedding_RepDel = nn.Embedding(12, 8)
        
        # Payment info embeddings with shared weights for similar features
        self.embedding_pay_info = nn.ModuleList([
            nn.Embedding(7, 8) for _ in range(11)
        ])
        
        # Calculate input size
        self.embedding_size = 64 + 32 + 16 + 8 + 16 + 16 + 8 + 11 * 8 + 11
        
        # Build shared feature extraction layers
        self.shared_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList()
        
        prev_size = self.embedding_size
        for i, hidden_size in enumerate(neurons):
            self.shared_layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            self.dropouts.append(nn.Dropout(dropout_rates[i] if i < len(dropout_rates) else 0.1))
            prev_size = hidden_size
        
        # Output layers for each time point with weight initialization
        self.output_layers = nn.ModuleList()
        for t in range(12):
            # Payment indicator head
            indicator_head = nn.Sequential(
                nn.Linear(prev_size, prev_size // 2),
                self.activation,
                nn.Dropout(0.1),
                nn.Linear(prev_size // 2, 1)
            )
            
            # Payment mean head  
            mean_head = nn.Sequential(
                nn.Linear(prev_size, prev_size // 2),
                self.activation,
                nn.Dropout(0.1),
                nn.Linear(prev_size // 2, 1)
            )
            
            # Initialize with starting values
            with torch.no_grad():
                indicator_head[-1].bias.data = torch.tensor([starting_values[t, 0]], dtype=torch.float32)
                mean_head[-1].bias.data = torch.tensor([starting_values[t, 1]], dtype=torch.float32)
            
            self.output_layers.append(nn.ModuleDict({
                'indicator': indicator_head,
                'mean': mean_head
            }))
        
        # Auxiliary prediction head for multi-task learning
        self.auxiliary_head = nn.Linear(prev_size, 1)
        
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass with improved regularization."""
        
        # Get embeddings
        embeddings = []
        embeddings.append(self.embedding_dropout(x['dropout'].long()).squeeze(1))
        embeddings.append(self.embedding_cc(x['cc'].long()).squeeze(1))
        embeddings.append(self.embedding_AY(x['AY'].long()).squeeze(1))
        embeddings.append(self.embedding_AQ(x['AQ'].long()).squeeze(1))
        embeddings.append(self.embedding_age(x['age'].long()).squeeze(1))
        embeddings.append(self.embedding_inj_part(x['inj_part'].long()).squeeze(1))
        embeddings.append(self.embedding_RepDel(x['RepDel'].long()).squeeze(1))
        
        # Payment info embeddings
        for i in range(11):
            key = f'Pay{i:02d}Info'
            embeddings.append(self.embedding_pay_info[i](x[key].long()).squeeze(1))
        
        # Time known features
        for i in range(1, 12):
            key = f'Time_Known{i:02d}'
            embeddings.append(x[key].unsqueeze(-1) if x[key].dim() == 1 else x[key])
        
        # Concatenate all embeddings
        h = torch.cat(embeddings, dim=-1)
        
        # Apply shared layers with residual connections
        for i, (layer, dropout) in enumerate(zip(self.shared_layers, self.dropouts)):
            h_prev = h
            h = layer(h)
            
            if self.batch_norms is not None:
                h = self.batch_norms[i](h)
            
            h = self.activation(h)
            h = dropout(h)
            
            # Residual connection if dimensions match
            if self.use_residual and h_prev.shape[-1] == h.shape[-1]:
                h = h + h_prev * 0.3
        
        # Generate outputs for each time point
        outputs = {}
        
        for t in range(12):
            # Get prediction indicators
            pred_ind_key = f'Time_Predict_Indicator{t:02d}'
            pred_pay_key = f'Time_Predict_Payment{t:02d}'
            
            # Payment indicator prediction
            ind_out = self.output_layers[t]['indicator'](h)
            ind_out = torch.sigmoid(ind_out)
            
            # Apply time mask
            ind_out = ind_out * x[pred_ind_key].unsqueeze(-1)
            outputs[f'payment_indicator{t:02d}_output'] = ind_out.squeeze(-1)
            
            # Payment mean prediction
            mean_out = self.output_layers[t]['mean'](h)
            
            # Apply time mask and add regularization
            mean_out = mean_out * x[pred_pay_key].unsqueeze(-1)
            
            # Add slight dampening to prevent extreme predictions
            mean_out = torch.tanh(mean_out / 10) * 10
            
            outputs[f'payment_mean{t:02d}_output'] = mean_out.squeeze(-1)
        
        # Add auxiliary output for regularization
        outputs['auxiliary'] = self.auxiliary_head(h).squeeze(-1)
        
        return outputs


class AdaptiveLoss(nn.Module):
    """
    Adaptive loss function that adjusts based on data characteristics.
    """
    
    def __init__(self, network_weights: torch.Tensor, alpha: float = 0.5):
        super().__init__()
        self.network_weights = network_weights
        self.alpha = alpha  # Balance between indicator and mean loss
        
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor],
                epoch: int = 0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate adaptive loss with dynamic weighting.
        """
        
        total_loss = 0
        loss_components = {}
        
        for t in range(12):
            # Get predictions and targets
            ind_pred = outputs[f'payment_indicator{t:02d}_output']
            mean_pred = outputs[f'payment_mean{t:02d}_output']
            
            ind_target = targets[f'PayInd{t:02d}']
            log_pay_target = targets[f'LogPay{t:02d}']
            
            # Indicator loss (BCE with label smoothing)
            eps = 0.05  # Label smoothing
            ind_target_smooth = ind_target * (1 - eps) + eps / 2
            ind_loss = F.binary_cross_entropy(ind_pred, ind_target_smooth, reduction='none')
            
            # Mean loss (Huber loss for robustness)
            mask = ind_target > 0
            if mask.sum() > 0:
                mean_loss = F.smooth_l1_loss(
                    mean_pred[mask], 
                    log_pay_target[mask], 
                    reduction='none',
                    beta=1.0
                )
            else:
                mean_loss = torch.tensor(0.0, device=mean_pred.device)
            
            # Apply network weights with adaptive scaling
            weight_ind = self.network_weights[t, 0].to(ind_pred.device)
            weight_mean = self.network_weights[t, 1].to(mean_pred.device)
            
            # Dynamic weight adjustment based on epoch
            epoch_factor = min(1.0, epoch / 20)  # Gradually increase mean loss weight
            
            # Combine losses
            weighted_ind_loss = ind_loss.mean() * weight_ind
            weighted_mean_loss = mean_loss.mean() * weight_mean * epoch_factor if mask.sum() > 0 else 0
            
            time_loss = self.alpha * weighted_ind_loss + (1 - self.alpha) * weighted_mean_loss
            total_loss += time_loss
            
            loss_components[f'loss_t{t:02d}'] = time_loss.item()
        
        # Add auxiliary loss for regularization
        if 'auxiliary' in outputs:
            aux_loss = outputs['auxiliary'].abs().mean() * 0.01
            total_loss += aux_loss
            loss_components['aux_loss'] = aux_loss.item()
        
        return total_loss, loss_components


class ModelWithUncertainty(nn.Module):
    """
    Wrapper for model with uncertainty estimation using MC Dropout.
    """
    
    def __init__(self, base_model: nn.Module, n_samples: int = 10):
        super().__init__()
        self.base_model = base_model
        self.n_samples = n_samples
        
    def forward(self, x: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass with uncertainty estimation.
        """
        # Enable dropout for MC sampling
        self.base_model.train()
        
        predictions = []
        for _ in range(self.n_samples):
            with torch.no_grad():
                pred = self.base_model(x)
                predictions.append(pred)
        
        # Calculate mean and std
        mean_pred = {}
        std_pred = {}
        
        for key in predictions[0].keys():
            stacked = torch.stack([p[key] for p in predictions])
            mean_pred[key] = stacked.mean(dim=0)
            std_pred[key] = stacked.std(dim=0)
        
        return mean_pred, std_pred