"""Strategies to Improve Categorical Diversity in Tabula Generation

This module provides several strategies to address the issue of models 
always generating the same category for certain variables.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
from collections import Counter
from dataclasses import dataclass
import pandas as pd


@dataclass
class DiversityConfig:
    """Configuration for categorical diversity strategies."""
    
    # Temperature scaling for categorical columns
    categorical_temperature: float = 1.2  # Higher = more diverse
    small_categorical_temperature: float = 1.5  # Even higher for small cats
    
    # Top-p (nucleus) sampling for categoricals  
    categorical_top_p: float = 0.95  # Keep top 95% probability mass
    
    # Top-k sampling as fallback
    categorical_top_k: int = 10  # Consider top 10 options
    
    # Frequency-based reweighting
    use_frequency_penalty: bool = True
    frequency_penalty: float = 0.5  # Penalty for overused categories
    
    # Training augmentations
    use_class_balancing: bool = True  # Balance loss by class frequency
    use_mixup: bool = False  # Mixup augmentation for training
    mixup_alpha: float = 0.2
    
    # Generation constraints
    enforce_minimum_diversity: bool = True
    min_unique_ratio: float = 0.3  # At least 30% of categories should appear
    
    # Curriculum learning
    use_curriculum: bool = True
    initial_temperature: float = 1.0
    final_temperature: float = 1.5
    warmup_steps: int = 1000


class CategoricalDiversityProcessor:
    """Enhanced processor for diverse categorical generation."""
    
    def __init__(self, 
                 column_stats: Dict[str, Dict],
                 config: DiversityConfig = None):
        """Initialize diversity processor.
        
        Args:
            column_stats: Statistics about categorical distributions per column
            config: Diversity configuration
        """
        self.column_stats = column_stats
        self.config = config or DiversityConfig()
        self.generation_history = {col: Counter() for col in column_stats}
        
    def compute_category_weights(self, column: str) -> Dict[str, float]:
        """Compute inverse frequency weights for categories.
        
        Args:
            column: Column name
            
        Returns:
            Dict mapping category to weight (rare categories get higher weight)
        """
        if column not in self.column_stats:
            return {}
            
        counts = self.column_stats[column].get('value_counts', {})
        total = sum(counts.values())
        
        if total == 0:
            return {}
            
        # Inverse frequency weighting with smoothing
        weights = {}
        for cat, count in counts.items():
            freq = count / total
            # Weight = 1 / (frequency + smoothing)
            weights[cat] = 1.0 / (freq + 0.01)
            
        # Normalize weights
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            weights = {k: v/weight_sum for k, v in weights.items()}
            
        return weights
    
    def apply_temperature_scaling(self, 
                                  logits: torch.Tensor,
                                  temperature: float) -> torch.Tensor:
        """Apply temperature scaling to logits.
        
        Args:
            logits: Raw model logits [batch_size, vocab_size]
            temperature: Temperature value (>1 increases diversity)
            
        Returns:
            Scaled logits
        """
        if temperature == 1.0:
            return logits
            
        return logits / temperature
    
    def apply_top_p_filtering(self,
                             logits: torch.Tensor,
                             top_p: float) -> torch.Tensor:
        """Apply nucleus (top-p) filtering.
        
        Args:
            logits: Model logits [batch_size, vocab_size]
            top_p: Cumulative probability threshold
            
        Returns:
            Filtered logits with low-probability tokens masked
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 0] = False
        
        # Scatter back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def apply_frequency_penalty(self,
                               logits: torch.Tensor,
                               column: str,
                               token_to_category: Dict[int, str]) -> torch.Tensor:
        """Apply penalty based on generation history.
        
        Args:
            logits: Model logits [batch_size, vocab_size]
            column: Current column name
            token_to_category: Mapping from token IDs to category values
            
        Returns:
            Logits with frequency penalty applied
        """
        if not self.config.use_frequency_penalty:
            return logits
            
        history = self.generation_history.get(column, Counter())
        if not history:
            return logits
            
        # Apply penalty based on how often each category was generated
        total_generated = sum(history.values())
        if total_generated == 0:
            return logits
            
        for token_id, category in token_to_category.items():
            if category in history:
                freq = history[category] / total_generated
                penalty = self.config.frequency_penalty * freq
                logits[:, token_id] -= penalty
                
        return logits
    
    def update_generation_history(self, column: str, generated_value: str):
        """Update history of generated values.
        
        Args:
            column: Column name
            generated_value: Generated category value
        """
        if column in self.generation_history:
            self.generation_history[column][generated_value] += 1


class TrainingDiversityEnhancer:
    """Enhance training for better categorical diversity."""
    
    def __init__(self, config: DiversityConfig = None):
        """Initialize training enhancer.
        
        Args:
            config: Diversity configuration
        """
        self.config = config or DiversityConfig()
        
    def compute_balanced_loss(self,
                             logits: torch.Tensor,
                             labels: torch.Tensor,
                             class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute class-balanced loss for categorical columns.
        
        Args:
            logits: Model predictions [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]
            class_weights: Per-class weights [vocab_size]
            
        Returns:
            Balanced loss value
        """
        if not self.config.use_class_balancing or class_weights is None:
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100
            )
        
        # Apply class weights
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            weight=class_weights,
            ignore_index=-100,
            reduction='none'
        )
        
        # Mask out padding
        mask = (labels != -100).float().reshape(-1)
        loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        
        return loss
    
    def apply_mixup(self,
                   inputs: Dict[str, torch.Tensor],
                   alpha: float = None) -> Dict[str, torch.Tensor]:
        """Apply mixup augmentation to training batch.
        
        Args:
            inputs: Training batch with input_ids, attention_mask, labels
            alpha: Mixup interpolation parameter
            
        Returns:
            Mixed batch
        """
        if not self.config.use_mixup:
            return inputs
            
        alpha = alpha or self.config.mixup_alpha
        batch_size = inputs['input_ids'].size(0)
        
        if batch_size < 2:
            return inputs
            
        # Sample lambda from Beta distribution
        lam = np.random.beta(alpha, alpha)
        
        # Random permutation for mixing
        index = torch.randperm(batch_size)
        
        # Mix embeddings (would need model's embedding layer)
        # For now, return unmixed - implement in trainer
        return inputs
    
    def get_curriculum_temperature(self, current_step: int) -> float:
        """Get temperature based on curriculum learning schedule.
        
        Args:
            current_step: Current training step
            
        Returns:
            Temperature value for current step
        """
        if not self.config.use_curriculum:
            return self.config.categorical_temperature
            
        if current_step >= self.config.warmup_steps:
            return self.config.final_temperature
            
        # Linear warmup
        progress = current_step / self.config.warmup_steps
        temp_range = self.config.final_temperature - self.config.initial_temperature
        return self.config.initial_temperature + progress * temp_range


class AdaptiveSamplingStrategy:
    """Adaptive sampling strategy that adjusts based on generated distribution."""
    
    def __init__(self, target_distribution: Dict[str, float]):
        """Initialize adaptive sampler.
        
        Args:
            target_distribution: Target distribution for categories
        """
        self.target_distribution = target_distribution
        self.generated_counts = Counter()
        self.total_generated = 0
        
    def get_sampling_weights(self) -> Dict[str, float]:
        """Get sampling weights to match target distribution.
        
        Returns:
            Dict mapping category to sampling weight
        """
        if self.total_generated == 0:
            # Initial weights = target distribution
            return self.target_distribution.copy()
            
        weights = {}
        for cat, target_prob in self.target_distribution.items():
            current_prob = self.generated_counts.get(cat, 0) / self.total_generated
            
            # Weight = target / current (with smoothing)
            if current_prob > 0:
                weights[cat] = target_prob / (current_prob + 0.01)
            else:
                weights[cat] = target_prob * 2.0  # Boost unseen categories
                
        # Normalize
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            weights = {k: v/weight_sum for k, v in weights.items()}
            
        return weights
    
    def update(self, generated_category: str):
        """Update with newly generated category.
        
        Args:
            generated_category: Category that was generated
        """
        self.generated_counts[generated_category] += 1
        self.total_generated += 1


def create_diversity_enhanced_trainer(base_trainer_class):
    """Create trainer class with diversity enhancements.
    
    Args:
        base_trainer_class: Base trainer class to extend
        
    Returns:
        Enhanced trainer class
    """
    
    class DiversityEnhancedTrainer(base_trainer_class):
        """Trainer with categorical diversity enhancements."""
        
        def __init__(self, *args, diversity_config: DiversityConfig = None, **kwargs):
            super().__init__(*args, **kwargs)
            self.diversity_config = diversity_config or DiversityConfig()
            self.diversity_enhancer = TrainingDiversityEnhancer(self.diversity_config)
            
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """Compute loss with diversity enhancements."""
            
            # Get base loss
            outputs = model(**inputs)
            logits = outputs.get("logits")
            labels = inputs.get("labels")
            
            if labels is not None and logits is not None:
                # Apply curriculum temperature
                current_temp = self.diversity_enhancer.get_curriculum_temperature(
                    self.state.global_step
                )
                
                # Scale logits by temperature during training
                if self.training and current_temp != 1.0:
                    logits = logits / current_temp
                    
                # Compute balanced loss
                loss = self.diversity_enhancer.compute_balanced_loss(
                    logits[..., :-1, :].contiguous(),
                    labels[..., 1:].contiguous()
                )
                
                outputs['loss'] = loss
                
            return (loss, outputs) if return_outputs else loss
            
    return DiversityEnhancedTrainer


# Practical usage example
def enhance_generation_with_diversity(
    tabula_model,
    n_samples: int,
    diversity_config: DiversityConfig = None
) -> pd.DataFrame:
    """Generate samples with enhanced diversity.
    
    Args:
        tabula_model: Trained Tabula model
        n_samples: Number of samples to generate
        diversity_config: Diversity configuration
        
    Returns:
        DataFrame with generated samples
    """
    config = diversity_config or DiversityConfig()
    
    # Override sampling parameters
    original_temp = tabula_model.temperature if hasattr(tabula_model, 'temperature') else 1.0
    
    try:
        # Set diversity-enhancing parameters
        if hasattr(tabula_model, 'temperature'):
            tabula_model.temperature = config.categorical_temperature
            
        # Generate with diversity strategies
        samples = tabula_model.sample(
            n_samples=n_samples,
            temperature=config.categorical_temperature,
            # Add top_p/top_k if supported by model
        )
        
        return samples
        
    finally:
        # Restore original settings
        if hasattr(tabula_model, 'temperature'):
            tabula_model.temperature = original_temp