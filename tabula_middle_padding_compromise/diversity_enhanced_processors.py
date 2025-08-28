"""Enhanced Generation Processors with Distribution-Aware Categorical Sampling

This module enhances the existing generation processors to improve categorical diversity
while maintaining constraints:
1. Only generate categories seen during training (trie enforcement)
2. Maintain fast generation speed for 10000+ samples
3. Better match training distributions through temperature-based reweighting
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass
import math

from .advanced_generation_processors import (
    ColumnProfile, ColumnType, FlattenedTrie, ColumnValueProcessor
)


@dataclass
class CategoryDistribution:
    """Store training distribution for a categorical column."""
    value_counts: Dict[str, int]
    value_to_tokens: Dict[str, List[int]]  # Map values to token sequences
    token_to_value: Dict[Tuple[int, ...], str]  # Map token sequences to values
    probabilities: Dict[str, float]
    entropy: float
    
    def get_distribution_weights(self) -> Dict[str, float]:
        """Get sampling weights based on training distribution."""
        return self.probabilities.copy()


class DistributionAwareTrie(FlattenedTrie):
    """Enhanced trie that tracks value probabilities for distribution matching."""
    
    def __init__(self, tokenized_values: List[List[int]], value_strings: List[str] = None):
        """Build trie with distribution tracking.
        
        Args:
            tokenized_values: List of token ID sequences
            value_strings: Original string values (for distribution tracking)
        """
        super().__init__(tokenized_values)
        
        # Store the training sequences for distribution matching
        self._training_sequences = list(tokenized_values)
        self._value_strings = value_strings or [str(i) for i in range(len(tokenized_values))]
        
        # Compute value distribution
        self.value_distribution = self._compute_distribution()
        
        # Track which token sequences map to which values
        self.sequence_to_value = {}
        for tokens, value in zip(tokenized_values, self._value_strings):
            self.sequence_to_value[tuple(tokens)] = value
    
    def _compute_distribution(self) -> Dict[str, float]:
        """Compute probability distribution of values."""
        value_counts = Counter(self._value_strings)
        total = sum(value_counts.values())
        return {val: count/total for val, count in value_counts.items()}
    
    def get_weighted_allowed_tokens(self, 
                                   step: int, 
                                   prefix_tokens: List[int],
                                   temperature: float = 1.0) -> Tuple[Set[int], Dict[int, float]]:
        """Get allowed tokens with distribution-based weights.
        
        Args:
            step: Current generation step
            prefix_tokens: Tokens generated so far
            temperature: Temperature for distribution sharpening/smoothing
            
        Returns:
            Tuple of (allowed_tokens, token_weights)
        """
        allowed = self.get_allowed_tokens(step, prefix_tokens)
        
        if not allowed or temperature == 0:
            return allowed, {}
        
        # Compute weights based on which values each token leads to
        token_weights = defaultdict(float)
        prefix_tuple = tuple(prefix_tokens)
        
        for full_sequence in self._training_sequences:
            # Check if this sequence matches our prefix
            if (len(full_sequence) > step and 
                tuple(full_sequence[:step]) == prefix_tuple):
                
                # This sequence is compatible with our prefix
                if step < len(full_sequence):
                    next_token = full_sequence[step]
                    if next_token in allowed:
                        # Get the value this sequence represents
                        value = self.sequence_to_value.get(tuple(full_sequence), "")
                        value_prob = self.value_distribution.get(value, 1.0 / len(self._training_sequences))
                        
                        # Add weighted by value probability
                        token_weights[next_token] += value_prob
        
        # Apply temperature scaling
        if token_weights and temperature != 1.0:
            # Convert to numpy for easier manipulation
            tokens = list(token_weights.keys())
            weights = np.array([token_weights[t] for t in tokens])
            
            # Apply temperature (higher temp = more uniform)
            if temperature > 0:
                weights = np.power(weights, 1.0 / temperature)
                weights = weights / weights.sum()
                
                token_weights = {t: w for t, w in zip(tokens, weights)}
        
        return allowed, token_weights


class DiversityEnhancedColumnValueProcessor(ColumnValueProcessor):
    """Enhanced processor with distribution-aware categorical sampling."""
    
    def __init__(self,
                 profiles: Dict[str, ColumnProfile],
                 special_ids: Dict[str, int],
                 columns: List[str],
                 tokenizer=None,
                 categorical_temperature: float = 1.2,
                 small_categorical_temperature: float = 1.5,
                 use_distribution_matching: bool = True):
        """Initialize enhanced processor.
        
        Args:
            profiles: Column profiles with distribution info
            special_ids: Special token IDs
            columns: Column names
            tokenizer: Tokenizer for debugging
            categorical_temperature: Temperature for categorical sampling
            small_categorical_temperature: Temperature for small categoricals
            use_distribution_matching: Whether to match training distributions
        """
        super().__init__(profiles, special_ids, columns, tokenizer)
        
        self.categorical_temperature = categorical_temperature
        self.small_categorical_temperature = small_categorical_temperature
        self.use_distribution_matching = use_distribution_matching
        
        # Track generation history for adaptive sampling
        self.generation_history = {col: Counter() for col in columns}
        self.total_generated = {col: 0 for col in columns}
        
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Process logits with distribution-aware sampling for categoricals.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            scores: Raw logits [batch_size, vocab_size]
            
        Returns:
            Processed scores with diversity enhancements
        """
        # Get current column
        current_col = self.column_state.current_column
        if current_col is None or current_col not in self.profiles:
            return super().__call__(input_ids, scores)
        
        profile = self.profiles[current_col]
        
        # Apply base processing first
        scores = super().__call__(input_ids, scores)
        
        # Apply distribution-aware enhancements for categorical columns
        if profile.column_type in [ColumnType.SMALL_CATEGORICAL, ColumnType.LARGE_CATEGORICAL]:
            scores = self._apply_categorical_enhancements(scores, profile, current_col)
        
        return scores
    
    def _apply_categorical_enhancements(self, 
                                       scores: torch.Tensor,
                                       profile: ColumnProfile,
                                       column: str) -> torch.Tensor:
        """Apply distribution-aware enhancements for categorical columns.
        
        Args:
            scores: Current scores [batch_size, vocab_size]
            profile: Column profile
            column: Column name
            
        Returns:
            Enhanced scores
        """
        # Determine temperature based on column type
        if profile.column_type == ColumnType.SMALL_CATEGORICAL:
            temperature = self.small_categorical_temperature
        else:
            temperature = self.categorical_temperature
        
        # Apply temperature scaling for diversity
        if temperature != 1.0:
            scores = scores / temperature
        
        # Apply distribution matching if we have a trie with distribution info
        if self.use_distribution_matching and hasattr(profile.trie, 'get_weighted_allowed_tokens'):
            scores = self._apply_distribution_weights(scores, profile, temperature)
        
        # Apply adaptive sampling based on generation history
        if self.total_generated[column] > 0:
            scores = self._apply_adaptive_weights(scores, column, profile)
        
        return scores
    
    def _apply_distribution_weights(self,
                                   scores: torch.Tensor,
                                   profile: ColumnProfile,
                                   temperature: float) -> torch.Tensor:
        """Apply weights based on training distribution.
        
        Args:
            scores: Current scores
            profile: Column profile with trie
            temperature: Current temperature
            
        Returns:
            Weighted scores
        """
        if not hasattr(self, '_trie_cursor') or self._trie_cursor is None:
            return scores
        
        # Get weighted allowed tokens from trie
        if isinstance(profile.trie, DistributionAwareTrie):
            allowed, weights = profile.trie.get_weighted_allowed_tokens(
                self._trie_cursor.step,
                self._trie_cursor.prefix_tokens,
                temperature
            )
            
            # Apply weights to scores
            if weights:
                for token_id, weight in weights.items():
                    # Boost tokens based on their distribution weight
                    # Use log-space addition to maintain numerical stability
                    boost = math.log(max(weight, 1e-10))
                    scores[:, token_id] += boost
        
        return scores
    
    def _apply_adaptive_weights(self,
                               scores: torch.Tensor,
                               column: str,
                               profile: ColumnProfile) -> torch.Tensor:
        """Apply adaptive weights based on generation history.
        
        Args:
            scores: Current scores
            column: Column name
            profile: Column profile
            
        Returns:
            Adaptively weighted scores
        """
        if not self.generation_history[column]:
            return scores
        
        # Get target distribution from profile
        if hasattr(profile.trie, 'value_distribution'):
            target_dist = profile.trie.value_distribution
            
            # Compute current generation distribution
            current_dist = {}
            total = self.total_generated[column]
            for value, count in self.generation_history[column].items():
                current_dist[value] = count / total if total > 0 else 0
            
            # Apply corrective weights to move toward target distribution
            # This is done at the token level if we have the mapping
            if hasattr(profile.trie, 'sequence_to_value'):
                for token_seq, value in profile.trie.sequence_to_value.items():
                    if value in target_dist:
                        target_prob = target_dist[value]
                        current_prob = current_dist.get(value, 0)
                        
                        # Compute corrective weight
                        if current_prob > 0:
                            # If overrepresented, reduce probability
                            if current_prob > target_prob * 1.2:  # 20% tolerance
                                penalty = math.log(current_prob / target_prob)
                                # Apply penalty to first token of this sequence
                                if len(token_seq) > 0:
                                    scores[:, token_seq[0]] -= penalty * 0.5
                            # If underrepresented, boost probability
                            elif current_prob < target_prob * 0.8:
                                boost = math.log(target_prob / (current_prob + 0.01))
                                if len(token_seq) > 0:
                                    scores[:, token_seq[0]] += boost * 0.5
        
        return scores
    
    def update_generation_history(self, column: str, value: str):
        """Update generation history after value is generated.
        
        Args:
            column: Column name
            value: Generated value
        """
        self.generation_history[column][value] += 1
        self.total_generated[column] += 1


def enhance_column_profiler_with_distribution(profiler_class):
    """Enhance the ColumnProfiler to track distributions.
    
    Args:
        profiler_class: Base ColumnProfiler class
        
    Returns:
        Enhanced profiler class
    """
    
    class DistributionAwareProfiler(profiler_class):
        """Profiler that tracks value distributions for diversity."""
        
        def profile_column(self, 
                          column_name: str,
                          values: List[str],
                          special_ids: Dict[str, int],
                          column_name_tokens: Set[int]) -> ColumnProfile:
            """Profile column with distribution tracking.
            
            Args:
                column_name: Name of column
                values: List of string values
                special_ids: Special token IDs
                column_name_tokens: Tokens from column name
                
            Returns:
                Enhanced ColumnProfile with distribution info
            """
            # Get base profile
            profile = super().profile_column(column_name, values, special_ids, column_name_tokens)
            
            # Enhance trie with distribution awareness for categoricals
            if profile.column_type in [ColumnType.SMALL_CATEGORICAL, ColumnType.LARGE_CATEGORICAL]:
                if profile.trie is not None:
                    # Rebuild trie with distribution tracking
                    tokenized_values = []
                    for val in values:
                        tokens = self.tokenizer.encode(str(val), add_special_tokens=False)
                        tokenized_values.append(tokens)
                    
                    # Create distribution-aware trie
                    profile.trie = DistributionAwareTrie(tokenized_values, values)
                    
                    # Add distribution stats to profile
                    value_counts = Counter(values)
                    total = len(values)
                    profile.value_distribution = {
                        val: count/total for val, count in value_counts.items()
                    }
                    
                    # Compute entropy for diversity monitoring
                    profile.entropy = -sum(
                        p * math.log(p + 1e-10) 
                        for p in profile.value_distribution.values()
                    )
            
            return profile
    
    return DistributionAwareProfiler


# Fast generation optimization for 10000+ samples
class BatchOptimizedProcessor(DiversityEnhancedColumnValueProcessor):
    """Optimized processor for fast batch generation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Pre-compute masks for common operations
        self._precompute_masks()
        
    def _precompute_masks(self):
        """Pre-compute masks for faster batch processing."""
        vocab_size = 50000  # Typical vocab size, adjust as needed
        
        # Pre-compute special token mask
        self.special_mask = torch.ones(vocab_size, dtype=torch.bool)
        for token_id in self.special_ban_set:
            if token_id < vocab_size:
                self.special_mask[token_id] = False
        
        # Pre-compute name token mask
        self.name_mask = torch.ones(vocab_size, dtype=torch.bool)
        for token_id in self.all_name_tokens:
            if token_id < vocab_size:
                self.name_mask[token_id] = False
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Optimized call for batch processing.
        
        Args:
            input_ids: Input IDs [batch_size, seq_len]
            scores: Logits [batch_size, vocab_size]
            
        Returns:
            Processed scores
        """
        batch_size, vocab_size = scores.shape
        device = scores.device
        
        # Move pre-computed masks to device if needed
        if self.special_mask.device != device:
            self.special_mask = self.special_mask.to(device)
            self.name_mask = self.name_mask.to(device)
        
        # Apply masks in batch (faster than iterating)
        if self.special_mask.shape[0] == vocab_size:
            scores = torch.where(
                self.special_mask.unsqueeze(0).expand(batch_size, -1),
                scores,
                torch.tensor(float('-inf'), device=device)
            )
        
        # Apply column-specific processing
        return super().__call__(input_ids, scores)