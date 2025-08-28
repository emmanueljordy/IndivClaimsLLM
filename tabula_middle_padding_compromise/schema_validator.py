"""
Schema learning and validation system for tabular data generation.

This module provides dataset-agnostic schema learning and post-processing validation
to ensure generated synthetic data maintains the structural integrity of training data.
"""

import logging
import re
import typing as tp
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


class ColumnType(Enum):
    """Column classification types."""
    CATEGORICAL_SMALL = "categorical_small"
    FIXED_DIGITS = "fixed_digits"
    VARIABLE_DIGITS = "variable_digits"
    FREE_TEXT = "free_text"


@dataclass
class ColumnConstraint:
    """Constraint specification for a single column."""
    column_name: str
    column_type: ColumnType
    
    # Length constraints (in characters)
    min_len: int = 0
    max_len: int = 1000
    expected_len: Optional[int] = None  # For fixed-length columns
    
    # Value constraints
    lexicon: Optional[Set[str]] = None  # For categorical
    min_val: Optional[float] = None     # For numeric
    max_val: Optional[float] = None     # For numeric
    soft_min: Optional[float] = None    # 1st percentile
    soft_max: Optional[float] = None    # 99th percentile
    
    # Character class constraints
    digits_only: bool = False
    allow_leading_space: bool = True
    
    # Frequency distribution for resampling
    value_frequencies: Optional[Counter] = None
    
    # Token-level constraints
    lexicon_tokens: Optional[Dict[str, List[int]]] = None  # Pre-tokenized lexicon
    name_tokens: Optional[Set[int]] = None  # Column name tokens to ban


@dataclass
class TokenMetadata:
    """Precomputed metadata for each token in vocabulary."""
    token_str: str
    char_len: int
    is_digit_only: bool
    is_space_only: bool
    is_separator: bool
    contains_space: bool
    is_alphanumeric: bool


class SchemaLearner:
    """Learn column constraints from training data."""
    
    def __init__(self, tokenizer: AutoTokenizer, sep_tokens: Dict[str, int]):
        self.tokenizer = tokenizer
        self.sep_tokens = sep_tokens
        self.token_metadata: Dict[int, TokenMetadata] = {}
        self._precompute_token_metadata()
    
    def _precompute_token_metadata(self) -> None:
        """Precompute metadata for all tokens in vocabulary."""
        logging.info("🔧 Precomputing token metadata for schema validation...")
        
        vocab_size = len(self.tokenizer.get_vocab())
        separator_ids = set(self.sep_tokens.values())
        
        for token_id in range(vocab_size):
            try:
                token_str = self.tokenizer.decode([token_id], skip_special_tokens=True)
                
                self.token_metadata[token_id] = TokenMetadata(
                    token_str=token_str,
                    char_len=len(token_str),
                    is_digit_only=token_str.isdigit() and len(token_str) > 0,
                    is_space_only=token_str.isspace() and len(token_str) > 0,
                    is_separator=token_id in separator_ids,
                    contains_space=bool(re.search(r'\s', token_str)),
                    is_alphanumeric=token_str.isalnum() and len(token_str) > 0
                )
            except Exception as e:
                # Handle edge cases in tokenizer decoding
                logging.warning(f"Failed to decode token {token_id}: {e}")
                self.token_metadata[token_id] = TokenMetadata(
                    token_str="",
                    char_len=0,
                    is_digit_only=False,
                    is_space_only=False,
                    is_separator=False,
                    contains_space=False,
                    is_alphanumeric=False
                )
        
        logging.info(f"✅ Precomputed metadata for {vocab_size} tokens")
    
    def learn_column_constraints(self, df: pd.DataFrame) -> Dict[str, ColumnConstraint]:
        """Learn constraints for each column from training data."""
        logging.info("📊 Learning column constraints from training data...")
        
        constraints = {}
        
        for col_name in df.columns:
            values = df[col_name].astype(str).tolist()  # Convert all to strings
            constraints[col_name] = self._analyze_column(col_name, values)
        
        logging.info(f"✅ Learned constraints for {len(constraints)} columns")
        return constraints
    
    def _analyze_column(self, col_name: str, values: List[str]) -> ColumnConstraint:
        """Analyze a single column to determine its constraints."""
        N = len(values)
        unique_values = list(set(values))
        uniq_count = len(unique_values)
        uniq_ratio = uniq_count / max(N, 1)
        
        # Calculate basic statistics
        lengths = [len(v) for v in values]
        min_len, max_len = min(lengths), max(lengths)
        value_frequencies = Counter(values)
        
        # Classify column type
        col_type, type_params = self._classify_column_type(values, uniq_count, uniq_ratio)
        
        # Create constraint object
        constraint = ColumnConstraint(
            column_name=col_name,
            column_type=col_type,
            min_len=min_len,
            max_len=max_len,
            value_frequencies=value_frequencies
        )
        
        # Set type-specific parameters
        if col_type == ColumnType.CATEGORICAL_SMALL:
            constraint.lexicon = set(unique_values)
            constraint.lexicon_tokens = self._precompute_lexicon_tokens(unique_values)
        
        elif col_type in [ColumnType.FIXED_DIGITS, ColumnType.VARIABLE_DIGITS]:
            constraint.digits_only = True
            constraint.allow_leading_space = False
            
            # Extract numeric values for range calculation
            numeric_values = []
            for v in values:
                if v.isdigit():
                    numeric_values.append(float(v))
            
            if numeric_values:
                constraint.min_val = min(numeric_values)
                constraint.max_val = max(numeric_values)
                # Soft boundaries (1st-99th percentiles)
                constraint.soft_min = np.percentile(numeric_values, 1)
                constraint.soft_max = np.percentile(numeric_values, 99)
            
            if col_type == ColumnType.FIXED_DIGITS:
                constraint.expected_len = type_params.get("len", max_len)
        
        # Precompute column name tokens to ban during generation
        constraint.name_tokens = self._get_column_name_tokens(col_name)
        
        return constraint
    
    def _classify_column_type(self, values: List[str], uniq_count: int, uniq_ratio: float) -> Tuple[ColumnType, Dict]:
        """Classify column type based on value patterns."""
        N = len(values)
        
        # Rule 1: Low-cardinality categorical
        categorical_threshold = min(2048, max(64, int(0.05 * N)))
        if uniq_count <= categorical_threshold and uniq_ratio < 0.25:
            return ColumnType.CATEGORICAL_SMALL, {}
        
        # Rule 2: Digit-heavy columns
        digit_values = [v for v in values if v.strip().isdigit()]
        if len(digit_values) >= 0.9 * N:
            lengths = {len(v) for v in digit_values}
            
            if len(lengths) == 1:
                # All same length
                return ColumnType.FIXED_DIGITS, {"len": next(iter(lengths))}
            elif (max(lengths) - min(lengths)) <= 2:
                # Tight length range
                return ColumnType.VARIABLE_DIGITS, {
                    "min_len": min(lengths), 
                    "max_len": max(lengths)
                }
        
        # Rule 3: Everything else is free text
        return ColumnType.FREE_TEXT, {}
    
    def _precompute_lexicon_tokens(self, values: List[str]) -> Dict[str, List[int]]:
        """Precompute token sequences for lexicon values."""
        lexicon_tokens = {}
        
        for value in values:
            try:
                tokens = self.tokenizer.encode(value, add_special_tokens=False)
                lexicon_tokens[value] = tokens
            except Exception as e:
                logging.warning(f"Failed to tokenize lexicon value '{value}': {e}")
                lexicon_tokens[value] = []
        
        return lexicon_tokens
    
    def _get_column_name_tokens(self, col_name: str) -> Set[int]:
        """Get token IDs that appear in the column name."""
        try:
            name_tokens = self.tokenizer.encode(col_name, add_special_tokens=False)
            return set(name_tokens)
        except Exception:
            return set()


class PostValidator:
    """Post-processing validation and repair for generated cells."""
    
    def __init__(self, constraints: Dict[str, ColumnConstraint], 
                 cell_resample_max: int = 2, row_resample_cap: int = 8):
        self.constraints = constraints
        self.cell_resample_max = cell_resample_max
        self.row_resample_cap = row_resample_cap
        
        # Cross-column dependency rules (can be extended)
        self.dependency_rules = []
    
    def validate_and_repair_row(self, row_data: Dict[str, str], 
                               resample_callback: Optional[tp.Callable] = None) -> Dict[str, str]:
        """Validate and repair a complete row of generated data."""
        repaired_row = row_data.copy()
        resample_budget = self.row_resample_cap
        
        # Step 1: Validate each cell independently
        for col_name, value in row_data.items():
            if col_name not in self.constraints:
                continue
            
            constraint = self.constraints[col_name]
            cell_attempts = 0
            
            while cell_attempts < self.cell_resample_max and resample_budget > 0:
                if self._validate_cell(value, constraint):
                    break
                
                # Try to repair the cell
                repaired_value = self._repair_cell(value, constraint)
                if self._validate_cell(repaired_value, constraint):
                    repaired_row[col_name] = repaired_value
                    break
                
                # Resample if callback provided
                if resample_callback:
                    value = resample_callback(col_name, constraint)
                    cell_attempts += 1
                    resample_budget -= 1
                else:
                    # Fallback: draw from empirical distribution
                    repaired_row[col_name] = self._fallback_sample(constraint)
                    break
            
            if cell_attempts >= self.cell_resample_max or resample_budget <= 0:
                # Final fallback
                repaired_row[col_name] = self._fallback_sample(constraint)
        
        # Step 2: Validate cross-column dependencies
        repaired_row = self._validate_dependencies(repaired_row)
        
        return repaired_row
    
    def _validate_cell(self, value: str, constraint: ColumnConstraint) -> bool:
        """Validate a single cell against its constraint."""
        if constraint.column_type == ColumnType.CATEGORICAL_SMALL:
            return value in constraint.lexicon
        
        elif constraint.column_type == ColumnType.FIXED_DIGITS:
            if not (value.isdigit() and len(value) == constraint.expected_len):
                return False
            if constraint.min_val is not None and constraint.max_val is not None:
                num_val = float(value)
                return constraint.soft_min <= num_val <= constraint.soft_max
        
        elif constraint.column_type == ColumnType.VARIABLE_DIGITS:
            if not value.isdigit():
                return False
            if not (constraint.min_len <= len(value) <= constraint.max_len):
                return False
            if constraint.min_val is not None and constraint.max_val is not None:
                num_val = float(value)
                return constraint.soft_min <= num_val <= constraint.soft_max
        
        elif constraint.column_type == ColumnType.FREE_TEXT:
            return constraint.min_len <= len(value) <= constraint.max_len
        
        return True
    
    def _repair_cell(self, value: str, constraint: ColumnConstraint) -> str:
        """Attempt to repair an invalid cell."""
        if constraint.column_type == ColumnType.FIXED_DIGITS:
            # Try truncation for over-length
            if len(value) > constraint.expected_len and value.isdigit():
                return value[:constraint.expected_len]
            # Try extracting digits for mixed content
            digits_only = ''.join(c for c in value if c.isdigit())
            if len(digits_only) == constraint.expected_len:
                return digits_only
        
        elif constraint.column_type == ColumnType.VARIABLE_DIGITS:
            # Extract digits and truncate if needed
            digits_only = ''.join(c for c in value if c.isdigit())
            if len(digits_only) > constraint.max_len:
                return digits_only[:constraint.max_len]
            elif len(digits_only) >= constraint.min_len:
                return digits_only
        
        elif constraint.column_type == ColumnType.CATEGORICAL_SMALL:
            # Try fuzzy matching (Levenshtein distance ≤ 1)
            best_match = self._fuzzy_match(value, constraint.lexicon)
            if best_match:
                return best_match
        
        # If repair fails, return original value
        return value
    
    def _fuzzy_match(self, value: str, lexicon: Set[str]) -> Optional[str]:
        """Find closest match in lexicon with Levenshtein distance ≤ 1."""
        def levenshtein_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        for lex_value in lexicon:
            if levenshtein_distance(value, lex_value) <= 1:
                return lex_value
        
        return None
    
    def _fallback_sample(self, constraint: ColumnConstraint) -> str:
        """Sample from empirical distribution as last resort."""
        if constraint.value_frequencies:
            # Frequency-weighted sampling
            values = list(constraint.value_frequencies.keys())
            weights = list(constraint.value_frequencies.values())
            return np.random.choice(values, p=np.array(weights) / sum(weights))
        
        # Ultimate fallback
        if constraint.column_type == ColumnType.FIXED_DIGITS and constraint.expected_len:
            return "0" * constraint.expected_len
        elif constraint.column_type == ColumnType.CATEGORICAL_SMALL and constraint.lexicon:
            return list(constraint.lexicon)[0]
        else:
            return "N/A"
    
    def _validate_dependencies(self, row_data: Dict[str, str]) -> Dict[str, str]:
        """Validate and repair cross-column dependencies."""
        # Placeholder for dependency validation
        # Can be extended with specific rules like:
        # - StartDate <= EndDate
        # - AY <= ReportYear
        # - Open_k <= Pay_k
        
        # For now, return row as-is
        return row_data
    
    def add_dependency_rule(self, rule_func: tp.Callable[[Dict[str, str]], bool], 
                           repair_func: tp.Callable[[Dict[str, str]], Dict[str, str]]):
        """Add a custom cross-column dependency rule."""
        self.dependency_rules.append((rule_func, repair_func))


class CharacterAwareLogitsProcessor:
    """Logits processor that enforces character-level constraints during generation."""
    
    def __init__(self, tokenizer: AutoTokenizer, constraints: Dict[str, ColumnConstraint],
                 token_metadata: Dict[int, TokenMetadata], sep_tokens: Dict[str, int]):
        self.tokenizer = tokenizer
        self.constraints = constraints
        self.token_metadata = token_metadata
        self.sep_tokens = sep_tokens
        
        # State tracking for current generation
        self.current_column = None
        self.current_value_str = ""
        self.column_order = list(constraints.keys())
        self.current_col_idx = 0
    
    def reset_for_new_row(self):
        """Reset state for generating a new row."""
        self.current_column = None
        self.current_value_str = ""
        self.current_col_idx = 0
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Apply character-aware constraints to logits."""
        if self.current_col_idx >= len(self.column_order):
            return scores  # No more columns to constrain
        
        current_col = self.column_order[self.current_col_idx]
        constraint = self.constraints[current_col]
        
        # Get top-k candidates for efficiency
        top_k = min(256, scores.size(-1))
        top_scores, top_ids = torch.topk(scores, top_k, dim=-1)
        
        # Apply character-aware filtering
        for batch_idx in range(scores.size(0)):
            batch_top_ids = top_ids[batch_idx]
            
            for i, token_id in enumerate(batch_top_ids):
                token_id_int = token_id.item()
                
                if not self._is_token_allowed(token_id_int, constraint):
                    scores[batch_idx, token_id_int] = float('-inf')
        
        return scores
    
    def _is_token_allowed(self, token_id: int, constraint: ColumnConstraint) -> bool:
        """Check if a token is allowed given current constraint and state."""
        if token_id not in self.token_metadata:
            return False
        
        metadata = self.token_metadata[token_id]
        
        # Always ban separators during value generation
        if metadata.is_separator:
            return False
        
        # Ban leading whitespace if not allowed
        if (len(self.current_value_str) == 0 and 
            not constraint.allow_leading_space and 
            metadata.is_space_only):
            return False
        
        # Character class constraints
        if constraint.digits_only and not metadata.is_digit_only:
            return False
        
        # Length constraints (character-aware)
        projected_len = len(self.current_value_str) + metadata.char_len
        if projected_len > constraint.max_len:
            return False
        
        # Ban column name tokens for non-free-text columns
        if (constraint.column_type != ColumnType.FREE_TEXT and 
            constraint.name_tokens and 
            token_id in constraint.name_tokens):
            return False
        
        return True
    
    def update_state(self, new_token_id: int):
        """Update processor state after a token is generated."""
        if new_token_id in self.token_metadata:
            metadata = self.token_metadata[new_token_id]
            
            if metadata.is_separator:
                # Move to next column
                self.current_col_idx += 1
                self.current_value_str = ""
            else:
                # Add to current value
                self.current_value_str += metadata.token_str