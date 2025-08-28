"""Tabula Generation Processors

LogitsProcessors for improved tabular data generation:
1. NoDoubleSepProcessor - Prevents consecutive separator tokens
2. AllowedSetProcessor - Restricts tokens to allowed sets per column
3. ColumnWiseProcessor - Manages column-specific constraints during generation
"""
import torch
from transformers import LogitsProcessor
from typing import Dict, List, Set, Optional, Union
import numpy as np


class NoDoubleSepProcessor(LogitsProcessor):
    """Prevent consecutive separator tokens during generation.
    
    This processor sets separator token logits to -inf if the previous token
    was already a separator, eliminating the consecutive separator issue.
    """
    
    def __init__(self, sep_ids: Union[List[int], Set[int]]):
        """Initialize the processor.
        
        Args:
            sep_ids: List or set of separator token IDs (e.g., [colsep_id, rowsep_id])
        """
        self.sep_ids = set(sep_ids) if not isinstance(sep_ids, set) else sep_ids
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Process logits to prevent consecutive separators.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            scores: Logits scores [batch_size, vocab_size]
            
        Returns:
            Modified scores with separator tokens masked where appropriate
        """
        if input_ids.size(1) == 0:
            return scores
            
        # Get last token for each sequence in batch
        last_tokens = input_ids[:, -1]  # [batch_size]
        
        # Check which sequences have separator as last token
        forbid_mask = torch.zeros(input_ids.size(0), dtype=torch.bool, device=scores.device)
        for sep_id in self.sep_ids:
            forbid_mask = forbid_mask | (last_tokens == sep_id)
        
        # Set separator logits to -inf for sequences that just emitted a separator
        if forbid_mask.any():
            for sep_id in self.sep_ids:
                scores[forbid_mask, sep_id] = -float("inf")
                
        return scores


class AllowedSetProcessor(LogitsProcessor):
    """Restrict generation to allowed token sets for specific columns.
    
    This processor masks out tokens that weren't seen in training data
    for the current column, helping ensure generated values are realistic.
    """
    
    def __init__(self, 
                 allowed_tokens: Set[int], 
                 extra_allowed: Optional[Set[int]] = None,
                 soft_penalty: bool = False,
                 penalty_weight: float = 5.0):
        """Initialize the processor.
        
        Args:
            allowed_tokens: Set of token IDs allowed for this column
            extra_allowed: Additional tokens to always allow (digits, punctuation)
            soft_penalty: If True, apply penalty instead of hard mask
            penalty_weight: Weight for soft penalty (only used if soft_penalty=True)
        """
        self.allowed_tokens = allowed_tokens.copy()
        if extra_allowed:
            self.allowed_tokens.update(extra_allowed)
        self.soft_penalty = soft_penalty
        self.penalty_weight = penalty_weight
        
        # Pre-compute allowed tensor for efficiency
        self.allowed_tensor = None
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Process logits to restrict to allowed tokens.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            scores: Logits scores [batch_size, vocab_size]
            
        Returns:
            Modified scores with disallowed tokens masked or penalized
        """
        if not self.allowed_tokens:
            return scores
            
        # Create allowed tensor on first call
        if self.allowed_tensor is None:
            self.allowed_tensor = torch.tensor(
                list(self.allowed_tokens), 
                dtype=torch.long, 
                device=scores.device
            )
        
        # Create mask for allowed tokens
        allowed_mask = torch.zeros(scores.size(-1), dtype=torch.bool, device=scores.device)
        allowed_mask[self.allowed_tensor] = True
        
        if self.soft_penalty:
            # Apply penalty to disallowed tokens
            penalty_mask = ~allowed_mask
            scores[:, penalty_mask] -= self.penalty_weight
        else:
            # Hard mask: set disallowed tokens to -inf
            disallowed_mask = ~allowed_mask
            scores[:, disallowed_mask] = -float("inf")
            
        return scores


class ColumnWiseProcessor(LogitsProcessor):
    """Manage column-specific generation constraints.
    
    This processor applies different constraints based on the current column
    being generated, supporting per-column token limits and allowed sets.
    """
    
    def __init__(self, 
                 column_configs: Dict[str, Dict],
                 special_ids: Dict[str, int],
                 current_column: str = None):
        """Initialize the processor.
        
        Args:
            column_configs: Dict mapping column names to configuration dicts
                          Each config can contain: 'max_tokens', 'allowed_tokens', 'type'
            special_ids: Dict with special token IDs (colsep_id, rowsep_id, etc.)
            current_column: Name of column currently being generated
        """
        self.column_configs = column_configs
        self.special_ids = special_ids
        self.current_column = current_column
        self.token_count = 0  # Track tokens generated for current column
        
    def set_column(self, column_name: str):
        """Set the current column being generated."""
        self.current_column = column_name
        self.token_count = 0
        
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Process logits based on current column constraints.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            scores: Logits scores [batch_size, vocab_size]
            
        Returns:
            Modified scores with column-specific constraints applied
        """
        if not self.current_column or self.current_column not in self.column_configs:
            return scores
            
        config = self.column_configs[self.current_column]
        
        # Apply max token limit for this column
        max_tokens = config.get('max_tokens', None)
        if max_tokens is not None and self.token_count >= max_tokens:
            # Force EOS (column separator) if we've hit the limit
            colsep_id = self.special_ids.get('colsep_id')
            rowsep_id = self.special_ids.get('rowsep_id')
            
            # Set all tokens to -inf except appropriate separator
            scores[:, :] = -float("inf")
            if colsep_id is not None:
                scores[:, colsep_id] = 0.0  # Allow column separator
            if rowsep_id is not None:
                scores[:, rowsep_id] = 0.0  # Allow row separator
                
        # Apply allowed token constraints
        allowed_tokens = config.get('allowed_tokens', None)
        if allowed_tokens is not None:
            # Create mask for allowed tokens
            allowed_mask = torch.zeros(scores.size(-1), dtype=torch.bool, device=scores.device)
            for token_id in allowed_tokens:
                if token_id < scores.size(-1):
                    allowed_mask[token_id] = True
                    
            # Always allow separators
            for sep_id in [self.special_ids.get('colsep_id'), self.special_ids.get('rowsep_id')]:
                if sep_id is not None and sep_id < scores.size(-1):
                    allowed_mask[sep_id] = True
                    
            # Apply mask
            disallowed_mask = ~allowed_mask
            scores[:, disallowed_mask] = -float("inf")
        
        # Increment token count (this is approximate for batch processing)
        self.token_count += 1
        
        return scores


class MinNonSpaceThenEOSProcessor(LogitsProcessor):
    """Prevent <COLSEP> generation until at least one non-space token is generated.
    
    This stateful processor tracks non-space tokens per sequence and prevents
    early termination that leads to empty cell values.
    """
    
    def __init__(self, colsep_id: int, space_id: int):
        """Initialize the processor.
        
        Args:
            colsep_id: Token ID for column separator
            space_id: Token ID for space character
        """
        self.colsep_id = colsep_id
        self.space_id = space_id
        
        # State tracking per batch element
        self.in_cell_step = None  # How many tokens generated for current cell
        self.nonspace_count = None  # Number of non-space tokens so far
        self.batch_size = 0
        
    def reset_for_new_column(self, batch_size: int):
        """Reset state for new column generation.
        
        Args:
            batch_size: Size of the current batch
        """
        self.batch_size = batch_size
        self.in_cell_step = torch.zeros(batch_size, dtype=torch.long)
        self.nonspace_count = torch.zeros(batch_size, dtype=torch.long)
        
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Process logits to prevent early column separation.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            scores: Logits scores [batch_size, vocab_size]
            
        Returns:
            Modified scores with <COLSEP> masked where appropriate
        """
        batch_size = input_ids.size(0)
        
        # Initialize state if needed
        if self.in_cell_step is None or self.in_cell_step.size(0) != batch_size:
            self.reset_for_new_column(batch_size)
        
        # For each sequence in batch
        for b in range(batch_size):
            # If we haven't generated any non-space tokens yet, block <COLSEP>
            if self.nonspace_count[b] == 0:
                scores[b, self.colsep_id] = -float("inf")
                
            # Add negative bias to <COLSEP> at first step for extra safety
            if self.in_cell_step[b] == 0:
                scores[b, self.colsep_id] -= 1.0  # Small negative bias under sampling
        
        # Update state for next call (track what will be generated)
        # This is approximate since we don't know what will actually be sampled
        # The exact tracking happens in update_state() called externally
        self.in_cell_step += 1
        
        return scores
    
    def update_state(self, new_token_ids: torch.Tensor):
        """Update state based on newly generated tokens.
        
        Args:
            new_token_ids: Newly generated token IDs [batch_size]
        """
        if self.nonspace_count is not None and new_token_ids is not None:
            # Count non-space tokens
            is_nonspace = (new_token_ids != self.space_id) & (new_token_ids != self.colsep_id)
            self.nonspace_count += is_nonspace.long()


class NoLeadingSpaceProcessor(LogitsProcessor):
    """Prevent space token as the first token of value generation.
    
    This processor blocks space tokens at the very first step of value generation
    to prevent values that become empty after .strip().
    """
    
    def __init__(self, space_id: int):
        """Initialize the processor.
        
        Args:
            space_id: Token ID for space character
        """
        self.space_id = space_id
        
        # State tracking
        self.in_cell_step = None
        self.batch_size = 0
        
    def reset_for_new_column(self, batch_size: int):
        """Reset state for new column generation."""
        self.batch_size = batch_size
        self.in_cell_step = torch.zeros(batch_size, dtype=torch.long)
        
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Process logits to prevent leading spaces.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            scores: Logits scores [batch_size, vocab_size]
            
        Returns:
            Modified scores with space masked at first step
        """
        batch_size = input_ids.size(0)
        
        # Initialize state if needed
        if self.in_cell_step is None or self.in_cell_step.size(0) != batch_size:
            self.reset_for_new_column(batch_size)
        
        # Block space token at first step of value generation
        for b in range(batch_size):
            if self.in_cell_step[b] == 0:
                scores[b, self.space_id] = -float("inf")
        
        # Update step counter
        self.in_cell_step += 1
        
        return scores


class NamePieceBlockerProcessor(LogitsProcessor):
    """Block column name tokens from appearing in generated values.
    
    This processor removes column name subwords from allowed token sets
    to prevent contamination like 'inj_part part 1' or 'Pay08_bin bin'.
    """
    
    def __init__(self, tokenizer, column_name_tokens: Dict[str, Set[int]], current_column: str = None):
        """Initialize the processor.
        
        Args:
            tokenizer: HuggingFace tokenizer (to get digit tokens)
            column_name_tokens: Dict mapping column names to sets of their token IDs
            current_column: Name of column currently being generated
        """
        self.tokenizer = tokenizer
        self.column_name_tokens = column_name_tokens
        self.current_column = current_column
        
        # Build set of digit tokens that should never be blocked
        self.digit_tokens = self._get_digit_tokens()
        
    def _get_digit_tokens(self) -> Set[int]:
        """Get token IDs for digit characters 0-9."""
        digit_tokens = set()
        for digit in "0123456789":
            try:
                token_ids = self.tokenizer.encode(digit, add_special_tokens=False)
                digit_tokens.update(token_ids)
            except Exception:
                continue
        return digit_tokens
    
    def set_column(self, column_name: str):
        """Set the current column being generated."""
        self.current_column = column_name
        
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Process logits to block column name tokens.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            scores: Logits scores [batch_size, vocab_size]
            
        Returns:
            Modified scores with name tokens masked
        """
        if not self.current_column or self.current_column not in self.column_name_tokens:
            return scores
            
        name_tokens = self.column_name_tokens[self.current_column]
        
        # Block name tokens, but never block digits
        tokens_to_block = name_tokens - self.digit_tokens
        
        for token_id in tokens_to_block:
            if token_id < scores.size(-1):
                scores[:, token_id] = -float("inf")
                
        return scores


class NoRowSepInCellProcessor(LogitsProcessor):
    """Hard-ban <ROWSEP> during cell value generation.
    
    This processor always masks <ROWSEP> to -inf during value generation
    to prevent it from sneaking into cell values and causing early row termination.
    """
    
    def __init__(self, rowsep_id: int):
        """Initialize the processor.
        
        Args:
            rowsep_id: Token ID for row separator
        """
        self.rowsep_id = rowsep_id
        
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Process logits to hard-ban <ROWSEP> in all cells.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            scores: Logits scores [batch_size, vocab_size]
            
        Returns:
            Modified scores with <ROWSEP> always masked
        """
        # Hard ban <ROWSEP> in all cell value generation
        scores[:, self.rowsep_id] = -float("inf")
        return scores


class CollapseGuardProcessor(LogitsProcessor):
    """Fallback to safe token sets when all other tokens are masked.
    
    This processor prevents complete mask collapse by providing minimal
    fallback token sets when all non-separator tokens are masked to -inf.
    """
    
    def __init__(self, tokenizer, special_ids: Dict[str, int], current_column: str = None, 
                 column_configs: Dict[str, Dict] = None):
        """Initialize the processor.
        
        Args:
            tokenizer: HuggingFace tokenizer
            special_ids: Dict with special token IDs (colsep_id, rowsep_id)
            current_column: Name of column currently being generated
            column_configs: Dict mapping column names to their configurations
        """
        self.tokenizer = tokenizer
        self.special_ids = special_ids
        self.current_column = current_column
        self.column_configs = column_configs or {}
        
        # Build fallback token sets
        self.digit_tokens = self._get_digit_tokens()
        self.binary_tokens = self._get_binary_tokens()
        
        # Counter for debugging
        self.collapse_fallback_hits = {}
        
    def _get_digit_tokens(self) -> Set[int]:
        """Get token IDs for digit characters 0-9."""
        digit_tokens = set()
        for digit in "0123456789":
            try:
                token_ids = self.tokenizer.encode(digit, add_special_tokens=False)
                digit_tokens.update(token_ids)
            except Exception:
                continue
        return digit_tokens
    
    def _get_binary_tokens(self) -> Set[int]:
        """Get token IDs for binary values 0 and 1."""
        binary_tokens = set()
        for val in ["0", "1"]:
            try:
                token_ids = self.tokenizer.encode(val, add_special_tokens=False)
                binary_tokens.update(token_ids)
            except Exception:
                continue
        return binary_tokens
    
    def set_column(self, column_name: str):
        """Set the current column being generated."""
        self.current_column = column_name
        
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Process logits to prevent complete mask collapse.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            scores: Logits scores [batch_size, vocab_size]
            
        Returns:
            Modified scores with fallback tokens enabled if needed
        """
        if not self.current_column:
            return scores
            
        # Check if all non-separator tokens are masked
        colsep_id = self.special_ids.get('colsep_id')
        rowsep_id = self.special_ids.get('rowsep_id')
        
        # Create mask of non-separator positions
        non_sep_mask = torch.ones(scores.size(-1), dtype=torch.bool, device=scores.device)
        if colsep_id is not None:
            non_sep_mask[colsep_id] = False
        if rowsep_id is not None:
            non_sep_mask[rowsep_id] = False
            
        # Check if all non-separator tokens are -inf for any sequence
        for b in range(scores.size(0)):
            non_sep_scores = scores[b, non_sep_mask]
            if torch.all(torch.isinf(non_sep_scores) & (non_sep_scores < 0)):
                # All non-separator tokens are masked! Apply fallback
                self._apply_fallback(scores[b:b+1], b)
                
        return scores
    
    def _apply_fallback(self, scores: torch.Tensor, batch_idx: int):
        """Apply fallback token set for collapsed masks.
        
        Args:
            scores: Logits for single sequence [1, vocab_size]
            batch_idx: Batch index for logging
        """
        if self.current_column not in self.collapse_fallback_hits:
            self.collapse_fallback_hits[self.current_column] = 0
        self.collapse_fallback_hits[self.current_column] += 1
        
        # Determine fallback set based on column type
        config = self.column_configs.get(self.current_column, {})
        col_type = config.get('type', 'text')
        
        fallback_tokens = set()
        
        if col_type == 'binary':
            fallback_tokens = self.binary_tokens
        elif col_type in ['numeric', 'categorical']:
            fallback_tokens = self.digit_tokens
        else:
            # For text columns, use top frequent tokens from training (if available)
            allowed_tokens = config.get('allowed_tokens', set())
            if allowed_tokens:
                # Use subset of most frequent tokens as fallback
                fallback_tokens = set(list(allowed_tokens)[:50])  # Top 50
            else:
                # Last resort: just digits
                fallback_tokens = self.digit_tokens
        
        # Enable fallback tokens
        for token_id in fallback_tokens:
            if token_id < scores.size(-1):
                scores[0, token_id] = 0.0  # Set to neutral logit
                
        # Log fallback activation for debugging
        if self.collapse_fallback_hits[self.current_column] <= 5:  # Don't spam
            print(f"CollapseGuard: Applied fallback for column '{self.current_column}' "
                  f"(type={col_type}, {len(fallback_tokens)} tokens enabled)")
                  
    def get_fallback_stats(self) -> Dict[str, int]:
        """Get statistics on fallback usage for debugging."""
        return dict(self.collapse_fallback_hits)


def build_numeric_allowed_tokens(tokenizer, include_negative: bool = True, include_decimal: bool = True) -> Set[int]:
    """Build allowed token set for numeric columns.
    
    Args:
        tokenizer: HuggingFace tokenizer
        include_negative: Whether to include negative sign
        include_decimal: Whether to include decimal point
        
    Returns:
        Set of token IDs for numeric characters
    """
    chars = list("0123456789")
    if include_negative:
        chars.extend(["-", "−"])  # regular and unicode minus
    if include_decimal:
        chars.extend([".", ","])  # period and comma decimals
        
    # Tokenize each character individually to get token IDs
    allowed_tokens = set()
    for char in chars:
        try:
            token_ids = tokenizer.encode(char, add_special_tokens=False)
            allowed_tokens.update(token_ids)
        except Exception:
            continue  # Skip problematic characters
            
    return allowed_tokens


def build_safe_allowed_tokens(tokenized_values: List[List[int]], special_ids: Dict[str, int], 
                             tokenizer, column_name_tokens: Set[int] = None) -> Set[int]:
    """Build allowed token set from values, excluding special tokens and name pieces.
    
    Args:
        tokenized_values: List of tokenized value sequences
        special_ids: Dict with special token IDs to exclude
        tokenizer: HuggingFace tokenizer (for digit detection)
        column_name_tokens: Optional set of column name tokens to exclude
        
    Returns:
        Set of safe allowed token IDs
    """
    # Collect all tokens from values
    value_tokens = set()
    for tokens in tokenized_values:
        value_tokens.update(tokens)
    
    # Always exclude special tokens
    special_tokens_to_exclude = set()
    for key in ['pad_id', 'eos_id', 'colsep_id', 'rowsep_id', 'space_id']:
        if key in special_ids and special_ids[key] is not None:
            special_tokens_to_exclude.add(special_ids[key])
    
    value_tokens -= special_tokens_to_exclude
    
    # Get digit tokens that should NEVER be excluded
    digit_tokens = set()
    for digit in "0123456789":
        try:
            token_ids = tokenizer.encode(digit, add_special_tokens=False)
            digit_tokens.update(token_ids)
        except Exception:
            continue
    
    # Exclude column name tokens, but preserve digits and common letters
    if column_name_tokens:
        # Never exclude digits even if they're in column names  
        safe_name_tokens = column_name_tokens - digit_tokens
        
        # Also preserve single common letters (Q, A, etc.) that might be valid values
        common_letters = set()
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            try:
                token_ids = tokenizer.encode(letter, add_special_tokens=False)
                common_letters.update(token_ids)
            except Exception:
                continue
        
        safe_name_tokens -= common_letters
        value_tokens -= safe_name_tokens
    
    return value_tokens


def infer_column_types(df, tokenizer, columns: Optional[List[str]] = None, 
                      special_ids: Dict[str, int] = None, 
                      column_name_tokens: Dict[str, Set[int]] = None) -> Dict[str, Dict]:
    """Automatically infer column types and build configurations.
    
    Args:
        df: Pandas DataFrame with training data
        tokenizer: HuggingFace tokenizer
        columns: List of column names to process (if None, use all)
        special_ids: Dict with special token IDs to exclude from allowed sets
        column_name_tokens: Dict mapping column names to their token sets (for name blocking)
        
    Returns:
        Dictionary mapping column names to configuration dictionaries
    """
    if columns is None:
        columns = df.columns.tolist()
    
    if special_ids is None:
        special_ids = {}
    
    if column_name_tokens is None:
        column_name_tokens = {}
        
    column_configs = {}
    
    for col_name in columns:
        if col_name not in df.columns:
            continue
            
        config = {}
        vals = df[col_name].astype(str).fillna("").tolist()
        
        # Tokenize all values to compute statistics
        tokenized = tokenizer(vals, add_special_tokens=False)["input_ids"]
        token_lengths = [len(tokens) for tokens in tokenized]
        
        # Compute max tokens (99th percentile + 1 for safety)
        max_tokens = int(np.percentile(token_lengths, 99)) + 1
        config['max_tokens'] = max_tokens
        
        # Determine column type and allowed tokens
        unique_vals = set(vals)
        
        # Get column name tokens for this column
        col_name_tokens = column_name_tokens.get(col_name, set())
        
        # Binary check
        if unique_vals <= {"0", "1", ""}:
            config['type'] = 'binary'
            # Use safe builder for binary tokens
            config['allowed_tokens'] = build_safe_allowed_tokens(
                tokenized, special_ids, tokenizer, col_name_tokens
            )
            config['max_tokens'] = 1
            
        # Categorical check (small vocabulary)
        elif len(unique_vals) <= 128:
            config['type'] = 'categorical'
            # Use safe builder for categorical tokens
            config['allowed_tokens'] = build_safe_allowed_tokens(
                tokenized, special_ids, tokenizer, col_name_tokens
            )
            
        # Numeric check
        else:
            numeric_vals = []
            for val in vals:
                try:
                    if val.strip():  # Skip empty strings
                        numeric_vals.append(float(val))
                except (ValueError, TypeError):
                    pass
                    
            numeric_ratio = len(numeric_vals) / len([v for v in vals if v.strip()]) if vals else 0
            
            if numeric_ratio > 0.98:  # 98% of non-empty values are numeric
                config['type'] = 'numeric'
                config['allowed_tokens'] = build_numeric_allowed_tokens(
                    tokenizer, 
                    include_negative=any(v < 0 for v in numeric_vals),
                    include_decimal=any('.' in str(v) for v in vals if v.strip())
                )
                # Remove special tokens from numeric allowed set
                special_tokens_to_exclude = set()
                for key in ['pad_id', 'eos_id', 'colsep_id', 'rowsep_id', 'space_id']:
                    if key in special_ids and special_ids[key] is not None:
                        special_tokens_to_exclude.add(special_ids[key])
                config['allowed_tokens'] -= special_tokens_to_exclude
            else:
                config['type'] = 'text'
                # Use safe builder for text tokens
                config['allowed_tokens'] = build_safe_allowed_tokens(
                    tokenized, special_ids, tokenizer, col_name_tokens
                )
                
        column_configs[col_name] = config
        
    return column_configs


def build_column_name_tokens(tokenizer, column_names: List[str]) -> Dict[str, Set[int]]:
    """Build token sets for column names to enable blocking name contamination.
    
    Args:
        tokenizer: HuggingFace tokenizer
        column_names: List of all column names
        
    Returns:
        Dict mapping column names to sets of their token IDs
    """
    column_name_tokens = {}
    
    for col_name in column_names:
        # Tokenize column name and collect all token IDs
        name_token_ids = set()
        
        try:
            # Tokenize the full column name
            tokens = tokenizer.encode(col_name, add_special_tokens=False)
            name_token_ids.update(tokens)
            
            # Also tokenize individual parts if underscore-separated
            if '_' in col_name:
                parts = col_name.split('_')
                for part in parts:
                    if part:  # Skip empty parts
                        part_tokens = tokenizer.encode(part, add_special_tokens=False)
                        name_token_ids.update(part_tokens)
            
            # Store the token set for this column
            column_name_tokens[col_name] = name_token_ids
            
        except Exception as e:
            print(f"Warning: Could not tokenize column name '{col_name}': {e}")
            column_name_tokens[col_name] = set()
    
    return column_name_tokens