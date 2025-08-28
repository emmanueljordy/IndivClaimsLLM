"""Advanced Generation Processors - FIXED VERSION

Fixed issues:
1. Early-close no longer fires too soon for small categorical columns
2. Proper position-wise masking enforcement for trie-based generation
3. Better handling of multi-token categorical values
4. Enhanced trie implementation prevents invalid token combinations (e.g., "2994" when only "1994-2005" are valid)

Key changes:
- should_force_close() now checks if we MUST close (no valid continuations) rather than if we CAN close
- Added should_definitely_close() helper to check if trie has no more valid tokens
- Improved trie cursor tracking
- Integration with EnhancedFlattenedTrie for better prefix-aware generation
"""

import torch
import numpy as np
from transformers import LogitsProcessor
from typing import Dict, List, Set, Optional, Union, Tuple, Counter as CounterType
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import pandas as pd

# Import enhanced trie for better categorical generation
try:
    from tabula_middle_padding_compromise.enhanced_trie import (
        EnhancedFlattenedTrie, ImprovedTrieCursor
    )
    USE_ENHANCED_TRIE = True
except ImportError:
    USE_ENHANCED_TRIE = False


class ColumnType(Enum):
    """Column type classification based on value analysis."""
    SMALL_CATEGORICAL = "small_categorical"    # Few unique values, exact trie matching
    LARGE_CATEGORICAL = "large_categorical"    # Many unique values, soft constraints  
    DISCRETE_NUMERIC = "discrete_numeric"      # Integer-like values
    CONTINUOUS_NUMERIC = "continuous_numeric"  # Float values with decimals
    FREE_TEXT = "free_text"                    # General text content


@dataclass
class ColumnProfile:
    """Complete profile for a single column derived from training values only."""
    
    # Basic metadata
    column_name: str
    column_type: ColumnType
    
    # Generation limits (value-only statistics)
    gen_limit: int  # p95/p99 of value token lengths + safety margin
    
    # Type-specific constraints
    trie: Optional['FlattenedTrie'] = None  # For small categoricals
    numeric_format: Optional[Dict[str, bool]] = None  # For numeric columns
    position_masks: Optional[Dict[int, Set[int]]] = None  # For large categoricals
    global_top_k: Optional[Set[int]] = None  # Column-level allowed tokens
    
    # Statistics for debugging
    value_token_lengths: CounterType[int] = None
    unique_ratio: float = 0.0
    char_classes: Set[str] = None
    
    # Instrumentation
    min_observed_len: int = 0
    max_observed_len: int = 0
    p95_len: float = 0.0
    p99_len: float = 0.0


class TrieCursor:
    """Minimal stateful wrapper for FlattenedTrie traversal."""
    
    def __init__(self, trie: Union['FlattenedTrie', 'EnhancedFlattenedTrie']):
        self.trie = trie
        self.step = 0
        self.prefix_tokens = []
        # Check if we're using the enhanced trie
        self.is_enhanced = isinstance(trie, EnhancedFlattenedTrie) if USE_ENHANCED_TRIE else False
    
    def reset(self):
        """Reset cursor to beginning of trie."""
        self.step = 0
        self.prefix_tokens = []
    
    def allowed_next_ids(self) -> Set[int]:
        """Get allowed token IDs at current step."""
        return self.trie.get_allowed_tokens(self.step, self.prefix_tokens)
    
    def advance(self, token_id: int) -> bool:
        """Advance cursor with token. Returns True if valid transition."""
        allowed = self.allowed_next_ids()
        if token_id in allowed:
            self.prefix_tokens.append(token_id)
            self.step += 1
            return True
        return False
    
    def can_close_here(self) -> bool:
        """Check if value can end at current step."""
        if self.is_enhanced:
            return self.trie.can_close_here(self.step, self.prefix_tokens)
        else:
            return self.trie.can_close_here(self.step)
    
    def must_close_here(self) -> bool:
        """Check if we MUST close here (no valid continuations)."""
        if self.is_enhanced:
            return self.trie.must_close_here(self.step, self.prefix_tokens)
        else:
            # Original logic for non-enhanced trie
            next_step_tokens = self.trie.get_allowed_tokens_for_next_step(self.step, self.prefix_tokens)
            return len(next_step_tokens) == 0


class FlattenedTrie:
    """Efficient trie implementation using per-position token sets.
    
    For small categoricals with exact token sequences. Faster than tree traversal
    and enables prefix validation without state explosion.
    """
    
    def __init__(self, tokenized_values: List[List[int]]):
        """Build flattened trie from tokenized value sequences.
        
        Args:
            tokenized_values: List of token ID sequences for this column
        """
        self.max_depth = max(len(tokens) for tokens in tokenized_values) if tokenized_values else 0
        self.step_allowed: Dict[int, Set[int]] = {}  # step -> allowed tokens
        self.can_close_at: Set[int] = set()  # steps where value can end
        self.tokenized_values = tokenized_values  # Store original sequences
        
        self._build_trie(tokenized_values)
    
    def _build_trie(self, tokenized_values: List[List[int]]):
        """Build per-position allowed token sets."""
        # Track all valid prefixes
        valid_prefixes = set()
        valid_prefixes.add(())  # Empty prefix
        
        for tokens in tokenized_values:
            if not tokens:
                self.can_close_at.add(0)
                continue
                
            # Add all prefixes of this value
            for i in range(len(tokens) + 1):
                prefix = tuple(tokens[:i])
                valid_prefixes.add(prefix)
                
                if i == len(tokens):
                    self.can_close_at.add(i)
        
        # Build step-wise allowed tokens
        for step in range(self.max_depth + 1):
            allowed_at_step = set()
            
            for prefix in valid_prefixes:
                if len(prefix) == step:
                    # Find all possible next tokens after this prefix
                    for full_tokens in tokenized_values:
                        if (len(full_tokens) > step and 
                            len(prefix) <= len(full_tokens) and
                            tuple(full_tokens[:step]) == prefix):
                            allowed_at_step.add(full_tokens[step])
            
            if allowed_at_step:
                self.step_allowed[step] = allowed_at_step
    
    def get_allowed_tokens(self, step: int, prefix_tokens: List[int]) -> Set[int]:
        """Get allowed tokens at given step with prefix validation.
        
        Args:
            step: Current generation step (0-indexed)
            prefix_tokens: Tokens generated so far for this value
            
        Returns:
            Set of allowed token IDs, empty if invalid prefix
        """
        # Handle the "peek ahead" case for must_close_here check
        if prefix_tokens and prefix_tokens[-1] is None:
            # This is a lookahead query, don't validate
            return self.step_allowed.get(step, set())
            
        # Validate prefix is still valid
        if len(prefix_tokens) != step:
            return set()
            
        # Check if current prefix exists in our trie
        prefix_tuple = tuple(prefix_tokens)
        prefix_valid = False
        
        # A prefix is valid if it's a prefix of at least one training value
        for full_tokens in self.tokenized_values:
            if (len(full_tokens) >= step and 
                tuple(full_tokens[:step]) == prefix_tuple):
                prefix_valid = True
                break
        
        if not prefix_valid and step > 0:  # Empty prefix is always valid
            return set()
            
        return self.step_allowed.get(step, set())
    
    def can_close_here(self, step: int) -> bool:
        """Check if value can end at this step."""
        return step in self.can_close_at
    
    def get_allowed_tokens_for_next_step(self, current_step: int, current_prefix: List[int]) -> Set[int]:
        """Get allowed tokens for the NEXT step given current prefix.
        
        Args:
            current_step: Current step (length of current prefix)
            current_prefix: Tokens generated so far
            
        Returns:
            Set of allowed token IDs for the next step, empty if no valid continuations
        """
        # We're at step `current_step` with prefix `current_prefix`
        # We want to know what tokens are allowed at step `current_step + 1`
        next_step = current_step + 1
        
        # Check if any training sequence can continue from this prefix
        allowed_next = set()
        current_tuple = tuple(current_prefix)
        
        for full_tokens in self.tokenized_values:
            # Check if this training sequence matches our current prefix
            # and has more tokens
            if (len(full_tokens) > current_step and 
                len(full_tokens) >= len(current_prefix) and
                tuple(full_tokens[:current_step]) == current_tuple):
                # This sequence matches our prefix and continues
                if len(full_tokens) > current_step:
                    # Add the next token from this sequence
                    allowed_next.add(full_tokens[current_step])
        
        return allowed_next


class ColumnProfiler:
    """Build-time profiler that analyzes training values to create ColumnProfiles."""
    
    def __init__(self, tokenizer, thresholds: Dict[str, float] = None):
        """Initialize profiler with tokenizer and classification thresholds.
        
        Args:
            tokenizer: HuggingFace tokenizer
            thresholds: Dict with classification thresholds (small_cat_ratio, etc.)
        """
        self.tokenizer = tokenizer
        self.thresholds = thresholds or {
            'small_categorical_ratio': 0.02,
            'small_categorical_max_unique': 50,
            'discrete_numeric_ratio': 0.1,
            'numeric_detection_ratio': 0.98,
            'top_k_coverage': 0.95,
            'position_hist_min_examples': 10
        }
        
        # Pre-compute common token sets
        self.digit_tokens = self._get_digit_tokens()
        self.space_tokens = self._get_space_tokens()
        self.punctuation_tokens = self._get_punctuation_tokens()
        
    def _get_digit_tokens(self) -> Set[int]:
        """Get token IDs for digits 0-9."""
        tokens = set()
        for digit in "0123456789":
            try:
                token_ids = self.tokenizer.encode(digit, add_special_tokens=False)
                tokens.update(token_ids)
            except Exception:
                continue
        return tokens
    
    def _get_space_tokens(self) -> Set[int]:
        """Get token IDs for various space characters."""
        tokens = set()
        for space_char in [" ", "\t", "\u00A0"]:  # space, tab, non-breaking space
            try:
                token_ids = self.tokenizer.encode(space_char, add_special_tokens=False)
                tokens.update(token_ids)
            except Exception:
                continue
        return tokens
    
    def _get_punctuation_tokens(self) -> Set[int]:
        """Get token IDs for common punctuation."""
        tokens = set()
        for punct in ".,;:!?-_()[]{}\"'":
            try:
                token_ids = self.tokenizer.encode(punct, add_special_tokens=False)
                tokens.update(token_ids)
            except Exception:
                continue
        return tokens
    
    def profile_column(self, column_name: str, values: List[str], 
                      special_ids: Dict[str, int] = None,
                      column_name_tokens: Set[int] = None) -> ColumnProfile:
        """Create complete profile for a single column.
        
        Args:
            column_name: Name of the column
            values: List of string values from training data
            special_ids: Dict of special token IDs to exclude
            column_name_tokens: Token IDs from column name to exclude
            
        Returns:
            ColumnProfile with complete analysis
        """
        special_ids = special_ids or {}
        column_name_tokens = column_name_tokens or set()
        
        # Clean and tokenize values (VALUE-ONLY, no column names)
        clean_values = [v.strip() for v in values if v.strip()]
        if not clean_values:
            return self._create_empty_profile(column_name)
        
        # Tokenize each value individually
        tokenized_values = []
        for value in clean_values:
            try:
                tokens = self.tokenizer.encode(value, add_special_tokens=False)
                tokenized_values.append(tokens)
            except Exception:
                tokenized_values.append([])
        
        # Compute value-only statistics
        token_lengths = [len(tokens) for tokens in tokenized_values]
        value_token_lengths = Counter(token_lengths)
        
        unique_values = set(clean_values)
        unique_ratio = len(unique_values) / len(clean_values) if clean_values else 0
        
        # Determine generation limit (value-only p95/p99)
        if token_lengths:
            p95_len = np.percentile(token_lengths, 95)
            p99_len = np.percentile(token_lengths, 99)
            min_len = min(token_lengths)
            max_len = max(token_lengths)
            # Use p95 + 1 for safety, minimum 1
            gen_limit = max(1, int(np.ceil(p95_len)) + 1)
        else:
            p95_len = p99_len = min_len = max_len = gen_limit = 1
        
        # Classify column type
        column_type = self._classify_column_type(
            clean_values, unique_ratio, len(unique_values)
        )
        
        # Clamp gen_limit for single-digit/binary columns
        if column_type == ColumnType.SMALL_CATEGORICAL and max_len <= 1:
            gen_limit = 1  # Binary and single-digit columns should be limited to 1 token
        elif column_type == ColumnType.DISCRETE_NUMERIC and max_len <= 1:
            gen_limit = 1  # Single-digit numeric columns
        
        # Build type-specific constraints
        profile = ColumnProfile(
            column_name=column_name,
            column_type=column_type,
            gen_limit=gen_limit,
            value_token_lengths=value_token_lengths,
            unique_ratio=unique_ratio,
            char_classes=self._analyze_char_classes(clean_values),
            min_observed_len=min_len,
            max_observed_len=max_len,
            p95_len=p95_len,
            p99_len=p99_len
        )
        
        # Add type-specific constraints
        if column_type == ColumnType.SMALL_CATEGORICAL:
            # Use enhanced trie if available for better categorical generation
            if USE_ENHANCED_TRIE:
                profile.trie = EnhancedFlattenedTrie(tokenized_values)
                # Debug: show some trie info
                if len(tokenized_values) > 0:
                    sample_values = tokenized_values[:5]
                    print(f"  [ENHANCED TRIE] {column_name}: {len(tokenized_values)} sequences, samples={sample_values}")
            else:
                profile.trie = FlattenedTrie(tokenized_values)
                # Debug: show some trie info
                if len(tokenized_values) > 0:
                    sample_values = tokenized_values[:5]
                    print(f"  [TRIE] {column_name}: {len(tokenized_values)} sequences, samples={sample_values}")
        
        elif column_type in [ColumnType.DISCRETE_NUMERIC, ColumnType.CONTINUOUS_NUMERIC]:
            profile.numeric_format = self._analyze_numeric_format(clean_values)
        
        elif column_type in [ColumnType.LARGE_CATEGORICAL, ColumnType.FREE_TEXT]:
            profile.global_top_k, profile.position_masks = self._build_soft_constraints(
                tokenized_values, special_ids, column_name_tokens
            )
        
        return profile
    
    def _create_empty_profile(self, column_name: str) -> ColumnProfile:
        """Create minimal profile for empty columns."""
        return ColumnProfile(
            column_name=column_name,
            column_type=ColumnType.FREE_TEXT,
            gen_limit=1,
            value_token_lengths=Counter({1: 1}),
            unique_ratio=0.0,
            char_classes=set(),
            global_top_k={},
        )
    
    def _classify_column_type(self, values: List[str], unique_ratio: float, 
                            unique_count: int) -> ColumnType:
        """Classify column type based on value analysis."""
        
        # Small categorical: low cardinality
        if (unique_ratio <= self.thresholds['small_categorical_ratio'] or 
            unique_count <= self.thresholds['small_categorical_max_unique']):
            return ColumnType.SMALL_CATEGORICAL
        
        # Numeric detection
        numeric_values = []
        for value in values:
            try:
                if value.strip():
                    numeric_values.append(float(value))
            except (ValueError, TypeError):
                pass
        
        numeric_ratio = len(numeric_values) / len(values) if values else 0
        
        if numeric_ratio >= self.thresholds['numeric_detection_ratio']:
            # Check if values are integer-like
            has_decimals = any('.' in str(v) for v in values if v.strip())
            if has_decimals or unique_ratio > self.thresholds['discrete_numeric_ratio']:
                return ColumnType.CONTINUOUS_NUMERIC
            else:
                return ColumnType.DISCRETE_NUMERIC
        
        # Large categorical vs free text
        if unique_ratio <= 0.5:  # Still somewhat categorical
            return ColumnType.LARGE_CATEGORICAL
        else:
            return ColumnType.FREE_TEXT
    
    def _analyze_char_classes(self, values: List[str]) -> Set[str]:
        """Analyze character classes present in values."""
        char_classes = set()
        
        for value in values:
            if any(c.isdigit() for c in value):
                char_classes.add('digits')
            if any(c.isalpha() for c in value):
                char_classes.add('letters')
            if any(c in '.,;:!?-_()[]{}"\'' for c in value):
                char_classes.add('punctuation')
            if any(c.isspace() for c in value):
                char_classes.add('spaces')
            if any(c in '+-' for c in value):
                char_classes.add('signs')
            if '.' in value:
                char_classes.add('decimal')
        
        return char_classes
    
    def _analyze_numeric_format(self, values: List[str]) -> Dict[str, bool]:
        """Analyze numeric format patterns."""
        format_info = {
            'has_negative': False,
            'has_decimal': False,
            'has_leading_zeros': False,
            'has_signs': False
        }
        
        for value in values:
            if value.startswith('-') or value.startswith('+'):
                format_info['has_signs'] = True
                if value.startswith('-'):
                    format_info['has_negative'] = True
            
            if '.' in value:
                format_info['has_decimal'] = True
            
            # Check for leading zeros (but not just "0")
            if len(value) > 1 and value[0] == '0' and value[1].isdigit():
                format_info['has_leading_zeros'] = True
        
        return format_info
    
    def _build_soft_constraints(self, tokenized_values: List[List[int]], 
                              special_ids: Dict[str, int],
                              column_name_tokens: Set[int]) -> Tuple[Set[int], Dict[int, Set[int]]]:
        """Build soft constraints for large categorical/text columns."""
        
        # Collect all value tokens
        all_value_tokens = set()
        for tokens in tokenized_values:
            all_value_tokens.update(tokens)
        
        # Remove special tokens and column name tokens
        special_tokens_to_exclude = set()
        for key in ['pad_id', 'eos_id', 'colsep_id', 'rowsep_id']:
            if key in special_ids and special_ids[key] is not None:
                special_tokens_to_exclude.add(special_ids[key])
        
        # Never exclude digits
        safe_tokens = all_value_tokens - special_tokens_to_exclude
        safe_name_tokens = column_name_tokens - self.digit_tokens
        global_top_k = safe_tokens - safe_name_tokens
        
        # Build position-specific masks if we have enough data
        position_masks = {}
        if len(tokenized_values) >= self.thresholds['position_hist_min_examples']:
            max_len = max(len(tokens) for tokens in tokenized_values) if tokenized_values else 0
            
            for step in range(min(max_len, 10)):  # Limit to first 10 positions
                step_tokens = []
                for tokens in tokenized_values:
                    if len(tokens) > step:
                        step_tokens.append(tokens[step])
                
                if step_tokens:
                    step_counter = Counter(step_tokens)
                    # Use top tokens that cover 95% of mass at this step
                    total = sum(step_counter.values())
                    coverage_threshold = total * self.thresholds['top_k_coverage']
                    
                    cumsum = 0
                    step_allowed = set()
                    for token, count in step_counter.most_common():
                        step_allowed.add(token)
                        cumsum += count
                        if cumsum >= coverage_threshold:
                            break
                    
                    position_masks[step] = step_allowed - special_tokens_to_exclude - safe_name_tokens
        
        return global_top_k, position_masks
    
    def profile_dataset(self, df: pd.DataFrame, 
                       special_ids: Dict[str, int] = None) -> Dict[str, ColumnProfile]:
        """Profile all columns in a dataset.
        
        Args:
            df: Pandas DataFrame with training data
            special_ids: Dict of special token IDs
            
        Returns:
            Dict mapping column names to ColumnProfiles
        """
        special_ids = special_ids or {}
        
        # Build column name token sets once
        column_name_tokens = {}
        for col_name in df.columns:
            name_tokens = set()
            try:
                tokens = self.tokenizer.encode(col_name, add_special_tokens=False)
                name_tokens.update(tokens)
                
                # Also tokenize parts if underscore-separated
                if '_' in col_name:
                    for part in col_name.split('_'):
                        if part:
                            part_tokens = self.tokenizer.encode(part, add_special_tokens=False)
                            name_tokens.update(part_tokens)
            except Exception:
                pass
            
            column_name_tokens[col_name] = name_tokens
        
        # Profile each column
        profiles = {}
        for col_name in df.columns:
            values = df[col_name].astype(str).fillna("").tolist()
            profile = self.profile_column(
                col_name, values, special_ids, column_name_tokens[col_name]
            )
            profiles[col_name] = profile
            
        return profiles


class ColumnState:
    """Track generation state for column-wise processing."""
    
    def __init__(self, columns: List[str]):
        """Initialize state tracking.
        
        Args:
            columns: List of column names in generation order
        """
        self.columns = columns
        self.current_column_idx = 0
        self.value_step = 0  # Step within current value
        self.emitted_nonspace_count = 0
        self.value_tokens = []  # Current value tokens
        
    @property
    def current_column(self) -> Optional[str]:
        """Get current column name."""
        if 0 <= self.current_column_idx < len(self.columns):
            return self.columns[self.current_column_idx]
        return None
    
    def advance_column(self):
        """Move to next column."""
        self.current_column_idx += 1
        self.value_step = 0
        self.emitted_nonspace_count = 0
        self.value_tokens = []
    
    def add_value_token(self, token_id: int, is_space: bool = False):
        """Add token to current value."""
        self.value_tokens.append(token_id)
        self.value_step += 1
        if not is_space:
            self.emitted_nonspace_count += 1
    
    def reset_for_new_sequence(self):
        """Reset state for new sequence generation."""
        self.current_column_idx = 0
        self.value_step = 0
        self.emitted_nonspace_count = 0
        self.value_tokens = []


class ColumnValueProcessor(LogitsProcessor):
    """Stateful logits processor with comprehensive value generation constraints.
    
    FIXED VERSION: Properly handles multi-token categorical values without early termination.
    """
    
    def __init__(self, 
                 profiles: Dict[str, ColumnProfile],
                 special_ids: Dict[str, int],
                 columns: List[str],
                 tokenizer=None):
        """Initialize processor.
        
        Args:
            profiles: Dict mapping column names to ColumnProfiles
            special_ids: Dict with special token IDs
            columns: List of column names in generation order
            tokenizer: HuggingFace tokenizer (for debugging)
        """
        self.profiles = profiles
        self.special_ids = special_ids
        self.columns = columns
        self.tokenizer = tokenizer
        
        # State tracking
        self.column_state = ColumnState(columns)
        
        # Pre-compute ban sets
        self.special_ban_set = self._build_special_ban_set()
        self.all_name_tokens = self._build_name_token_ban_set()
        
        # Debug logging flag
        self._debug_logging = False
        
        # Instrumentation
        self.stats = {
            'mask_collapses': defaultdict(int),
            'fallback_activations': defaultdict(int),
            'early_closes': defaultdict(int),
            'limit_enforcements': defaultdict(int),
            'empty_mask_events': defaultdict(int)
        }
        
    def _build_special_ban_set(self) -> Set[int]:
        """Build set of special tokens to always ban during value generation."""
        ban_set = set()
        for key in ['pad_id', 'eos_id', 'colsep_id', 'rowsep_id']:
            if key in self.special_ids and self.special_ids[key] is not None:
                ban_set.add(self.special_ids[key])
        return ban_set
    
    def _build_name_token_ban_set(self) -> Set[int]:
        """Build set of all column name tokens to ban."""
        name_tokens = set()
        if not self.tokenizer:
            return name_tokens
            
        for col_name in self.columns:
            try:
                tokens = self.tokenizer.encode(col_name, add_special_tokens=False)
                name_tokens.update(tokens)
                
                if '_' in col_name:
                    for part in col_name.split('_'):
                        if part:
                            part_tokens = self.tokenizer.encode(part, add_special_tokens=False)
                            name_tokens.update(part_tokens)
            except Exception:
                continue
        return name_tokens
    
    def reset_for_new_sequence(self):
        """Reset state for new sequence generation."""
        self.column_state.reset_for_new_sequence()
    
    def enable_debug_logging(self, enabled: bool = True):
        """Enable or disable debug logging for mask collapse analysis."""
        self._debug_logging = enabled
    
    def reset_for_new_column(self, col_idx: int):
        """Reset processor state for new column without appending space.
        
        Use this when space is already appended to context before tensor conversion.
        
        Args:
            col_idx: Index of column being entered
        """
        # Set column state - RESET to 0
        self.column_state.current_column_idx = col_idx
        self.column_state.value_step = 0
        self.column_state.emitted_nonspace_count = 0
        self.column_state.value_tokens = []
        
        # Reset trie state if applicable
        self._reset_trie_state(col_idx)
        
        # DO NOT append space - it should already be in the context
        # DO NOT call update_state_with_token for anything
    
    def _reset_trie_state(self, col_idx: int):
        """Reset trie cursor for the given column."""
        current_col = self.columns[col_idx] if col_idx < len(self.columns) else None
        if current_col and current_col in self.profiles:
            profile = self.profiles[current_col]
            if profile.trie is not None:
                if not hasattr(self, '_trie_cursors'):
                    self._trie_cursors = {}
                if current_col not in self._trie_cursors:
                    self._trie_cursors[current_col] = TrieCursor(profile.trie)
                self._trie_cursors[current_col].reset()
    
    def _advance_trie_state(self, token_id: int):
        """Advance trie state with given token."""
        current_col = self.column_state.current_column
        if current_col and hasattr(self, '_trie_cursors') and current_col in self._trie_cursors:
            self._trie_cursors[current_col].advance(token_id)
    
    def should_force_close(self) -> bool:
        """Check if current column should be forcibly closed.
        
        FIXED: Only close when we MUST (no valid continuations), not when we CAN.
        """
        current_col = self.column_state.current_column
        if not current_col or current_col not in self.profiles:
            return False
            
        profile = self.profiles[current_col]
        
        # Force close if we've hit the generation limit
        if self.column_state.value_step >= profile.gen_limit:
            self.stats['limit_enforcements'][current_col] += 1
            if self._debug_logging:
                print(f"[FORCE_CLOSE] {current_col}: Hit gen_limit={profile.gen_limit} at step={self.column_state.value_step}")
            return True
            
        # For small categoricals with trie: check if we MUST close (no valid continuations)
        if (profile.column_type == ColumnType.SMALL_CATEGORICAL and 
            profile.trie and 
            hasattr(self, '_trie_cursors') and 
            current_col in self._trie_cursors):
            
            cursor = self._trie_cursors[current_col]
            
            # Only force close if:
            # 1. We have emitted at least one non-space token AND
            # 2. There are NO valid continuations (must_close_here)
            if self.column_state.emitted_nonspace_count > 0:
                must_close = cursor.must_close_here()
                can_close = cursor.can_close_here()
                
                if self._debug_logging:
                    print(f"[CLOSE_CHECK] {current_col}: step={cursor.step}, prefix={cursor.prefix_tokens}, can_close={can_close}, must_close={must_close}")
                
                if must_close:
                    self.stats['early_closes'][current_col] += 1
                    if self._debug_logging:
                        print(f"[FORCE_CLOSE] {current_col}: Must close (no continuations) at step={cursor.step}")
                    return True
        
        return False
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Process logits with comprehensive constraints.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            scores: Logits scores [batch_size, vocab_size]
            
        Returns:
            Modified scores with all constraints applied
        """
        current_col = self.column_state.current_column
        if not current_col or current_col not in self.profiles:
            return scores
        
        profile = self.profiles[current_col]
        batch_size = scores.size(0)
        
        # 1) HARD GUARD: Cannot close a cell before emitting any non-space value token
        if self.column_state.emitted_nonspace_count == 0:
            colsep_id = self.special_ids.get('colsep_id')
            rowsep_id = self.special_ids.get('rowsep_id')
            if colsep_id is not None and colsep_id < scores.size(-1):
                scores[:, colsep_id] = -float("inf")
            if rowsep_id is not None and rowsep_id < scores.size(-1):
                scores[:, rowsep_id] = -float("inf")
        
        # 2) Always ban other special tokens during value generation
        pad_id = self.special_ids.get('pad_id')
        eos_id = self.special_ids.get('eos_id')
        if pad_id is not None and pad_id < scores.size(-1):
            scores[:, pad_id] = -float("inf")
        if eos_id is not None and eos_id < scores.size(-1):
            scores[:, eos_id] = -float("inf")
        
        # 3) Ban name tokens with smart filtering
        self._apply_name_contamination_filter(scores, current_col)
        
        # 4) Apply type-specific constraints
        scores = self._apply_type_constraints(scores, profile)
        
        # Debug logging for first few steps
        if self.column_state.value_step == 0 and hasattr(self, '_debug_logging') and self._debug_logging:
            ctx_tail = input_ids[0, -8:].tolist() if input_ids.size(1) >= 8 else input_ids[0].tolist()
            space_id = self.special_ids.get('space_id')
            ends_with_space = len(ctx_tail) > 0 and ctx_tail[-1] == space_id
            print(f"[DBG] {current_col} ctx_tail={ctx_tail}")
            print(f"[DBG] ends_with_space={ends_with_space}")
        
        # Check for mask collapse and apply fallbacks
        scores = self._apply_fallback_if_needed(scores, profile)
        
        return scores
    
    def _get_digit_tokens(self) -> Set[int]:
        """Get digit token IDs."""
        digit_tokens = set()
        if self.tokenizer:
            for digit in "0123456789":
                try:
                    tokens = self.tokenizer.encode(digit, add_special_tokens=False)
                    digit_tokens.update(tokens)
                except Exception:
                    continue
        return digit_tokens
    
    def _apply_type_constraints(self, scores: torch.Tensor, profile: ColumnProfile) -> torch.Tensor:
        """Apply type-specific masking constraints.
        
        FIXED: Proper position-wise masking for trie-based columns.
        """
        
        if profile.column_type == ColumnType.SMALL_CATEGORICAL and profile.trie:
            # Exact trie matching with proper step tracking
            if hasattr(self, '_trie_cursors') and profile.column_name in self._trie_cursors:
                cursor = self._trie_cursors[profile.column_name]
                allowed_tokens = cursor.allowed_next_ids()
            else:
                # Fallback to direct trie query
                allowed_tokens = profile.trie.get_allowed_tokens(
                    self.column_state.value_step, 
                    self.column_state.value_tokens
                )
            
            if allowed_tokens:
                # Create mask for allowed tokens only
                allowed_mask = torch.zeros(scores.size(-1), dtype=torch.bool, device=scores.device)
                for token_id in allowed_tokens:
                    if token_id < scores.size(-1):
                        allowed_mask[token_id] = True
                
                # IMPORTANT: Also allow separators if we CAN close here (but don't force it)
                if (hasattr(self, '_trie_cursors') and 
                    profile.column_name in self._trie_cursors and
                    self._trie_cursors[profile.column_name].can_close_here() and
                    self.column_state.emitted_nonspace_count > 0):
                    
                    colsep_id = self.special_ids.get('colsep_id')
                    rowsep_id = self.special_ids.get('rowsep_id')
                    if colsep_id is not None and colsep_id < scores.size(-1):
                        allowed_mask[colsep_id] = True
                    if rowsep_id is not None and rowsep_id < scores.size(-1):
                        allowed_mask[rowsep_id] = True
                
                disallowed_mask = ~allowed_mask
                scores[:, disallowed_mask] = -float("inf")
        
        elif profile.column_type in [ColumnType.DISCRETE_NUMERIC, ColumnType.CONTINUOUS_NUMERIC]:
            # Numeric constraints
            allowed_tokens = self._get_numeric_allowed_tokens(profile)
            
            if allowed_tokens:
                allowed_mask = torch.zeros(scores.size(-1), dtype=torch.bool, device=scores.device)
                for token_id in allowed_tokens:
                    if token_id < scores.size(-1):
                        allowed_mask[token_id] = True
                
                # Always allow separators for numeric columns
                colsep_id = self.special_ids.get('colsep_id')
                rowsep_id = self.special_ids.get('rowsep_id')
                if colsep_id is not None and colsep_id < scores.size(-1):
                    allowed_mask[colsep_id] = True
                if rowsep_id is not None and rowsep_id < scores.size(-1):
                    allowed_mask[rowsep_id] = True
                
                disallowed_mask = ~allowed_mask
                scores[:, disallowed_mask] = -float("inf")
        
        elif profile.column_type in [ColumnType.LARGE_CATEGORICAL, ColumnType.FREE_TEXT]:
            # Soft constraints: position masks or global top-k
            if (profile.position_masks and 
                self.column_state.value_step in profile.position_masks):
                # Use position-specific mask
                allowed_tokens = profile.position_masks[self.column_state.value_step]
            elif profile.global_top_k:
                # Use global top-k
                allowed_tokens = profile.global_top_k
            else:
                # No constraints
                allowed_tokens = None
            
            if allowed_tokens:
                allowed_mask = torch.zeros(scores.size(-1), dtype=torch.bool, device=scores.device)
                for token_id in allowed_tokens:
                    if token_id < scores.size(-1):
                        allowed_mask[token_id] = True
                
                # Always allow separators
                colsep_id = self.special_ids.get('colsep_id')
                rowsep_id = self.special_ids.get('rowsep_id')
                if colsep_id is not None and colsep_id < scores.size(-1):
                    allowed_mask[colsep_id] = True
                if rowsep_id is not None and rowsep_id < scores.size(-1):
                    allowed_mask[rowsep_id] = True
                
                disallowed_mask = ~allowed_mask
                scores[:, disallowed_mask] = -float("inf")
        
        return scores
    
    def _get_numeric_allowed_tokens(self, profile: ColumnProfile) -> Set[int]:
        """Get allowed tokens for numeric columns."""
        if not profile.numeric_format or not self.tokenizer:
            return set()
        
        allowed_chars = list("0123456789")
        
        if profile.numeric_format.get('has_decimal', False):
            allowed_chars.extend([".", ","])
        
        if profile.numeric_format.get('has_signs', False):
            allowed_chars.extend(["+", "-", "−"])
        
        allowed_tokens = set()
        for char in allowed_chars:
            try:
                tokens = self.tokenizer.encode(char, add_special_tokens=False)
                allowed_tokens.update(tokens)
            except Exception:
                continue
        
        return allowed_tokens
    
    def _apply_fallback_if_needed(self, scores: torch.Tensor, profile: ColumnProfile) -> torch.Tensor:
        """Apply fallback strategies if mask collapses."""
        
        # Check for complete mask collapse (all non-separator tokens are -inf)
        for b in range(scores.size(0)):
            non_sep_mask = torch.ones(scores.size(-1), dtype=torch.bool, device=scores.device)
            
            # Don't count separators in collapse check
            for sep_id in self.special_ban_set:
                if sep_id < scores.size(-1):
                    non_sep_mask[sep_id] = False
            
            non_sep_scores = scores[b, non_sep_mask]
            
            if torch.all(torch.isinf(non_sep_scores) & (non_sep_scores < 0)):
                # Mask collapsed! Apply fallback
                self.stats['mask_collapses'][profile.column_name] += 1
                self._apply_fallback_strategy(scores[b:b+1], profile)
        
        return scores
    
    def _apply_fallback_strategy(self, scores: torch.Tensor, profile: ColumnProfile):
        """Apply graduated fallback strategy for mask collapse.
        
        Args:
            scores: Logits for single sequence [1, vocab_size]
            profile: Column profile
        """
        self.stats['fallback_activations'][profile.column_name] += 1
        
        # Debug mask collapse
        if hasattr(self, '_debug_logging') and self._debug_logging:
            print(f"[DBG] MASK COLLAPSE col={profile.column_name} step={self.column_state.value_step} nonspace={self.column_state.emitted_nonspace_count}")
        
        # 0) HARD GUARDS: Always block separators until first non-space value
        if self.column_state.emitted_nonspace_count == 0:
            colsep_id = self.special_ids.get('colsep_id')
            rowsep_id = self.special_ids.get('rowsep_id')
            if colsep_id is not None and colsep_id < scores.size(-1):
                scores[0, colsep_id] = -float("inf")
            if rowsep_id is not None and rowsep_id < scores.size(-1):
                scores[0, rowsep_id] = -float("inf")
        
        # 1) Type-aware step-0 backstop if mask collapsed at very start
        if self.column_state.value_step == 0 and self.column_state.emitted_nonspace_count == 0:
            return self._first_token_backstop(scores, profile)
        
        # 2) Otherwise, apply existing graduated strategies
        return self._existing_fallback_strategies(scores, profile)
    
    def _first_token_backstop(self, scores: torch.Tensor, profile: ColumnProfile):
        """Type-specific fallback for step 0 mask collapse."""
        vocab_size = scores.size(-1)
        
        # Reset all scores to -inf
        scores[0, :] = -float("inf")
        
        if profile.column_type == ColumnType.SMALL_CATEGORICAL:
            # For small categoricals, allow first tokens from trie
            if profile.trie:
                allowed = profile.trie.get_allowed_tokens(0, [])
                for token_id in allowed:
                    if token_id < vocab_size:
                        scores[0, token_id] = 0.0
                if allowed:  # If we found valid tokens, return
                    return scores
        
        # Binary columns: allow '0', '1'
        if profile.column_type in [ColumnType.DISCRETE_NUMERIC, ColumnType.CONTINUOUS_NUMERIC]:
            digit_tokens = self._get_digit_tokens()
            for token_id in digit_tokens:
                if token_id < vocab_size:
                    scores[0, token_id] = 0.0
            
            # For continuous numeric, also allow signs and decimals
            if (profile.column_type == ColumnType.CONTINUOUS_NUMERIC and 
                profile.numeric_format):
                if profile.numeric_format.get('has_signs', False):
                    for char in ["+", "-", "−"]:
                        try:
                            tokens = self.tokenizer.encode(char, add_special_tokens=False)
                            for token_id in tokens:
                                if token_id < vocab_size:
                                    scores[0, token_id] = 0.0
                        except Exception:
                            continue
                if profile.numeric_format.get('has_decimal', False):
                    for char in [".", ","]:
                        try:
                            tokens = self.tokenizer.encode(char, add_special_tokens=False)
                            for token_id in tokens:
                                if token_id < vocab_size:
                                    scores[0, token_id] = 0.0
                        except Exception:
                            continue
            return scores
        
        # Text-like columns: use global top-k but still block separators
        if profile.global_top_k:
            safe_tokens = list(profile.global_top_k)[:50]  # Top 50
            for token_id in safe_tokens:
                if token_id < vocab_size:
                    scores[0, token_id] = 0.0
        else:
            # Last resort: common single chars but no separators
            common_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            for char in common_chars[:20]:  # First 20
                try:
                    tokens = self.tokenizer.encode(char, add_special_tokens=False)
                    for token_id in tokens:
                        if token_id < vocab_size:
                            scores[0, token_id] = 0.0
                except Exception:
                    continue
        
        return scores
    
    def _existing_fallback_strategies(self, scores: torch.Tensor, profile: ColumnProfile):
        """Apply existing graduated fallback strategies for non-step-0 cases."""
        
        # Strategy 1: For small categoricals, move up trie level
        if (profile.column_type == ColumnType.SMALL_CATEGORICAL and 
            profile.trie and self.column_state.value_step > 0):
            # Get all possible tokens at previous step
            prev_allowed = profile.trie.get_allowed_tokens(
                self.column_state.value_step - 1, 
                self.column_state.value_tokens[:-1] if self.column_state.value_tokens else []
            )
            if prev_allowed:
                for token_id in prev_allowed:
                    if token_id < scores.size(-1):
                        scores[0, token_id] = 0.0
                return scores
        
        # Strategy 2: Use global top-k if available
        if profile.global_top_k:
            safe_tokens = list(profile.global_top_k)[:50]  # Limit to top 50
            for token_id in safe_tokens:
                if token_id < scores.size(-1):
                    scores[0, token_id] = 0.0
            return scores
        
        # Strategy 3: Type-specific fallbacks
        if profile.column_type in [ColumnType.DISCRETE_NUMERIC, ColumnType.CONTINUOUS_NUMERIC]:
            # Allow digits
            digit_tokens = self._get_digit_tokens()
            for token_id in digit_tokens:
                if token_id < scores.size(-1):
                    scores[0, token_id] = 0.0
        else:
            # Last resort: allow most common single character tokens
            common_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            for char in common_chars[:10]:  # Just first 10
                try:
                    tokens = self.tokenizer.encode(char, add_special_tokens=False)
                    for token_id in tokens:
                        if token_id < scores.size(-1):
                            scores[0, token_id] = 0.0
                except Exception:
                    continue
        
        return scores
    
    def update_state_with_token(self, token_id: int):
        """Update processor state after token generation.
        
        CRITICAL: Only call this for ACTUAL VALUE TOKENS generated by the model.
        Do NOT call for forced spaces or separators we inject ourselves.
        
        Args:
            token_id: The token that was just generated by the model
        """
        space_id = self.special_ids.get('space_id')
        colsep_id = self.special_ids.get('colsep_id')
        rowsep_id = self.special_ids.get('rowsep_id')
        
        # Count ONLY actual value tokens (ignore spaces and separators entirely)
        if token_id not in (space_id, colsep_id, rowsep_id):
            self.column_state.value_step += 1
            self.column_state.emitted_nonspace_count += 1
            # Add to value tokens list for trie tracking
            self.column_state.value_tokens.append(token_id)
            # Advance trie state if applicable
            self._advance_trie_state(token_id)
        
        # If it's a space or separator, we don't increment any counters
        # This keeps our "empty vs non-empty" check accurate
    
    def notify_column_complete(self):
        """Notify processor that current column is complete."""
        self.column_state.advance_column()
    
    def _apply_name_contamination_filter(self, scores: torch.Tensor, current_col: str):
        """Apply smart name contamination filtering.
        
        Args:
            scores: Logits tensor to modify
            current_col: Current column name
        """
        # Never subtract at position 0 (first value token)
        if self.column_state.value_step == 0:
            return
        
        # Get column profile to check if digit-only
        if current_col not in self.profiles:
            return
            
        profile = self.profiles[current_col]
        digit_tokens = self._get_digit_tokens()
        
        # For digit-only columns, only subtract alphabetic name tokens
        if profile.column_type in [ColumnType.DISCRETE_NUMERIC, ColumnType.CONTINUOUS_NUMERIC]:
            # Keep digits available; subtract only alpha-ish name tokens
            alpha_name_tokens = self.all_name_tokens - digit_tokens
            for token_id in alpha_name_tokens:
                if token_id < scores.size(-1):
                    scores[:, token_id] = -float("inf")
        else:
            # For other columns, subtract all name tokens but preserve digits
            safe_name_tokens = self.all_name_tokens - digit_tokens
            for token_id in safe_name_tokens:
                if token_id < scores.size(-1):
                    scores[:, token_id] = -float("inf")
    
    def get_instrumentation_stats(self) -> Dict:
        """Get debugging and performance statistics."""
        return {
            'mask_collapses_by_column': dict(self.stats['mask_collapses']),
            'fallback_activations_by_column': dict(self.stats['fallback_activations']),
            'early_closes_by_column': dict(self.stats['early_closes']),
            'limit_enforcements_by_column': dict(self.stats['limit_enforcements']),
            'empty_mask_events_by_column': dict(self.stats['empty_mask_events']),
            'total_mask_collapses': sum(self.stats['mask_collapses'].values()),
            'total_fallback_activations': sum(self.stats['fallback_activations'].values())
        }


def print_profiling_summary(profiles: Dict[str, ColumnProfile], tokenizer=None):
    """Print comprehensive summary of column profiling results.
    
    Args:
        profiles: Dict mapping column names to ColumnProfiles
        tokenizer: Optional tokenizer for debugging output
    """
    print("\n" + "="*80)
    print("COLUMN PROFILING SUMMARY")
    print("="*80)
    
    for col_name, profile in profiles.items():
        print(f"\nColumn: {col_name}")
        print(f"  Type: {profile.column_type.value}")
        print(f"  Gen Limit: {profile.gen_limit} tokens (p95={profile.p95_len:.1f}, p99={profile.p99_len:.1f})")
        print(f"  Value Lengths: min={profile.min_observed_len}, max={profile.max_observed_len}")
        print(f"  Unique Ratio: {profile.unique_ratio:.3f}")
        print(f"  Char Classes: {', '.join(profile.char_classes) if profile.char_classes else 'none'}")
        
        # Print length distribution
        if profile.value_token_lengths:
            length_dist = dict(profile.value_token_lengths.most_common(5))
            print(f"  Length Distribution: {length_dist}")
        
        # Type-specific details
        if profile.column_type == ColumnType.SMALL_CATEGORICAL and profile.trie:
            # Handle both original and enhanced trie types
            if hasattr(profile.trie, 'can_close_at'):
                close_positions = sorted(profile.trie.can_close_at)[:5]
                print(f"  Trie: max_depth={profile.trie.max_depth}, can_close_at={close_positions}")
            else:
                print(f"  Trie: max_depth={profile.trie.max_depth}")
        
        elif profile.numeric_format:
            print(f"  Numeric Format: {profile.numeric_format}")
        
        elif profile.global_top_k:
            print(f"  Global Top-K: {len(profile.global_top_k)} tokens")
            if profile.position_masks:
                print(f"  Position Masks: {len(profile.position_masks)} positions")
    
    print("\n" + "="*80)