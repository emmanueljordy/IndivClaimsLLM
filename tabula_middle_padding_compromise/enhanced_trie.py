"""Enhanced Trie for Categorical Generation

This module provides an improved trie implementation that properly tracks
prefix-conditional allowed tokens to prevent invalid combinations like "2994"
when the valid years are only 1994-2005.
"""

import torch
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict


class PrefixAwareTrie:
    """Enhanced trie that tracks allowed tokens conditioned on the exact prefix.
    
    Unlike the flattened trie which allows any token that appears at a position,
    this implementation only allows tokens that can follow the specific prefix
    that has been generated so far.
    """
    
    def __init__(self, tokenized_values: List[List[int]]):
        """Build prefix-aware trie from tokenized value sequences.
        
        Args:
            tokenized_values: List of token ID sequences for valid categorical values
        """
        self.root = {}  # Nested dict representing the trie
        self.tokenized_values = tokenized_values  # Store for debugging
        self.max_depth = max(len(tokens) for tokens in tokenized_values) if tokenized_values else 0
        
        # Build the trie structure
        for tokens in tokenized_values:
            self._insert(tokens)
    
    def _insert(self, tokens: List[int]):
        """Insert a token sequence into the trie."""
        node = self.root
        for token in tokens:
            if token not in node:
                node[token] = {}
            node = node[token]
        # Mark end of valid sequence
        node['__END__'] = True
    
    def get_allowed_next_tokens(self, prefix: List[int]) -> Set[int]:
        """Get tokens that can validly follow the given prefix.
        
        Args:
            prefix: Token sequence generated so far
            
        Returns:
            Set of token IDs that can follow this prefix
        """
        node = self.root
        
        # Navigate to the node representing this prefix
        for token in prefix:
            if token not in node:
                # Invalid prefix - no valid continuations
                return set()
            node = node[token]
        
        # Return all possible next tokens (except the end marker)
        return set(k for k in node.keys() if k != '__END__')
    
    def can_end_at_prefix(self, prefix: List[int]) -> bool:
        """Check if the given prefix is a complete valid value.
        
        Args:
            prefix: Token sequence to check
            
        Returns:
            True if this prefix forms a complete valid value
        """
        node = self.root
        
        # Navigate to the node representing this prefix
        for token in prefix:
            if token not in node:
                return False
            node = node[token]
        
        # Check if this node marks the end of a valid sequence
        return '__END__' in node
    
    def has_valid_continuations(self, prefix: List[int]) -> bool:
        """Check if there are any valid continuations from this prefix.
        
        Args:
            prefix: Token sequence generated so far
            
        Returns:
            True if there are valid tokens that can follow this prefix
        """
        allowed_next = self.get_allowed_next_tokens(prefix)
        return len(allowed_next) > 0


class EnhancedFlattenedTrie:
    """Improved flattened trie with better prefix tracking.
    
    This version maintains a mapping from (position, prefix_tuple) -> allowed_tokens
    to ensure only valid combinations are generated.
    """
    
    def __init__(self, tokenized_values: List[List[int]]):
        """Build enhanced flattened trie from tokenized value sequences.
        
        Args:
            tokenized_values: List of token ID sequences for this column
        """
        self.max_depth = max(len(tokens) for tokens in tokenized_values) if tokenized_values else 0
        self.tokenized_values = tokenized_values
        
        # Map from (step, prefix_tuple) -> Set[allowed_next_tokens]
        self.prefix_to_allowed: Dict[Tuple[int, Tuple[int, ...]], Set[int]] = defaultdict(set)
        
        # Set of (step, prefix_tuple) where values can end
        self.valid_endings: Set[Tuple[int, Tuple[int, ...]]] = set()
        
        # For backward compatibility with original FlattenedTrie
        self.can_close_at: Set[int] = set()  # Steps where at least one value can end
        
        self._build_prefix_map()
    
    def _build_prefix_map(self):
        """Build the prefix-conditional allowed token map."""
        for tokens in self.tokenized_values:
            # Process each position in the sequence
            for i in range(len(tokens) + 1):
                prefix = tuple(tokens[:i])
                
                # Mark valid ending if this is the full sequence
                if i == len(tokens):
                    self.valid_endings.add((i, prefix))
                    self.can_close_at.add(i)  # For backward compatibility
                
                # Add next token to allowed set if not at end
                if i < len(tokens):
                    self.prefix_to_allowed[(i, prefix)].add(tokens[i])
    
    def get_allowed_tokens(self, step: int, prefix_tokens: List[int]) -> Set[int]:
        """Get allowed tokens at given step for the specific prefix.
        
        Args:
            step: Current generation step (0-indexed)
            prefix_tokens: Tokens generated so far for this value
            
        Returns:
            Set of allowed token IDs for this exact prefix
        """
        if len(prefix_tokens) != step:
            return set()
        
        prefix_tuple = tuple(prefix_tokens)
        return self.prefix_to_allowed.get((step, prefix_tuple), set())
    
    def can_close_here(self, step: int, prefix_tokens: List[int]) -> bool:
        """Check if value can end at this step with this prefix."""
        if len(prefix_tokens) != step:
            return False
        
        prefix_tuple = tuple(prefix_tokens)
        return (step, prefix_tuple) in self.valid_endings
    
    def must_close_here(self, step: int, prefix_tokens: List[int]) -> bool:
        """Check if we MUST close here (no valid continuations)."""
        allowed_next = self.get_allowed_tokens(step, prefix_tokens)
        return len(allowed_next) == 0 and self.can_close_here(step, prefix_tokens)


class ImprovedTrieCursor:
    """Cursor for navigating the enhanced trie during generation."""
    
    def __init__(self, trie: EnhancedFlattenedTrie):
        self.trie = trie
        self.step = 0
        self.prefix_tokens = []
    
    def reset(self):
        """Reset cursor to beginning of trie."""
        self.step = 0
        self.prefix_tokens = []
    
    def allowed_next_ids(self) -> Set[int]:
        """Get allowed token IDs at current position with current prefix."""
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
        """Check if value can end at current position."""
        return self.trie.can_close_here(self.step, self.prefix_tokens)
    
    def must_close_here(self) -> bool:
        """Check if we MUST close here (no valid continuations)."""
        return self.trie.must_close_here(self.step, self.prefix_tokens)