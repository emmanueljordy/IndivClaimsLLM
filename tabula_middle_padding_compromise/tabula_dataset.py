import random
import typing as tp
from datasets import Dataset
from dataclasses import dataclass
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase, PreTrainedModel
import torch


# Special tokens configuration - no hard-coded IDs
SPECIALS = {
    "additional_special_tokens": ["<COLSEP>", "<ROWSEP>"]  # column/row separators
}


def normalize_tokenizer_and_model(tokenizer: PreTrainedTokenizerBase,
                                  model: PreTrainedModel = None):
    """Ensure tokenizer has pad/eos and our separators; resize model if needed.
    
    Args:
        tokenizer: HuggingFace tokenizer to normalize
        model: Optional model to resize if tokens are added
        
    Returns:
        dict: Dictionary with special token IDs
    """
    added = False
    orig = len(tokenizer)

    # Ensure EOS exists (most models have one; if not, add)
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|eos|>"})
        added = True
        print(f"Added EOS token: <|eos|>")

    # Force a dedicated PAD (not aliasing EOS) - this is crucial for proper training
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        added = True
        print(f"Added dedicated PAD token: <|pad|>")

    # Add separators as single tokens
    to_add = []
    for t in ["<COLSEP>", "<ROWSEP>"]:
        if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id:
            to_add.append(t)
    if to_add:
        current_additional = tokenizer.additional_special_tokens or []
        tokenizer.add_special_tokens({"additional_special_tokens": current_additional + to_add})
        added = True
        print(f"Added separator tokens: {to_add}")

    # Resize model embeddings if tokens were added
    if added and model is not None:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resizing model embeddings: {orig} -> {len(tokenizer)}")

    # Convenience IDs - always computed dynamically
    ids = {
        "pad_id": tokenizer.pad_token_id,
        "eos_id": tokenizer.eos_token_id,
        "colsep_id": tokenizer.convert_tokens_to_ids("<COLSEP>"),
        "rowsep_id": tokenizer.convert_tokens_to_ids("<ROWSEP>"),
        "space_id": tokenizer.encode(" ", add_special_tokens=False)[0],  # First token for space
    }
    
    # Critical assertion: PAD ≠ EOS and all required tokens exist
    assert ids["pad_id"] is not None and ids["eos_id"] is not None and ids["pad_id"] != ids["eos_id"], f"PAD must not equal EOS: {ids}"
    assert ids["colsep_id"] != tokenizer.unk_token_id and ids["rowsep_id"] != tokenizer.unk_token_id, f"Separators must be known tokens: {ids}"
    
    print(f"Special token IDs: {ids}")
    return ids


class TabulaDataset(Dataset):
    """Tabula Dataset - Tokenizer Agnostic Version
    
    The TabulaDataset overwrites the _getitem function of the HuggingFace Dataset Class 
    to include the permutation step, but now works with any tokenizer without hard-coded IDs.
    
    Attributes:
        tokenizer (AutoTokenizer): Tokenizer from HuggingFace
        special_ids (dict): Dictionary of special token IDs
    """
    
    def set_tokenizer(self, tokenizer, special_ids: dict = None):
        """Set the Tokenizer and Special IDs
        
        Args:
            tokenizer: Tokenizer from HuggingFace
            special_ids: Dictionary of special token IDs (pad_id, eos_id, etc.)
        """
        self.tokenizer = tokenizer
        self.special_ids = special_ids or {
            "pad_id": tokenizer.pad_token_id,
            "eos_id": tokenizer.eos_token_id,
            "colsep_id": tokenizer.convert_tokens_to_ids("<COLSEP>"),
            "rowsep_id": tokenizer.convert_tokens_to_ids("<ROWSEP>"),
            "space_id": tokenizer.encode(" ", add_special_tokens=False)[0],
        }

    def _getitem(self, key: tp.Union[int, slice, str], decoded: bool = True, **kwargs) -> tp.Union[tp.Dict, tp.List]:
        """Get Item from Tabular Data
        
        Get one instance of the tabular data, converted to text and tokenized.
        Returns list of token ID lists (one per column).
        """
        # Get single row
        row = self._data.fast_slice(key, 1)

        # Build a list of columnwise strings: ["ColName value", "value", "value", ...]
        data_row_text_list = []
        for i in range(row.num_columns):
            val = str(row.columns[i].to_pylist()[0]).strip()
            if i == 0:
                # First column includes column name
                data_row_text_list.append(f"{row.column_names[i]} {val}")
            else:
                # Other columns just the value
                data_row_text_list.append(val)

        # Tokenize each column separately and return list of input_ids lists
        # This maintains the per-column structure needed by the collator
        return self.tokenizer(data_row_text_list, add_special_tokens=False)["input_ids"]
        
    def __getitems__(self, keys: tp.Union[int, slice, str, list]):
        """Get multiple items"""
        if isinstance(keys, list):
            return [self._getitem(key) for key in keys]
        else:
            return self._getitem(keys)


@dataclass
class TabulaDataCollator(DataCollatorWithPadding):
    """Tabula Data Collator - Tokenizer Agnostic Version
    
    Pads per column to token_list_length[i] using PAD for mid columns
    and EOS for the last column, then flattens columns into a single sequence.
    
    No hard-coded token IDs - uses tokenizer-provided IDs.
    """
    token_list_length: tp.List[int] = None
    special_ids: tp.Dict[str, int] = None
    debug: bool = False  # Enable debug output during training
    _dbg_printed: bool = False  # Track whether we've printed debug info

    def set_token_list_length(self, token_list_length):
        """Set the token list length for each column
        
        Args:
            token_list_length: List where each element represents the longest 
                             token sequence size for the corresponding column
        """
        self.token_list_length = token_list_length

    def set_special_ids(self, special_ids: dict):
        """Set special token IDs from tokenizer
        
        Args:
            special_ids: Dictionary with keys: pad_id, eos_id, colsep_id, rowsep_id
        """
        self.special_ids = special_ids

    def __call__(self, features):
        """Process batch of features into padded tensors with separator injection and PAD masking
        
        Args:
            features: List of samples, each sample is a list-of-lists (per column token ids)
            
        Returns:
            dict: Batch with input_ids, attention_mask, and labels (PAD tokens masked from loss)
        """
        assert self.token_list_length is not None, "token_list_length must be set via set_token_list_length"
        assert self.special_ids is not None, "special_ids must be set via set_special_ids"

        pad_id  = self.special_ids["pad_id"]
        col_id  = self.special_ids["colsep_id"]
        row_id  = self.special_ids["rowsep_id"]

        B = len(features)  # batch size
        C = len(self.token_list_length)  # number of columns

        # Build per-sample sequences with injected separators
        encoded_text = []
        for b in range(B):
            seq = []
            for c in range(C):
                toks = features[b][c][:]  # copy to avoid modifying original
                
                # Leave room for 1 separator: content can use at most (span_length - 1) tokens
                max_no_sep = max(0, self.token_list_length[c] - 1)
                if len(toks) > max_no_sep:
                    toks = toks[:max_no_sep]  # truncate if too long
                
                # Append appropriate separator
                sep = row_id if c == C - 1 else col_id  # <ROWSEP> for last column, <COLSEP> for others
                toks = toks + [sep]
                
                # Pad to full span (separator already included)
                if len(toks) < self.token_list_length[c]:
                    toks = toks + [pad_id] * (self.token_list_length[c] - len(toks))
                
                seq.extend(toks)  # add this column's tokens to the sequence
            encoded_text.append(seq)

        # Convert to tensors
        input_ids = torch.tensor(encoded_text, dtype=torch.long)
        
        # Build attention mask: 1 for non-PAD tokens, 0 for PAD tokens
        attention_mask = (input_ids != pad_id).long()

        # Build labels for language modeling: mask PAD tokens from loss, but keep separators in loss
        labels = input_ids.clone()
        labels[input_ids == pad_id] = -100  # ignore PAD in loss (keep COL/ROW in loss)

        # Create batch dictionary
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
        # FIX: One-time debug verification of training batch separators
        if self.debug and not self._dbg_printed:
            ids = batch["input_ids"][0].tolist()
            try:
                txt = self.tokenizer.decode(ids, skip_special_tokens=False)
                print(f"TRAIN SEQ EX: {txt[:400]}{'...' if len(txt) > 400 else ''}")
                
                # Count separators using token decoding (more reliable than string matching)
                colsep_token = self.tokenizer.decode([col_id], skip_special_tokens=False)
                rowsep_token = self.tokenizer.decode([row_id], skip_special_tokens=False)
                colsep_count = txt.count(colsep_token)
                rowsep_count = txt.count(rowsep_token)
                
                expected_colseps = C - 1  # number of columns - 1
                expected_rowseps = 1
                
                print(f"COUNTS: <COLSEP>={colsep_count} (expected: {expected_colseps}), <ROWSEP>={rowsep_count} (expected: {expected_rowseps})")
                
                # Verify separator counts
                if colsep_count == expected_colseps and rowsep_count == expected_rowseps:
                    print("✅ TRAINING BATCH SEPARATORS: CORRECT")
                else:
                    print("❌ TRAINING BATCH SEPARATORS: MISMATCH - check collator logic!")
                    
            except Exception as e:
                print(f"TRAIN SEQ DEBUG ERROR: {e}")
            
            # Set flag so we only print this once per training session
            self._dbg_printed = True
        
        return batch