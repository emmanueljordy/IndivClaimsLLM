import typing as tp
from collections import defaultdict, Counter
import logging

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer


def _array_to_dataframe(data: tp.Union[pd.DataFrame, np.ndarray], columns=None) -> pd.DataFrame:
    """ Converts a Numpy Array to a Pandas DataFrame

    Args:
        data: Pandas DataFrame or Numpy NDArray
        columns: If data is a Numpy Array, columns needs to be a list of all column names

    Returns:
        Pandas DataFrame with the given data
    """
    if isinstance(data, pd.DataFrame):
        return data

    assert isinstance(data, np.ndarray), "Input needs to be a Pandas DataFrame or a Numpy NDArray"
    assert columns, "To convert the data into a Pandas DataFrame, a list of column names has to be given!"
    assert len(columns) == len(data[0]), \
        "%d column names are given, but array has %d columns!" % (len(columns), len(data[0]))

    return pd.DataFrame(data=data, columns=columns)


def _get_column_distribution(df: pd.DataFrame, col: str) -> tp.Union[list, dict]:
    """ Returns the distribution of a given column. If continuous, returns a list of all values.
        If categorical, returns a dictionary in form {"A": 0.6, "B": 0.4}

    Args:
        df: pandas DataFrame
        col: name of the column

    Returns:
        Distribution of the column
    """
    if df[col].dtype == "float":
        col_dist = df[col].to_list()
    else:
        col_dist = df[col].value_counts(1).to_dict()
    return col_dist


def _parse_cell_with_longest_prefix(text: str, known_columns: list) -> tuple[str, str]:
    """FIX D: Parse cell text using longest-prefix matching for robust key-value extraction.
    
    Args:
        text: Cell text to parse (may be "ColumnName value" format)
        known_columns: List of valid column names to match against
        
    Returns:
        tuple: (column_name, value) where column_name="" if no match found
    """
    if not text.strip():
        return "", ""
    
    tokens = text.split()
    if not tokens:
        return "", ""
    
    # Try to find longest matching column name prefix
    name = None
    split_at = None
    for i in range(len(tokens), 0, -1):
        candidate = " ".join(tokens[:i])
        if candidate in known_columns:
            name = candidate
            split_at = i
            break
    
    # If no known column found, use first token as name
    if name is None:
        name = tokens[0] if tokens else ""
        value = " ".join(tokens[1:]) if len(tokens) > 1 else ""
    else:
        value = " ".join(tokens[split_at:]) if split_at < len(tokens) else ""
    
    return name, value


def _convert_tokens_to_dataframe_by_seps(tokens: tp.List[torch.Tensor],
                                         tokenizer,
                                         column_list: list,
                                         special_ids: dict,
                                         df_gen: pd.DataFrame,
                                         start_col: str = None,
                                         debug: bool = False) -> pd.DataFrame:
    """Convert tokens to dataframe using separator-aware parsing.
    
    Args:
        tokens: List of generated token tensors (continuation only, no prompt)
        tokenizer: HuggingFace tokenizer
        column_list: List of column names in original order
        special_ids: Dictionary with special token IDs (pad_id, colsep_id, rowsep_id, eos_id)
        df_gen: DataFrame to append results to
        start_col: Starting column name (for rotation handling)
        debug: Enable detailed debug printing
    
    Returns:
        Updated dataframe with generated rows
    """
    pad_id  = special_ids["pad_id"]
    col_id  = special_ids["colsep_id"]
    row_id  = special_ids["rowsep_id"]
    eos_id  = special_ids["eos_id"]

    # Rotate columns if we started elsewhere
    cols = list(column_list)
    if start_col and start_col in cols:
        i = cols.index(start_col)
        cols = cols[i:] + cols[:i]  # rotate: [start_col, ..., last_col, first_col, ..., before_start]

    def split_on(ids, sep):
        """Split token IDs on separator, filtering out PAD tokens."""
        out, cur = [], []
        for t in ids:
            if t == sep:
                out.append(cur)
                cur = []
            else:
                if t != pad_id:  # ignore PAD tokens
                    cur.append(t)
        out.append(cur)  # add final segment
        return out

    rows = []
    for i, t in enumerate(tokens):
        ids = t.tolist() if hasattr(t, 'tolist') else t

        # Cut at first <ROWSEP> or EOS (stop generation point)
        cut = len(ids)
        for stop_id in [row_id, eos_id]:
            if stop_id is not None and stop_id in ids:
                cut = min(cut, ids.index(stop_id))
        gen = ids[:cut]  # generated tokens until stop

        if debug and i < 3:
            try:
                decoded_gen = tokenizer.decode(gen, skip_special_tokens=False)
                decoded_gen_safe = decoded_gen.encode('ascii', 'replace').decode('ascii')
                print(f"[DECODE] row {i}: cut at pos {cut}, generated: '{decoded_gen_safe}'")
            except Exception as e:
                print(f"[DECODE] row {i}: decode error - {e}")

        # Split the row into columns by <COLSEP>
        col_spans = split_on(gen, col_id)
        
        # Keep exactly len(cols) cells (pad missing with empty, truncate excess)
        if len(col_spans) < len(cols):
            col_spans += [[] for _ in range(len(cols) - len(col_spans))]  # pad with empty
        else:
            col_spans = col_spans[:len(cols)]  # truncate excess

        # FIX D: Use key-value parsing instead of positional mapping
        # This is more robust to rotation, missing separators, and cell order changes
        cell_dict = {}
        
        for j, span in enumerate(col_spans):
            try:
                cell_text = tokenizer.decode(span, skip_special_tokens=True).strip()
                if debug and i < 3:
                    print(f"[DECODE]   col {j}: {span[:10]}{'...' if len(span) > 10 else ''} -> '{cell_text}'")
                
                # Parse using robust key-value extraction
                col_name, col_value = _parse_cell_with_longest_prefix(cell_text, column_list)
                
                if col_name and col_name in column_list:
                    cell_dict[col_name] = col_value
                    if debug and i < 3:
                        print(f"[DECODE]     parsed: '{col_name}' = '{col_value}'")
                elif j < len(cols):
                    # Fallback: if key-value parsing failed, use positional (rotated order)
                    cell_dict[cols[j]] = cell_text
                    if debug and i < 3:
                        print(f"[DECODE]     fallback positional: '{cols[j]}' = '{cell_text}'")
                        
            except Exception as e:
                if debug and i < 3:
                    print(f"[DECODE]   col {j}: decode error -> skip")

        # Create row with all columns, filling missing with empty strings
        row = {}
        for c in column_list:  # use original column order
            row[c] = cell_dict.get(c, "")  # get from parsed dict or default to empty

        rows.append(row)

        if debug and i < 3:
            try:
                row_str = str(row).encode('ascii', 'replace').decode('ascii')
                print(f"[DECODE] final row {i}: {row_str}")
            except Exception as e:
                print(f"[DECODE] final row {i}: <encoding error - {e}>")

    # Combine results
    if rows:
        new_df = pd.DataFrame(rows)
        df_gen = pd.concat([df_gen, new_df], ignore_index=True)
        
    if debug:
        print(f"[DECODE] Added {len(rows)} rows to dataframe")
        
    return df_gen


def _convert_tokens_to_dataframe_legacy(tokens: tp.List[torch.Tensor], tokenizer: AutoTokenizer, token_list_length: list, column_list: list, df_gen, debug=False):
    """
    Convert tokens to dataframe with enhanced parsing and debug output.
    
    Args:
        tokens: List of generated token tensors
        tokenizer: HuggingFace tokenizer
        token_list_length: List of token spans for each column
        column_list: List of column names
        df_gen: DataFrame to append results to
        debug: Enable detailed debug printing
    
    Returns:
        Updated dataframe with generated rows
    """
    result_list = []
    
    # Statistics for debugging
    total_rows = len(tokens)
    successful_parses = 0
    parsing_errors = 0
    
    if debug:
        print(f"DEBUG: Converting {len(tokens)} token sequences to dataframe")
        print(f"DEBUG: Expected columns: {column_list}")
        print(f"DEBUG: Token spans: {token_list_length}")
    
    for token_idx, t in enumerate(tokens):
        token_list_cursor = 0
        td = dict.fromkeys(column_list)
        row_successful = True
        
        if debug:
            print(f"\nDEBUG: Processing token sequence {token_idx + 1}/{len(tokens)}")
            print(f"DEBUG: Token length: {len(t)}")
            try:
                decoded_full = tokenizer.decode(t, skip_special_tokens=False)
                # Handle Unicode characters that might not display properly
                decoded_full_safe = decoded_full.encode('ascii', 'replace').decode('ascii')
                print(f"DEBUG: Full decoded text: '{decoded_full_safe}'")
            except Exception as e:
                print(f"DEBUG: Error decoding tokens: {e}")
        
        for idx, token_list_span in enumerate(token_list_length):
            if token_list_cursor + token_list_span > len(t):
                if debug:
                    print(f"DEBUG: Warning - token span {idx} exceeds token length")
                row_successful = False
                break
                
            decoded_text = tokenizer.decode(t[token_list_cursor:token_list_cursor+token_list_span])
            decoded_text = ("").join(decoded_text)
            token_list_cursor = token_list_cursor + token_list_span
            
            # Clean text
            cleaned_text = decoded_text.replace("<|endoftext|>", "")
            cleaned_text = cleaned_text.replace("\n", " ")
            cleaned_text = cleaned_text.replace("\r", "")
            
            if debug:
                try:
                    # Handle Unicode characters safely
                    decoded_text_safe = decoded_text.encode('ascii', 'replace').decode('ascii')
                    cleaned_text_safe = cleaned_text.encode('ascii', 'replace').decode('ascii')
                    print(f"DEBUG: Column {idx} ({column_list[idx]}): raw='{decoded_text_safe}' clean='{cleaned_text_safe}'")
                except Exception as e:
                    print(f"DEBUG: Column {idx} ({column_list[idx]}): encoding error - {e}")
            
            if idx == 0:
                values = cleaned_text.strip().split(" ")
                try:
                    if len(values) > 1:
                        td[column_list[idx]] = [values[1]]
                        if debug:
                            print(f"DEBUG: First column value extracted: '{values[1]}'")
                    else:
                        td[column_list[idx]] = [cleaned_text.strip()]
                        if debug:
                            print(f"DEBUG: First column fallback: '{cleaned_text.strip()}'")
                except Exception as e:
                    print(f"Index Error occurred in first column: {e}")
                    td[column_list[idx]] = [""]
                    row_successful = False
            else:
                try:
                    td[column_list[idx]] = [cleaned_text.strip()]
                    if debug:
                        print(f"DEBUG: Column {idx} value: '{cleaned_text.strip()}'")
                except Exception as e:
                    print(f"Index Error occurred in column {idx}: {e}")
                    td[column_list[idx]] = [""]
                    row_successful = False
        
        if row_successful:
            successful_parses += 1
        else:
            parsing_errors += 1
            
        if debug:
            print(f"DEBUG: Row {token_idx + 1} result: {td}")
            print(f"DEBUG: Row successful: {row_successful}")
        
        result_list.append(pd.DataFrame(td))

    # Combine results
    if result_list:
        generated_df = pd.concat(result_list, ignore_index=True, axis=0)
        df_gen = pd.concat([df_gen, generated_df], ignore_index=True, axis=0)
    
    # Print summary statistics
    success_rate = successful_parses / total_rows * 100 if total_rows > 0 else 0
    print(f"PARSING SUMMARY: {successful_parses}/{total_rows} successful ({success_rate:.1f}%), {parsing_errors} errors")
    
    if debug:
        print(f"DEBUG: Final dataframe shape: {df_gen.shape}")
        print(f"DEBUG: Final dataframe columns: {df_gen.columns.tolist()}")
    
    return df_gen




def _convert_tokens_to_text(tokens: tp.List[torch.Tensor], tokenizer: AutoTokenizer) -> tp.List[str]:
    """ Decodes the tokens back to strings

    Args:
        tokens: List of tokens to decode
        tokenizer: Tokenizer used for decoding

    Returns:
        List of decoded strings
    """
    # Convert tokens to text
    text_data = [tokenizer.decode(t) for t in tokens]

    # Clean text
    text_data = [d.replace("<|endoftext|>", "") for d in text_data]
    text_data = [d.replace("\n", " ") for d in text_data]
    text_data = [d.replace("\r", "") for d in text_data]

    return text_data



def _convert_text_to_tabular_data(text: tp.List[str], df_gen: pd.DataFrame) -> pd.DataFrame:
    """ Converts the sentences back to tabular data

    Args:
        text: List of the tabular data in text form
        df_gen: Pandas DataFrame where the tabular data is appended

    Returns:
        Pandas DataFrame with the tabular data from the text appended
    """
    columns = df_gen.columns.to_list()
    result_list = []
    # Convert text to tabular data
    for t in text:
        features = t.split(",")
        td = dict.fromkeys(columns)
        
        # Transform all features back to tabular data
        for f in features:
            values = f.strip().split(" ")
            if values[0] in columns and not td[values[0]]:
                try:
                    td[values[0]] = [values[1]]
                except IndexError:
                    #print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
                    pass
        result_list.append(pd.DataFrame(td))   
        # df_gen = pd.concat([df_gen, pd.DataFrame(td)], ignore_index=True, axis=0)
    generated_df = pd.concat(result_list, ignore_index=True, axis=0)
    df_gen = pd.concat([df_gen, generated_df], ignore_index=True, axis=0)
    return df_gen