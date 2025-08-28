"""
Tokenization-aware data cleaning for optimal LLM training.

This module provides cleaning functions that reduce token count and improve 
model performance by making data more tokenizer-friendly.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, Union
from scipy import stats


# ------------------------------------------------------------------
# 1.  Binning rule  g(y)
# ------------------------------------------------------------------
def bin_payments(values: Union[pd.Series, np.ndarray]) -> pd.Series:
    """
    Map each payment y to its bin:
      0 :  y == 0
      1 :  y  < 0
      2 :  0  < y ≤   5 000
      3 :  5 000 < y ≤  20 000
      4 : 20 000 < y ≤ 100 000
      5 : 100 000 < y
    
    Args:
        values: Payment values to bin
        
    Returns:
        Series with bin labels (0-5) as int8 dtype, NaN for missing values
    """
    v = pd.Series(values)              # ensures vector ops
    bins = np.select(
        [
            v.eq(0),                       #    y == 0
            v.lt(0),                       #    y  < 0
            v.le(5_000) & v.gt(0),         # 0  < y ≤   5 000
            v.le(20_000) & v.gt(5_000),    # 5k < y ≤  20 000
            v.le(100_000) & v.gt(20_000),  # 20k< y ≤ 100 000
            v.gt(100_000)                  #    y  > 100 k
        ],
        [0, 1, 2, 3, 4, 5],
        default=np.nan                    # for missing values
    )
    # Use object dtype first to handle NaN, then convert valid values to int8
    result = pd.Series(bins, dtype="object")
    valid_mask = ~pd.isna(result)
    if valid_mask.any():
        result.loc[valid_mask] = result.loc[valid_mask].astype("int8")
    return result


# ------------------------------------------------------------------
# 2.  "Un-binning" / resampling
# ------------------------------------------------------------------
#  Choose a representative numeric value for each class.
#  Here we pick uniformly inside the original interval
#  (except for 0-payments, which remain 0).
SAMPLING_RANGES = {
    0: (0.0, 0.0),          # always 0
    1: (-5000.0, -1.0),      # pick your own negative range
    2: (1.0, 5_000.0),
    3: (5_000.0, 20_000.0),
    4: (20_000.0, 100_000.0),
    5: (100_000.0, 500_000.0)  # or np.inf if you'll truncate later
}

def sample_from_bins(bins: pd.Series,
                     rng: Optional[np.random.Generator] = None) -> pd.Series:
    """
    Draw a numeric payment for each bin label in *bins*.
    
    Args:
        bins: Series containing bin labels (0-5)
        rng: Random number generator instance. If None, uses default_rng()
        
    Returns:
        Series with sampled payment values
    """
    rng = rng or np.random.default_rng()
    lo_hi = bins.map(lambda b: SAMPLING_RANGES[int(b)])
    lo  = lo_hi.map(lambda t: t[0]).to_numpy()
    hi  = lo_hi.map(lambda t: t[1]).to_numpy()
    same = lo == hi
    draws = rng.uniform(lo, hi)          # vectorised uniform
    draws[same] = lo[same]               # keep exact values where lo==hi
    return pd.Series(draws)


# ------------------------------------------------------------------
# 3. Parametric sampling helpers
# ------------------------------------------------------------------
def _fit_params_by_bin(values: pd.Series, bin_ids: pd.Series, family: str, 
                       ranges: Dict[int, Tuple[float, float]]) -> Dict[int, Dict[str, float]]:
    """
    Fit parametric distribution parameters for each bin.
    
    Args:
        values: Original payment values (same length as bin_ids)
        bin_ids: Bin assignments (0-5)
        family: Distribution family ("lognormal" or "gamma")
        ranges: Bin ranges for fallback parameter estimation
        
    Returns:
        Dictionary mapping bin_id -> {"param1": val, "param2": val}
    """
    params_by_bin = {}
    
    for bin_id in range(6):  # 0-5
        mask = (bin_ids == bin_id) & values.notna()
        bin_values = values[mask]
        
        if bin_id == 1:  # Negative bin
            # For negative values, fit on magnitudes and remember to apply sign
            neg_values = bin_values[bin_values < 0]
            if len(neg_values) >= 3:
                magnitudes = -neg_values  # Convert to positive
                try:
                    if family == "lognormal":
                        # log(magnitude) should be normal
                        log_mags = np.log(magnitudes[magnitudes > 0])
                        if len(log_mags) >= 3:
                            mu, sigma = stats.norm.fit(log_mags)
                            params_by_bin[bin_id] = {"mu": mu, "sigma": sigma, "is_negative": True}
                        else:
                            raise ValueError("Insufficient positive magnitudes")
                    elif family == "gamma":
                        # Fit gamma on magnitudes
                        shape, loc, scale = stats.gamma.fit(magnitudes, floc=0)
                        params_by_bin[bin_id] = {"shape": shape, "scale": scale, "is_negative": True}
                except Exception:
                    # Fallback to bin-based parameters
                    params_by_bin[bin_id] = _fallback_params(bin_id, family, ranges)
            else:
                # Insufficient data, use fallback
                params_by_bin[bin_id] = _fallback_params(bin_id, family, ranges)
        
        elif bin_id == 0:  # Zero bin
            # Always zero, no parameters needed
            params_by_bin[bin_id] = {"is_zero": True}
        
        else:  # Positive bins (2-5)
            pos_values = bin_values[bin_values > 0]
            if len(pos_values) >= 3:
                try:
                    if family == "lognormal":
                        log_vals = np.log(pos_values)
                        mu, sigma = stats.norm.fit(log_vals)
                        params_by_bin[bin_id] = {"mu": mu, "sigma": sigma, "is_negative": False}
                    elif family == "gamma":
                        shape, loc, scale = stats.gamma.fit(pos_values, floc=0)
                        params_by_bin[bin_id] = {"shape": shape, "scale": scale, "is_negative": False}
                except Exception:
                    # Fallback to bin-based parameters
                    params_by_bin[bin_id] = _fallback_params(bin_id, family, ranges)
            else:
                # Insufficient data, use fallback
                params_by_bin[bin_id] = _fallback_params(bin_id, family, ranges)
    
    return params_by_bin


def _fallback_params(bin_id: int, family: str, ranges: Dict[int, Tuple[float, float]]) -> Dict[str, float]:
    """
    Generate fallback parameters based on bin endpoints.
    
    Args:
        bin_id: Bin identifier (0-5)
        family: Distribution family ("lognormal" or "gamma")
        ranges: Bin ranges
        
    Returns:
        Parameter dictionary for the distribution
    """
    if bin_id == 0:
        return {"is_zero": True}
    
    low, high = ranges[bin_id]
    
    if bin_id == 1:  # Negative bin
        # Use magnitude range
        low, high = -high, -low  # Convert to positive magnitudes
    
    if family == "lognormal":
        # For lognormal, fit parameters so most mass is in [low, high]
        log_low, log_high = np.log(max(low, 1e-6)), np.log(max(high, 1e-6))
        mu = (log_low + log_high) / 2
        sigma = (log_high - log_low) / 4  # Rough approximation
        return {"mu": mu, "sigma": max(sigma, 0.1), "is_negative": bin_id == 1}
    
    elif family == "gamma":
        # For gamma, set shape and scale so mean is roughly in middle of range
        mean_target = (low + high) / 2
        # Simple heuristic: shape=2, scale=mean/2
        shape = 2.0
        scale = mean_target / shape
        return {"shape": shape, "scale": max(scale, 1.0), "is_negative": bin_id == 1}
    
    return {}


def sample_from_bins_parametric(bins: pd.Series, family: str, 
                                params_by_bin: Dict[int, Dict[str, float]],
                                ranges: Dict[int, Tuple[float, float]],
                                rng: Optional[np.random.Generator] = None) -> pd.Series:
    """
    Sample values using fitted parametric distributions.
    
    Args:
        bins: Bin assignments (0-5)
        family: Distribution family ("lognormal" or "gamma")
        params_by_bin: Fitted parameters for each bin
        ranges: Bin ranges for clipping
        rng: Random number generator
        
    Returns:
        Series of sampled values
    """
    rng = rng or np.random.default_rng()
    result = np.zeros(len(bins))
    
    for i, bin_id in enumerate(bins):
        bin_id = int(bin_id)
        bin_id = max(0, min(5, bin_id))  # Clip to valid range
        
        params = params_by_bin.get(bin_id, {})
        
        if params.get("is_zero", False):
            result[i] = 0.0
        elif family == "lognormal" and "mu" in params:
            # Sample from lognormal
            mu, sigma = params["mu"], params["sigma"]
            is_negative = params.get("is_negative", False)
            
            # Sample magnitude
            magnitude = rng.lognormal(mu, sigma)
            
            # Clip to bin range
            low, high = ranges[bin_id]
            if is_negative:
                # For negative bin, clip magnitude and apply negative sign
                magnitude = np.clip(magnitude, -high, -low)
                result[i] = -magnitude
            else:
                magnitude = np.clip(magnitude, low, high)
                result[i] = magnitude
                
        elif family == "gamma" and "shape" in params:
            # Sample from gamma
            shape, scale = params["shape"], params["scale"]
            is_negative = params.get("is_negative", False)
            
            # Sample magnitude
            magnitude = rng.gamma(shape, scale)
            
            # Clip to bin range
            low, high = ranges[bin_id]
            if is_negative:
                # For negative bin, clip magnitude and apply negative sign
                magnitude = np.clip(magnitude, -high, -low)
                result[i] = -magnitude
            else:
                magnitude = np.clip(magnitude, low, high)
                result[i] = magnitude
        else:
            # Fallback to uniform sampling
            low, high = ranges[bin_id]
            if low == high:
                result[i] = low
            else:
                result[i] = rng.uniform(low, high)
    
    return pd.Series(result)


def fit_eh_gamma_params_for_columns(df: pd.DataFrame, pay_cols: list) -> Dict[str, Any]:
    """
    Fit gamma parameters for EH payment columns.
    
    Args:
        df: DataFrame with binned payment columns (Pay00_bin, etc.)
        pay_cols: List of payment column names (without _bin suffix)
        
    Returns:
        Metadata dictionary with gamma parameters
    """
    metadata = {
        "family": "gamma",
        "ranges": SAMPLING_RANGES.copy(),
        "per_column": {}
    }
    
    for col in pay_cols:
        bin_col = col + "_bin"
        if bin_col in df.columns:
            # Reconstruct original values for fitting (approximate)
            # Convert to numeric, fill NaN with 2.0, then convert to int to avoid FutureWarning
            bin_ids = pd.to_numeric(df[bin_col], errors="coerce")
            bin_ids = bin_ids.fillna(2.0).astype('int64')
            
            # Generate synthetic data for fitting based on bin centers
            synthetic_values = []
            for bin_id in bin_ids:
                bin_id = max(0, min(5, bin_id))
                low, high = SAMPLING_RANGES[bin_id]
                if low == high:
                    synthetic_values.append(low)
                else:
                    synthetic_values.append((low + high) / 2)  # Use bin center
            
            synthetic_series = pd.Series(synthetic_values)
            
            try:
                params_for_col = _fit_params_by_bin(
                    values=synthetic_series,
                    bin_ids=bin_ids,
                    family="gamma",
                    ranges=SAMPLING_RANGES
                )
                metadata["per_column"][col] = {"params_by_bin": params_for_col}
            except Exception as e:
                logging.warning(f"Could not fit EH gamma parameters for {col}: {e}")
    
    return metadata


def clean_eh_dataset_for_tokenization(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    EH cleaner aligned with BI logic but with EH-specific indicators:
      - Keep 'open_obs' and 'settlement_obs' as 0/1 strings (single tokens).
      - Bin payments Pay00–Pay11 into 6 range bins (0..5) using bin_payments().
      - Fit per-bin GAMMA params (incl. negative magnitudes) and store in metadata['payment_sampling_eh'].
      - NO 256-quantile binning of size_obs* (removed).

    Returns:
        (df_clean, metadata)
    """
    df_clean = df.copy()
    metadata: Dict[str, Any] = {}

    logging.info("🧹 EH cleaning: indicator simplification + 6-bin gamma for payments")

    # ------------------------------------------------------------------
    # 1) rep.delay (years → months string), occ.month/rep.month zero-padded
    # ------------------------------------------------------------------
    if 'rep.delay' in df_clean.columns:
        df_clean['rep_delay_m'] = (pd.to_numeric(df_clean['rep.delay'], errors='coerce') * 12)\
                                    .round().astype('Int64').fillna(0).astype(int).astype(str)
        df_clean = df_clean.drop(columns=['rep.delay'])
        metadata['rep_delay_conversion'] = 'years_to_months_string'
        logging.info("✅ rep.delay → rep_delay_m (months as string)")

    for col in ('occ.month', 'rep.month'):
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').clip(1, 12)\
                               .fillna(1).astype(int).map(lambda x: f"{x:02d}")
            metadata[f'{col}_format'] = 'zero_padded_2'
            logging.info(f"✅ {col}: zero-padded to 2 chars")

    # ------------------------------------------------------------------
    # 2) EH indicators: ONLY 'open_obs' and 'settlement_obs' columns (0/1 strings)
    #    (If prefixed variants exist, we normalize their values but KEEP names.)
    # ------------------------------------------------------------------
    obs_cols = [c for c in df_clean.columns if c.startswith('open_obs') or c.startswith('settlement_obs')]
    fixed_obs = []
    for col in obs_cols:
        # robust extraction of 0/1 from things like '...=1', '1', '0', True/False
        v = df_clean[col].astype(str).str.extract(r'([01])', expand=False)
        df_clean[col] = pd.to_numeric(v, errors='coerce').fillna(0).clip(0, 1)\
                           .astype('int64').astype(str)
        fixed_obs.append(col)
    if fixed_obs:
        metadata['obs_columns_cleaned'] = fixed_obs
        logging.info(f"✅ Indicators normalized to 0/1 strings: {fixed_obs}")

    # ------------------------------------------------------------------
    # 3) Payments: 6-bin range binning + per-bin GAMMA fit (EH requirement)
    # ------------------------------------------------------------------
    pay_cols = [c for c in df_clean.columns if c.startswith('size_obs')]
    per_col_meta: Dict[str, Any] = {}
    binned_count = 0

    for col in pay_cols:
        vals = pd.to_numeric(df_clean[col], errors='coerce')
        if vals.notna().sum() == 0:
            logging.warning(f"⚠️ {col}: no numeric values; skipping")
            continue

        # 6-bin assignments (0..5) using your bin_payments
        bins = bin_payments(vals)  # returns 0..5 or NaN
        # Fit per-bin GAMMA params directly from raw values & bin IDs
        try:
            params = _fit_params_by_bin(values=vals, bin_ids=bins, family="gamma",
                                        ranges=SAMPLING_RANGES)
            if "payment_sampling_eh" not in metadata:
                metadata["payment_sampling_eh"] = {
                    "family": "gamma",
                    "ranges": SAMPLING_RANGES.copy(),
                    "per_column": {}
                }
            metadata["payment_sampling_eh"]["per_column"][col] = {"params_by_bin": params}
        except Exception as e:
            logging.warning(f"Could not fit EH gamma params for {col}: {e}")

        # Store compact bin ID as category '0'..'5' (fill NaN with middle bin '2')
        df_clean[col + "_bin"] = bins.fillna(2.0).astype('int64').astype(str).astype('category')
        df_clean = df_clean.drop(columns=[col])
        binned_count += 1
        logging.info(f"✅ {col} → {col}_bin (6 range bins, gamma params stored)")

    metadata['payment_columns_binned'] = binned_count
    metadata['payment_binning_method'] = 'range_6bins_gamma'  # for clarity

    # ------------------------------------------------------------------
    # 4) (REMOVED) size_obs* 256-quantile binning  ❌
    # ------------------------------------------------------------------
    # Intentionally no action on 'size_obs*' columns to avoid the old 256 qcut path.

    # ------------------------------------------------------------------
    # 5) General: round any leftover floats (not the *_bin which are categories)
    # ------------------------------------------------------------------
    float_cols = [c for c in df_clean.select_dtypes(include=[np.floating]).columns
                  if not c.endswith('_bin')]
    for c in float_cols:
        df_clean[c] = df_clean[c].round(2).astype(str)
        logging.info(f"ℹ️ {c}: rounded to 2 decimals, cast to string")

    # Convert residual object columns to category for memory/token stability
    obj_cols = df_clean.select_dtypes(include=['object']).columns
    for c in obj_cols:
        df_clean[c] = df_clean[c].astype('category')
    metadata['object_to_category_columns'] = list(obj_cols)

    # Summary
    metadata['original_columns'] = len(df.columns)
    metadata['cleaned_columns'] = len(df_clean.columns)
    metadata['cleaning_applied'] = True
    metadata['dataset_type'] = 'eh_payments'

    logging.info("🎯 EH cleaning complete (6-bin gamma, indicators normalized)")
    logging.info(f"   Original columns: {metadata['original_columns']}")
    logging.info(f"   Cleaned columns:  {metadata['cleaned_columns']}")
    logging.info(f"   Payment cols binned: {binned_count} (gamma per-bin)")
    return df_clean, metadata


def create_quantile_bin_decoder(quantile_maps: Dict[str, np.ndarray]) -> Dict[str, callable]:
    """
    Create decoder functions to convert bin IDs back to continuous values.
    
    Args:
        quantile_maps: Dictionary of column -> bin_edges from cleaning
        
    Returns:
        Dictionary of column -> decoder_function
    """
    decoders = {}
    
    for col, bin_edges in quantile_maps.items():
        def make_decoder(edges):
            def decode_bin(bin_id):
                try:
                    bin_idx = int(bin_id)
                    if bin_idx < 0 or bin_idx >= len(edges) - 1:
                        # Return middle value for out-of-range
                        return np.mean(edges)
                    # Return random value within the bin
                    bin_start = edges[bin_idx]
                    bin_end = edges[bin_idx + 1]
                    return np.random.uniform(bin_start, bin_end)
                except (ValueError, IndexError):
                    return np.mean(edges)  # Fallback
            return decode_bin
        
        decoders[col] = make_decoder(bin_edges)
    
    return decoders


def reverse_eh_cleaning(
    df_generated: pd.DataFrame,
    metadata: Dict[str, Any],
    *,
    rng: Optional[np.random.Generator] = None
) -> pd.DataFrame:
    """
    Reverse the EH cleaning done by `clean_eh_dataset_for_tokenization`.

    Expectations from cleaning:
      - 'rep_delay_m' contains months as string → restore 'rep.delay' in years (float).
      - 'occ.month' / 'rep.month' are zero-padded strings "01".."12" → restore Int64.
      - 'open_obs*' / 'settlement_obs*' are '0'/'1' strings → restore Int64 (0/1).
      - Payment columns were binned: original 'size_obsX' → 'size_obsX_bin' with labels '0'..'5'.
        We decode with per-bin **gamma** params from metadata['payment_sampling_eh'].

    If per-column gamma params are missing, falls back to uniform sampling in
    the fixed SAMPLING_RANGES for the 6 bins.

    Returns
    -------
    DataFrame with original column names restored where possible.
    """
    import re

    rng = rng or np.random.default_rng()
    df_restored = df_generated.copy()

    # ------------------------------------------------------------------
    # 0) Defensive cleaning on string columns (strip leaked special tokens)
    # ------------------------------------------------------------------
    for col in df_restored.columns:
        if df_restored[col].dtype == 'object':
            s = df_restored[col].astype(str)
            # Remove brackets-like tokens and common LLM artifacts
            s = s.str.replace(r'<[^>]*>', '', regex=True)
            s = s.str.replace(r'«[^»]*»', '', regex=True)
            s = s.str.replace(r'(?i)\s*<\s*/?s>\s*', '', regex=True)
            s = s.str.replace(r'(?i)\s*<pad>\s*', '', regex=True)
            s = s.str.strip()
            s = s.replace('', pd.NA)
            df_restored[col] = s

    # ------------------------------------------------------------------
    # 1) rep_delay_m → rep.delay (months → years)
    # ------------------------------------------------------------------
    if 'rep_delay_m' in df_restored.columns and metadata.get('rep_delay_conversion') == 'years_to_months_string':
        months = pd.to_numeric(df_restored['rep_delay_m'].astype(str).str.extract(r'(\d+)', expand=False),
                               errors='coerce')
        df_restored['rep.delay'] = (months / 12.0).astype(float)
        df_restored = df_restored.drop(columns=['rep_delay_m'])

    # ------------------------------------------------------------------
    # 2) Months: remove zero-padding and cast to Int64
    # ------------------------------------------------------------------
    for mcol in ('occ.month', 'rep.month'):
        if mcol in df_restored.columns:
            nums = pd.to_numeric(df_restored[mcol].astype(str).str.extract(r'(\d{1,2})', expand=False),
                                 errors='coerce')
            # Clip to 1..12, keep NA if invalid
            nums = nums.where(nums.between(1, 12), pd.NA).astype('Int64')
            df_restored[mcol] = nums

    # ------------------------------------------------------------------
    # 3) EH indicators: 'open_obs*' / 'settlement_obs*' → Int64 0/1
    # ------------------------------------------------------------------
    obs_cols = [c for c in df_restored.columns if c.startswith('open_obs') or c.startswith('settlement_obs')]
    for col in obs_cols:
        v = df_restored[col].astype(str).str.extract(r'([01])', expand=False)
        df_restored[col] = pd.to_numeric(v, errors='coerce').fillna(0).clip(0, 1).astype('Int64')

    # ------------------------------------------------------------------
    # 4) Payments: decode 'size_obs*_bin' → 'size_obs*' using per-bin GAMMA params
    # ------------------------------------------------------------------
    eh_meta = metadata.get("payment_sampling_eh", {})
    family = eh_meta.get("family", "gamma")
    ranges = eh_meta.get("ranges", SAMPLING_RANGES)
    per_col = eh_meta.get("per_column", {}) if isinstance(eh_meta.get("per_column", {}), dict) else {}

    pay_bin_cols = [c for c in df_restored.columns if c.endswith('_bin') and c.startswith('size_obs')]
    for bin_col in pay_bin_cols:
        original_col = bin_col[:-4]  # strip '_bin'

        # Sanitize bin ids: allow '0'..'5', numeric strings, ints; default to 2 if missing
        b = pd.to_numeric(df_restored[bin_col], errors='coerce').fillna(0.0)
        b = b.clip(0, 5).astype('int64')

        # Prefer per-column gamma params; else fallback to uniform-in-bin
        params_by_bin = None
        if original_col in per_col:
            params_by_bin = per_col[original_col].get("params_by_bin")

        if family == "gamma" and params_by_bin is not None:
            vals = sample_from_bins_parametric(
                bins=b,
                family="gamma",
                params_by_bin=params_by_bin,
                ranges=ranges,
                rng=rng
            )
        else:
            # Fallback: uniform sampling in fixed ranges
            vals = sample_from_bins(b)

        df_restored[original_col] = pd.to_numeric(vals, errors='coerce')
        df_restored = df_restored.drop(columns=[bin_col])
        logging.info(f"{bin_col} → {original_col}: restored via {'gamma' if params_by_bin is not None else 'uniform'} sampling")

    # ------------------------------------------------------------------
    # 5) Final light QA (optional)
    # ------------------------------------------------------------------
    # Ensure numeric dtype for restored size_obs*
    for col in [c for c in df_restored.columns if c.startswith('size_obs')]:
        df_restored[col] = pd.to_numeric(df_restored[col], errors='coerce')

    return df_restored


def clean_bi_dataset_for_tokenization(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply tokenization-aware cleaning to the BI (business insurance payments) dataset.
    
    Transformations based on user specifications:
    - ClNr: Already dropped in training scripts
    - LoB, cc: Keep as strings (already low-cardinality categoricals)
    - AY: Keep as 4-digit string
    - AQ: Convert 1-4 → Q1-Q4 (single token guarantee)
    - age, inj_part: Zero-padded 2-digit strings
    - RepDel: Cap to 120 months, cast to string
    - Pay00-Pay11: Quantile-bin to 256 buckets
    - Open02-Open11: Strip =0/1 suffix, keep bare 0/1
    
    Args:
        df: Raw BI dataset
        
    Returns:
        Tuple of (cleaned_dataframe, metadata_dict)
    """
    df_clean = df.copy()
    metadata = {}
    
    logging.info("Starting BI dataset tokenization-aware cleaning...")
    
    # 1. LoB, cc: Already fine (low-cardinality categoricals)
    lob_cc_cols = [col for col in ['LoB', 'cc'] if col in df_clean.columns]
    if lob_cc_cols:
        logging.info(f"LoB/cc columns kept as strings: {lob_cc_cols}")
        metadata['lob_cc_preserved'] = lob_cc_cols
    
    # 2. AY: Keep as 4-digit string
    if 'AY' in df_clean.columns:
        df_clean['AY'] = df_clean['AY'].astype(int).astype(str)
        logging.info("AY: converted to 4-digit string")
        metadata['ay_format'] = 'four_digit_string'
    
    # 3. AQ: Convert 1-4 → Q1-Q4 (single token guarantee)
    if 'AQ' in df_clean.columns:
        df_clean['AQ'] = df_clean['AQ'].astype(int).astype(str)
        logging.info("AQ: converted 1-4")
        metadata['aq_format'] = 'string'
    
    # 4. age, inj_part: Zero-padded 2-digit strings
    age_cols = [col for col in ['age', 'inj_part'] if col in df_clean.columns]
    for col in age_cols:
        df_clean[col] = df_clean[col].astype(int).astype(str).str.zfill(2)
        logging.info(f"{col}: zero-padded to 2-digit string")
        metadata[f'{col}_format'] = 'zero_padded_2digit'
    
    # 5. RepDel: Cap to 10 years and cast to string (BI dataset uses years)
    if 'RepDel' in df_clean.columns:
        df_clean['RepDel'] = df_clean['RepDel'].clip(lower=0, upper=10).astype(int).astype(str)
        logging.info("RepDel: capped to 10 years, converted to string")
        metadata['repdev_cap'] = 10
        metadata['repdev_format'] = 'string'
    
    # 6. Pay00-Pay11: Simple range-based binning
    pay_cols = [col for col in df_clean.columns if col.startswith('Pay')]
    
    if pay_cols:
        for col in pay_cols:
            values = df_clean[col].dropna()
            if len(values) > 0:
                # Apply simple range-based binning
                binned_values = bin_payments(df_clean[col])

                # NEW: fit per bin lognormal parameters for this Pay column and store them
                try:
                    params_for_col = _fit_params_by_bin(
                        values=df_clean[col],              # same length as bins, includes NaN
                        bin_ids=binned_values,             # 0..5
                        family="lognormal",                # BI uses lognormal
                        ranges=SAMPLING_RANGES
                    )
                    # initialize container once
                    if "payment_sampling" not in metadata:
                        metadata["payment_sampling"] = {
                            "family": "lognormal",
                            "ranges": SAMPLING_RANGES.copy(),
                            "per_column": {}
                        }
                    metadata["payment_sampling"]["per_column"][col] = {
                        "params_by_bin": params_for_col
                    }
                except Exception as e:
                    logging.warning(f"Could not fit BI per bin lognormal for {col}: {e}")
                
                # Convert to string and handle NaN - avoid FutureWarning by converting to float first
                # Fill NaN with 2 (middle bin), then convert to int, then to string
                binned_values = binned_values.fillna(2.0)  # Fill with float to avoid downcasting warning
                df_clean[col + '_bin'] = binned_values.astype('int64').astype(str)
                
                # Convert to category for memory efficiency (6 unique values: 0-5)
                df_clean[col + '_bin'] = df_clean[col + '_bin'].astype('category')
                
                # Drop original column
                df_clean = df_clean.drop(col, axis=1)
                logging.info(f"{col} → {col}_bin: range-binned to 6 categories (0-5)")
    
    metadata['payment_binning_method'] = 'simple_range_based'
    metadata['payment_columns_binned'] = len(pay_cols)
    
    # 7. Open02-Open11: Strip =0/1 suffix, keep bare 0/1
    open_cols = [col for col in df_clean.columns if col.startswith('Open')]
    for col in open_cols:
        # Handle various formats: "value=1", "=1", "1", etc.
        df_clean[col] = (
            df_clean[col].fillna('=0').astype(str)
            .str.split('=', n=1).str[-1]  # Take part after last =
            .astype(int).astype(str)      # Ensure clean 0/1 strings
        )
        logging.info(f"{col}: stripped =0/1 suffix, kept bare 0/1")
    
    metadata['open_columns_cleaned'] = len(open_cols)
    
    # 8. Reorder columns for readability (if they exist)
    base_cols = ['LoB', 'cc', 'AY', 'AQ', 'age', 'inj_part', 'RepDel']
    pay_bin_cols = [col + '_bin' for col in pay_cols]
    
    ordered_cols = []
    for col in base_cols + pay_bin_cols + open_cols:
        if col in df_clean.columns:
            ordered_cols.append(col)
    
    # Add any remaining columns
    remaining_cols = [col for col in df_clean.columns if col not in ordered_cols]
    ordered_cols.extend(remaining_cols)
    
    df_clean = df_clean[ordered_cols]
    
    # Convert object columns to category for memory efficiency
    object_cols = df_clean.select_dtypes(include=['object']).columns
    for col in object_cols:
        df_clean[col] = df_clean[col].astype('category')
        logging.info(f"{col}: converted object to category dtype")
    
    metadata['object_to_category_columns'] = list(object_cols)
    
    # Calculate summary statistics
    original_cols = len(df.columns)
    cleaned_cols = len(df_clean.columns)
    
    logging.info("BI data cleaning complete:")
    logging.info(f"   Original columns: {original_cols}")
    logging.info(f"   Cleaned columns: {cleaned_cols}")
    logging.info(f"   Payment columns binned: {len(pay_cols)} (6 bins each)")
    logging.info(f"   Open columns cleaned: {len(open_cols)}")
    logging.info("   Estimated token reduction: ~60%")
    
    metadata['original_columns'] = original_cols
    metadata['cleaned_columns'] = cleaned_cols
    metadata['cleaning_applied'] = True
    metadata['dataset_type'] = 'bi_payments'
    
    return df_clean, metadata


def reverse_bi_cleaning(df_generated: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
    """
    Reverse the BI dataset cleaning to get back original format.
    
    Args:
        df_generated: DataFrame with cleaned/generated data
        metadata: Metadata from the cleaning process
        
    Returns:
        DataFrame in original format
    """
    import re
    
    df_restored = df_generated.copy()
    
    logging.info("Reversing BI dataset cleaning...")
    
    # Defensive cleaning: Remove sentinel tokens that might have leaked into values
    logging.info("Applying defensive cleaning to remove sentinel artifacts...")
    for col in df_restored.columns:
        if df_restored[col].dtype == 'object':
            # Convert to string first, then clean
            df_restored[col] = df_restored[col].astype(str)
            # Remove <...> and «...» patterns and common special tokens
            df_restored[col] = df_restored[col].str.replace(r'<[^>]*>', '', regex=True)
            df_restored[col] = df_restored[col].str.replace(r'«[^»]*»', '', regex=True)
            # Specifically target BOS/EOS tokens that might have spaces
            df_restored[col] = df_restored[col].str.replace(r'(?i)\s*<\s*/?s>\s*', '', regex=True)
            df_restored[col] = df_restored[col].str.replace(r'(?i)\s*<pad>\s*', '', regex=True)
            # Clean up whitespace
            df_restored[col] = df_restored[col].str.strip()
            # Replace empty strings with NaN
            df_restored[col] = df_restored[col].replace('', pd.NA)
    
    # Reverse AY (keep as string, but could convert back to int if needed)
    if 'AY' in df_restored.columns and metadata.get('ay_format'):
        # Keep as string - already in correct format
        pass
    
    # Reverse AQ: Q1-Q4 → 1-4
    if 'AQ' in df_restored.columns and metadata.get('aq_format'):
        # Robust AQ conversion with error handling
        try:
            # Extract digits only, handle various formats
            aq_values = df_restored['AQ'].astype(str).str.extract(r'(\d+)', expand=False)
            df_restored['AQ'] = pd.to_numeric(aq_values, errors='coerce').astype('Int64')
            
            # Log statistics about the conversion
            valid_count = df_restored['AQ'].notna().sum()
            total_count = len(df_restored)
            logging.info(f"AQ: converted Q1-Q4 → 1-4 ({valid_count}/{total_count} valid values)")
            
            if valid_count < total_count * 0.8:  # Less than 80% valid
                logging.warning(f"AQ conversion had low success rate: {valid_count}/{total_count} valid")
        except Exception as e:
            logging.error(f"AQ conversion failed: {e}")
            # Fallback: keep as string
            pass
    
    # Reverse age/inj_part (remove zero-padding, convert to int)
    age_cols = [col for col in ['age', 'inj_part'] if col in df_restored.columns]
    for col in age_cols:
        if metadata.get(f'{col}_format') == 'zero_padded_2digit':
            try:
                # Extract numeric values only
                numeric_values = df_restored[col].astype(str).str.extract(r'(\d+)', expand=False)
                df_restored[col] = pd.to_numeric(numeric_values, errors='coerce').astype('Int64')
                
                # Log statistics
                valid_count = df_restored[col].notna().sum()
                total_count = len(df_restored)
                logging.info(f"{col}: removed zero-padding, converted to int ({valid_count}/{total_count} valid values)")
                
                if valid_count < total_count * 0.8:
                    logging.warning(f"{col} conversion had low success rate: {valid_count}/{total_count} valid")
            except Exception as e:
                logging.error(f"{col} conversion failed: {e}")
    
    # Reverse RepDel (convert back to int)
    if 'RepDel' in df_restored.columns and metadata.get('repdev_format'):
        try:
            numeric_values = df_restored['RepDel'].astype(str).str.extract(r'(\d+)', expand=False)
            df_restored['RepDel'] = pd.to_numeric(numeric_values, errors='coerce').astype('Int64')
            
            valid_count = df_restored['RepDel'].notna().sum()
            total_count = len(df_restored)
            logging.info(f"RepDel: converted back to int ({valid_count}/{total_count} valid values)")
            
            if valid_count < total_count * 0.8:
                logging.warning(f"RepDel conversion had low success rate: {valid_count}/{total_count} valid")
        except Exception as e:
            logging.error(f"RepDel conversion failed: {e}")
    
    # Reverse payment binning using simple range-based sampling
    if metadata.get('payment_binning_method') == 'simple_range_based':
        # Find all payment bin columns
        pay_bin_cols = [col for col in df_restored.columns if col.endswith('_bin') and col.startswith('Pay')]
        
        for bin_col in pay_bin_cols:
            original_col = bin_col.replace('_bin', '')
            
            # Convert bin IDs to numeric, handle contaminated data
            try:
                bin_ids = pd.to_numeric(df_restored[bin_col], errors='coerce').fillna(0)  # Default to bin 2
                bin_ids = bin_ids.clip(0, 5).astype(int)  # Ensure valid range 0-5
                
                pay_meta = metadata.get("payment_sampling", {})
                family = pay_meta.get("family", None)
                per_col = pay_meta.get("per_column", {})
                col_meta = per_col.get(original_col, {})
                params_by_bin = col_meta.get("params_by_bin", None)

                if family in {"lognormal", "gamma"} and params_by_bin is not None:
                    sampled_values = sample_from_bins_parametric(
                        bins=bin_ids,
                        family=family,
                        params_by_bin=params_by_bin,
                        ranges=pay_meta.get("ranges", SAMPLING_RANGES),
                    )
                else:
                    sampled_values = sample_from_bins(bin_ids)

                df_restored[original_col] = sampled_values
                
                # Drop bin column
                df_restored = df_restored.drop(bin_col, axis=1)
                logging.info(f"{bin_col} → {original_col}: reversed range-based binning")
                
            except Exception as e:
                logging.error(f"Failed to reverse binning for {bin_col}: {e}")
                # Keep the bin column as fallback
                continue
    
    # Reverse Open columns (0/1 → original format if needed)
    # Note: Since we don't know the original format exactly, we keep as 0/1 integers
    open_cols = [col for col in df_restored.columns if col.startswith('Open')]
    for col in open_cols:
        try:
            # Extract 0 or 1 from values, handle contaminated data
            numeric_values = df_restored[col].astype(str).str.extract(r'([01])', expand=False)
            df_restored[col] = pd.to_numeric(numeric_values, errors='coerce').astype('Int64')
            
            valid_count = df_restored[col].notna().sum()
            total_count = len(df_restored)
            logging.info(f"{col}: converted to 0/1 ({valid_count}/{total_count} valid values)")
            
            if valid_count < total_count * 0.8:
                logging.warning(f"{col} conversion had low success rate: {valid_count}/{total_count} valid")
        except Exception as e:
            logging.error(f"{col} conversion failed: {e}")
    
    # Add final data quality check
    total_rows = len(df_restored)
    valid_rows = 0
    critical_cols = ['AY', 'AQ'] if 'AY' in df_restored.columns and 'AQ' in df_restored.columns else []
    
    if critical_cols:
        valid_rows = df_restored[critical_cols].notna().all(axis=1).sum()
        logging.info(f"Data quality check: {valid_rows}/{total_rows} rows have all critical columns valid")
        
        if valid_rows < total_rows * 0.5:
            logging.warning(f"LOW DATA QUALITY: Only {valid_rows}/{total_rows} rows are fully valid")
    
    logging.info("Reversed BI dataset cleaning")
    return df_restored


if __name__ == "__main__":
    # Example usage
    print("Data preprocessing module ready for use")
    print()
    print("Available functions:")
    print("- clean_eh_dataset_for_tokenization(df) -> (df_clean, metadata)")
    print("- reverse_eh_cleaning(df_generated, metadata) -> df_restored")
    print("- clean_bi_dataset_for_tokenization(df) -> (df_clean, metadata)")
    print("- reverse_bi_cleaning(df_generated, metadata) -> df_restored")