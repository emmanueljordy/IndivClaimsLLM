#!/usr/bin/env python3
"""Optimized Qwen/Qwen3-0.6B training script with QLoRA and advanced optimizations."""

import pandas as pd
import numpy as np
import torch
import time
import json
import pathlib
import argparse
import sys
import os
from tqdm import tqdm
from typing import Tuple, Dict, Sequence
from transformers import AutoTokenizer

import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Import Tabula with optimizations
from tabula_middle_padding_compromise.tabula import Tabula
from tabula_middle_padding_compromise.data_preprocessing import (
    clean_bi_dataset_for_tokenization, reverse_bi_cleaning,
    clean_eh_dataset_for_tokenization, reverse_eh_cleaning
)
from azure_model_utils import create_model_manager

def signed_log1p(x: np.ndarray, eps: float) -> np.ndarray:
    """Apply signed log1p transform to handle negative values."""
    return np.sign(x) * np.log1p(np.abs(x) / eps)

def inverse_signed_log1p(z: np.ndarray, eps: float) -> np.ndarray:
    """Inverse of signed log1p transform."""
    return np.sign(z) * eps * (np.exp(np.abs(z)) - 1)

def prep_for_llm(
    df: pd.DataFrame,
    open_prefixes: Sequence[str] = ("Open",),
    size_prefixes: Sequence[str] = ("Pay",),
    epsilon: float | None = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Transform DataFrame for LLM training."""
    df_trans = df.copy()

    # Locate columns
    open_cols = [
        c for c in df_trans.columns
        if any(c.startswith(p) for p in open_prefixes)
    ]
    size_cols = [
        c for c in df_trans.columns
        if any(c.startswith(p) for p in size_prefixes)
    ]

    # Choose epsilon if not supplied
    if epsilon is None:
        pos_vals = pd.concat([df_trans[c][df_trans[c] > 0] for c in size_cols])
        if pos_vals.empty:
            raise ValueError("No strictly positive payments to determine epsilon.")
        epsilon = 0.5 * pos_vals.min()

    # OPEN → categorical tokens
    for col in open_cols:
        df_trans[col] = (
            df_trans[col]
            .fillna(-1)
            .astype(int)
            .map({0: f"{col}=0", 1: f"{col}=1", -1: f"{col}=nan"})
        )

    # SIZE → signed log transform
    for col in size_cols:
        df_trans[col] = signed_log1p(df_trans[col], epsilon)

    meta = {
        "epsilon": epsilon,
        "open_cols": open_cols,
        "size_cols": size_cols,
        "open_prefixes": tuple(open_prefixes),
        "size_prefixes": tuple(size_prefixes),
    }
    return df_trans, meta

def recover_from_llm(df_gen: pd.DataFrame, meta: Dict) -> pd.DataFrame:
    """Recover DataFrame from LLM output to original scale."""
    df_orig = df_gen.copy()
    ε = meta["epsilon"]

    # OPEN tokens → numeric (extract digits after = sign)
    for col in meta["open_cols"]:
        df_orig[col] = (
            df_orig[col]
            .astype(str)
            .str.extract(r"=(\d+)", expand=False)  # Extract digits after = sign
            .astype("float")
            .astype("Int64")
        )

    # SIZE signed log transform → original scale
    for col in meta["size_cols"]:
        df_orig[col] = inverse_signed_log1p(df_orig[col], ε)

    return df_orig

def hms(seconds: float | int) -> str:
    """Convert seconds → H:MM:SS."""
    seconds = int(round(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"

def main():
    parser = argparse.ArgumentParser(description="Train Qwen/Qwen3-0.6B with QLoRA optimizations")
    parser.add_argument("--dataset", required=True, choices=["bi", "eh"], help="Dataset to use")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--logging_dir", required=True, help="Logging directory")
    parser.add_argument("--run_name", required=True, help="Experiment run name")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of synthetic samples")
    parser.add_argument("--hf_token", type=str, help="HuggingFace token for accessing gated models")
    parser.add_argument("--continue_training", action="store_true", help="Continue training from existing model")
    parser.add_argument("--fresh_training", action="store_true", help="Force fresh training (skip model registry check)")
    parser.add_argument("--num_epochs", type=int, help="Number of additional epochs for continue training (overrides default 40)")
    parser.add_argument("--use_qlora", action="store_true", help="Enable QLoRA for memory-efficient training (default: False for better performance)")
    parser.add_argument("--use_diversity_enhancement", action="store_true", help="Enable diversity enhancement for categorical columns")
    parser.add_argument("--no_diversity_enhancement", action="store_true", help="Disable diversity enhancement for categorical columns")
    
    args = parser.parse_args()
    
    # Validate that either continue_training or fresh_training is specified, but not both
    if args.continue_training and args.fresh_training:
        print("❌ ERROR: Cannot specify both --continue_training and --fresh_training")
        sys.exit(1)
    if not args.continue_training and not args.fresh_training:
        print("❌ ERROR: Must specify either --continue_training or --fresh_training")
        sys.exit(1)
    
    # Initialize Azure ML model manager
    model_manager = create_model_manager()
    
    # Setup HuggingFace authentication if token provided
    if args.hf_token:
        os.environ['HF_TOKEN'] = args.hf_token
        try:
            from huggingface_hub import login
            login(token=args.hf_token)
            print("HuggingFace authentication successful")
        except ImportError:
            print("WARNING: huggingface_hub not available, relying on environment variable")
        except Exception as e:
            print(f"WARNING: HuggingFace login failed: {e}")
    elif os.getenv('HF_TOKEN'):
        print("Using HF_TOKEN from environment")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load and prepare data with tokenization-aware cleaning
    print("Loading and preparing data with tokenization-aware cleaning...")
    if args.dataset == "bi":
        data_all = pd.read_csv("../data/raw/BI_10000_short.txt", sep=";")
        data = data_all.iloc[:, 1:]  # Remove first column
        
        # Apply BI-specific tokenization-aware cleaning
        print("Applying BI dataset tokenization-aware cleaning...")
        df_clean, cleaning_metadata = clean_bi_dataset_for_tokenization(data)
        
    else:  # eh
        data_all = pd.read_csv("../data/raw/EH_10000_short.txt", sep=";")
        data = data_all.iloc[:, 1:]  # Remove first column
        data = data.drop(['claim.nr','occ.date', 'rep.date', 'settlement.date', 'occ.year', 'rep.year'], axis=1, errors='ignore')
       
        
        # Apply EH-specific tokenization-aware cleaning
        print("Applying EH dataset tokenization-aware cleaning...")
        df_clean, cleaning_metadata = clean_eh_dataset_for_tokenization(data)
    
    # Use cleaned data for training
    df_llm = df_clean
    
    # Set categorical columns based on cleaned data
    if args.dataset == "bi":
        categorical_columns = ['LoB', 'cc']
    else:  # eh
        categorical_columns = ['type', 'hidden'] 
    # Convert object columns to category for memory efficiency
    # (data preprocessing already converted most, but ensure consistency)
    for col in categorical_columns:
        if col in df_llm.columns and df_llm[col].dtype == 'object':
            df_llm[col] = df_llm[col].astype('category')
            print(f"   Converted {col} from object to category")
    
    # Convert any remaining object columns to category
    remaining_object_cols = df_llm.select_dtypes(include=['object']).columns
    if not remaining_object_cols.empty:
        for col in remaining_object_cols:
            df_llm[col] = df_llm[col].astype('category')
            print(f"   Converted {col} from object to category")
    
    print(f"Data cleaning complete - Shape: {df_llm.shape}")
    print(f"Columns after cleaning: {list(df_llm.columns)[:10]}...")  # Show first 10 columns
    print(f"Data types: {df_llm.dtypes.value_counts().to_dict()}")
    print(f"Memory usage: {df_llm.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # Tokenizer setup
    print("Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    
    SEP = " | "
    def row_to_text(row):
        return SEP.join(str(v) for v in row)
    
    # Debug: Print sample training format
    sample_text = row_to_text(df_llm.iloc[0])
    print(f"Training text format: {sample_text[:200]}...")
    print(f"Using separator: '{SEP}'")

    # Compute token lengths
    lengths = []
    for _, row in tqdm(df_llm.iterrows(), total=len(df_llm), desc="Computing token lengths"):
        text = row_to_text(row)
        lengths.append(len(tokenizer(text).input_ids))

    # Dataset-specific MAX_TOKENS - increased for better generation
    # Use 95th percentile + buffer for complete record generation
    recommended_tokens = int(np.percentile(lengths, 95) * 1.5)  # 50% buffer
    MAX_TOKENS = max(256, recommended_tokens)  # Minimum 256 tokens
    tokenizer.model_max_length = MAX_TOKENS
    
    print(f"Token length stats - min: {np.min(lengths)}, mean: {np.mean(lengths):.1f}, "
          f"95th pct: {np.percentile(lengths, 95):.1f}, max: {np.max(lengths)}")
    print(f"Using MAX_TOKENS: {MAX_TOKENS}")
    
    # Calculate separate generation length for better completion
    GENERATION_MAX_LENGTH = max(MAX_TOKENS * 3, 600)
    print(f"Using GENERATION_MAX_LENGTH: {GENERATION_MAX_LENGTH}")

    # QLoRA Configuration optimized for Qwen/Qwen3-0.6B (GPU only)
    if device == "cuda":
        qlora_quantization_config = {
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16  # Qwen3 optimized for bfloat16
        }
    else:
        # Disable quantization for CPU
        qlora_quantization_config = None
        print("CPU detected: Disabling quantization for compatibility")

    qlora_lora_config = {
        "r": 64,                    # Optimized rank for 0.6B model (reduced from 96)
        "lora_alpha": 128,          # Proportional alpha (reduced from 192)
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Qwen3 specific modules
        "lora_dropout": 0.08,       # Slightly higher dropout for smaller model
        "bias": "none",
        "use_rslora": True,
        "use_dora": False,          # Disable DoRA with quantization for stability
        "init_lora_weights": True   # Use standard Kaiming uniform initialization
    }

    # Create output directories
    output_path = pathlib.Path(args.output_dir)
    logging_path = pathlib.Path(args.logging_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logging_path.mkdir(parents=True, exist_ok=True)

    # Check for existing model (local cache or Azure ML registry)
    model_path = output_path / "model"
    
    if args.fresh_training:
        print("=" * 60)
        print("FRESH TRAINING: Starting Qwen/Qwen3-0.6B with QLoRA (4-bit + LoRA)")
        print("=" * 60)
        final_model_path = str(model_path)
        needs_training = True
    else:
        final_model_path, needs_training = model_manager.get_or_create_model(
            base_model="qwen3_0_6b",  # Changed to distinguish from old Qwen2-1.5B models
            dataset=args.dataset,
            local_model_path=str(model_path)
        )
    
    # Handle different training modes
    if args.continue_training:
        if pathlib.Path(final_model_path).exists():
            print("=" * 60)
            print("CONTINUE TRAINING: Loading existing Qwen/Qwen3-0.6B QLoRA model")
            print("=" * 60)
            print(f"Found existing model at: {final_model_path}")
            
            start_time = time.time()
            tabula_qlora = Tabula.load_from_dir(str(final_model_path))
            load_time = time.time() - start_time
            
            print(f"Existing Qwen/Qwen3-0.6B model loaded in {load_time:.2f}s")
            print(f"Previous training config:")
            print(f"   Original epochs: {getattr(tabula_qlora, 'epochs', 'unknown')}")
            print(f"   Original batch size: {getattr(tabula_qlora, 'batch_size', 'unknown')}")
            
            # Check column compatibility for continue training
            original_columns = set(tabula_qlora.columns) if hasattr(tabula_qlora, 'columns') else set()
            current_columns = set(df_llm.columns)
            
            if original_columns and original_columns != current_columns:
                print(f"⚠️  WARNING: Column mismatch detected!")
                print(f"   Original model columns: {len(original_columns)}")
                print(f"   Current data columns: {len(current_columns)}")
                missing_in_current = original_columns - current_columns
                extra_in_current = current_columns - original_columns
                if missing_in_current:
                    print(f"   Missing in current data: {list(missing_in_current)[:5]}...")
                if extra_in_current:
                    print(f"   Extra in current data: {list(extra_in_current)[:5]}...")
                print(f"   This may cause training errors. Consider using fresh training instead.")
            else:
                print(f"✅ Column structure matches original model")
            
            # Update training parameters for continuation
            if args.num_epochs is not None:
                print(f"Continuing training for {args.num_epochs} additional epochs")
                tabula_qlora.epochs = args.num_epochs
            else:
                print(f"Continuing training for 40 additional epochs")
                tabula_qlora.epochs = 40
                
            # Update batch size
            tabula_qlora.batch_size = 32
            
            # Continue training
            print("Continuing Qwen/Qwen3-0.6B training...")
            start_time = time.time()
            try:
                trainer_qlora = tabula_qlora.fit(df_llm, debug_training=True, use_qlora=args.use_qlora)
            except Exception as e:
                print(f"❌ Continue training failed: {e}")
                print(f"This is likely due to data format mismatch.")
                print(f"Consider starting fresh training instead of continuing.")
                raise
            training_time = time.time() - start_time
            print(f"Qwen/Qwen3-0.6B continue training completed in {training_time:.2f}s")
            
            # Save and register the updated model
            print(f"Saving and registering continued training model...")
            model_manager.save_and_register_model(
                tabula_model=tabula_qlora,
                local_model_path=str(model_path),
                base_model="qwen3_0_6b",
                dataset=args.dataset
            )
            print("Continued training model saved and registered successfully")
            
        else:
            print("Continue training requested but no model found. Starting fresh training...")
            needs_training = True
    
    elif not needs_training:
        print("=" * 60)
        print("LOADING: Existing Qwen/Qwen3-0.6B QLoRA model")
        print("=" * 60)
        print(f"Found existing model at: {final_model_path}")
        
        start_time = time.time()
        # Load the complete trained model with all attributes
        tabula_qlora = Tabula.load_from_dir(str(final_model_path))
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f}s")
        print("Skipping training - using existing model")
        print(f"Loaded model columns: {tabula_qlora.columns}")
        print(f"Loaded categorical columns: {tabula_qlora.categorical_columns}")
        training_time = 0
        
    elif needs_training:
        if args.fresh_training:
            print("Starting fresh training - skipping model registry checks...")
        else:
            print("=" * 60)
            print("INITIALIZING: Qwen/Qwen3-0.6B with QLoRA (4-bit + LoRA)")
            print("=" * 60)
        
        # Determine diversity enhancement setting
        use_diversity = False  # Default disabled for safety
        if hasattr(args, 'use_diversity_enhancement') and args.use_diversity_enhancement:
            use_diversity = True
            print("⚠️  Diversity enhancement: ENABLED (WARNING: may generate invalid categories)")
        else:
            print("Diversity enhancement: DISABLED (default for safety)")
            
        start_time = time.time()
        tabula_qlora = Tabula(
            llm='Qwen/Qwen3-0.6B',
            tokenizer=tokenizer,
            experiment_dir=str(logging_path),
            epochs=args.num_epochs if args.num_epochs else 40,  # More epochs for smaller model
            batch_size=32,                  # Larger batch for 0.6B model
            categorical_columns=categorical_columns,
            use_lora=True,
            lora_config=qlora_lora_config,
            quantization_config=qlora_quantization_config,
            # Diversity enhancement settings
            use_diversity_enhancement=use_diversity,
            categorical_temperature=1.2 if use_diversity else 1.0,  # Moderate temperature for Qwen
            small_categorical_temperature=1.5 if use_diversity else 1.0,
            max_seq_length=MAX_TOKENS,
            gradient_accumulation_steps=8,  # Reduced for smaller model
            learning_rate=2e-4,             # Higher LR for smaller model
            warmup_steps=200,               # More warmup for smaller model
            weight_decay=0.01,
            lr_scheduler_type="cosine",
        )

        init_time = time.time() - start_time
        print(f"Qwen/Qwen3-0.6B QLoRA initialization time: {init_time:.2f}s")
        print(f"Trainable parameters: {sum(p.numel() for p in tabula_qlora.model.parameters() if p.requires_grad):,}")
        
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        # Training
        print("Training Qwen/Qwen3-0.6B with QLoRA...")
        start_time = time.time()
        
        trainer_qlora = tabula_qlora.fit(df_llm, debug_training=True, use_qlora=args.use_qlora)  # Moderate subset
        
        training_time = time.time() - start_time
        print(f"Qwen/Qwen3-0.6B QLoRA training completed in {training_time:.2f}s")
        
        # Save and register the trained model
        if args.fresh_training:
            print(f"Saving fresh training model locally...")
            tabula_qlora.save(str(model_path))
            print("Fresh training model saved locally (not registered to avoid conflicts)")
        else:
            print(f"Saving and registering trained model...")
            model_manager.save_and_register_model(
                tabula_model=tabula_qlora,
                local_model_path=str(model_path),
                base_model="qwen3_0_6b",  # Changed to match the registry key above
                dataset=args.dataset
            )
            print("Model saved and registered successfully")

    # Generation with optimizations
    print("\nGenerating synthetic samples with Qwen3-0.6B QLoRA and optimizations...")
    print(f"Target samples: {args.num_samples}")
    print(f"Max length: {GENERATION_MAX_LENGTH} tokens")
    print(f"Temperature: 0.8")
    print(f"Generation batch size: 96")
    start_time = time.time()
    
    synthetic_data = tabula_qlora.sample(
        n_samples=args.num_samples,
        temperature=0.9,            # Higher temperature for diversity in small model
        max_length=GENERATION_MAX_LENGTH,
        device=device,
        use_flash_attention=True,   # Enable for 0.6B model efficiency
        use_kv_cache=True,
        batch_size_generation=96,   # Larger batch for smaller model
        debug=True,                 # Enable detailed debug output
        verbose=True,               # Enable verbose logging
        debug_interval=20           # Show debug info every 20 attempts
    )
    
    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.2f}s")
    print(f"Generated samples: {len(synthetic_data)}")
    print(f"Success rate: {len(synthetic_data)/args.num_samples*100:.1f}%")
    print(f"Speed: {len(synthetic_data)/generation_time:.1f} samples/second")

    if len(synthetic_data) == 0:
        print("ERROR: No valid samples generated! Check generation parameters.")
        sys.exit(1)
    elif len(synthetic_data) < args.num_samples * 0.1:  # Less than 10% success
        print("WARNING: Very low success rate. Consider:")
        print("   - Increasing max_length")
        print("   - Reducing temperature")
        print("   - Training for more epochs")

    # Reverse tokenization-aware cleaning to get original scale data
    try:
        if args.dataset == "bi":
            print("Reversing BI dataset cleaning to original scale...")
            df_synth = reverse_bi_cleaning(synthetic_data, cleaning_metadata)
        else:  # eh
            print("Reversing EH dataset cleaning to original scale...")
            df_synth = reverse_eh_cleaning(synthetic_data, cleaning_metadata)
            
        print(f"Successfully reversed cleaning for {len(df_synth)} samples to original scale")
    except Exception as e:
        print(f"ERROR: Failed to reverse data cleaning to original scale: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Validate synthetic data before saving
    if df_synth.empty:
        print("ERROR: Recovered synthetic data is empty")
        sys.exit(1)
    
    if df_synth.isnull().all().all():
        print("ERROR: Recovered synthetic data contains only null values")
        sys.exit(1)
    
    # Save synthetic data with error handling
    synthetic_path = output_path / "synthetic_data_qwen3.csv"
    try:
        df_synth.to_csv(synthetic_path, index=False)
        print(f"Synthetic data saved successfully to: {synthetic_path}")
        
        # Verify file was created and has content
        if not synthetic_path.exists():
            raise FileNotFoundError(f"CSV file was not created at {synthetic_path}")
        
        if synthetic_path.stat().st_size == 0:
            raise ValueError(f"CSV file is empty at {synthetic_path}")
            
    except Exception as e:
        print(f"ERROR: Failed to save synthetic data to {synthetic_path}: {e}")
        sys.exit(1)
    
    # Model already saved during training (or loaded from existing)
    
    # Save experiment results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": "Qwen/Qwen3-0.6B",
        "dataset": args.dataset,
        "run_name": args.run_name,
        "max_tokens": MAX_TOKENS,
        "num_samples": args.num_samples,
        "training_secs": training_time,
        "training_hms": hms(training_time),
        "generation_secs": generation_time,
        "generation_hms": hms(generation_time),
        "samples_per_second": len(synthetic_data) / max(generation_time, 0.001),  # Avoid division by zero
        "trainable_params": sum(p.numel() for p in tabula_qlora.model.parameters() if p.requires_grad)
    }
    
    # Save experiment results with error handling
    results_path = output_path / "experiment_results.json"
    try:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Experiment results saved successfully to: {results_path}")
        
        # Verify file was created and has content
        if not results_path.exists():
            raise FileNotFoundError(f"JSON file was not created at {results_path}")
        
        if results_path.stat().st_size == 0:
            raise ValueError(f"JSON file is empty at {results_path}")
            
        # Validate JSON content
        with open(results_path, "r") as f:
            json.load(f)  # This will raise an exception if JSON is invalid
            
    except Exception as e:
        print(f"ERROR: Failed to save experiment results to {results_path}: {e}")
        sys.exit(1)
    
    print(f"\nExperiment completed successfully!")
    print(f"Model saved to: {model_path}")
    print(f"Synthetic data saved to: {synthetic_path}")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()