import os
import warnings
import json
import typing as tp
import logging

import numpy as np
import pandas as pd

from tqdm import tqdm

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    from transformers import BitsAndBytesConfig
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False

import torch
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          TrainingArguments,
                          AutoConfig)

from tabula_middle_padding_compromise.tabula_dataset import TabulaDataset, TabulaDataCollator, normalize_tokenizer_and_model
from tabula_middle_padding_compromise.tabula_start import TabulaStart, CategoricalStart, ContinuousStart, RandomStart, CanonicalStart
from tabula_middle_padding_compromise.tabula_trainer import TabulaTrainer
from tabula_middle_padding_compromise.tabula_utils import _convert_tokens_to_dataframe_by_seps, _convert_tokens_to_dataframe_legacy, _array_to_dataframe, _get_column_distribution, _convert_tokens_to_text, \
    _convert_text_to_tabular_data
from tabula_middle_padding_compromise.generation_processors import (
    NoDoubleSepProcessor, AllowedSetProcessor, ColumnWiseProcessor, 
    infer_column_types, build_numeric_allowed_tokens, build_column_name_tokens,
    MinNonSpaceThenEOSProcessor, NoLeadingSpaceProcessor, NamePieceBlockerProcessor, 
    CollapseGuardProcessor, NoRowSepInCellProcessor
)
try:
    # Try to import the fixed version first
    from tabula_middle_padding_compromise.advanced_generation_processors_fixed import (
        ColumnProfiler, ColumnValueProcessor, print_profiling_summary
    )
    print("Using FIXED advanced generation processors")
except ImportError:
    # Fall back to original if fixed version not found
    from tabula_middle_padding_compromise.advanced_generation_processors import (
        ColumnProfiler, ColumnValueProcessor, print_profiling_summary
    )
    print("Using original advanced generation processors")
from tabula_middle_padding_compromise.schema_validator import (
    SchemaLearner, PostValidator, CharacterAwareLogitsProcessor, ColumnConstraint, ColumnType
)


class Tabula:
    """ Tabula Class

    The Tabula class handles the whole generation flow. It is used to fine-tune a large language model for tabular data,
    and to sample synthetic tabular data.

    Attributes:
        llm (str): HuggingFace checkpoint of a pretrained large language model, used a basis of our model
        tokenizer (AutoTokenizer): Tokenizer, automatically downloaded from llm-checkpoint
        model (AutoModelForCausalLM): Large language model, automatically downloaded from llm-checkpoint
        experiment_dir (str): Directory, where the training checkpoints will be saved
        epochs (int): Number of epochs to fine-tune the model
        batch_size (int): Batch size used for fine-tuning
        train_hyperparameters (dict): Additional hyperparameters added to the TrainingArguments used by the
         HuggingFaceLibrary, see here the full list of all possible values
         https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
        columns (list): List of all features/columns of the tabular dataset
        num_cols (list): List of all numerical features/columns of the tabular dataset
        conditional_col (str): Name of a feature/column on which the sampling can be conditioned
        conditional_col_dist (dict | list): Distribution of the feature/column specified by condtional_col
    """

    def __init__(self, llm: str, experiment_dir: str = "trainer_tabula", epochs: int = 100,
                 batch_size: int = 8, use_lora: bool = False, lora_config: tp.Optional[dict] = None,
                 quantization_config: tp.Optional[dict] = None, 
                 use_diversity_enhancement: bool = False,  # Disabled by default to prevent invalid categories
                 categorical_temperature: float = 1.2,
                 small_categorical_temperature: float = 1.5,
                 **train_kwargs):
        """ Initializes Tabula.

        Args:
            llm: HuggingFace checkpoint of a pretrained large language model, used a basis of our model
            experiment_dir:  Directory, where the training checkpoints will be saved
            epochs: Number of epochs to fine-tune the model
            batch_size: Batch size used for fine-tuning
            use_lora: Whether to use LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
            lora_config: Configuration dictionary for LoRA. If None, uses default LoRA settings
            quantization_config: Configuration dictionary for model quantization (4-bit/8-bit)
            use_diversity_enhancement: Whether to use enhanced categorical diversity sampling (WARNING: may generate invalid categories)
            categorical_temperature: Temperature for categorical column generation (higher = more diverse)
            small_categorical_temperature: Temperature for small categorical columns
            train_kwargs: Additional hyperparameters added to the TrainingArguments used by the HuggingFaceLibrary,
             see here the full list of all possible values
             https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
        """
        # Load Model and Tokenizer from HuggingFace
        self.llm = llm
        
        # Store diversity enhancement settings
        self.use_diversity_enhancement = use_diversity_enhancement
        self.categorical_temperature = categorical_temperature
        self.small_categorical_temperature = small_categorical_temperature
        
        # Handle custom tokenizer if provided, otherwise use default
        custom_tokenizer = train_kwargs.pop('tokenizer', None)
        if custom_tokenizer is not None:
            self.tokenizer = custom_tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm, trust_remote_code=True)
            # Don't set pad_token = eos_token here - let normalize_tokenizer_and_model handle it
            
        self.config = AutoConfig.from_pretrained(self.llm, trust_remote_code=True)
        
        # Determine appropriate torch dtype with robust fallback
        torch_dtype = torch.float32  # Safe default
        
        if hasattr(self.config, 'torch_dtype') and self.config.torch_dtype is not None:
            # Handle string representations of dtypes
            if isinstance(self.config.torch_dtype, str):
                if self.config.torch_dtype == "auto":
                    torch_dtype = torch.float32  # Safe fallback for "auto"
                    logging.info("Config specified 'auto' dtype, using float32")
                elif self.config.torch_dtype in ["float32", "float16", "bfloat16", "float64"]:
                    try:
                        torch_dtype = getattr(torch, self.config.torch_dtype)
                        logging.info(f"Using config-specified dtype: {self.config.torch_dtype}")
                    except AttributeError:
                        logging.warning(f"Invalid dtype '{self.config.torch_dtype}' in config, using float32")
                        torch_dtype = torch.float32
                else:
                    logging.warning(f"Unrecognized dtype string '{self.config.torch_dtype}' in config, using float32")
                    torch_dtype = torch.float32
            elif isinstance(self.config.torch_dtype, torch.dtype):
                # It's already a torch dtype object
                torch_dtype = self.config.torch_dtype
                logging.info(f"Using config dtype object: {torch_dtype}")
            elif hasattr(self.config.torch_dtype, '__name__'):
                # It's some other dtype object
                torch_dtype = self.config.torch_dtype
                logging.info(f"Using config dtype object: {torch_dtype}")
            else:
                logging.warning(f"Unrecognized dtype type in config: {type(self.config.torch_dtype)}, using float32")
                torch_dtype = torch.float32
        else:
            logging.info("No torch_dtype in config, using default float32")
        
        logging.info(f"Final torch dtype: {torch_dtype}")
        
        self.model = AutoModelForCausalLM.from_config(self.config, torch_dtype=torch_dtype, trust_remote_code=True)
        
        # Normalize tokenizer and model - add missing special tokens and resize model if needed
        self.special_ids = normalize_tokenizer_and_model(self.tokenizer, self.model)

        # Extract other custom parameters that shouldn't go to TrainingArguments
        self.categorical_columns = train_kwargs.pop('categorical_columns', None)
        self.use_lora = use_lora
        self.lora_config = lora_config
        self.quantization_config = quantization_config
        self.max_seq_length = train_kwargs.pop('max_seq_length', None)

        # Set the training hyperparameters (remaining kwargs go to TrainingArguments)
        self.experiment_dir = experiment_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_hyperparameters = train_kwargs

        # Needed for the sampling process
        self.columns = None
        self.num_cols = None
        self.conditional_col = None
        self.conditional_col_dist = None
        self.token_list_length = []
        
        # Quick fixes: column configurations and generation processors
        self.column_configs = {}  # Per-column type info and constraints
        
        # Schema validation components
        self.schema_learner = None
        self.post_validator = None
        self.column_constraints = None
        self.enable_schema_validation = train_kwargs.pop('enable_schema_validation', True)
        self.train_span_tokens = []  # Full span length (name + value + sep) for training
        self.max_cell_tokens = {}  # Value-only length limits for generation

    def fit(self, data: tp.Union[pd.DataFrame, np.ndarray], column_names: tp.Optional[tp.List[str]] = None,
            conditional_col: tp.Optional[str] = None, resume_from_checkpoint: tp.Union[bool, str] = False,
            debug_training: bool = False, use_qlora: bool = False) -> TabulaTrainer:
        """ Fine-tune Tabula using tabular data.

        Args:
            data: Pandas DataFrame or Numpy Array that contains the tabular data
            column_names: If data is Numpy Array, the feature names have to be defined. If data is Pandas
            DataFrame, the value is ignored
            conditional_col: If given, the distribution of this column is saved and used as a starting
            point for the generation process later. If None, the first column is considered as conditional feature
            resume_from_checkpoint: Resume training from checkpoint (bool or path)  
            debug_training: Enable detailed training debug output
            use_qlora: Enable QLoRA for memory-efficient training (default False for better performance)

        Returns:
            TabulaTrainer used for the fine-tuning process
        """
        if debug_training:
            print(f"DEBUG: Starting fit with data shape: {data.shape}")
            print(f"DEBUG: Data columns: {list(data.columns)}")
            print(f"DEBUG: Conditional column: {conditional_col}")
            print(f"DEBUG: Tokenizer vocab size: {len(self.tokenizer)}")
            print(f"DEBUG: Model vocab size: {self.model.config.vocab_size}")
            print(f"DEBUG: Special token IDs: {self.special_ids}")
            
        # Show tokenizer normalization results
        print(f"TOKENIZER NORMALIZATION:")
        print(f"   Tokenizer vocab size: {len(self.tokenizer)}")
        print(f"   Model vocab size: {self.model.config.vocab_size}")
        print(f"   Special tokens: {self.special_ids}")
            
        # Enhanced token analysis similar to original tabula_middle_padding
        print(f"TOKEN ANALYSIS:")
        print(f"   Dataset shape: {data.shape}")
        print(f"   Columns: {list(data.columns)}")
        
        # Show sample data format that will be tokenized
        sample_row = data.iloc[0]
        print(f"   Sample row: {dict(sample_row)}")
        
        # ADVANCED COLUMN PROFILING: Use new value-only profiler
        print(f"\nADVANCED COLUMN PROFILING (VALUE-ONLY):")
        # Use enhanced profiler if diversity mode enabled
        if getattr(self, 'use_diversity_enhancement', False):
            try:
                from .diversity_enhanced_processors import enhance_column_profiler_with_distribution
                EnhancedProfiler = enhance_column_profiler_with_distribution(ColumnProfiler)
                self.column_profiler = EnhancedProfiler(self.tokenizer)
                print("Using diversity-enhanced profiler (experimental)")
            except ImportError:
                self.column_profiler = ColumnProfiler(self.tokenizer)
                print("Diversity profiler not available, using standard profiler")
        else:
            self.column_profiler = ColumnProfiler(self.tokenizer)
        self.column_profiles = self.column_profiler.profile_dataset(data, self.special_ids)
        
        # Print comprehensive profiling summary
        print_profiling_summary(self.column_profiles, self.tokenizer)
        
        # LEGACY COMPATIBILITY: Build old-style configs for existing code
        print(f"\nBUILDING LEGACY COMPATIBILITY LAYER:")
        self.column_name_tokens = build_column_name_tokens(self.tokenizer, list(data.columns))
        self.column_configs = self._build_legacy_configs_from_profiles()
        
        for var in data.columns:
            # TRAINING SPAN: Still compute for collator compatibility (name + value + separator)
            encoded_term = [f"{var} {str(data[var][i]).strip()}" for i in data.index]
            token_list = self.tokenizer(encoded_term)
            max_len = max(len(l) for l in token_list['input_ids'])
            train_span = max_len + 1  # +1 for separator
            self.token_list_length.append(train_span)
            self.train_span_tokens.append(train_span)
            
            # GENERATION LIMIT: Use value-only limit from advanced profiler
            profile = self.column_profiles.get(var)
            if profile:
                value_only_limit = profile.gen_limit
                col_type = profile.column_type.value
            else:
                # Fallback for missing profiles
                value_only_limit = 1
                col_type = 'unknown'
            
            self.max_cell_tokens[var] = value_only_limit
            
            # Enhanced debug output
            print(f"   Column '{var}' (type: {col_type}): train_span={train_span}, gen_limit={value_only_limit}")
            if profile:
                print(f"     Value stats: p95={profile.p95_len:.1f}, p99={profile.p99_len:.1f}, unique_ratio={profile.unique_ratio:.3f}")
            
            if debug_training:
                print(f"   Sample text: '{encoded_term[0]}'")
                sample_tokens = token_list['input_ids'][0]
                print(f"   Sample token IDs: {sample_tokens[:15]}{'...' if len(sample_tokens) > 15 else ''}")
                
                # Show allowed tokens for constrained columns (use legacy config)
                legacy_config = self.column_configs.get(var, {})
                if 'allowed_tokens' in legacy_config:
                    allowed_count = len(legacy_config['allowed_tokens'])
                    print(f"   Allowed tokens: {allowed_count} tokens")
                
                # Show token-to-text mapping for first few tokens (only for first column to avoid spam)
                is_first_column = var == data.columns[0]
                if is_first_column:
                    print(f"   Token breakdown:")
                    for i, token_id in enumerate(sample_tokens[:10]):
                        try:
                            decoded = self.tokenizer.decode([token_id])
                            print(f"      Token {i}: ID={token_id} -> '{decoded}'")
                        except Exception as e:
                            print(f"      Token {i}: ID={token_id} -> ERROR: {e}")
        
        print(f"   Final training spans: {self.train_span_tokens}")
        print(f"   Generation limits: {self.max_cell_tokens}")
        print(f"   Total sequence length: {sum(self.token_list_length)}")
        print(f"   Expected separators per row: {len(data.columns)-1} <COLSEP> + 1 <ROWSEP>")
        
        # Check for common separator tokens that might appear in data
        if debug_training:
            print(f"SEPARATOR TOKEN ANALYSIS:")
            for test_id in [216, 220, 32000, 32001]:  # Check common separator tokens
                try:
                    decoded = self.tokenizer.decode([test_id])
                    print(f"   Token {test_id} -> '{decoded}'")
                except Exception as e:
                    print(f"   Token {test_id} -> ERROR: {e}")
                    
            # Show what common punctuation tokenizes to
            test_strings = [" | ", ", ", "; ", " , ", "  ", "\t", "\n"]
            print(f"   Common separators:")
            for test_str in test_strings:
                try:
                    tokens = self.tokenizer(test_str)['input_ids']
                    print(f"   '{test_str}' -> {tokens}")
                except Exception as e:
                    print(f"   '{test_str}' -> ERROR: {e}")


        df = _array_to_dataframe(data, columns=column_names)
        self._update_column_information(df)
        self._update_conditional_information(df, conditional_col)
        
        # Check QLoRA compatibility before applying
        self._check_model_compatibility()
        
        # Apply QLoRA configuration if requested (optional for performance)
        if use_qlora:
            print("Applying QLoRA configuration...")
            self._apply_qlora_configuration()
        else:
            print("QLoRA disabled - using full fine-tuning for better performance")
        
        # Store column medians for better sample generation
        self._column_medians = {}
        for col in self.num_cols:
            if col in df.columns:
                self._column_medians[col] = df[col].median()
                if debug_training:
                    print(f"   Stored median for {col}: {self._column_medians[col]}")
  

        # Convert DataFrame into HuggingFace dataset object
        logging.info("Convert data into HuggingFace dataset object...")
        tabula_ds = TabulaDataset.from_pandas(df)
        tabula_ds.set_tokenizer(self.tokenizer, self.special_ids)

        data_collator = TabulaDataCollator(self.tokenizer, debug=debug_training)
        data_collator.set_token_list_length(self.token_list_length)
        data_collator.set_special_ids(self.special_ids)
        # tabula_ds.set_token_list_length(self.token_list_length)        

        # Set training hyperparameters
        logging.info("Create Tabula Trainer...")
        training_args = TrainingArguments(self.experiment_dir,
                                          num_train_epochs=self.epochs,
                                          per_device_train_batch_size=self.batch_size,
                                          save_strategy="no",
                                          report_to=[],  # Disable automatic logging to avoid Azure ML limits
                                          **self.train_hyperparameters)
        
        # Create trainer with debug option and REDUCED separator loss weighting
        # QUICK FIX: Reduce separator loss weight from 2.0 to 1.1 to prevent overemphasis
        tabula_trainer = TabulaTrainer(model=self.model, 
                                     args=training_args, 
                                     train_dataset=tabula_ds, 
                                     tokenizer=self.tokenizer,
                                     data_collator=data_collator, 
                                     debug_training=debug_training,
                                     special_ids=self.special_ids,
                                     sep_loss_weight=1.1)  # Reduced from 2.0 to prevent consecutive separators
        
        if debug_training:
            print(f"DEBUG: Created TabulaTrainer with debug enabled")
            print(f"DEBUG: Training arguments: epochs={self.epochs}, batch_size={self.batch_size}")
            print(f"DEBUG: Dataset size: {len(tabula_ds)} samples")

        # Start training
        logging.info("Start training...")
        if debug_training:
            print(f"DEBUG: Starting training...")
        tabula_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        return tabula_trainer
    
    def _build_legacy_configs_from_profiles(self) -> dict:
        """Build legacy column configs from new ColumnProfiles for backward compatibility."""
        legacy_configs = {}
        
        for col_name, profile in self.column_profiles.items():
            config = {
                'max_tokens': profile.gen_limit,
                'type': profile.column_type.value,
            }
            
            # Add type-specific fields for legacy compatibility
            if profile.column_type.value in ['categorical', 'small_categorical']:
                # Convert trie to allowed tokens set if available
                if profile.trie and hasattr(profile.trie, '_training_sequences'):
                    all_tokens = set()
                    for tokens in profile.trie._training_sequences:
                        all_tokens.update(tokens)
                    config['allowed_tokens'] = all_tokens
                elif profile.global_top_k:
                    config['allowed_tokens'] = profile.global_top_k
            
            elif profile.column_type.value in ['numeric', 'discrete_numeric', 'continuous_numeric']:
                if profile.global_top_k:
                    config['allowed_tokens'] = profile.global_top_k
            
            elif profile.global_top_k:
                config['allowed_tokens'] = profile.global_top_k
            
            legacy_configs[col_name] = config
        
        return legacy_configs
    
    def _generate_value_tokens_manual(self, running_ids_batch: tp.List[tp.List[int]], max_tokens: int, 
                                     processor, temperature: float, device: str, debug: bool = False) -> tp.List[tp.List[int]]:
        """Generate value tokens manually (token-by-token) to avoid EOS issues.
        
        Args:
            running_ids_batch: List of token sequences (already contains col name + space)
            max_tokens: Maximum tokens to generate per sequence
            processor: ColumnValueProcessor for constraints
            temperature: Sampling temperature
            device: Device for tensors
            debug: Enable debug logging
            
        Returns:
            List of generated token lists for each sequence
        """
        from torch.nn.functional import softmax
        
        batch_size = len(running_ids_batch)
        generated_tokens_batch = [[] for _ in range(batch_size)]
        
        # Manual token-by-token loop
        for step in range(max_tokens):
            # Pad and convert to tensor
            max_len = max(len(seq) for seq in running_ids_batch)
            padded_sequences = []
            attention_masks = []
            
            for seq in running_ids_batch:
                pad_length = max_len - len(seq)
                padded_seq = [self.special_ids["pad_id"]] * pad_length + seq
                padded_sequences.append(padded_seq)
                
                attention_mask = [0] * pad_length + [1] * len(seq)
                attention_masks.append(attention_mask)
            
            input_ids = torch.tensor(padded_sequences, dtype=torch.long, device=device)
            attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)
            
            # Forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            
            # Apply processor constraints
            masked_logits = processor(input_ids, logits)
            
            # Extra safety: hard-ban separators (redundant with processor but defensive)
            colsep_id = self.special_ids.get('colsep_id')
            rowsep_id = self.special_ids.get('rowsep_id')
            if colsep_id is not None:
                masked_logits[:, colsep_id] = -float("inf")
            if rowsep_id is not None:
                masked_logits[:, rowsep_id] = -float("inf")
            
            # Sample next tokens
            if temperature > 0:
                probs = softmax(masked_logits / temperature, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [batch_size]
            else:
                next_token_ids = torch.argmax(masked_logits, dim=-1)  # [batch_size]
            
            # Debug logging
            if debug:
                if step == 0:
                    top_tokens = torch.topk(masked_logits[0], 5).indices.tolist()
                    colsep_allowed = masked_logits[0, colsep_id].item() > -1e30 if colsep_id is not None else "N/A"
                    print(f"[MANUAL] step {step} top5={top_tokens}, colsep_allowed={colsep_allowed}")
                
                # Show sampled tokens
                sample_tokens = [int(next_token_ids[i].item()) for i in range(min(4, batch_size))]
                print(f"[STEP] step {step} next_ids[:4] -> {sample_tokens}")
            
            # Append tokens and update state
            for b in range(batch_size):
                token_id = int(next_token_ids[b].item())
                running_ids_batch[b].append(token_id)
                generated_tokens_batch[b].append(token_id)
                
            # Update processor state ONLY for the first sequence (representative)
            # The processor tracks column-level constraints, not per-sequence state
            if batch_size > 0:
                token_id = int(next_token_ids[0].item())
                space_id = self.special_ids.get('space_id')
                if token_id not in (space_id, colsep_id, rowsep_id):
                    processor.update_state_with_token(token_id)
            
            # Early stopping: check if processor says we should close
            if processor.should_force_close():
                if debug:
                    print(f"[MANUAL] Early close at step {step}, value_step={processor.column_state.value_step}, emitted={processor.column_state.emitted_nonspace_count}")
                break
        
        if debug:
            final_lengths = [len(tokens) for tokens in generated_tokens_batch[:4]]
            print(f"[MANUAL] Final generated lengths: {final_lengths}")
        
        return generated_tokens_batch
    
    def _sample_segmented(self, n_samples: int, start_col: str, start_col_dist, temperature: float,
                         device: str, debug: bool, verbose: bool, use_kv_cache: bool,
                         use_flash_attention: bool, batch_size_generation: int, use_allowed_mask: str) -> pd.DataFrame:
        """SEGMENT-WISE GENERATION: Generate tabular data column by column.
        
        This method generates each column separately, injecting column names and separators
        ourselves while letting the model only generate the values. This guarantees
        correct structure and eliminates consecutive separator issues.
        
        Args:
            All parameters from main sample method
            
        Returns:
            DataFrame with generated samples
        """
        if verbose:
            print(f"\n=== SEGMENT-WISE GENERATION ===")
            print(f"Columns: {self.columns}")
            print(f"Max cell tokens per column: {self.max_cell_tokens}")
        
        self.model.to(device)
        
        # Apply speed optimizations
        if use_flash_attention:
            try:
                if hasattr(self.model.config, 'attn_implementation'):
                    self.model.config.attn_implementation = "flash_attention_2"
                if verbose:
                    print("Flash Attention 2 enabled")
            except Exception as e:
                if verbose:
                    print(f"Flash Attention 2 not available: {e}")
        
        # Initialize results
        results = []
        
        with tqdm(total=n_samples, desc="Generating samples (segment-wise)") as pbar:
            batch_start = 0
            
            while batch_start < n_samples:
                batch_end = min(batch_start + batch_size_generation, n_samples)
                actual_batch_size = batch_end - batch_start
                
                if verbose and batch_start == 0:
                    print(f"\nProcessing batch {batch_start}-{batch_end} (size: {actual_batch_size})")
                
                # Initialize running prompts for this batch - all start empty
                running_ids = [[] for _ in range(actual_batch_size)]
                
                # Generate each column sequentially
                for col_idx, col_name in enumerate(self.columns):
                    is_last_column = (col_idx == len(self.columns) - 1)
                    
                    if verbose and batch_start == 0:
                        print(f"  Column {col_idx+1}/{len(self.columns)}: {col_name}")
                    
                    # ENTER COLUMN: Batch-wide idempotent logic
                    col_name_tokens = self.tokenizer.encode(col_name, add_special_tokens=False)
                    space_id = self.special_ids.get('space_id')
                    
                    # Debug: Check sequences before entering column
                    if verbose and batch_start == 0 and col_idx <= 4:
                        tails = [seq[-3:] if len(seq) >= 3 else seq for seq in running_ids[:2]]
                        print(f"[ENTER] {col_name} before: tails -> {tails}")
                    
                    # 1) Append column name tokens to ALL sequences
                    for i in range(actual_batch_size):
                        running_ids[i].extend(col_name_tokens)
                    
                    # 2) Force exactly one space for ALL sequences
                    if space_id is not None:
                        for i in range(actual_batch_size):
                            if not running_ids[i] or running_ids[i][-1] != space_id:
                                running_ids[i].append(space_id)
                    
                    # 3) Initialize processor if needed BEFORE reset
                    if not hasattr(self, '_column_value_processor') or col_idx == 0:
                        # Use enhanced processor if diversity mode is enabled
                        if getattr(self, 'use_diversity_enhancement', False):
                            try:
                                from .diversity_enhanced_processors import BatchOptimizedProcessor
                                self._column_value_processor = BatchOptimizedProcessor(
                                    profiles=self.column_profiles,
                                    special_ids=self.special_ids,
                                    columns=self.columns,
                                    tokenizer=self.tokenizer,
                                    categorical_temperature=getattr(self, 'categorical_temperature', 1.2),
                                    small_categorical_temperature=getattr(self, 'small_categorical_temperature', 1.5),
                                    use_distribution_matching=True
                                )
                                if verbose:
                                    print("Using diversity-enhanced processor (experimental)")
                            except ImportError:
                                # Fall back to standard processor if enhanced one not available
                                self._column_value_processor = ColumnValueProcessor(
                                    profiles=self.column_profiles,
                                    special_ids=self.special_ids,
                                    columns=self.columns,
                                    tokenizer=self.tokenizer
                                )
                                if verbose:
                                    print("Diversity processor not available, using standard processor")
                        else:
                            self._column_value_processor = ColumnValueProcessor(
                                profiles=self.column_profiles,
                                special_ids=self.special_ids,
                                columns=self.columns,
                                tokenizer=self.tokenizer
                            )
                        if col_idx == 0:  # Reset for new batch
                            self._column_value_processor.reset_for_new_sequence()
                    
                    # 4) Reset processor state (DO NOT update state with forced space)
                    self._column_value_processor.reset_for_new_column(col_idx)
                    
                    # Debug: Check sequences after entering column
                    if verbose and batch_start == 0 and col_idx <= 4:
                        tails = [seq[-3:] if len(seq) >= 3 else seq for seq in running_ids[:2]]
                        print(f"[ENTER] {col_name} after: tails -> {tails} (last must be {space_id})")
                    
                    # 5) Convert to tensor (rebuild from SAME list objects we are mutating)
                    # Ensure we're working with independent list copies to avoid aliasing
                    working_sequences = [list(seq) for seq in running_ids]  # Deep copy to avoid aliasing
                    
                    max_prompt_len = max(len(seq) for seq in working_sequences)
                    padded_prompts = []
                    attention_masks = []
                    
                    for seq in working_sequences:
                        # Left-pad to max length (consistent with our -1 indexing)
                        pad_length = max_prompt_len - len(seq)
                        padded_seq = [self.special_ids["pad_id"]] * pad_length + seq
                        padded_prompts.append(padded_seq)
                        
                        # Create attention mask (0 for padding, 1 for real tokens)
                        attention_mask = [0] * pad_length + [1] * len(seq)
                        attention_masks.append(attention_mask)
                    
                    input_ids = torch.tensor(padded_prompts, dtype=torch.long, device=device)
                    attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)
                    
                    # 4) HARD ASSERT: Verify ALL sequences end with space (simple check)
                    if space_id is not None:
                        # Simple check: last token in each sequence should be space_id
                        # (Works for both left and right padding since we check the actual last token)
                        last_tokens = input_ids[:, -1]  # [batch_size] - last token of each sequence
                        space_check = (last_tokens == space_id)
                        
                        if not space_check.all():
                            # Debug failed sequences
                            failed_seqs = (~space_check).nonzero(as_tuple=True)[0]
                            for b in failed_seqs[:3]:  # Show first 3 failures
                                seq_content = running_ids[b][-10:] if len(running_ids[b]) >= 10 else running_ids[b]
                                print(f"❌ SPACE CHECK FAILED for seq {b}:")
                                print(f"   running_ids tail: {seq_content}")
                                print(f"   last token in tensor: {last_tokens[b].item()}, expected: {space_id}")
                            assert False, f"Guard: {failed_seqs.numel()} sequences don't end with space_id={space_id}"
                        
                        if verbose and batch_start == 0 and col_idx <= 4:
                            print(f"✅ All {actual_batch_size} sequences end with space_id={space_id}")
                    
                    # Set up generation parameters for this column
                    max_tokens_for_col = self.max_cell_tokens.get(col_name, 10)  # fallback
                    eos_token_id = self.special_ids["rowsep_id"] if is_last_column else self.special_ids["colsep_id"]
                    
                    # ADVANCED GENERATION PROCESSOR: Use new value-only system
                    from transformers import LogitsProcessorList
                    processors = LogitsProcessorList()
                    
                    # (Processor already initialized and reset above in enter column logic)
                    
                    # Add the main processor
                    processors.append(self._column_value_processor)
                    
                    # Add legacy double-sep prevention as backup
                    sep_ids = [self.special_ids["colsep_id"], self.special_ids["rowsep_id"]]
                    sep_ids = [sid for sid in sep_ids if sid is not None]
                    if sep_ids:
                        processors.append(NoDoubleSepProcessor(sep_ids))
                    
                    # SANITY LOGS: Check state before generation
                    if verbose and batch_start == 0 and col_idx <= 4:
                        profile = self.column_profiles.get(col_name)
                        if profile:
                            print(f"    Advanced processor: type={profile.column_type.value}, gen_limit={profile.gen_limit}")
                            print(f"    Constraint info: trie={profile.trie is not None}, global_top_k={len(profile.global_top_k) if profile.global_top_k else 0}")
                        
                        # Log context state before generation
                        for i in range(min(1, actual_batch_size)):  # Just first sequence for brevity
                            last8 = input_ids[i, -8:].tolist() if input_ids.size(1) >= 8 else input_ids[i].tolist()
                            space_id = self.special_ids.get('space_id')
                            want_space_last = len(last8) > 0 and last8[-1] == space_id
                            print(f"[GEN] {col_name} step0 last8={last8} (want last token == space_id={space_id})")
                            print(f"[GEN] {col_name} context ends with space: {want_space_last}")
                            print(f"[GEN] {col_name} processor state: step={self._column_value_processor.column_state.value_step}, nonspace={self._column_value_processor.column_state.emitted_nonspace_count}")
                    
                    # Dynamic temperature: higher for flexible columns, lower for constrained ones
                    profile = self.column_profiles.get(col_name)
                    if profile and profile.column_type.value in ['small_categorical', 'discrete_numeric']:
                        segment_temperature = min(temperature * 0.5, 0.1)  # Very low for exact matching
                    else:
                        segment_temperature = min(temperature, 0.3)  # Moderate for others
                    
                    # Enable debug logging for processor if this is first batch and early columns
                    if verbose and batch_start == 0 and col_idx < 3:
                        self._column_value_processor.enable_debug_logging(True)
                    else:
                        self._column_value_processor.enable_debug_logging(False)
                    
                    # Generate value tokens for this column
                    generation_kwargs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "max_new_tokens": max_tokens_for_col,
                        "do_sample": True if segment_temperature > 0 else False,  # Greedy if temp=0
                        "temperature": segment_temperature,
                        "top_p": 0.9,
                        "eos_token_id": eos_token_id,
                        "pad_token_id": self.special_ids["pad_id"],
                        "repetition_penalty": 1.05,
                        "use_cache": use_kv_cache,
                        "logits_processor": processors,  # This ensures our masks are applied
                        "return_dict_in_generate": True,
                        "output_scores": True,  # Enable for sanity check logging
                    }
                    
                    # Debug processor state at column start
                    if verbose and batch_start == 0 and col_idx < 3:
                        print(f"    Segment temperature: {segment_temperature} (from {temperature})")
                    
                    # Generate tokens manually (token-by-token) to avoid EOS issues
                    with torch.no_grad():
                        generated_tokens_batch = self._generate_value_tokens_manual(
                            running_ids, max_tokens_for_col, processors[0], segment_temperature, device, verbose and batch_start == 0 and col_idx <= 4
                        )
                        
                    # (Manual generation - no need to extract sequences)
                    
                    # Debug: Show lengths after manual generation
                    if verbose and batch_start == 0 and col_idx <= 4:
                        lengths = [len(seq) for seq in running_ids[:4]]
                        print(f"[VALUE] {col_name} lengths after generation: {lengths}")
                    
                    # Process the manually generated tokens and append to running_ids
                    for i in range(actual_batch_size):
                        generated_tokens = generated_tokens_batch[i]
                        
                        # Debug proof that tokens were appended
                        if verbose and batch_start == 0 and col_idx <= 4 and i < 2:
                            print(f"[APPLY] {col_name} seq {i} appended -> {generated_tokens}")
                            print(f"[TAIL]  {col_name} seq {i} context tail -> {running_ids[i][-8:]}")
                        
                        # Add the appropriate separator (append manually, not via EOS)
                        running_ids[i].append(eos_token_id)
                        
                        # Debug: Show final sequence after separator
                        if verbose and batch_start == 0 and col_idx <= 4 and i < 2:
                            print(f"[FINAL] {col_name} seq {i} final tail -> {running_ids[i][-8:]}")
                    
                    # Notify processor that column is complete
                    self._column_value_processor.notify_column_complete()
                    
                    # Debug output for first batch and first few columns
                    if verbose and batch_start == 0 and col_idx <= 4:
                        # Use manually generated tokens for debug display
                        if generated_tokens_batch and len(generated_tokens_batch[0]) > 0:
                            sample_continuation = generated_tokens_batch[0]
                        else:
                            sample_continuation = []
                        if sample_continuation:
                            sample_text = self.tokenizer.decode(sample_continuation, skip_special_tokens=False)
                        else:
                            sample_text = '(empty)'
                        print(f"    Sample generated: '{sample_text[:50]}{'...' if len(sample_text) > 50 else ''}'")
                
                # Convert completed sequences to dataframe rows
                batch_rows = []
                for i in range(actual_batch_size):
                    # Decode the full sequence
                    full_sequence = running_ids[i]
                    try:
                        decoded_text = self.tokenizer.decode(full_sequence, skip_special_tokens=False)
                        
                        # Parse using the existing separator-based parsing
                        # Convert to tensor format expected by the parser
                        seq_tensor = torch.tensor(full_sequence)
                        temp_df = pd.DataFrame(columns=self.columns)
                        
                        parsed_df = _convert_tokens_to_dataframe_by_seps(
                            [seq_tensor],
                            tokenizer=self.tokenizer,
                            column_list=self.columns,
                            special_ids=self.special_ids,
                            df_gen=temp_df,
                            start_col=start_col or self.columns[0],
                            debug=(debug and len(batch_rows) == 0)
                        )
                        
                        if len(parsed_df) > 0:
                            batch_rows.append(parsed_df.iloc[0].to_dict())
                        else:
                            # Fallback: create empty row
                            batch_rows.append({col: "" for col in self.columns})
                            
                    except Exception as e:
                        if verbose:
                            print(f"    Error parsing sequence {i}: {e}")
                        # Create empty row as fallback
                        batch_rows.append({col: "" for col in self.columns})
                
                # Add batch to results
                results.extend(batch_rows)
                pbar.update(actual_batch_size)
                batch_start = batch_end
        
        # Convert results to DataFrame
        final_df = pd.DataFrame(results)
        
        # Clean up numerical columns
        for col in self.num_cols:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        
        if verbose:
            print(f"\nSegment-wise generation complete: {len(final_df)} samples")
            
            # Print advanced processor statistics
            if hasattr(self, '_column_value_processor'):
                stats = self._column_value_processor.get_instrumentation_stats()
                print(f"\nADVANCED PROCESSOR STATISTICS:")
                print(f"  Total mask collapses: {stats['total_mask_collapses']}")
                print(f"  Total fallback activations: {stats['total_fallback_activations']}")
                if stats['limit_enforcements_by_column']:
                    print(f"  Gen limit enforcements: {dict(list(stats['limit_enforcements_by_column'].items())[:3])}")
                if stats['early_closes_by_column']:
                    print(f"  Early closes (exact matches): {dict(list(stats['early_closes_by_column'].items())[:3])}")
            
            if debug:
                print("Sample results:")
                print(final_df.head(3))
        
        return final_df.head(n_samples)  # Ensure exact count

    def sample(self, n_samples: int,
               start_col: tp.Optional[str] = "", start_col_dist: tp.Optional[tp.Union[dict, list]] = None,
               temperature: float = 0.7, k: int = 100, max_length: int = 100, device: str = "cuda", debug: bool = False,
               use_flash_attention: bool = False, use_kv_cache: bool = False, batch_size_generation: tp.Optional[int] = None,
               verbose: bool = True, debug_interval: int = 50, segmented_decode: bool = True, 
               use_allowed_mask: str = 'categorical') -> pd.DataFrame:
        """ Generate synthetic tabular data samples

        Args:
            n_samples: Number of synthetic samples to generate
            start_col: Feature to use as starting point for the generation process. If not given, the target
             learned during the fitting is used as starting point
            start_col_dist: Feature distribution of the starting feature. Should have the format
             "{F1: p1, F2: p2, ...}" for discrete columns or be a list of possible values for continuous columns.
             If not given, the target distribution learned during the fitting is used as starting point
            temperature: The generation samples each token from the probability distribution given by a softmax
             function. The temperature parameter controls the softmax function. A low temperature makes it sharper
             (0 equals greedy search), a high temperature brings more diversity but also uncertainty into the output.
             See this blog article (https://huggingface.co/blog/how-to-generate) to read more about the generation
             process
            k: Sampling Batch Size. Set as high as possible. Speeds up the generation process significantly
            max_length: Maximal number of tokens to generate - has to be long enough to not cut any information!
            device: Set to "cpu" if the GPU should not be used. You can also specify the concrete GPU
            debug: Enable debug output during generation
            use_flash_attention: Enable flash attention for memory efficiency (currently for compatibility)
            use_kv_cache: Enable key-value caching for faster generation (currently for compatibility)
            batch_size_generation: Override batch size for generation (if None, uses k parameter)
            verbose: Enable detailed logging of generation attempts and rejection reasons
            debug_interval: How often to log detailed generation information
            segmented_decode: Use segment-wise generation (recommended, True by default)
            use_allowed_mask: Apply token constraints ('categorical', 'all', or 'none')

        Returns:
            Pandas DataFrame with n_samples rows of generated data
        """
        tabula_start = self._get_start_sampler(start_col, start_col_dist)

        # Handle batch size parameter
        effective_batch_size = batch_size_generation if batch_size_generation is not None else k

        # Move model to device
        self.model.to(device)
        
        # MODEL DIAGNOSTIC: Check if model seems undertrained
        if debug or verbose:
            print(f"MODEL DIAGNOSTIC:")
            print(f"   Model has {sum(p.numel() for p in self.model.parameters())} total parameters")
            print(f"   Model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} trainable parameters")
            
            # QLoRA parameter efficiency check
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            if hasattr(self, 'use_lora') and self.use_lora:
                param_efficiency = (trainable_params / total_params) * 100
                print(f"   🔧 QLoRA efficiency: {param_efficiency:.2f}% parameters trainable")
                if param_efficiency > 10:
                    print(f"   ⚠️  WARNING: QLoRA efficiency seems low ({param_efficiency:.2f}% > 10%)")
                    print(f"   💡 Consider reducing LoRA rank or target modules for better efficiency")
                elif param_efficiency < 0.1:
                    print(f"   ⚠️  WARNING: QLoRA efficiency extremely low ({param_efficiency:.2f}% < 0.1%)")
                    print(f"   💡 Consider increasing LoRA rank or adding more target modules")
                else:
                    print(f"   ✅ QLoRA efficiency looks good ({param_efficiency:.2f}%)")
            
            # Check if model weights look reasonable (not all zeros or random)
            with torch.no_grad():
                # Try to find the output layer (different models use different names)
                output_layer = None
                if hasattr(self.model, 'lm_head') and self.model.lm_head is not None:
                    output_layer = self.model.lm_head
                elif hasattr(self.model, 'output') and self.model.output is not None:
                    output_layer = self.model.output
                elif hasattr(self.model, 'classifier') and self.model.classifier is not None:
                    output_layer = self.model.classifier
                elif hasattr(self.model, 'head') and self.model.head is not None:
                    output_layer = self.model.head
                
                if output_layer is not None and hasattr(output_layer, 'weight'):
                    lm_head_mean = output_layer.weight.mean().item()
                    lm_head_std = output_layer.weight.std().item()
                    print(f"   Output layer weight stats: mean={lm_head_mean:.4f}, std={lm_head_std:.4f}")
                    
                    if abs(lm_head_mean) < 0.001 and lm_head_std < 0.01:
                        print("   WARNING: Output layer weights appear close to zero - model may be undertrained!")
                    elif lm_head_std > 1.0:
                        print("   WARNING: Output layer weights have high variance - model may be poorly initialized!")
                else:
                    # If we can't find the output layer, just skip this check
                    if verbose:
                        print("   (Output layer weight check skipped - layer not found)")
        
        # Apply speed optimizations
        if use_flash_attention:
            try:
                # Enable Flash Attention 2 if available
                self.model = self.model.to(device)
                if hasattr(self.model.config, 'attn_implementation'):
                    self.model.config.attn_implementation = "flash_attention_2"
                if debug or verbose:
                    print("Flash Attention 2 enabled")
            except Exception as e:
                if debug or verbose:
                    print(f"Flash Attention 2 not available: {e}")
        
        if debug:
            print(f"DEBUG: Starting generation for {n_samples} samples")
            print(f"DEBUG: Using device: {device}")
            print(f"DEBUG: Model parameters: temperature={temperature}, max_length={max_length}, batch_size={effective_batch_size}")
            print(f"DEBUG: Advanced features: flash_attention={use_flash_attention}, kv_cache={use_kv_cache}")
            print(f"DEBUG: Expected columns: {self.columns}")
            print(f"DEBUG: Token list length: {self.token_list_length}")

        # Init empty DataFrame for the generated samples
        df_gen = pd.DataFrame(columns=self.columns)
        
        # Account for potential losses during categorical decoding
        # Generate extra samples if we have categorical columns
        if hasattr(self, 'categorical_columns') and self.categorical_columns:
            # Estimate 20% loss during categorical decoding, so generate 25% extra
            target_raw_samples = int(n_samples * 1.25)
        else:
            target_raw_samples = n_samples
            
        if verbose:
            print(f"Target: {n_samples} final samples, generating {target_raw_samples} raw samples")
            print(f"Using {'segment-wise' if segmented_decode else 'one-pass'} generation")
            
        # SEGMENT-WISE GENERATION: Primary method (recommended)
        if segmented_decode:
            return self._sample_segmented(
                n_samples=n_samples,
                start_col=start_col,
                start_col_dist=start_col_dist,
                temperature=temperature,
                device=device,
                debug=debug,
                verbose=verbose,
                use_kv_cache=use_kv_cache,
                use_flash_attention=use_flash_attention,
                batch_size_generation=effective_batch_size,
                use_allowed_mask=use_allowed_mask
            )
        
        # Start generation process with guaranteed n_samples output
        max_total_attempts = n_samples * 50  # Generous upper limit to prevent infinite loops
        attempt_count = 0

        # Start generation process
        with tqdm(total=target_raw_samples, desc="Generating valid samples") as pbar:
            already_generated = 0
            consecutive_failures = 0
            
            while len(df_gen) < target_raw_samples and attempt_count < max_total_attempts:
                # Calculate how many more samples we need
                remaining_samples = target_raw_samples - len(df_gen)
                
                # Adaptive batch size: generate more if success rate is low
                if consecutive_failures > 3:
                    # Increase batch size when having difficulties
                    adaptive_batch_size = min(effective_batch_size * 2, 128)
                    consecutive_failures = 0  # Reset counter
                else:
                    # Generate 2-3x what we need to account for rejections
                    success_rate_estimate = max(0.1, (len(df_gen) + 1) / max(1, attempt_count))
                    adaptive_batch_size = min(
                        max(effective_batch_size, int(remaining_samples / success_rate_estimate * 2)),
                        128
                    )
                
                attempt_count += 1
                
                start_tokens = tabula_start.get_start_tokens(adaptive_batch_size)
                start_tokens = torch.tensor(start_tokens).to(device)
                
                # Create attention mask to handle pad tokens properly
                attention_mask = (start_tokens != self.special_ids["pad_id"]).long().to(device)
                
                # VERBOSE: Show start tokens for debugging (first few attempts)
                if verbose and attempt_count <= 3:
                    print(f"\nSTART TOKEN ANALYSIS (Attempt {attempt_count}):")
                    for i in range(min(3, len(start_tokens))):
                        try:
                            start_text = self.tokenizer.decode(start_tokens[i], skip_special_tokens=False)
                            start_text_safe = start_text.encode('ascii', 'replace').decode('ascii')
                            print(f"   Start Prompt {i+1}: '{start_text_safe}'")
                        except Exception as e:
                            print(f"   Start Prompt {i+1}: decode error - {e}")
                        start_token_ids = start_tokens[i][:10] if len(start_tokens[i]) > 10 else start_tokens[i]
                        print(f"   Start Token IDs: {start_token_ids}{'...' if len(start_tokens[i]) > 10 else ''}")

                # FIX E: Use only <ROWSEP> as EOS for cleaner stopping (avoid pretrain EOS interference)
                eos_ids = [self.special_ids["rowsep_id"]]
                # Temporarily disable general EOS to prevent early stopping on pretrain patterns
                # if self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id != self.special_ids["rowsep_id"]:
                #     eos_ids.append(self.tokenizer.eos_token_id)

                # QUICK FIX: Add LogitsProcessors to prevent consecutive separators
                from transformers import LogitsProcessorList
                
                logits_processors = LogitsProcessorList()
                
                # 1. Prevent consecutive separators - HIGH IMPACT
                sep_ids = [self.special_ids["colsep_id"], self.special_ids["rowsep_id"]]
                sep_ids = [sid for sid in sep_ids if sid is not None]  # Filter out None values
                if sep_ids:
                    logits_processors.append(NoDoubleSepProcessor(sep_ids))
                    if verbose and attempt_count == 1:
                        print(f"   Added NoDoubleSepProcessor for tokens: {sep_ids}")
                
                # 2. Character-aware schema validation - NEW
                if self.enable_schema_validation and self.column_constraints and self.schema_learner:
                    sep_tokens = {
                        'COL_SEP': self.special_ids.get('colsep_id'),
                        'ROW_SEP': self.special_ids.get('rowsep_id')
                    }
                    char_processor = CharacterAwareLogitsProcessor(
                        tokenizer=self.tokenizer,
                        constraints=self.column_constraints,
                        token_metadata=self.schema_learner.token_metadata,
                        sep_tokens=sep_tokens
                    )
                    logits_processors.append(char_processor)
                    if verbose and attempt_count == 1:
                        print(f"   Added CharacterAwareLogitsProcessor with {len(self.column_constraints)} column constraints")
                
                # Adjust temperature for constrained columns
                if self.enable_schema_validation and self.column_constraints:
                    # Use lower temperature for better constraint adherence
                    constrained_temp = min(temperature, 0.5)
                    if verbose and attempt_count == 1:
                        print(f"   Using constrained temperature: {constrained_temp} (original: {temperature})")
                    temperature = constrained_temp
                
                generation_kwargs = {
                    "input_ids":          start_tokens,
                    "attention_mask":     attention_mask,
                    "max_new_tokens":     max(max_length, 100),    
                    "do_sample":          True,
                    "temperature":        temperature,
                    "top_p":              0.9,      # nucleus sampling
                    "pad_token_id":       self.special_ids["pad_id"],
                    "repetition_penalty": 1.1,
                    "return_dict_in_generate": True,
                    "use_cache":          use_kv_cache,
                    "logits_processor":   logits_processors,  # Add our processors
                }
                
                # Try to use list of EOS tokens if supported, otherwise use stopping criteria
                try:
                    # Test if the current transformers version supports list of EOS tokens
                    generation_kwargs["eos_token_id"] = eos_ids
                    if verbose and attempt_count == 1:
                        print(f"   Using EOS token list: {eos_ids}")
                except Exception:
                    # Fallback to custom stopping criteria for older transformers
                    from transformers import StoppingCriteria, StoppingCriteriaList
                    
                    class StopOnAnyId(StoppingCriteria):
                        def __init__(self, stop_ids: set):
                            self.stop_ids = stop_ids
                        def __call__(self, input_ids, scores, **kwargs):
                            if input_ids.size(1) > 0:
                                last = input_ids[0, -1].item()
                                return last in self.stop_ids
                            return False
                    
                    stopping = StoppingCriteriaList([StopOnAnyId(set(eos_ids))])
                    generation_kwargs["stopping_criteria"] = stopping
                    generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id  # fallback
                    if verbose and attempt_count == 1:
                        print(f"   Using stopping criteria for IDs: {eos_ids}")
                
                # Add additional speed optimizations for supported models
                if hasattr(self.model.config, 'use_flash_attention_2'):
                    generation_kwargs["use_flash_attention_2"] = use_flash_attention

                # Generate tokens
                if debug and attempt_count <= 3:
                    print(f"DEBUG: Running model.generate() with kwargs: {list(generation_kwargs.keys())}")
                    
                tokens_out = self.model.generate(**generation_kwargs)
                
                # Handle different return formats
                if hasattr(tokens_out, 'sequences'):
                    seqs = tokens_out.sequences  # [B, prompt+new]
                else:
                    seqs = tokens_out  # fallback for older format

                # Slice continuations per sample (remove prompt from each sequence)
                prompt_lens = attention_mask.sum(dim=1).tolist()
                continuations = [seqs[i, prompt_lens[i]:].tolist() for i in range(seqs.size(0))]

                # VERBOSE: Show raw generated continuations before parsing
                if verbose and attempt_count <= 3:
                    print(f"\nRAW CONTINUATION ANALYSIS (Attempt {attempt_count}):")
                    for i in range(min(3, len(continuations))):
                        cont = continuations[i]
                        print(f"   Continuation {i+1}: {cont[:30]}{'...' if len(cont) > 30 else ''}")
                        
                        # Check for separators in the continuation
                        colsep_count = cont.count(self.special_ids["colsep_id"]) if self.special_ids["colsep_id"] in cont else 0
                        rowsep_count = cont.count(self.special_ids["rowsep_id"]) if self.special_ids["rowsep_id"] in cont else 0
                        print(f"   + <COLSEP> count: {colsep_count}, <ROWSEP> count: {rowsep_count}")
                        
                        # Show decoded text
                        try:
                            decoded_text = self.tokenizer.decode(cont, skip_special_tokens=False)
                            decoded_text_safe = decoded_text.encode('ascii', 'replace').decode('ascii')
                            print(f"   + Decoded: '{decoded_text_safe[:100]}{'...' if len(decoded_text_safe) > 100 else ''}'")
                        except Exception as e:
                            print(f"   + Decode error: {e}")

                # Convert tokens to dataframe using separator-aware parsing
                batch_before_conversion = len(df_gen)
                df_gen = _convert_tokens_to_dataframe_by_seps(
                    [torch.tensor(cont) for cont in continuations],
                    tokenizer=self.tokenizer,
                    column_list=self.columns,
                    special_ids=self.special_ids,
                    df_gen=df_gen,
                    start_col=start_col or (self.columns[0] if self.columns else None),
                    debug=(debug and attempt_count <= 3)
                )
                
                # Apply post-processing validation and repair
                if self.enable_schema_validation and self.post_validator and len(df_gen) > batch_before_conversion:
                    pre_validation_rows = len(df_gen)
                    validated_rows = []
                    
                    for idx in range(batch_before_conversion, len(df_gen)):
                        row_data = df_gen.iloc[idx].to_dict()
                        
                        # Validate and repair the row
                        repaired_row = self.post_validator.validate_and_repair_row(
                            row_data=row_data,
                            resample_callback=None  # No resampling callback for now
                        )
                        
                        validated_rows.append(repaired_row)
                    
                    # Replace the batch with validated data
                    if validated_rows:
                        validated_df = pd.DataFrame(validated_rows)
                        # Replace the newly added rows with validated ones
                        df_gen = pd.concat([
                            df_gen.iloc[:batch_before_conversion],
                            validated_df
                        ], ignore_index=True)
                    
                    if verbose and attempt_count <= 10:
                        print(f"   Post-validation: {pre_validation_rows - batch_before_conversion} rows processed")
                
                batch_size_before = len(df_gen) - batch_before_conversion
                
                # Enhanced debugging with detailed generation inspection + structure validation metrics
                if verbose and (attempt_count % debug_interval == 0 or attempt_count <= 10):
                    print(f"\nGENERATION DEBUG (Attempt {attempt_count}):")
                    print(f"   Batch size: {adaptive_batch_size}, Generated continuations: {len(continuations)}")
                    print(f"   Parsed valid rows: {batch_size_before}")
                    
                    # Schema validation metrics
                    if self.enable_schema_validation and self.column_constraints:
                        validation_stats = {"categorical_valid": 0, "digits_valid": 0, "length_valid": 0, "total_cells": 0}
                        
                        # Check newly added rows
                        for idx in range(max(0, len(df_gen) - batch_size_before), len(df_gen)):
                            if idx < len(df_gen):
                                row = df_gen.iloc[idx]
                                for col_name, value in row.items():
                                    if col_name in self.column_constraints:
                                        constraint = self.column_constraints[col_name]
                                        validation_stats["total_cells"] += 1
                                        
                                        if constraint.column_type.value == "categorical_small":
                                            if str(value) in constraint.lexicon:
                                                validation_stats["categorical_valid"] += 1
                                        elif constraint.column_type.value in ["fixed_digits", "variable_digits"]:
                                            if str(value).isdigit():
                                                validation_stats["digits_valid"] += 1
                                        
                                        if constraint.min_len <= len(str(value)) <= constraint.max_len:
                                            validation_stats["length_valid"] += 1
                        
                        if validation_stats["total_cells"] > 0:
                            print(f"   Schema validation: categorical {validation_stats['categorical_valid']}/{validation_stats['total_cells']}, "
                                  f"digits {validation_stats['digits_valid']}/{validation_stats['total_cells']}, "
                                  f"length {validation_stats['length_valid']}/{validation_stats['total_cells']}")
                    
                    
                    # FIX F: Add structure validation metrics
                    expected_colseps = len(self.columns) - 1
                    colsep_id = self.special_ids["colsep_id"]
                    rowsep_id = self.special_ids["rowsep_id"]
                    
                    structure_stats = {
                        "samples_with_rowsep": 0,
                        "correct_colsep_count": 0,
                        "total_colseps": 0
                    }
                    
                    # Show some generated samples (decoded text) + structure analysis
                    for i, token_sequence in enumerate(continuations[:min(3, len(continuations))]):  # Show first 3 samples
                        decoded = self.tokenizer.decode(token_sequence, skip_special_tokens=False)
                        decoded_safe = decoded.encode('ascii', 'replace').decode('ascii')
                        
                        # Count structure tokens
                        colsep_count = token_sequence.count(colsep_id)
                        rowsep_count = token_sequence.count(rowsep_id)
                        has_rowsep = rowsep_count > 0
                        
                        print(f"   Sample {i+1}: {decoded_safe[:200]}{'...' if len(decoded_safe) > 200 else ''}")
                        print(f"   + Tokens: {len(token_sequence)}, COLSEPs: {colsep_count}, ROWSEPs: {rowsep_count}")
                        
                        # Update stats for all samples in batch
                        if i < len(continuations):
                            if has_rowsep:
                                structure_stats["samples_with_rowsep"] += 1
                            if colsep_count == expected_colseps:
                                structure_stats["correct_colsep_count"] += 1
                            structure_stats["total_colseps"] += colsep_count
                    
                    # Show structure metrics for this batch
                    batch_size = len(continuations)
                    if batch_size > 0:
                        rowsep_rate = structure_stats["samples_with_rowsep"] / batch_size * 100
                        correct_colsep_rate = structure_stats["correct_colsep_count"] / batch_size * 100
                        avg_colseps = structure_stats["total_colseps"] / batch_size
                        
                        print(f"   STRUCTURE METRICS:")
                        print(f"     Samples with <ROWSEP>: {structure_stats['samples_with_rowsep']}/{batch_size} ({rowsep_rate:.1f}%)")
                        print(f"     Correct <COLSEP> count ({expected_colseps}): {structure_stats['correct_colsep_count']}/{batch_size} ({correct_colsep_rate:.1f}%)")
                        print(f"     Average <COLSEP> count: {avg_colseps:.1f}")

                elif attempt_count <= 10:
                    if verbose:
                        print(f"Attempt {attempt_count}: Generated {len(continuations)} continuations, parsed {batch_size_before} valid rows")

                # Track rejection reasons for verbose logging
                rejection_reasons = []
                original_batch_size = len(df_gen)
                
                # More lenient numerical validation - try to fix instead of reject
                for i_num_cols in self.num_cols:
                    if i_num_cols in df_gen.columns:
                        # Try to convert, fill invalid with median from training data
                        before_numeric = len(df_gen)
                        numeric_vals = pd.to_numeric(df_gen[i_num_cols], errors='coerce')
                        invalid_count = numeric_vals.isna().sum()
                        
                        if hasattr(self, '_column_medians') and i_num_cols in self._column_medians:
                            df_gen[i_num_cols] = numeric_vals.fillna(self._column_medians[i_num_cols])
                            if verbose and invalid_count > 0:
                                rejection_reasons.append(f"Fixed {invalid_count} invalid {i_num_cols} values with median")
                        else:
                            df_gen = df_gen[numeric_vals.notnull()]
                            if len(df_gen) < before_numeric:
                                rejected = before_numeric - len(df_gen)
                                rejection_reasons.append(f"Dropped {rejected} rows due to invalid {i_num_cols} values")

                # More lenient missing value handling - only drop if too many missing
                missing_threshold = 0.7  # Allow up to 70% of columns to be missing
                if len(df_gen) > 0:
                    before_missing = len(df_gen)
                    missing_ratio = df_gen.isna().sum(axis=1) / len(df_gen.columns)
                    df_gen = df_gen[missing_ratio <= missing_threshold]
                    if len(df_gen) < before_missing:
                        rejected = before_missing - len(df_gen)
                        rejection_reasons.append(f"Dropped {rejected} rows due to >70% missing values")

                # Convert numerical columns
                for col in self.num_cols:
                    if col in df_gen.columns:
                        before_conversion = len(df_gen)
                        df_gen[col] = pd.to_numeric(df_gen[col], errors='coerce')
                        # Check for conversion failures
                        invalid_after = df_gen[col].isna().sum()
                        if invalid_after > 0 and verbose:
                            rejection_reasons.append(f"{invalid_after} values in {col} became NaN after conversion")
                
                # Log rejection reasons if verbose
                if verbose and (attempt_count % debug_interval == 0 or attempt_count <= 10) and rejection_reasons:
                    print(f"   REJECTION REASONS:")
                    for reason in rejection_reasons:
                        print(f"      • {reason}")
                    print(f"   Final batch: {original_batch_size} + {len(df_gen)} valid rows")

                # Add valid samples to main dataframe
                valid_samples = len(df_gen) - batch_before_conversion
                
                if valid_samples == 0:
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0
                
                # Update progress bar
                new_samples = len(df_gen) - already_generated
                pbar.update(new_samples)
                already_generated = len(df_gen)
                
                if debug:
                    print(f"DEBUG: Progress: {already_generated}/{target_raw_samples} samples ({already_generated/target_raw_samples*100:.1f}%)")

        df_gen = df_gen.reset_index(drop=True)
        final_df = df_gen.head(n_samples)
        
        if debug or verbose:
            print(f"Final result: {final_df.shape}")
            print(f"Success rate: {len(final_df)/n_samples*100:.1f}%")
            if debug:
                print(f"DEBUG: Sample of final data:")
                print(final_df.head(3))
        
        return final_df

    def _apply_qlora_configuration(self):
        """Apply quantization and LoRA configuration to the model."""
        # Check if model embeddings need resize after tokenizer normalization
        tokenizer_vocab_size = len(self.tokenizer)
        model_vocab_size = self.model.config.vocab_size
        
        if tokenizer_vocab_size > model_vocab_size:
            print(f"🔧 Resizing model embeddings from {model_vocab_size} to {tokenizer_vocab_size}...")
            self.model.resize_token_embeddings(tokenizer_vocab_size)
            print(f"✅ Model embeddings resized to {tokenizer_vocab_size}")
        
        # NOW apply quantization after embedding resize
        if self.quantization_config and QUANTIZATION_AVAILABLE:
            print(f"🔧 Applying quantization after embedding resize...")
            from peft.utils.other import prepare_model_for_kbit_training
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)
            print(f"✅ Quantization applied")
        
        # NOW apply LoRA after quantization
        if self.use_lora and self.lora_config and PEFT_AVAILABLE:
            print(f"🔧 Applying LoRA after quantization...")
            peft_config = LoraConfig(**self.lora_config)
            self.model = get_peft_model(self.model, peft_config)
            print(f"✅ LoRA applied with config: {self.lora_config}")
        elif self.use_lora and not self.lora_config and PEFT_AVAILABLE:
            # Use default LoRA configuration
            print(f"🔧 Applying LoRA with default configuration...")
            model_type = getattr(self.config, 'model_type', 'unknown').lower()
            
            # Set target modules based on model architecture
            if model_type in ["gpt2", "gpt_neo", "gpt_neox"]:
                target_modules = ["c_attn", "c_proj"]
            elif model_type in ["llama", "mistral", "phi", "qwen", "gemma"]:
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif "openelm" in model_type:
                target_modules = ["qkv_proj", "out_proj", "proj_1", "proj_2"]
            else:
                # Generic fallback - try common attention projection names
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
                print(f"⚠️  Unknown model type '{model_type}', using fallback target_modules: {target_modules}")
            
            default_lora_config = {
                "task_type": TaskType.CAUSAL_LM,
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "target_modules": target_modules,
                "bias": "none"
            }
            
            peft_config = LoraConfig(**default_lora_config)
            self.model = get_peft_model(self.model, peft_config)
            print(f"✅ LoRA applied with default config: {default_lora_config}")
        elif self.use_lora and not PEFT_AVAILABLE:
            raise ImportError("LoRA requested but 'peft' library not available. Install with: pip install peft")

    def _check_model_compatibility(self):
        """Check if the model supports the requested optimizations."""
        if self.use_lora and not PEFT_AVAILABLE:
            raise ImportError("LoRA requested but 'peft' library not available. Install with: pip install peft")
        
        if self.quantization_config and not QUANTIZATION_AVAILABLE:
            raise ImportError("Quantization requested but 'bitsandbytes' not available. Install with: pip install bitsandbytes")
        
        # Check model architecture compatibility
        model_type = getattr(self.config, 'model_type', 'unknown').lower()
        
        if self.use_lora:
            compatible_architectures = ["gpt2", "gpt_neo", "gpt_neox", "llama", "mistral", "phi", "qwen", "gemma", "openelm"]
            if model_type not in compatible_architectures:
                warnings.warn(f"Model type '{model_type}' may not be fully compatible with LoRA. "
                             f"Supported types: {compatible_architectures}")
        
        if self.quantization_config:
            # Most transformer models support quantization, but warn about potential issues
            if hasattr(self.config, 'torch_dtype') and self.config.torch_dtype == torch.float16:
                logging.info("Model uses float16 - quantization will be applied appropriately")
        
        logging.info(f"Model compatibility check passed for {model_type}")

    def validate_baseline_generation(self, n_samples: int = 10, temperature: float = 0.4, device: str = "cuda") -> dict:
        """FIX B: Validate baseline generation starting from first column with tighter parameters.
        
        This helps isolate whether the issue is format mismatch (should work after Fix A) 
        vs deeper training/collator issues.
        
        Args:
            n_samples: Number of test samples to generate
            temperature: Lower temperature for more focused generation
            device: Device to run on
            
        Returns:
            dict: Validation statistics including separator counts and structure metrics
        """
        if not hasattr(self, 'columns') or not self.columns:
            raise ValueError("Model must be fitted before validation. Call fit() first.")
            
        print(f"\n=== BASELINE VALIDATION (Fix B) ===")
        print(f"Testing generation from first column: {self.columns[0]}")
        print(f"Expected format after Fix A: '{self.columns[0]} <value> <COLSEP>'")
        print(f"Parameters: temperature={temperature}, n_samples={n_samples}")
        
        # Force first column start for baseline test
        self.model.to(device)
        
        # Get start sampler for first column
        tabula_start = self._get_start_sampler(
            start_col=self.columns[0], 
            start_col_dist=None  # Use default distribution
        )
        
        # Generate with conservative parameters
        start_tokens = tabula_start.get_start_tokens(n_samples)
        start_tokens = torch.tensor(start_tokens).to(device)
        attention_mask = (start_tokens != self.special_ids["pad_id"]).long().to(device)
        
        print(f"\nSample start prompt: '{self.tokenizer.decode(start_tokens[0], skip_special_tokens=False)}'")
        
        generation_kwargs = {
            "input_ids": start_tokens,
            "attention_mask": attention_mask,
            "max_new_tokens": 100,  # Conservative length
            "temperature": temperature,
            "top_p": 0.9,
            "do_sample": True,
            "eos_token_id": self.special_ids["rowsep_id"],  # Stop only on <ROWSEP>
            "pad_token_id": self.special_ids["pad_id"],
            "repetition_penalty": 1.1,
        }
        
        # Generate
        with torch.no_grad():
            tokens_out = self.model.generate(**generation_kwargs)
            
        # Extract continuations
        prompt_lens = attention_mask.sum(dim=1).tolist()
        continuations = [tokens_out[i, prompt_lens[i]:].tolist() for i in range(tokens_out.size(0))]
        
        # Analyze structure
        stats = {
            "total_samples": len(continuations),
            "samples_with_rowsep": 0,
            "samples_with_correct_colseps": 0,
            "avg_colsep_count": 0,
            "samples": []
        }
        
        expected_colseps = len(self.columns) - 1
        colsep_id = self.special_ids["colsep_id"]
        rowsep_id = self.special_ids["rowsep_id"]
        
        total_colseps = 0
        
        for i, cont in enumerate(continuations):
            sample_stats = {
                "index": i,
                "length": len(cont),
                "colsep_count": cont.count(colsep_id),
                "rowsep_count": cont.count(rowsep_id),
                "has_rowsep": rowsep_id in cont,
                "text": ""
            }
            
            try:
                decoded = self.tokenizer.decode(cont, skip_special_tokens=False)
                sample_stats["text"] = decoded[:200] + ("..." if len(decoded) > 200 else "")
            except:
                sample_stats["text"] = "<decode_error>"
            
            # Update totals
            if sample_stats["has_rowsep"]:
                stats["samples_with_rowsep"] += 1
            
            if sample_stats["colsep_count"] == expected_colseps:
                stats["samples_with_correct_colseps"] += 1
                
            total_colseps += sample_stats["colsep_count"]
            stats["samples"].append(sample_stats)
        
        stats["avg_colsep_count"] = total_colseps / len(continuations) if continuations else 0
        
        # Print results
        print(f"\n=== VALIDATION RESULTS ===")
        print(f"Samples with <ROWSEP>: {stats['samples_with_rowsep']}/{stats['total_samples']} ({stats['samples_with_rowsep']/stats['total_samples']*100:.1f}%)")
        print(f"Samples with correct <COLSEP> count ({expected_colseps}): {stats['samples_with_correct_colseps']}/{stats['total_samples']} ({stats['samples_with_correct_colseps']/stats['total_samples']*100:.1f}%)")
        print(f"Average <COLSEP> count: {stats['avg_colsep_count']:.1f} (expected: {expected_colseps})")
        
        print(f"\n=== SAMPLE OUTPUTS ===")
        for i, sample in enumerate(stats["samples"][:5]):  # Show first 5
            print(f"Sample {i+1}: COLSEP={sample['colsep_count']}, ROWSEP={sample['rowsep_count']}")
            print(f"  Text: {sample['text']}")
            print()
        
        return stats

    def tabula_sample(self, starting_prompts: tp.Union[str, list[str]], temperature: float = 0.7, max_length: int = 100,
                     device: str = "cuda") -> pd.DataFrame:
        """ Generate synthetic tabular data samples conditioned on a given input.

        Args:
            starting_prompts: String or List of Strings on which the output is conditioned.
             For example, "Sex is female, Age is 26"
            temperature: The generation samples each token from the probability distribution given by a softmax
             function. The temperature parameter controls the softmax function. A low temperature makes it sharper
             (0 equals greedy search), a high temperature brings more diversity but also uncertainty into the output.
             See this blog article (https://huggingface.co/blog/how-to-generate) to read more about the generation
             process.
            max_length: Maximal number of tokens to generate - has to be long enough to not cut any information
            device: Set to "cpu" if the GPU should not be used. You can also specify the concrete GPU.

         Returns:
            Pandas DataFrame with synthetic data generated based on starting_prompts
        """
        # ToDo: Add n_samples argument to generate more samples for one conditional input.

        self.model.to(device)
        starting_prompts = [starting_prompts] if isinstance(starting_prompts, str) else starting_prompts
        generated_data = []

        # Generate a sample for each starting point
        for prompt in tqdm(starting_prompts):
            start_token = torch.tensor(self.tokenizer(prompt)["input_ids"]).to(device)
            print("start_token: ", start_token)
            # Generate tokens
            gen = self.model.generate(input_ids=torch.unsqueeze(start_token, 0), max_length=max_length,
                                      do_sample=True, temperature=temperature, pad_token_id=self.tokenizer.pad_token_id)
            print("gen: ", gen)
            generated_data.append(torch.squeeze(gen))

        # Convert Text back to Tabular Data
        decoded_data = _convert_tokens_to_text(generated_data, self.tokenizer)
        print("decoded_data: ", decoded_data)
        df_gen = _convert_text_to_tabular_data(decoded_data, pd.DataFrame(columns=self.columns))

        return df_gen

    def save(self, path: str):
        """ Save Tabula Model

        Saves the model weights and a configuration file in the given directory.

        Args:
            path: Path where to save the model
        """
        # Make directory
        if os.path.isdir(path):
            warnings.warn(f"Directory {path} already exists and is overwritten now.")
        else:
            os.mkdir(path)

        # Save attributes
        with open(path + "/config.json", "w") as f:
            attributes = self.__dict__.copy()
            
            # Remove non-serializable objects
            non_serializable_keys = ["tokenizer", "model", "config", "lora_config", "quantization_config", 
                                    "schema_learner", "post_validator", "column_constraints", 
                                    "column_profiler", "column_profiles", "_column_value_processor"]
            for key in non_serializable_keys:
                attributes.pop(key, None)  # Use pop with default to avoid KeyError

            # NDArray is not JSON serializable and therefore has to be converted into a list.
            if "conditional_col_dist" in attributes and isinstance(attributes["conditional_col_dist"], np.ndarray):
                attributes["conditional_col_dist"] = list(attributes["conditional_col_dist"])

            # QUICK FIX: Convert sets in column_configs to lists for JSON serialization
            if "column_configs" in attributes and isinstance(attributes["column_configs"], dict):
                column_configs_serializable = {}
                for col_name, config in attributes["column_configs"].items():
                    config_copy = config.copy()
                    # Convert set to list for JSON serialization
                    if "allowed_tokens" in config_copy and isinstance(config_copy["allowed_tokens"], set):
                        config_copy["allowed_tokens"] = list(config_copy["allowed_tokens"])
                    column_configs_serializable[col_name] = config_copy
                attributes["column_configs"] = column_configs_serializable

            # EMPTY CELL FIX: Convert sets in column_name_tokens to lists for JSON serialization  
            if "column_name_tokens" in attributes and isinstance(attributes["column_name_tokens"], dict):
                column_name_tokens_serializable = {}
                for col_name, token_set in attributes["column_name_tokens"].items():
                    if isinstance(token_set, set):
                        column_name_tokens_serializable[col_name] = list(token_set)
                    else:
                        column_name_tokens_serializable[col_name] = token_set
                attributes["column_name_tokens"] = column_name_tokens_serializable

            json.dump(attributes, f)

        # Save model weights
        torch.save(self.model.state_dict(), path + "/model.pt")

    def load_finetuned_model(self, path: str):
        """ Load fine-tuned model

        Load the weights of a fine-tuned large language model into the Tabula pipeline

        Args:
            path: Path to the fine-tuned model
        """
        self.model.load_state_dict(torch.load(path))

    @classmethod
    def load_from_dir(cls, path: str):
        """ Load Tabula class

        Load trained Tabula model from directory.

        Args:
            path: Directory where Tabula model is saved

        Returns:
            New instance of Tabula loaded from directory
        """
        assert os.path.isdir(path), f"Directory {path} does not exist."

        # Load attributes
        with open(path + "/config.json", "r") as f:
            attributes = json.load(f)

        # Create new be_tabula model instance
        tabula = cls(attributes["llm"])

        # Set all attributes
        for k, v in attributes.items():
            setattr(tabula, k, v)
            
        # Set default values for attributes that weren't serialized
        if not hasattr(tabula, 'categorical_columns'):
            tabula.categorical_columns = None
        if not hasattr(tabula, 'use_lora'):
            tabula.use_lora = False
        if not hasattr(tabula, 'lora_config'):
            tabula.lora_config = None
        if not hasattr(tabula, 'quantization_config'):
            tabula.quantization_config = None
        if not hasattr(tabula, 'max_seq_length'):
            tabula.max_seq_length = None
            
        # QUICK FIX: Initialize new attributes if not present (backward compatibility)
        if not hasattr(tabula, 'column_configs'):
            tabula.column_configs = {}
        if not hasattr(tabula, 'train_span_tokens'):
            tabula.train_span_tokens = []
        if not hasattr(tabula, 'max_cell_tokens'):
            tabula.max_cell_tokens = {}
            
        # Initialize schema validation components (they're not serialized)
        if not hasattr(tabula, 'schema_learner'):
            tabula.schema_learner = None
        if not hasattr(tabula, 'post_validator'):
            tabula.post_validator = None
        if not hasattr(tabula, 'column_constraints'):
            tabula.column_constraints = None
        if not hasattr(tabula, 'enable_schema_validation'):
            tabula.enable_schema_validation = True
            
        # QUICK FIX: Convert lists back to sets in column_configs after loading
        if hasattr(tabula, 'column_configs') and isinstance(tabula.column_configs, dict):
            for col_name, config in tabula.column_configs.items():
                if "allowed_tokens" in config and isinstance(config["allowed_tokens"], list):
                    config["allowed_tokens"] = set(config["allowed_tokens"])

        # EMPTY CELL FIX: Convert lists back to sets in column_name_tokens after loading
        if not hasattr(tabula, 'column_name_tokens'):
            tabula.column_name_tokens = {}
        elif isinstance(tabula.column_name_tokens, dict):
            for col_name, token_list in tabula.column_name_tokens.items():
                if isinstance(token_list, list):
                    tabula.column_name_tokens[col_name] = set(token_list)

        # Load model weights
        tabula.model.load_state_dict(torch.load(path + "/model.pt", map_location="cpu"))

        # Normalize tokenizer and model again after loading - ensure special tokens are available
        tabula.special_ids = normalize_tokenizer_and_model(tabula.tokenizer, tabula.model)

        return tabula

    def _update_column_information(self, df: pd.DataFrame):
        # Update the column names (and numerical columns for some sanity checks after sampling)
        self.columns = df.columns.to_list()
        self.num_cols = df.select_dtypes(include=np.number).columns.to_list()
        
        # Initialize schema validation components if enabled
        if self.enable_schema_validation:
            logging.info("🔧 Initializing schema validation system...")
            
            # Define separator tokens (assuming they exist in special_ids)
            sep_tokens = {}
            if hasattr(self, 'special_ids') and self.special_ids:
                sep_tokens = {
                    'COL_SEP': self.special_ids.get('COL_SEP', None),
                    'ROW_SEP': self.special_ids.get('ROW_SEP', None)
                }
            
            # Initialize schema learner
            self.schema_learner = SchemaLearner(self.tokenizer, sep_tokens)
            
            # Learn column constraints from training data
            self.column_constraints = self.schema_learner.learn_column_constraints(df)
            
            # Initialize post-validator
            self.post_validator = PostValidator(
                constraints=self.column_constraints,
                cell_resample_max=2,
                row_resample_cap=8
            )
            
            logging.info("✅ Schema validation system initialized")

    def _update_conditional_information(self, df: pd.DataFrame, conditional_col: tp.Optional[str] = None):
        assert conditional_col is None or isinstance(conditional_col, str), \
            f"The column name has to be a string and not {type(conditional_col)}"
        assert conditional_col is None or conditional_col in df.columns, \
            f"The column name {conditional_col} is not in the feature names of the given dataset"

        # Take the distribution of the conditional column for a starting point in the generation process
        # FIX: Default to FIRST column instead of last to match training format (no rotation)
        self.conditional_col = conditional_col if conditional_col else df.columns[0]
        self.conditional_col_dist = _get_column_distribution(df, self.conditional_col)

    def _get_start_sampler(self, start_col: tp.Optional[str],
                           start_col_dist: tp.Optional[tp.Union[tp.Dict, tp.List]]) -> TabulaStart:
        if start_col and start_col_dist is None:
            raise ValueError(f"Start column {start_col} was given, but no corresponding distribution.")
        if start_col_dist is not None and not start_col:
            raise ValueError(f"Start column distribution {start_col} was given, the column name is missing.")

        assert start_col is None or isinstance(start_col, str), \
            f"The column name has to be a string and not {type(start_col)}"
        assert start_col_dist is None or isinstance(start_col_dist, dict) or isinstance(start_col_dist, list), \
            f"The distribution of the start column on has to be a list or a dict and not {type(start_col_dist)}"

        # FIX: Default to first column and always use CanonicalStart for exact format match
        if not start_col:
            start_col = self.columns[0]  # Force first column (matches training without rotation)
            start_col_dist = self.conditional_col_dist  # Use the stored distribution
            
        # Always use CanonicalStart to ensure correct separator usage and format matching
        return CanonicalStart(
            tokenizer=self.tokenizer,
            all_columns=self.columns,
            start_col=start_col,
            special_ids=self.special_ids,
            start_col_dist=start_col_dist if start_col_dist else self.conditional_col_dist
        )
