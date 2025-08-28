import random
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import Trainer
from transformers.trainer_callback import TrainerCallback


def _seed_worker(_):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)


class TabulaTrainer(Trainer):
    """ Tabula Trainer

    Enhanced trainer that supports:
    1. Preserving "unused" columns needed for tabular data
    2. Advanced optimizer settings with separate learning rates for embeddings
    3. Support for LoRA, quantization, flash attention, and kv_caching
    4. General training enhancements for better tabular data modeling
    
    This version removes all sentinel token functionality to work with simple 
    comma-separated tabular data format.
    """
    
    def __init__(self, model=None, args=None, embedding_lr_multiplier=2.0, debug_training=False, 
                 special_ids=None, sep_loss_weight=2.0, **kwargs):
        """Initialize enhanced TabulaTrainer."""
        super().__init__(model=model, args=args, **kwargs)
        self.embedding_lr_multiplier = embedding_lr_multiplier  # Higher LR for embeddings
        self.debug_training = debug_training  # Enable debug output during training
        self.last_batch = None  # Store last batch for analysis
        self.step_count = 0  # Track training steps
        self.special_ids = special_ids or {}  # Special token IDs for loss weighting
        self.sep_loss_weight = sep_loss_weight  # Weight for separator tokens in loss
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Create optimizer with separate learning rates for embeddings vs other parameters."""
        if self.optimizer is None:
            # Get embedding parameters (for higher learning rate)
            embedding_params = []
            other_params = []
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                # Target embedding layers with higher LR
                if any(x in name.lower() for x in ["embed", "wte"]):
                    embedding_params.append(param)
                else:
                    other_params.append(param)
            
            # Create parameter groups with different learning rates
            base_lr = self.args.learning_rate
            embedding_lr = base_lr * self.embedding_lr_multiplier
            
            param_groups = []
            if embedding_params:
                param_groups.append({
                    "params": embedding_params, 
                    "lr": embedding_lr,
                    "weight_decay": self.args.weight_decay
                })
                print(f"Embedding parameters: {len(embedding_params)} with LR {embedding_lr:.2e} ({self.embedding_lr_multiplier}x)")
            
            if other_params:
                param_groups.append({
                    "params": other_params, 
                    "lr": base_lr,
                    "weight_decay": self.args.weight_decay
                })
                print(f"Other parameters: {len(other_params)} with LR {base_lr:.2e} (1.0x)")
            
            # Use AdamW optimizer
            try:
                from transformers.optimization import AdamW
            except ImportError:
                from torch.optim import AdamW
            self.optimizer = AdamW(param_groups, eps=self.args.adam_epsilon)
            
        if self.lr_scheduler is None:
            self.lr_scheduler = self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)
        
        return self.optimizer, self.lr_scheduler

    def get_train_dataloader(self) -> DataLoader:
        """Get training dataloader without removing unused columns."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        data_collator = self.data_collator
        # IMPORTANT: Don't remove unused columns for tabular data!
        train_dataset = self.train_dataset  # self._remove_unused_columns(self.train_dataset, description="training")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=_seed_worker,
        )
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Enhanced loss computation with improved numerical stability and debugging.
        
        Args:
            model: The model being trained
            inputs: Input batch from data collator
            return_outputs: Whether to return model outputs along with loss
            num_items_in_batch: Number of items in batch (for newer HF versions)
        """
        # Store the last batch for debugging/analysis
        self.last_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        self.step_count += 1
        
        labels = inputs.get("labels")
        
        # Debug output for training
        if self.debug_training and self.step_count % 100 == 0:
            batch_size = inputs['input_ids'].shape[0]
            seq_len = inputs['input_ids'].shape[1] 
            print(f"DEBUG: Training step {self.step_count}, batch_size={batch_size}, seq_len={seq_len}")
            
            if labels is not None:
                valid_labels = (labels != -100).sum().item()
                total_labels = labels.numel()
                print(f"DEBUG: Valid labels: {valid_labels}/{total_labels} ({valid_labels/total_labels*100:.1f}%)")
        
        # Forward pass
        outputs = model(**inputs)
        
        # Enhanced loss computation with separator token weighting
        if labels is not None:
            logits = outputs.get("logits")
            if logits is not None:
                # Shift for causal language modeling
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # FIX: Implement loss weighting for special tokens (upweight <COLSEP>/<ROWSEP>)
                if self.special_ids and self.sep_loss_weight != 1.0:
                    vocab_size = shift_logits.size(-1)
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                    loss_flat = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
                    
                    # Create per-token weights: upweight separators, standard weight for others
                    weights = torch.ones_like(shift_labels, dtype=loss_flat.dtype)
                    
                    colsep_id = self.special_ids.get("colsep_id")
                    rowsep_id = self.special_ids.get("rowsep_id")
                    
                    if colsep_id is not None:
                        weights = torch.where(shift_labels == colsep_id, self.sep_loss_weight, weights)
                    if rowsep_id is not None:
                        weights = torch.where(shift_labels == rowsep_id, self.sep_loss_weight, weights)
                    
                    # Apply mask for ignored labels (-100)
                    mask = (shift_labels != -100).float()
                    weights = (weights * mask).view(-1)
                    
                    # Weighted average loss
                    loss = (loss_flat * weights).sum() / weights.sum().clamp_min(1.0)
                    
                    # Debug separator weighting
                    if self.debug_training and self.step_count % 100 == 0:
                        with torch.no_grad():
                            total_tokens = mask.sum().item()
                            weighted_sep_tokens = ((shift_labels == colsep_id) | (shift_labels == rowsep_id)).sum().item() if colsep_id or rowsep_id else 0
                            print(f"DEBUG: Weighted {weighted_sep_tokens} separator tokens out of {total_tokens} total (weight={self.sep_loss_weight})")
                            
                else:
                    # Standard loss without weighting
                    loss_fct = torch.nn.CrossEntropyLoss(
                        ignore_index=-100,
                        label_smoothing=getattr(self.args, 'label_smoothing_factor', 0.0)
                    )
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                
                # Debug loss information
                if self.debug_training and self.step_count % 10 == 0:
                    print(f"DEBUG: Loss = {loss.item():.4f}")
                    
                    # Additional token analysis
                    with torch.no_grad():
                        # Get predictions
                        preds = torch.argmax(shift_logits, dim=-1)
                        # Calculate accuracy on valid labels
                        valid_mask = shift_labels != -100
                        if valid_mask.sum() > 0:
                            accuracy = (preds[valid_mask] == shift_labels[valid_mask]).float().mean()
                            print(f"DEBUG: Token accuracy: {accuracy.item():.3f}")
                
            else:
                loss = torch.tensor(0.0, requires_grad=True, device=model.device)
        else:
            loss = outputs.get("loss", torch.tensor(0.0, requires_grad=True))
        
        return (loss, outputs) if return_outputs else loss
        
# Removed custom training_step to avoid method signature conflicts
# The parent Trainer class handles training steps correctly


class EpochCallback(TrainerCallback):
    """Callback to track current epoch for data collator."""
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Update data collator with current epoch info."""
        trainer = kwargs.get('trainer')
        if trainer and hasattr(trainer, 'data_collator'):
            # Update epoch for any curriculum learning in data collator
            if hasattr(trainer.data_collator, '_epoch'):
                trainer.data_collator._epoch = state.epoch
                print(f"Training epoch {state.epoch}")