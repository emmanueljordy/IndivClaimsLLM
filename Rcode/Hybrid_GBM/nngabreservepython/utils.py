"""Utility functions for saving and loading configurations."""

import json
from pathlib import Path


def save_config(config: dict, lob: int):
    """Save configuration to file."""
    config_dir = Path(f"./Results/Config/LoB{lob}")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = config_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {config_file}")


def load_config(lob: int) -> dict:
    """Load configuration from file."""
    config_file = Path(f"./Results/Config/LoB{lob}/config.json")
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"Configuration loaded from {config_file}")
        return config
    else:
        print(f"Warning: No saved configuration found for LoB {lob}, using defaults")
        # Return default configuration
        return {
            'neurons': [40, 30, 10],
            'dropout_rates': [0.00000000001] * 10,
            'batch_size': 10000,
            'seed': 100,
        }