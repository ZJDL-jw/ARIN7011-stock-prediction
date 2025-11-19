"""I/O utilities for loading configs and saving/loading data."""
import yaml
import os
from pathlib import Path
import json
import pickle


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Handle base_config references
    if 'base_config' in config:
        base = load_config(config['base_config'])
        config = {**base, **config}
        del config['base_config']
    
    # Handle factors_config references
    if 'factors_config' in config:
        factors = load_config(config['factors_config'])
        if 'factors' not in config:
            config['factors'] = {}
        config['factors'].update(factors)
        del config['factors_config']
    
    return config


def ensure_dir(path: str):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: str):
    """Save dictionary as JSON."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> dict:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(obj, path: str):
    """Save object as pickle."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    """Load pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)

