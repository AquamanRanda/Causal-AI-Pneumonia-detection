"""
Configuration utilities for CausalXray framework.

This module provides configuration loading, validation, and default configuration
generation for the CausalXray model training and inference.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration for CausalXray model.
    
    Returns:
        Default configuration dictionary
    """
    config = {
        'data': {
            'image_size': [224, 224],
            'batch_size': 32,
            'num_workers': 4,
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1,
            'augmentation': True
        },
        'model': {
            'backbone': {
                'architecture': 'densenet121',
                'pretrained': True,
                'num_classes': 2,
                'feature_dims': [1024, 512, 256],
                'dropout_rate': 0.3
            }
        },
        'causal': {
            'confounders': {},
            'hidden_dims': [512, 256],
            'dropout_rate': 0.3,
            'use_variational': True
        },
        'attribution': {
            'feature_layers': ['backbone'],
            'attribution_methods': ['intervention', 'counterfactual', 'gradcam'],
            'patch_size': 16,
            'num_patches': None
        },
        'training': {
            'epochs': 100,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'optimizer': 'adam',
            'scheduler': 'cosine',
            'early_stopping_patience': 10,
            'save_best_only': True
        },
        'loss': {
            'classification_weight': 1.0,
            'causal_weight': 0.1,
            'disentanglement_weight': 0.05
        },
        'logging': {
            'log_dir': './logs',
            'checkpoint_dir': './checkpoints',
            'tensorboard': True,
            'wandb': False
        }
    }
    
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if configuration is valid, False otherwise
    """
    required_sections = ['data', 'model', 'causal', 'training']
    
    for section in required_sections:
        if section not in config:
            print(f"Missing required section: {section}")
            return False
    
    # Validate data section
    data = config.get('data', {})
    if 'image_size' not in data:
        print("Missing image_size in data section")
        return False
    
    # Validate model section
    model = config.get('model', {})
    if 'backbone' not in model:
        print("Missing backbone in model section")
        return False
    
    # Validate training section
    training = config.get('training', {})
    if 'epochs' not in training:
        print("Missing epochs in training section")
        return False
    
    return True


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                _merge_dict(base[key], value)
            else:
                base[key] = value
    
    _merge_dict(merged, override_config)
    return merged


def get_config_from_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create configuration from command line arguments.
    
    Args:
        args: Command line arguments dictionary
        
    Returns:
        Configuration dictionary
    """
    config = create_default_config()
    
    # Override with command line arguments
    if 'config_path' in args and args['config_path']:
        if os.path.exists(args['config_path']):
            file_config = load_config(args['config_path'])
            config = merge_configs(config, file_config)
    
    # Override specific values from command line
    if 'batch_size' in args:
        config['data']['batch_size'] = args['batch_size']
    
    if 'learning_rate' in args:
        config['training']['learning_rate'] = args['learning_rate']
    
    if 'epochs' in args:
        config['training']['epochs'] = args['epochs']
    
    return config
