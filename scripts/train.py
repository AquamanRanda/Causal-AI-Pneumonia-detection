# CausalXray/scripts/train.py
"""Training script for CausalXray model."""

import argparse
import torch
import yaml
from pathlib import Path
import sys

# Add package to path
sys.path.append(str(Path(__file__).parent.parent))

from causalxray import CausalXray
from causalxray.data import create_dataloader, NIHChestXray14, RSNAPneumonia, PediatricDataset
from causalxray.data import CausalTransforms
from causalxray.training import CausalTrainer
from causalxray.utils import load_config, setup_logger


def main():
    parser = argparse.ArgumentParser(description='Train CausalXray model')
    parser.add_argument('--config', type=str, required=True, help='Configuration file path')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--resume', type=str, help='Checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logger('CausalXray_Training', args.output_dir)
    logger.info(f"Starting CausalXray training with config: {args.config}")
    
    # Create datasets
    dataset_name = config['data']['dataset']
    
    if dataset_name == 'nih':
        dataset_class = NIHChestXray14
    elif dataset_name == 'rsna':
        dataset_class = RSNAPneumonia
    elif dataset_name == 'pediatric':
        dataset_class = PediatricDataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create transforms
    train_transforms = CausalTransforms(mode='train', **config['data']['transforms'])
    val_transforms = CausalTransforms(mode='val', **config['data']['transforms'])
    
    # Create datasets
    train_dataset = dataset_class(
        data_dir=args.data_dir,
        split='train',
        transform=train_transforms,
        include_confounders=True
    )
    
    val_dataset = dataset_class(
        data_dir=args.data_dir,
        split='val',
        transform=val_transforms,
        include_confounders=True
    )
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4)
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4)
    )
    
    # Create model
    model = CausalXray(
        backbone_config=config['model']['backbone'],
        causal_config=config['model']['causal'],
        attribution_config=config['model'].get('attribution', {}),
        training_phase='backbone'
    )
    
    # Create trainer
    trainer = CausalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        device=args.device,
        logger=logger
    )
    
    # Train model
    history = trainer.train(
        num_epochs=config['training']['num_epochs'],
        resume_from=args.resume
    )
    
    logger.info("Training completed successfully!")
    
    # Save final model
    final_model_path = Path(args.output_dir) / 'final_model.pth'
    model.save_checkpoint(str(final_model_path), trainer.current_epoch)
    logger.info(f"Final model saved to: {final_model_path}")


if __name__ == '__main__':
    main()
