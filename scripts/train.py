# CausalXray/scripts/train.py
"""Training script for CausalXray model."""
import seaborn as sns
import argparse
import plotly.graph_objects as go
import torch
import yaml
from pathlib import Path
import sys

# Add package to path
sys.path.append(str(Path(__file__).parent.parent))

from causalxray.models.backbone import CausalBackbone
from causalxray.data.datasets import create_dataloader, NIHChestXray14, RSNAPneumonia, PediatricDataset
from causalxray.data.transforms import CausalTransforms
from causalxray.training.trainer import CausalTrainer
from causalxray.utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description='Train CausalXray model')
    parser.add_argument('--config', type=str, required=True, help='Configuration file path')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--resume', type=str, help='Checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
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
    # TODO: Specify the correct architecture and parameters as per your experiment
    model = CausalBackbone(
        architecture=config['model']['backbone'].get('architecture', 'densenet121'),
        pretrained=config['model']['backbone'].get('pretrained', True),
        num_classes=config['model']['backbone'].get('num_classes', 2),
        feature_dims=config['model']['backbone'].get('feature_dims', [1024, 512, 256]),
        dropout_rate=config['model']['backbone'].get('dropout_rate', 0.3)
    )
    
    # Create trainer
    trainer = CausalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        device=args.device,
        logger=logger.logger
    )
    
    # Train model
    history = trainer.train(
        num_epochs=config['training']['num_epochs'],
        resume_from=args.resume
    )
    
    logger.info("Training completed successfully!")
    
    # Final model is saved via trainer checkpoints. If you want to save manually, use torch.save(model.state_dict(), path)


if __name__ == '__main__':
    main()
