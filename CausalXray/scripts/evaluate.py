# CausalXray/scripts/evaluate.py
"""Evaluation script for CausalXray model."""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys
import json

# Add package to path
sys.path.append(str(Path(__file__).parent.parent))

from causalxray import CausalXray
from causalxray.data import create_dataloader, NIHChestXray14, RSNAPneumonia, PediatricDataset
from causalxray.data import CausalTransforms
from causalxray.training import CausalMetrics
from causalxray.utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description='Evaluate CausalXray model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--dataset', type=str, choices=['nih', 'rsna', 'pediatric'], 
                       required=True, help='Dataset to evaluate on')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', 
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger('CausalXray_Evaluation', args.output_dir)
    
    # Load model
    model, checkpoint = CausalXray.load_checkpoint(args.checkpoint, args.device)
    model.eval()
    
    logger.info(f"Loaded model from: {args.checkpoint}")
    logger.info(f"Model trained for {checkpoint['epoch']} epochs")
    
    # Create dataset
    if args.dataset == 'nih':
        dataset_class = NIHChestXray14
    elif args.dataset == 'rsna':
        dataset_class = RSNAPneumonia
    elif args.dataset == 'pediatric':
        dataset_class = PediatricDataset
    
    # Create transforms
    test_transforms = CausalTransforms(mode='test')
    
    # Create test dataset
    test_dataset = dataset_class(
        data_dir=args.data_dir,
        split='test',
        transform=test_transforms,
        include_confounders=True
    )
    
    # Create data loader
    test_loader = create_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize metrics
    metrics_calculator = CausalMetrics()
    
    # Evaluation
    logger.info("Starting evaluation...")
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(args.device)
            labels = batch['label'].to(args.device)
            
            # Forward pass
            outputs = model(images)
            
            # Collect predictions
            predictions = torch.argmax(outputs['probabilities'], dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(outputs['probabilities'].cpu().numpy())
    
    # Compute metrics
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    
    results = metrics_calculator.compute_epoch_metrics(all_probabilities, all_labels)
    
    # Log results
    logger.info("Evaluation Results:")
    for metric, value in results.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save results
    results_path = Path(args.output_dir) / f'{args.dataset}_evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {results_path}")


if __name__ == '__main__':
    main()
