# CausalXray/scripts/inference.py
"""Inference script for CausalXray model with attribution visualization."""

import argparse
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add package to path
sys.path.append(str(Path(__file__).parent.parent))

from causalxray import CausalXray
from causalxray.data import CausalTransforms
from causalxray.utils import AttributionVisualizer


def main():
    parser = argparse.ArgumentParser(description='CausalXray inference with attribution')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--output_dir', type=str, default='./inference_results', 
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--show_attributions', action='store_true', 
                       help='Generate attribution visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, checkpoint = CausalXray.load_checkpoint(args.checkpoint, args.device)
    model.eval()
    
    print(f"Loaded model from: {args.checkpoint}")
    
    # Load and preprocess image
    image = Image.open(args.image)
    transforms = CausalTransforms(mode='test')
    image_tensor = transforms(image).unsqueeze(0).to(args.device)
    
    # Inference
    with torch.no_grad():
        predictions = model.predict(
            image_tensor,
            return_probabilities=True,
            return_attributions=args.show_attributions
        )
    
    # Get results
    predicted_class = predictions['predicted_class'].item()
    probabilities = predictions['probabilities'].cpu().numpy()[0]
    
    class_names = ['Normal', 'Pneumonia']
    predicted_label = class_names[predicted_class]
    confidence = probabilities[predicted_class]
    
    print(f"\nPrediction Results:")
    print(f"Predicted Class: {predicted_label}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Probabilities: Normal={probabilities[0]:.4f}, Pneumonia={probabilities[1]:.4f}")
    
    # Save results
    results = {
        'predicted_class': predicted_label,
        'confidence': float(confidence),
        'probabilities': {
            'normal': float(probabilities[0]),
            'pneumonia': float(probabilities[1])
        }
    }
    
    # Visualization
    if args.show_attributions and 'attributions' in predictions:
        print("\nGenerating attribution visualizations...")
        
        visualizer = AttributionVisualizer()
        
        # Prepare image for visualization
        original_image = np.array(image)
        if len(original_image.shape) == 3:
            original_image = original_image.mean(axis=2)  # Convert to grayscale
        
        attributions = predictions['attributions']
        
        # Create attribution comparison plot
        fig = visualizer.visualize_attribution_comparison(
            original_image,
            {k: v.cpu().numpy()[0] for k, v in attributions.items()},
            prediction=results
        )
        
        # Save visualization
        attribution_path = output_dir / 'attribution_comparison.png'
        fig.savefig(attribution_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Attribution visualization saved to: {attribution_path}")
    
    # Save prediction results
    results_path = output_dir / 'prediction_results.json'
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")


if __name__ == '__main__':
    main()
