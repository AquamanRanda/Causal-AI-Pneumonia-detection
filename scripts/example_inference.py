#!/usr/bin/env python3
"""
Example script for using CausalXray inference with causal attributions.

This script demonstrates how to:
1. Load a trained CausalXray model
2. Perform inference on X-ray images
3. Generate causal attributions
4. Visualize results

Usage:
    python example_inference.py --model_path /path/to/model.pth --image_path /path/to/xray.jpg
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.inference_with_attributions import CausalXrayInference


def example_single_image_inference(model_path: str, image_path: str, output_dir: str = None):
    """
    Example: Analyze a single X-ray image with causal attributions.
    
    Args:
        model_path: Path to the trained model checkpoint
        image_path: Path to the X-ray image
        output_dir: Directory to save results (optional)
    """
    print("🔍 Single Image Inference Example")
    print("=" * 50)
    
    # Initialize inference engine
    print("Loading model...")
    inference_engine = CausalXrayInference(
        model_path=model_path,
        device=None  # Auto-detect device
    )
    
    # Perform inference with attributions
    print(f"\nAnalyzing image: {image_path}")
    results = inference_engine.generate_attributions(
        image_path=image_path,
        save_visualization=True,
        output_dir=output_dir
    )
    
    # Print detailed results
    print("\n📊 Detailed Results:")
    print(f"Image: {results['image_path']}")
    print(f"Prediction: {results['predicted_class']}")
    print(f"Confidence: {results['confidence']:.4f}")
    print(f"Probabilities:")
    print(f"  Normal: {results['probabilities']['normal']:.4f}")
    print(f"  Pneumonia: {results['probabilities']['pneumonia']:.4f}")
    print(f"Attribution methods: {', '.join(results['attribution_methods'])}")
    
    return results


def example_batch_inference(model_path: str, image_dir: str, output_dir: str = None):
    """
    Example: Analyze multiple X-ray images in batch.
    
    Args:
        model_path: Path to the trained model checkpoint
        image_dir: Directory containing X-ray images
        output_dir: Directory to save results (optional)
    """
    print("🔍 Batch Inference Example")
    print("=" * 50)
    
    # Initialize inference engine
    print("Loading model...")
    inference_engine = CausalXrayInference(
        model_path=model_path,
        device=None  # Auto-detect device
    )
    
    # Find all image files
    image_dir_path = Path(image_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_paths = [
        str(f) for f in image_dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    print(f"Found {len(image_paths)} images for batch processing")
    
    # Perform batch inference
    results = inference_engine.batch_inference(
        image_paths=image_paths,
        save_attributions=True,
        output_dir=output_dir
    )
    
    # Print summary
    inference_engine.print_summary(results)
    
    return results


def example_quick_prediction(model_path: str, image_path: str):
    """
    Example: Quick prediction without attributions (faster).
    
    Args:
        model_path: Path to the trained model checkpoint
        image_path: Path to the X-ray image
    """
    print("⚡ Quick Prediction Example")
    print("=" * 50)
    
    # Initialize inference engine
    print("Loading model...")
    inference_engine = CausalXrayInference(
        model_path=model_path,
        device=None  # Auto-detect device
    )
    
    # Perform quick prediction
    print(f"\nAnalyzing image: {image_path}")
    results = inference_engine.predict(image_path)
    
    # Print results
    print(f"\nPrediction: {results['predicted_class']}")
    print(f"Confidence: {results['confidence']:.4f}")
    print(f"Normal probability: {results['probabilities']['normal']:.4f}")
    print(f"Pneumonia probability: {results['probabilities']['pneumonia']:.4f}")
    
    return results


def example_target_class_attribution(model_path: str, image_path: str, target_class: int, output_dir: str = None):
    """
    Example: Generate attributions for a specific target class.
    
    Args:
        model_path: Path to the trained model checkpoint
        image_path: Path to the X-ray image
        target_class: Target class (0=Normal, 1=Pneumonia)
        output_dir: Directory to save results (optional)
    """
    print("🎯 Target Class Attribution Example")
    print("=" * 50)
    
    class_names = ['Normal', 'Pneumonia']
    print(f"Generating attributions for target class: {class_names[target_class]}")
    
    # Initialize inference engine
    print("Loading model...")
    inference_engine = CausalXrayInference(
        model_path=model_path,
        device=None  # Auto-detect device
    )
    
    # Generate attributions for specific target class
    results = inference_engine.generate_attributions(
        image_path=image_path,
        target_class=target_class,
        save_visualization=True,
        output_dir=output_dir
    )
    
    print(f"\nResults for target class '{class_names[target_class]}':")
    print(f"Actual prediction: {results['predicted_class']}")
    print(f"Confidence: {results['confidence']:.4f}")
    print(f"Attribution methods: {', '.join(results['attribution_methods'])}")
    
    return results


def main():
    """Main function with command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CausalXray Inference Examples"
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the trained model checkpoint (.pth file)'
    )
    
    parser.add_argument(
        '--image_path',
        type=str,
        help='Path to a single X-ray image for analysis'
    )
    
    parser.add_argument(
        '--image_dir',
        type=str,
        help='Directory containing multiple X-ray images for batch analysis'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./inference_results',
        help='Directory to save results and visualizations'
    )
    
    parser.add_argument(
        '--example',
        type=str,
        choices=['single', 'batch', 'quick', 'target'],
        default='single',
        help='Type of example to run'
    )
    
    parser.add_argument(
        '--target_class',
        type=int,
        choices=[0, 1],
        default=1,
        help='Target class for attribution (0=Normal, 1=Pneumonia)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.example == 'single' and not args.image_path:
        parser.error("--image_path is required for single image example")
    
    if args.example == 'batch' and not args.image_dir:
        parser.error("--image_dir is required for batch example")
    
    if args.example == 'target' and not args.image_path:
        parser.error("--image_path is required for target class example")
    
    # Run examples
    try:
        if args.example == 'single':
            results = example_single_image_inference(
                model_path=args.model_path,
                image_path=args.image_path,
                output_dir=args.output_dir
            )
        
        elif args.example == 'batch':
            results = example_batch_inference(
                model_path=args.model_path,
                image_dir=args.image_dir,
                output_dir=args.output_dir
            )
        
        elif args.example == 'quick':
            results = example_quick_prediction(
                model_path=args.model_path,
                image_path=args.image_path
            )
        
        elif args.example == 'target':
            results = example_target_class_attribution(
                model_path=args.model_path,
                image_path=args.image_path,
                target_class=args.target_class,
                output_dir=args.output_dir
            )
        
        print(f"\n✅ Example completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error running example: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main()) 