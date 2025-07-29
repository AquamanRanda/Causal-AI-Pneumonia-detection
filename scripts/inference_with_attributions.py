#!/usr/bin/env python3
"""
CausalXray Inference Script

This script loads trained CausalXray models and performs inference with causal attributions
on X-ray images to detect pneumonia and provide interpretable explanations.

Usage:
    python inference_with_attributions.py --model_path /path/to/model.pth --image_path /path/to/xray.jpg
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

# Optional imports
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not available. Some visualizations may be limited.")

# Import our custom modules
from causalxray.models.causalxray import CausalXrayModel
from causalxray.data.transforms import CausalTransforms
from causalxray.models.attribution import CausalAttribution
from causalxray.utils.visualization import AttributionVisualizer
from causalxray.utils.config import load_config, create_default_config


class CausalXrayInference:
    """Inference class for CausalXray models with causal attributions."""
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the trained model checkpoint
            config_path: Path to the model configuration file
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        self.device = self._setup_device(device)
        
        # Load configuration
        self.config = self._load_configuration()
        
        # Initialize components
        self.model = None
        self.attribution_module = None
        self.visualizer = None
        
        # Setup logging
        self._setup_logging()
        
        # Load model
        self._load_model()
        
        # Setup attribution
        self._setup_attribution()
    
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Setup the device for inference."""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        device = torch.device(device)
        print(f"Using device: {device}")
        
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        return device
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load model configuration."""
        if self.config_path and self.config_path.exists():
            config = load_config(str(self.config_path))
            print(f"Loaded configuration from {self.config_path}")
        else:
            config = create_default_config()
            print("Using default configuration")
        
        return config
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('CausalXrayInference')
    
    def _load_model(self):
        """Load the trained model from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        print(f"Loading model from {self.model_path}")
        
        # Load checkpoint with weights_only=False for compatibility
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"Warning: Failed to load with weights_only=False: {e}")
            print("Trying with weights_only=True...")
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            except Exception as e2:
                print(f"Error loading checkpoint: {e2}")
                raise
        
        # Extract configuration from checkpoint if available
        if 'config' in checkpoint:
            self.config = checkpoint['config']
            print("Loaded configuration from checkpoint")
        
        # Create model
        backbone_config = self.config['model']['backbone']
        causal_config = self.config['causal']
        
        print(f"Creating model with backbone config: {backbone_config}")
        print(f"Creating model with causal config: {causal_config}")
        
        try:
            self.model = CausalXrayModel(
                backbone_config=backbone_config,
                causal_config=causal_config
            )
        except Exception as e:
            print(f"Error creating model: {e}")
            raise
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("Loaded model weights successfully")
            except Exception as e:
                print(f"Error loading model state dict: {e}")
                print("Available keys in checkpoint:", list(checkpoint.keys()))
                raise
        else:
            print("Available keys in checkpoint:", list(checkpoint.keys()))
            raise ValueError("No model_state_dict found in checkpoint")
        
        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded with {total_params:,} parameters")
        
        # Print checkpoint info
        if 'best_metric' in checkpoint:
            print(f"Best validation metric: {checkpoint['best_metric']:.4f}")
        if 'epoch' in checkpoint:
            print(f"Trained for {checkpoint['epoch']} epochs")
    
    def _setup_attribution(self):
        """Setup causal attribution module."""
        attribution_config = self.config.get('attribution', {})
        
        # The attribution module will be initialized when needed
        # Store the config for later use
        self.attribution_config = attribution_config
        self.attribution_module = None  # Will be initialized when needed
        
        self.visualizer = AttributionVisualizer()
        print("Attribution module initialized")
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess an image for inference.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (preprocessed_tensor, original_image_array)
        """
        # Load image
        image = Image.open(image_path)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Keep original for visualization
        original_image = np.array(image)
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            original_image = np.mean(original_image, axis=2)  # Convert to grayscale
        
        # Apply transforms
        image_size = tuple(self.config['data']['image_size'])
        transforms = CausalTransforms(mode='test', image_size=image_size)
        
        # Preprocess
        image_tensor = transforms(image).unsqueeze(0).to(self.device)
        
        return image_tensor, original_image
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Perform prediction on a single image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing prediction results
        """
        print(f"Analyzing image: {image_path}")
        
        # Preprocess image
        image_tensor, original_image = self.preprocess_image(image_path)
        
        # Perform inference
        self.model.eval()
        with torch.no_grad():
            
            
            outputs = self.model(image_tensor)
            probabilities = outputs['probabilities'].cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
        
        # Get results
        class_names = ['Normal', 'Pneumonia']
        predicted_label = class_names[predicted_class]
        confidence = probabilities[predicted_class]
        
        results = {
            'image_path': image_path,
            'predicted_class': predicted_label,
            'predicted_class_id': int(predicted_class),
            'confidence': float(confidence),
            'probabilities': {
                'normal': float(probabilities[0]),
                'pneumonia': float(probabilities[1])
            },
            'original_image': original_image
        }
        
        print(f"Prediction: {predicted_label} (confidence: {confidence:.4f})")
        print(f"Probabilities - Normal: {probabilities[0]:.4f}, Pneumonia: {probabilities[1]:.4f}")
        
        return results
    
    def generate_attributions(
        self,
        image_path: str,
        target_class: Optional[int] = None,
        save_visualization: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate causal attributions for an image.
        
        Args:
            image_path: Path to the input image
            target_class: Target class for attribution (None for predicted class)
            save_visualization: Whether to save attribution visualizations
            output_dir: Directory to save visualizations (uses image directory if None)
            
        Returns:
            Dictionary containing attribution results
        """
        print(f"Generating causal attributions for: {image_path}")
        
        # Get prediction first
        prediction_results = self.predict(image_path)
        
        # Preprocess image
        image_tensor, original_image = self.preprocess_image(image_path)
        
        # Generate attributions using the model's attribution method
        with torch.no_grad():
            attributions_tensor = self.model.generate_attributions(
                image_tensor, 
                target_class=target_class or prediction_results['predicted_class_id']
            )
        
        # Convert to numpy
        attributions_np = {}
        for method, attr_tensor in attributions_tensor.items():
            if isinstance(attr_tensor, torch.Tensor):
                attributions_np[method] = attr_tensor.squeeze().cpu().numpy()
            else:
                print(f"Warning: {method} returned non-tensor: {type(attr_tensor)}")
                if isinstance(attr_tensor, dict):
                    print(f"Warning: {method} dict keys: {list(attr_tensor.keys())}")
        
        # Debug: print what we have
        print(f"Available attribution methods: {list(attributions_np.keys())}")
        for method, attr in attributions_np.items():
            if hasattr(attr, 'shape'):
                print(f"  {method}: shape {attr.shape}")
            else:
                print(f"  {method}: type {type(attr)} (no shape attribute)")
        
        # Create prediction info for visualization
        prediction_info = {
            'predicted_class': prediction_results['predicted_class'],
            'confidence': prediction_results['confidence'],
            'probabilities': prediction_results['probabilities']
        }
        
        # Generate visualizations
        if self.visualizer and save_visualization:
            # Create output directory
            if output_dir is None:
                output_dir = Path(image_path).parent / 'attributions'
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save comparison visualization
            comparison_path = output_dir / f"{Path(image_path).stem}_attributions.png"
            try:
                fig = self.visualizer.visualize_attribution_comparison(
                    original_image, attributions_np, prediction_info
                )
                fig.savefig(comparison_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"ERROR: Failed to create comparison visualization: {e}")
            
            # Save statistics visualization
            stats_path = output_dir / f"{Path(image_path).stem}_statistics.png"
            try:
                fig = self.visualizer.plot_attribution_statistics(attributions_np)
                fig.savefig(stats_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"ERROR: Failed to create statistics visualization: {e}")
            
            print(f"Attribution visualizations saved to: {output_dir}")
        
        # Combine results
        results = {
            **prediction_results,
            'attributions': attributions_np,
            'attribution_methods': list(attributions_np.keys())
        }
        
        return results
    
    def batch_inference(
        self,
        image_paths: List[str],
        save_attributions: bool = True,
        output_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform batch inference on multiple images.
        
        Args:
            image_paths: List of image paths
            save_attributions: Whether to generate attributions
            output_dir: Directory to save results
            
        Returns:
            List of results for each image
        """
        print(f"Performing batch inference on {len(image_paths)} images")
        
        if output_dir is None:
            output_dir = Path.cwd() / 'inference_results'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"\nProcessing image {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                if save_attributions:
                    results = self.generate_attributions(
                        image_path,
                        output_dir=output_dir
                    )
                else:
                    results = self.predict(image_path)
                
                all_results.append(results)
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                all_results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        # Save batch results
        results_path = output_dir / 'batch_results.json'
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = []
            for result in all_results:
                json_result = result.copy()
                if 'attributions' in json_result:
                    json_result['attributions'] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in json_result['attributions'].items()
                    }
                if 'original_image' in json_result:
                    del json_result['original_image']  # Don't save large arrays
                json_results.append(json_result)
            
            json.dump(json_results, f, indent=2)
        
        print(f"\nBatch inference completed. Results saved to: {results_path}")
        
        return all_results
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Print a summary of inference results."""
        print("\n" + "="*60)
        print("INFERENCE SUMMARY")
        print("="*60)
        
        # Count predictions
        predictions = {}
        errors = 0
        
        for result in results:
            if 'error' in result:
                errors += 1
                continue
            
            pred_class = result['predicted_class']
            predictions[pred_class] = predictions.get(pred_class, 0) + 1
        
        # Print statistics
        total_processed = len(results)
        successful = total_processed - errors
        
        print(f"Total images processed: {total_processed}")
        print(f"Successful predictions: {successful}")
        print(f"Errors: {errors}")
        
        if predictions:
            print("\nPredictions:")
            for class_name, count in predictions.items():
                percentage = (count / successful) * 100
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Average confidence
        if successful > 0:
            confidences = [r['confidence'] for r in results if 'confidence' in r]
            avg_confidence = np.mean(confidences)
            print(f"\nAverage confidence: {avg_confidence:.4f}")
        
        print("="*60)


def main():
    """Main function for inference with attributions."""
    parser = argparse.ArgumentParser(description='CausalXray Inference with Attributions')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--config_path', type=str, help='Path to configuration file')
    parser.add_argument('--device', type=str, help='Device to use (cpu/cuda)')
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    parser.add_argument('--target_class', type=int, help='Target class for attribution')
    parser.add_argument('--batch_mode', action='store_true', help='Process multiple images')
    
    args = parser.parse_args()
    
    try:
        # Initialize inference engine
        inference_engine = CausalXrayInference(
            model_path=args.model_path,
            config_path=args.config_path,
            device=args.device
        )
        
        if args.batch_mode:
            # Batch processing
            if os.path.isdir(args.image_path):
                image_paths = [os.path.join(args.image_path, f) for f in os.listdir(args.image_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
            else:
                # Assume it's a file with list of paths
                with open(args.image_path, 'r') as f:
                    image_paths = [line.strip() for line in f.readlines()]
            
            results = inference_engine.batch_inference(
                image_paths=image_paths,
                save_attributions=True,
                output_dir=args.output_dir
            )
        else:
            # Single image processing
            results = inference_engine.generate_attributions(
                image_path=args.image_path,
                target_class=args.target_class,
                save_visualization=True,
                output_dir=args.output_dir
            )
            results = [results]  # Convert to list for consistency
        
        inference_engine.print_summary(results)
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    exit(main()) 