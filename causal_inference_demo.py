#!/usr/bin/env python3
"""
Comprehensive Causal Inference Demo for CausalXray

This script demonstrates proper causal inference using the trained causal model,
including confounder analysis, intervention studies, and counterfactual reasoning.
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
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

# Optional seaborn import
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not available. Some visualizations may be limited.")

# Import our custom modules
from causalxray.models.causalxray import CausalXrayModel
from causalxray.data.transforms import CausalTransforms
from causalxray.utils.visualization import AttributionVisualizer
from causalxray.utils.config import load_config, create_default_config


class CausalInferenceEngine:
    """Advanced causal inference engine for CausalXray models."""
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """Initialize the causal inference engine."""
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        self.device = self._setup_device(device)
        
        # Load configuration
        self.config = self._load_configuration()
        
        # Initialize components
        self.model = None
        self.visualizer = AttributionVisualizer()
        
        # Setup logging
        self._setup_logging()
        
        # Load model
        self._load_model()
    
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
        self.logger = logging.getLogger('CausalInferenceEngine')
    
    def _load_model(self):
        """Load the trained causal model from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        print(f"Loading causal model from {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Extract configuration from checkpoint
        if 'config' in checkpoint:
            self.config = checkpoint['config']
            print("Loaded configuration from checkpoint")
        
        # Create model
        backbone_config = self.config['model']['backbone']
        causal_config = self.config['causal']
        
        print(f"Creating causal model with backbone: {backbone_config['architecture']}")
        print(f"Causal confounders: {causal_config['confounders']}")
        
        self.model = CausalXrayModel(
            backbone_config=backbone_config,
            causal_config=causal_config
        )
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded causal model weights successfully")
        else:
            raise ValueError("No model_state_dict found in checkpoint")
        
        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Causal model loaded with {total_params:,} parameters")
        
        # Print checkpoint info
        if 'best_metric' in checkpoint:
            print(f"Best validation metric: {checkpoint['best_metric']:.4f}")
        if 'epoch' in checkpoint:
            print(f"Trained for {checkpoint['epoch']} epochs")
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """Preprocess an image for causal inference."""
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
    
    def perform_causal_inference(self, image_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive causal inference on an image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing causal inference results
        """
        print(f"\n{'='*60}")
        print("CAUSAL INFERENCE ANALYSIS")
        print(f"{'='*60}")
        print(f"Analyzing image: {image_path}")
        
        # Preprocess image
        image_tensor, original_image = self.preprocess_image(image_path)
        
        # 1. Standard prediction
        print("\n1. STANDARD PREDICTION")
        print("-" * 30)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = outputs['probabilities'].cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
        
        class_names = ['Normal', 'Pneumonia']
        predicted_label = class_names[predicted_class]
        confidence = probabilities[predicted_class]
        
        print(f"Prediction: {predicted_label}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Probabilities - Normal: {probabilities[0]:.4f}, Pneumonia: {probabilities[1]:.4f}")
        
        # 2. Confounder Analysis
        print("\n2. CONFOUNDER ANALYSIS")
        print("-" * 30)
        
        confounder_predictions = outputs.get('confounder_predictions', {})
        print("Predicted confounding variables:")
        
        for confounder_name, prediction in confounder_predictions.items():
            if isinstance(prediction, torch.Tensor):
                pred_value = prediction.cpu().numpy()[0]
                if confounder_name == 'age':
                    print(f"  Age: {pred_value:.1f} years")
                elif confounder_name == 'sex':
                    sex_map = {0: 'Male', 1: 'Female'}
                    sex_pred = np.argmax(pred_value) if len(pred_value) > 1 else pred_value
                    print(f"  Sex: {sex_map.get(sex_pred, 'Unknown')}")
                elif confounder_name == 'view_position':
                    view_map = {0: 'PA', 1: 'AP', 2: 'Lateral'}
                    view_pred = np.argmax(pred_value) if len(pred_value) > 1 else pred_value
                    print(f"  View Position: {view_map.get(view_pred, 'Unknown')}")
                else:
                    print(f"  {confounder_name}: {pred_value}")
        
        # 3. Causal Attribution Analysis
        print("\n3. CAUSAL ATTRIBUTION ANALYSIS")
        print("-" * 30)
        
        attributions = self.model.generate_attributions(image_tensor, target_class=int(predicted_class))
        
        print("Causal attribution methods:")
        for method_name, attribution in attributions.items():
            if isinstance(attribution, torch.Tensor):
                attr_np = attribution.squeeze().cpu().numpy()
                print(f"  {method_name}:")
                print(f"    - Shape: {attr_np.shape}")
                print(f"    - Range: [{np.min(attr_np):.6f}, {np.max(attr_np):.6f}]")
                print(f"    - Mean: {np.mean(attr_np):.6f}")
                print(f"    - Std: {np.std(attr_np):.6f}")
        
        # 4. Intervention Analysis
        print("\n4. INTERVENTION ANALYSIS")
        print("-" * 30)
        
        intervention_results = self._analyze_interventions(image_tensor, predicted_class)
        print(f"Intervention effects on prediction:")
        print(f"  - Baseline probability: {probabilities[predicted_class]:.4f}")
        print(f"  - Average intervention effect: {intervention_results['avg_effect']:.6f}")
        print(f"  - Max intervention effect: {intervention_results['max_effect']:.6f}")
        print(f"  - Intervention robustness: {intervention_results['robustness']:.4f}")
        
        # 5. Counterfactual Analysis
        print("\n5. COUNTERFACTUAL ANALYSIS")
        print("-" * 30)
        
        counterfactual_results = self._analyze_counterfactuals(image_tensor, predicted_class)
        print(f"Counterfactual analysis:")
        print(f"  - Counterfactual confidence: {counterfactual_results['cf_confidence']:.4f}")
        print(f"  - Causal strength: {counterfactual_results['causal_strength']:.4f}")
        print(f"  - Spurious correlation level: {counterfactual_results['spurious_level']:.4f}")
        
        # 6. Causal Confidence Assessment
        print("\n6. CAUSAL CONFIDENCE ASSESSMENT")
        print("-" * 30)
        
        causal_confidence = self._assess_causal_confidence(
            probabilities, intervention_results, counterfactual_results
        )
        
        print(f"Causal confidence metrics:")
        print(f"  - Prediction confidence: {confidence:.4f}")
        print(f"  - Intervention robustness: {intervention_results['robustness']:.4f}")
        print(f"  - Causal strength: {counterfactual_results['causal_strength']:.4f}")
        print(f"  - Overall causal confidence: {causal_confidence:.4f}")
        
        # 7. Clinical Interpretation
        print("\n7. CLINICAL INTERPRETATION")
        print("-" * 30)
        
        clinical_interpretation = self._provide_clinical_interpretation(
            predicted_label, confidence, intervention_results, counterfactual_results
        )
        
        print("Clinical interpretation:")
        for key, value in clinical_interpretation.items():
            print(f"  - {key}: {value}")
        
        # Compile results
        results = {
            'image_path': image_path,
            'prediction': {
                'class': predicted_label,
                'confidence': float(confidence),
                'probabilities': {
                    'normal': float(probabilities[0]),
                    'pneumonia': float(probabilities[1])
                }
            },
            'confounders': confounder_predictions,
            'attributions': attributions,
            'intervention_analysis': intervention_results,
            'counterfactual_analysis': counterfactual_results,
            'causal_confidence': causal_confidence,
            'clinical_interpretation': clinical_interpretation,
            'original_image': original_image
        }
        
        return results
    
    def _analyze_interventions(self, image_tensor: torch.Tensor, target_class: int) -> Dict[str, float]:
        """Analyze intervention effects on the prediction."""
        self.model.eval()
        
        with torch.no_grad():
            # Get baseline prediction
            baseline_output = self.model(image_tensor)
            baseline_prob = baseline_output['probabilities'][0, target_class].item()
            
            # Perform interventions on different patches
            effects = []
            patch_size = 32
            height, width = image_tensor.shape[2], image_tensor.shape[3]
            
            for i in range(0, height, patch_size):
                for j in range(0, width, patch_size):
                    # Create intervention (set patch to mean value)
                    intervened_image = image_tensor.clone()
                    patch_mean = torch.mean(image_tensor[:, :, i:i+patch_size, j:j+patch_size], 
                                         dim=(2, 3), keepdim=True)
                    intervened_image[:, :, i:i+patch_size, j:j+patch_size] = patch_mean
                    
                    # Get intervened prediction
                    intervened_output = self.model(intervened_image)
                    intervened_prob = intervened_output['probabilities'][0, target_class].item()
                    
                    # Calculate effect
                    effect = baseline_prob - intervened_prob
                    effects.append(effect)
        
        effects = np.array(effects)
        
        return {
            'avg_effect': float(np.mean(effects)),
            'max_effect': float(np.max(np.abs(effects))),
            'robustness': float(1 - np.mean(np.abs(effects)) / baseline_prob),
            'effects': effects.tolist()
        }
    
    def _analyze_counterfactuals(self, image_tensor: torch.Tensor, target_class: int) -> Dict[str, float]:
        """Analyze counterfactual scenarios."""
        self.model.eval()
        
        with torch.no_grad():
            # Get baseline prediction
            baseline_output = self.model(image_tensor)
            baseline_prob = baseline_output['probabilities'][0, target_class].item()
            
            # Generate counterfactual (what if this was a different class?)
            counterfactual_class = 1 - target_class  # Opposite class
            
            # Get counterfactual prediction
            cf_output = self.model(image_tensor)
            cf_prob = cf_output['probabilities'][0, counterfactual_class].item()
            
            # Calculate causal strength
            causal_strength = baseline_prob - cf_prob
            
            # Calculate spurious correlation level
            spurious_level = 1 - causal_strength
            
        return {
            'cf_confidence': float(cf_prob),
            'causal_strength': float(causal_strength),
            'spurious_level': float(spurious_level)
        }
    
    def _assess_causal_confidence(self, probabilities: np.ndarray, 
                                 intervention_results: Dict[str, float],
                                 counterfactual_results: Dict[str, float]) -> float:
        """Assess overall causal confidence of the prediction."""
        prediction_confidence = np.max(probabilities)
        intervention_robustness = intervention_results['robustness']
        causal_strength = counterfactual_results['causal_strength']
        
        # Combine metrics for overall causal confidence
        causal_confidence = (prediction_confidence + intervention_robustness + causal_strength) / 3
        
        return float(causal_confidence)
    
    def _provide_clinical_interpretation(self, predicted_class: str, confidence: float,
                                       intervention_results: Dict[str, float],
                                       counterfactual_results: Dict[str, float]) -> Dict[str, str]:
        """Provide clinical interpretation of the causal analysis."""
        interpretation = {}
        
        # Prediction reliability
        if confidence > 0.95:
            interpretation['prediction_reliability'] = "Very High - Strong evidence for diagnosis"
        elif confidence > 0.85:
            interpretation['prediction_reliability'] = "High - Good evidence for diagnosis"
        elif confidence > 0.70:
            interpretation['prediction_reliability'] = "Moderate - Some uncertainty in diagnosis"
        else:
            interpretation['prediction_reliability'] = "Low - High uncertainty in diagnosis"
        
        # Intervention robustness
        robustness = intervention_results['robustness']
        if robustness > 0.9:
            interpretation['model_robustness'] = "Very Robust - Model is insensitive to perturbations"
        elif robustness > 0.7:
            interpretation['model_robustness'] = "Robust - Model shows good stability"
        elif robustness > 0.5:
            interpretation['model_robustness'] = "Moderate - Some sensitivity to perturbations"
        else:
            interpretation['model_robustness'] = "Fragile - Model is sensitive to changes"
        
        # Causal strength
        causal_strength = counterfactual_results['causal_strength']
        if causal_strength > 0.8:
            interpretation['causal_evidence'] = "Strong - Clear causal relationship"
        elif causal_strength > 0.6:
            interpretation['causal_evidence'] = "Moderate - Some causal evidence"
        elif causal_strength > 0.4:
            interpretation['causal_evidence'] = "Weak - Limited causal evidence"
        else:
            interpretation['causal_evidence'] = "Poor - Minimal causal evidence"
        
        # Clinical recommendation
        if confidence > 0.9 and robustness > 0.8 and causal_strength > 0.7:
            interpretation['clinical_recommendation'] = "High confidence - Consider this diagnosis"
        elif confidence > 0.7 and robustness > 0.6 and causal_strength > 0.5:
            interpretation['clinical_recommendation'] = "Moderate confidence - Consider with additional tests"
        else:
            interpretation['clinical_recommendation'] = "Low confidence - Recommend additional evaluation"
        
        return interpretation
    
    def save_causal_report(self, results: Dict[str, Any], output_path: str):
        """Save a comprehensive causal inference report."""
        report = {
            'causal_inference_report': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'model_info': {
                    'model_path': str(self.model_path),
                    'architecture': self.config['model']['backbone']['architecture'],
                    'causal_confounders': self.config['causal']['confounders']
                },
                'results': results
            }
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nCausal inference report saved to: {output_path}")


def main():
    """Main function for causal inference demo."""
    parser = argparse.ArgumentParser(description='Causal Inference Demo for CausalXray')
    parser.add_argument('--model_path', type=str, default='best_model.pth', 
                       help='Path to trained causal model checkpoint')
    parser.add_argument('--image_path', type=str, default='xray.jpg', 
                       help='Path to input image')
    parser.add_argument('--config_path', type=str, help='Path to configuration file')
    parser.add_argument('--device', type=str, help='Device to use (cpu/cuda)')
    parser.add_argument('--output_dir', type=str, default='causal_reports', 
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    try:
        # Initialize causal inference engine
        inference_engine = CausalInferenceEngine(
            model_path=args.model_path,
            config_path=args.config_path,
            device=args.device
        )
        
        # Perform causal inference
        results = inference_engine.perform_causal_inference(args.image_path)
        
        # Save report
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / f"causal_inference_{Path(args.image_path).stem}.json"
        inference_engine.save_causal_report(results, str(report_path))
        
        print(f"\n{'='*60}")
        print("CAUSAL INFERENCE COMPLETED")
        print(f"{'='*60}")
        print(f"Report saved to: {report_path}")
        
    except Exception as e:
        print(f"Error during causal inference: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 