"""
Causal attribution mechanisms for CausalXray framework.

This module implements intervention-based attribution methods that provide genuine causal
explanations for model predictions, going beyond correlation-based saliency maps to
establish causal relationships between image features and diagnostic outcomes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import numpy as np
from scipy import ndimage

# Optional imports
try:
    from captum.attr import IntegratedGradients, LayerGradCam, LayerConductance
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("Warning: captum not available. Some attribution methods will be disabled.")

# Import the other model components
from .backbone import CausalBackbone
from .causal_heads import CausalHeads


class CausalAttribution(nn.Module):
    """
    Causal attribution module implementing intervention-based explanations
    using Pearl's do-calculus and counterfactual reasoning.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_layers: List[str],
        attribution_methods: List[str] = ['intervention', 'counterfactual', 'gradcam'],
        patch_size: int = 16,
        num_patches: Optional[int] = None
    ):
        """
        Initialize causal attribution module.

        Args:
            model: The trained CausalXray model
            feature_layers: Names of layers to extract features from
            attribution_methods: List of attribution methods to use
            patch_size: Size of image patches for intervention analysis
            num_patches: Number of patches to analyze (if None, use all patches)
        """
        super(CausalAttribution, self).__init__()

        self.model = model
        self.feature_layers = feature_layers
        self.attribution_methods = attribution_methods
        self.patch_size = patch_size
        self.num_patches = num_patches

        # Initialize attribution methods
        self.attributors = {}
        if 'gradcam' in attribution_methods and CAPTUM_AVAILABLE:
            self.attributors['gradcam'] = LayerGradCam(model, self._get_gradcam_layer(model))
        if 'integrated_gradients' in attribution_methods and CAPTUM_AVAILABLE:
            self.attributors['integrated_gradients'] = IntegratedGradients(model)

    def forward(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Generate causal attributions for input images.

        Args:
            x: Input images of shape (batch_size, channels, height, width)
            target_class: Target class for attribution (if None, use predicted class)
            return_intermediate: Whether to return intermediate attribution results

        Returns:
            Dictionary containing attribution maps and scores
        """
        batch_size, channels, height, width = x.shape
        device = x.device

        # Get model predictions
        with torch.no_grad():
            model_output = self.model(x)
            if target_class is None:
                target_class_tensor = torch.argmax(model_output['probabilities'], dim=1)
            else:
                if isinstance(target_class, int):
                    target_class_tensor = torch.full((batch_size,), target_class, dtype=torch.long, device=device)
                elif isinstance(target_class, torch.Tensor):
                    target_class_tensor = target_class.to(device)
                else:
                    raise ValueError("target_class must be int, None, or torch.Tensor")

        attributions = {}

        # Intervention-based attribution
        if 'intervention' in self.attribution_methods:
            intervention_attr = self._intervention_attribution(x, target_class_tensor)
            attributions['intervention'] = intervention_attr

        # Counterfactual attribution
        if 'counterfactual' in self.attribution_methods:
            counterfactual_attr = self._counterfactual_attribution(x, target_class_tensor)
            attributions['counterfactual'] = counterfactual_attr

        # GradCAM attribution
        if 'gradcam' in self.attribution_methods:
            gradcam_attr = self._gradcam_attribution(x, target_class_tensor)
            attributions['gradcam'] = gradcam_attr

        # Integrated gradients attribution
        if 'integrated_gradients' in self.attribution_methods:
            ig_attr = self._integrated_gradients_attribution(x, target_class_tensor)
            attributions['integrated_gradients'] = ig_attr

        # Aggregate attributions if multiple methods
        if len(attributions) > 1:
            aggregated_attr = self._aggregate_attributions(attributions)
            attributions['aggregated'] = aggregated_attr

        return attributions

    def _intervention_attribution(
        self,
        x: torch.Tensor,
        target_class: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate intervention-based attributions by systematically perturbing
        image patches and measuring causal effects.
        """
        batch_size, channels, height, width = x.shape
        device = x.device

        # Create patch grid
        patch_size = self.patch_size
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        total_patches = num_patches_h * num_patches_w

        # Limit number of patches if specified
        if self.num_patches is not None:
            total_patches = min(total_patches, self.num_patches)

        attribution_map = torch.zeros((batch_size, height, width), device=device)

        # Sample patches for intervention
        patch_indices = torch.randperm(num_patches_h * num_patches_w)[:total_patches]

        for patch_idx in patch_indices:
            # Calculate patch coordinates
            patch_h = (patch_idx // num_patches_w) * patch_size
            patch_w = (patch_idx % num_patches_w) * patch_size

            # Create intervened image (replace patch with normal patch)
            intervened_x = x.clone()
            normal_patch = self._generate_normal_patch(
                x[:, :, patch_h:patch_h + patch_size, patch_w:patch_w + patch_size]
            )
            intervened_x[:, :, patch_h:patch_h + patch_size, patch_w:patch_w + patch_size] = normal_patch

            # Get predictions for original and intervened images
            with torch.no_grad():
                original_output = self.model(x)
                intervened_output = self.model(intervened_x)

                original_probs = F.softmax(original_output['logits'], dim=1)
                intervened_probs = F.softmax(intervened_output['logits'], dim=1)

                # Calculate causal effect
                causal_effect = original_probs[torch.arange(batch_size), target_class] - \
                               intervened_probs[torch.arange(batch_size), target_class]

                # Update attribution map
                attribution_map[:, patch_h:patch_h + patch_size, patch_w:patch_w + patch_size] = \
                    causal_effect.unsqueeze(1).unsqueeze(2).expand(-1, patch_size, patch_size)

        return attribution_map

    def _counterfactual_attribution(
        self,
        x: torch.Tensor,
        target_class: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate counterfactual attributions by creating alternative scenarios
        and measuring the difference in predictions.
        """
        batch_size, channels, height, width = x.shape
        device = x.device

        # Create counterfactual image (invert the image)
        counterfactual_x = 1.0 - x

        # Get predictions for original and counterfactual images
        with torch.no_grad():
            original_output = self.model(x)
            counterfactual_output = self.model(counterfactual_x)

            original_probs = F.softmax(original_output['logits'], dim=1)
            counterfactual_probs = F.softmax(counterfactual_output['logits'], dim=1)

            # Calculate counterfactual effect
            counterfactual_effect = original_probs[torch.arange(batch_size), target_class] - \
                                   counterfactual_probs[torch.arange(batch_size), target_class]

            # Create attribution map based on counterfactual effect
            attribution_map = counterfactual_effect.unsqueeze(1).unsqueeze(2).expand(-1, height, width)

        return attribution_map

    def _generate_normal_patch(self, patch: torch.Tensor) -> torch.Tensor:
        """Generate a normal patch by applying Gaussian noise."""
        normal_patch = torch.randn_like(patch) * 0.1 + 0.5
        return torch.clamp(normal_patch, 0, 1)

    def _gradcam_attribution(
        self,
        x: torch.Tensor,
        target_class: torch.Tensor
    ) -> torch.Tensor:
        """Generate GradCAM attributions."""
        if not CAPTUM_AVAILABLE or 'gradcam' not in self.attributors:
            return torch.zeros_like(x[:, 0])  # Return zero attribution if not available

        attributions = self.attributors['gradcam'].attribute(
            x, target=target_class
        )
        return attributions

    def _integrated_gradients_attribution(
        self,
        x: torch.Tensor,
        target_class: torch.Tensor
    ) -> torch.Tensor:
        """Generate Integrated Gradients attributions."""
        if not CAPTUM_AVAILABLE or 'integrated_gradients' not in self.attributors:
            return torch.zeros_like(x[:, 0])  # Return zero attribution if not available

        attributions = self.attributors['integrated_gradients'].attribute(
            x, target=target_class
        )
        return attributions

    def _aggregate_attributions(
        self,
        attributions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Aggregate multiple attribution methods using weighted average."""
        aggregated = torch.zeros_like(list(attributions.values())[0])
        
        # Simple average aggregation
        for attribution in attributions.values():
            aggregated += attribution
        
        return aggregated / len(attributions)

    def _get_gradcam_layer(self, model: nn.Module) -> nn.Module:
        """Get the appropriate layer for GradCAM."""
        # Try to find the last convolutional layer
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Conv2d):
                return module
        # Fallback to the model itself
        return model

    def generate_attribution_heatmap(
        self,
        attribution: Union[torch.Tensor, np.ndarray],
        original_image: Union[torch.Tensor, np.ndarray],
        colormap: str = 'jet',
        alpha: float = 0.6
    ) -> np.ndarray:
        """
        Generate a heatmap visualization of attributions overlaid on the original image.

        Args:
            attribution: Attribution map (H, W) or (B, H, W)
            original_image: Original image (H, W) or (B, H, W)
            colormap: Matplotlib colormap name
            alpha: Transparency for overlay

        Returns:
            Heatmap visualization as numpy array
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        # Convert to numpy if needed
        if isinstance(attribution, torch.Tensor):
            attribution = attribution.detach().cpu().numpy()
        if isinstance(original_image, torch.Tensor):
            original_image = original_image.detach().cpu().numpy()

        # Handle batch dimension
        if attribution.ndim == 3:
            attribution = attribution[0]
        if original_image.ndim == 3:
            original_image = original_image[0]

        # Normalize attribution
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)

        # Create colormap
        cmap = cm.get_cmap(colormap)
        attribution_colored = cmap(attribution)[:, :, :3]  # Remove alpha channel

        # Normalize original image
        if original_image.max() > 1.0:
            original_image = original_image / 255.0

        # Create overlay
        if original_image.ndim == 2:
            original_image = np.stack([original_image] * 3, axis=-1)

        heatmap = alpha * attribution_colored + (1 - alpha) * original_image
        heatmap = np.clip(heatmap, 0, 1)

        return heatmap


class AttributionQualityMetrics:
    """Metrics for evaluating attribution quality and consistency."""

    @staticmethod
    def sensitivity_correlation(
        attribution1: torch.Tensor,
        attribution2: torch.Tensor
    ) -> float:
        """Calculate correlation between two attribution methods."""
        flat1 = attribution1.flatten()
        flat2 = attribution2.flatten()
        
        correlation = torch.corrcoef(torch.stack([flat1, flat2]))[0, 1]
        return correlation.item()

    @staticmethod
    def attribution_consistency(
        attributions: List[torch.Tensor],
        threshold: float = 0.1
    ) -> float:
        """Calculate consistency across multiple attribution methods."""
        if len(attributions) < 2:
            return 1.0

        # Binarize attributions
        binarized = []
        for attr in attributions:
            binary = (attr > threshold).float()
            binarized.append(binary.flatten())

        # Calculate intersection over union
        intersection = torch.stack(binarized).min(dim=0)[0]
        union = torch.stack(binarized).max(dim=0)[0]
        
        iou = intersection.sum() / (union.sum() + 1e-8)
        return iou.item()

    @staticmethod
    def attribution_sparsity(attribution: torch.Tensor, percentile: float = 90) -> float:
        """Calculate sparsity of attribution map."""
        flat_attr = attribution.flatten()
        threshold = torch.quantile(flat_attr, percentile / 100)
        sparsity = (flat_attr < threshold).float().mean()
        return sparsity.item()


class CausalXrayModel(nn.Module):
    """
    Main CausalXray model that combines backbone, causal heads, and attribution.
    
    This model implements causal reasoning for X-ray image analysis with
    interpretable attributions and confounding variable prediction.
    """

    def __init__(
        self,
        backbone_config: Dict[str, Any],
        causal_config: Dict[str, Any],
        attribution_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the CausalXray model.

        Args:
            backbone_config: Configuration for the backbone network
            causal_config: Configuration for causal heads and reasoning
            attribution_config: Configuration for attribution methods
        """
        super(CausalXrayModel, self).__init__()

        # Initialize backbone
        self.backbone = CausalBackbone(
            architecture=backbone_config.get('architecture', 'densenet121'),
            pretrained=backbone_config.get('pretrained', True),
            num_classes=backbone_config.get('num_classes', 2),
            feature_dims=backbone_config.get('feature_dims', [1024, 512, 256]),
            dropout_rate=backbone_config.get('dropout_rate', 0.3)
        )

        # Initialize causal heads
        confounders = causal_config.get('confounders', {})
        self.causal_heads = CausalHeads(
            input_dim=backbone_config.get('feature_dims', [1024, 512, 256])[-1],
            confounders=confounders,
            hidden_dims=causal_config.get('hidden_dims', [512, 256]),
            dropout_rate=causal_config.get('dropout_rate', 0.3),
            use_variational=causal_config.get('use_variational', True)
        )

        # Store attribution config for lazy initialization
        self.attribution_config = attribution_config or {}
        self.attribution = None  # Will be initialized when needed

        # Model configuration
        self.config = {
            'backbone': backbone_config,
            'causal': causal_config,
            'attribution': attribution_config
        }

    def forward(self, x: torch.Tensor, confounders: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Forward pass through the CausalXray model.

        Args:
            x: Input images of shape (batch_size, channels, height, width)
            confounders: Optional confounding variables

        Returns:
            Dictionary containing model outputs
        """
        # Extract features from backbone
        backbone_output = self.backbone(x, confounders)
        # Use the final causal features (256-dimensional) for causal heads
        features = backbone_output['causal_features'][-1]  # Get the last causal feature layer

        # Process through causal heads
        causal_output = self.causal_heads(features)

        # Combine outputs
        outputs = {
            'logits': backbone_output['logits'],
            'probabilities': F.softmax(backbone_output['logits'], dim=1),
            'features': features,
            'causal_features': causal_output.get('causal_features', features),
            'confounder_predictions': causal_output.get('confounder_predictions', {}),
            'latent_variables': causal_output.get('latent_variables', {})
        }

        return outputs

    def predict(self, x: torch.Tensor) -> Dict[str, Any]:
        """Make predictions on input images."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def generate_attributions(
        self, 
        x: torch.Tensor, 
        target_class: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Generate causal attributions for input images."""
        # Initialize attribution module if not already done
        if self.attribution is None:
            from .attribution import CausalAttribution
            self.attribution = CausalAttribution(
                model=self,
                feature_layers=self.attribution_config.get('feature_layers', ['backbone']),
                attribution_methods=self.attribution_config.get('attribution_methods', 
                                                             ['intervention', 'counterfactual', 'gradcam']),
                patch_size=self.attribution_config.get('patch_size', 16),
                num_patches=self.attribution_config.get('num_patches', None)
            )
        
        return self.attribution(x, target_class=target_class)

    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract feature maps from the model."""
        return self.backbone.get_feature_maps(x)
