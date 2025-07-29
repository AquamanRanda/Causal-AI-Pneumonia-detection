"""
Causal attribution mechanisms for CausalXray framework.

This module implements intervention-based attribution methods that provide genuine causal
explanations for model predictions, going beyond correlation-based saliency maps to
establish causal relationships between image features and diagnostic outcomes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from scipy import ndimage

# Optional imports
try:
    from captum.attr import IntegratedGradients, LayerGradCam
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("Warning: captum not available. Some attribution methods will be disabled.")


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
                # If target_class is int, make it a tensor
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

        # Traditional attribution methods for comparison
        if 'gradcam' in self.attribution_methods and CAPTUM_AVAILABLE:
            gradcam_attr = self._gradcam_attribution(x, target_class_tensor)
            attributions['gradcam'] = gradcam_attr

        if 'integrated_gradients' in self.attribution_methods and CAPTUM_AVAILABLE:
            ig_attr = self._integrated_gradients_attribution(x, target_class_tensor)
            attributions['integrated_gradients'] = ig_attr

        # Aggregate attribution scores
        aggregated_attr = self._aggregate_attributions(attributions)
        attributions['aggregated'] = aggregated_attr

        return attributions

    def _intervention_attribution(
        self,
        x: torch.Tensor,
        target_class: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute intervention-based attribution using do-calculus.

        This method systematically intervenes on image patches and measures
        the causal effect on the prediction.
        """
        batch_size, channels, height, width = x.shape
        device = x.device

        # Create patch grid
        patches_h = height // self.patch_size
        patches_w = width // self.patch_size

        # Initialize attribution map
        attribution_map = torch.zeros(batch_size, height, width, device=device)

        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(x)
            baseline_probs = baseline_output['probabilities']

        # Iterate through patches
        for i in range(patches_h):
            for j in range(patches_w):
                # Define patch boundaries
                h_start = i * self.patch_size
                h_end = min((i + 1) * self.patch_size, height)
                w_start = j * self.patch_size
                w_end = min((j + 1) * self.patch_size, width)

                # Create intervention (set patch to mean value)
                x_intervened = x.clone()
                patch_mean = torch.mean(x[:, :, h_start:h_end, w_start:w_end], dim=(2, 3), keepdim=True)
                x_intervened[:, :, h_start:h_end, w_start:w_end] = patch_mean

                # Compute intervened prediction
                with torch.no_grad():
                    intervened_output = self.model(x_intervened)
                    intervened_probs = intervened_output['probabilities']

                # Compute causal effect
                for b in range(batch_size):
                    target_idx = target_class[b].item()
                    causal_effect = baseline_probs[b, target_idx] - intervened_probs[b, target_idx]
                    attribution_map[b, h_start:h_end, w_start:w_end] = causal_effect

        return attribution_map

    def _counterfactual_attribution(
        self,
        x: torch.Tensor,
        target_class: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute counterfactual attribution using structural causal models.

        This method asks "What would the prediction be if this region appeared normal?"
        """
        batch_size, channels, height, width = x.shape
        device = x.device

        # Initialize attribution map
        attribution_map = torch.zeros(batch_size, height, width, device=device)

        # Get model's causal representation
        with torch.no_grad():
            model_output = self.model(x)
            causal_features = model_output.get('causal_features', [])

            if len(causal_features) == 0:
                # Fallback to standard features if causal features not available
                causal_features = [model_output['features']]

        # Generate counterfactual scenarios
        for patch_idx in range(0, height * width, self.patch_size**2):
            i = (patch_idx // width) // self.patch_size
            j = (patch_idx % width) // self.patch_size

            if i >= height // self.patch_size or j >= width // self.patch_size:
                continue

            h_start = i * self.patch_size
            h_end = min((i + 1) * self.patch_size, height)
            w_start = j * self.patch_size
            w_end = min((j + 1) * self.patch_size, width)

            # Create counterfactual image (replace patch with normal tissue pattern)
            x_counterfactual = x.clone()
            normal_patch = self._generate_normal_patch(x[:, :, h_start:h_end, w_start:w_end])
            x_counterfactual[:, :, h_start:h_end, w_start:w_end] = normal_patch

            # Compute counterfactual prediction
            with torch.no_grad():
                cf_output = self.model(x_counterfactual)
                cf_probs = cf_output['probabilities']
                original_probs = model_output['probabilities']

            # Compute counterfactual effect
            for b in range(batch_size):
                target_idx = target_class[b].item()
                cf_effect = original_probs[b, target_idx] - cf_probs[b, target_idx]
                attribution_map[b, h_start:h_end, w_start:w_end] = cf_effect

        return attribution_map

    def _generate_normal_patch(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Generate a 'normal' version of a patch for counterfactual analysis.

        This could be implemented using various strategies:
        - Statistical normalization
        - Generative models
        - Domain knowledge
        """
        # Simple implementation: use patch mean and add controlled noise
        patch_mean = torch.mean(patch, dim=(2, 3), keepdim=True)
        noise = torch.randn_like(patch) * 0.1 * torch.std(patch, dim=(2, 3), keepdim=True)
        normal_patch = patch_mean + noise

        # Clamp to valid pixel range
        normal_patch = torch.clamp(normal_patch, 0, 1)

        return normal_patch

    def _gradcam_attribution(
        self,
        x: torch.Tensor,
        target_class: torch.Tensor
    ) -> torch.Tensor:
        """Compute GradCAM attribution for comparison."""
        if 'gradcam' not in self.attributors:
            return torch.zeros(x.shape[0], x.shape[2], x.shape[3], device=x.device)

        attributions = []
        for i, target in enumerate(target_class):
            attr = self.attributors['gradcam'].attribute(
                x[i:i+1], 
                target=target.item()
            )
            attributions.append(attr.squeeze(0).squeeze(0))

        return torch.stack(attributions)

    def _integrated_gradients_attribution(
        self,
        x: torch.Tensor,
        target_class: torch.Tensor
    ) -> torch.Tensor:
        """Compute Integrated Gradients attribution for comparison."""
        if 'integrated_gradients' not in self.attributors:
            return torch.zeros_like(x)

        # Create baseline (typically zeros or mean image)
        baseline = torch.zeros_like(x)

        attributions = []
        for i, target in enumerate(target_class):
            attr = self.attributors['integrated_gradients'].attribute(
                x[i:i+1],
                baseline[i:i+1],
                target=target.item(),
                n_steps=50
            )
            # Sum across channels for visualization
            attr_summed = torch.sum(attr.squeeze(0), dim=0)
            attributions.append(attr_summed)

        return torch.stack(attributions)

    def _aggregate_attributions(
        self,
        attributions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Aggregate multiple attribution maps into a single consensus map.
        """
        if not attributions:
            return torch.zeros(1)

        # Normalize each attribution map
        normalized_attrs = {}
        for method, attr_map in attributions.items():
            if method != 'aggregated':  # Avoid recursion
                # Normalize to [0, 1] range
                attr_min = torch.min(attr_map.view(attr_map.size(0), -1), dim=1)[0].unsqueeze(-1).unsqueeze(-1)
                attr_max = torch.max(attr_map.view(attr_map.size(0), -1), dim=1)[0].unsqueeze(-1).unsqueeze(-1)

                attr_range = attr_max - attr_min
                attr_range[attr_range == 0] = 1  # Avoid division by zero

                normalized_attr = (attr_map - attr_min) / attr_range
                normalized_attrs[method] = normalized_attr

        if not normalized_attrs:
            return torch.zeros(1)

        # Weighted average (prioritize causal methods)
        weights = {
            'intervention': 0.4,
            'counterfactual': 0.4,
            'gradcam': 0.1,
            'integrated_gradients': 0.1
        }

        aggregated = torch.zeros_like(list(normalized_attrs.values())[0])
        total_weight = 0.0

        for method, attr_map in normalized_attrs.items():
            weight = weights.get(method, 0.1)
            aggregated += weight * attr_map
            total_weight += weight

        aggregated = aggregated / total_weight if total_weight > 0 else aggregated

        return aggregated

    def generate_attribution_heatmap(
        self,
        attribution: Union[torch.Tensor, np.ndarray],
        original_image: Union[torch.Tensor, np.ndarray],
        colormap: str = 'jet',
        alpha: float = 0.6
    ) -> np.ndarray:
        """
        Generate visualization heatmap overlaid on original image.

        Args:
            attribution: Attribution map (torch.Tensor or np.ndarray)
            original_image: Original input image (torch.Tensor or np.ndarray)
            colormap: Colormap for heatmap
            alpha: Transparency for overlay

        Returns:
            Heatmap visualization as numpy array
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        # Convert to numpy and normalize
        if torch.is_tensor(attribution):
            attribution = attribution.detach().cpu().numpy()
        if torch.is_tensor(original_image):
            original_image = original_image.detach().cpu().numpy()

        # Normalize attribution
        attr_norm = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)

        # Apply colormap
        cmap = cm.get_cmap(colormap)
        heatmap = cmap(attr_norm)

        # Handle image format (assuming CHW format)
        if original_image.ndim == 3 and original_image.shape[0] in [1, 3]:
            if original_image.shape[0] == 1:
                original_image = np.repeat(original_image, 3, axis=0)
            original_image = np.transpose(original_image, (1, 2, 0))

        # Normalize image
        img_norm = (original_image - original_image.min()) / (original_image.max() - original_image.min() + 1e-8)

        # Overlay heatmap on image
        overlay = alpha * np.array(heatmap)[..., :3] + (1 - alpha) * img_norm
        overlay = np.clip(overlay, 0, 1)

        return overlay

    def _get_gradcam_layer(self, model: nn.Module) -> nn.Module:
        """
        Helper to get the last convolutional layer for GradCAM based on backbone architecture.
        """
        backbone = getattr(model, 'backbone', None)
        if backbone is None:
            raise AttributeError("Model does not have a 'backbone' attribute.")
        # For DenseNet
        if hasattr(backbone.backbone, 'features') and hasattr(backbone.backbone.features, 'denseblock4'):
            return backbone.backbone.features.denseblock4
        # For ResNet
        if hasattr(backbone.backbone, 'layer4'):
            return backbone.backbone.layer4
        # Fallback: try to get the last module
        layers = list(backbone.backbone.modules())
        for layer in reversed(layers):
            if isinstance(layer, nn.Conv2d):
                return layer
        raise AttributeError("Could not find a suitable layer for GradCAM.")


class AttributionQualityMetrics:
    """
    Metrics for evaluating the quality of causal attributions.
    """

    @staticmethod
    def sensitivity_correlation(
        attribution1: torch.Tensor,
        attribution2: torch.Tensor
    ) -> float:
        """Compute correlation between different attribution methods."""
        attr1_flat = attribution1.flatten()
        attr2_flat = attribution2.flatten()

        correlation = torch.corrcoef(torch.stack([attr1_flat, attr2_flat]))[0, 1]
        return correlation.item()

    @staticmethod
    def attribution_consistency(
        attributions: List[torch.Tensor],
        threshold: float = 0.1
    ) -> float:
        """Measure consistency of attributions across similar inputs."""
        if len(attributions) < 2:
            return 1.0

        correlations = []
        for i in range(len(attributions)):
            for j in range(i + 1, len(attributions)):
                corr = AttributionQualityMetrics.sensitivity_correlation(
                    attributions[i], attributions[j]
                )
                correlations.append(corr)

        return float(np.mean(correlations))

    @staticmethod
    def attribution_sparsity(attribution: torch.Tensor, percentile: float = 90) -> float:
        """Measure sparsity of attribution (how focused it is)."""
        # Use numpy percentile since torch.percentile does not exist
        attr_np = attribution.detach().cpu().numpy()
        threshold = np.percentile(attr_np.flatten(), percentile)
        sparse_attr = (attr_np > threshold).astype(float)
        sparsity = np.sum(sparse_attr) / sparse_attr.size
        return float(sparsity)
