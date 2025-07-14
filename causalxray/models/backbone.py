"""
CNN Backbone architectures for CausalXray framework.

This module implements DenseNet-121 and ResNet-50 backbones with modifications
for causal reasoning integration and medical imaging applications.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Optional, Tuple, Any, Union


class CausalBackbone(nn.Module):
    """
    CNN backbone with causal integration capabilities.
    Supports DenseNet-121 and ResNet-50 architectures with medical imaging optimizations.
    """

    def __init__(
        self,
        architecture: str = "densenet121",
        pretrained: bool = True,
        num_classes: int = 2,
        feature_dims: List[int] = [1024, 512, 256],
        dropout_rate: float = 0.3
    ):
        """
        Initialize the causal backbone network.

        Args:
            architecture: CNN architecture ("densenet121" or "resnet50")
            pretrained: Whether to use ImageNet pretrained weights
            num_classes: Number of output classes (2 for pneumonia binary classification)
            feature_dims: Dimensions for intermediate feature layers
            dropout_rate: Dropout probability for regularization
        """
        super(CausalBackbone, self).__init__()

        self.architecture = architecture
        self.num_classes = num_classes
        self.feature_dims = feature_dims

        # Initialize base CNN architecture
        if architecture == "densenet121":
            self.backbone = models.densenet121(pretrained=pretrained)
            self.feature_size = self.backbone.classifier.in_features
            # Replace classifier with a dummy linear layer (type-safe)
            self.backbone.classifier = nn.Linear(self.feature_size, self.feature_size)

        elif architecture == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feature_size = self.backbone.fc.in_features
            # Replace fc with a dummy linear layer (type-safe)
            self.backbone.fc = nn.Linear(self.feature_size, self.feature_size)

        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Causal-aware feature processing layers
        self.causal_features = self._build_causal_layers()

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dims[-1], feature_dims[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dims[-1] // 2, num_classes)
        )

        # Initialize weights for new layers
        self._initialize_weights()

    def _build_causal_layers(self) -> nn.ModuleList:
        """Build causal-aware feature processing layers."""
        layers = nn.ModuleList()

        input_dim = self.feature_size
        for output_dim in self.feature_dims:
            layers.append(nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ))
            input_dim = output_dim

        return layers

    def _initialize_weights(self):
        """Initialize weights for newly added layers."""
        for module in [self.causal_features, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass through the causal backbone.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Dictionary containing:
                - 'features': Raw backbone features
                - 'causal_features': List of processed causal features
                - 'logits': Classification logits
                - 'probabilities': Softmax probabilities
        """
        # Extract raw backbone features
        raw_features = self.backbone(x)

        # Process through causal layers
        causal_features = []
        current_features = raw_features

        for layer in self.causal_features:
            current_features = layer(current_features)
            causal_features.append(current_features)

        # Generate classification predictions
        logits = self.classifier(causal_features[-1])
        probabilities = torch.softmax(logits, dim=1)

        return {
            'features': raw_features,
            'causal_features': causal_features,
            'logits': logits,
            'probabilities': probabilities
        }

    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract feature maps for attribution analysis.
        """
        if self.architecture == "densenet121":
            return self._extract_densenet_features(x)
        elif self.architecture == "resnet50":
            return self._extract_resnet_features(x)
        else:
            return {}

    def _extract_densenet_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:  # type: ignore
        features = getattr(self.backbone, 'features', None)
        if not isinstance(features, nn.Module):
            raise TypeError("Expected self.backbone.features to be nn.Module, got {}".format(type(features)))
        conv0 = getattr(features, 'conv0')  # type: ignore
        norm0 = getattr(features, 'norm0')  # type: ignore
        relu0 = getattr(features, 'relu0')  # type: ignore
        pool0 = getattr(features, 'pool0')  # type: ignore
        denseblock1 = getattr(features, 'denseblock1')  # type: ignore
        transition1 = getattr(features, 'transition1')  # type: ignore
        denseblock2 = getattr(features, 'denseblock2')  # type: ignore
        transition2 = getattr(features, 'transition2')  # type: ignore
        denseblock3 = getattr(features, 'denseblock3')  # type: ignore
        transition3 = getattr(features, 'transition3')  # type: ignore
        denseblock4 = getattr(features, 'denseblock4')  # type: ignore

        feature_maps = {}
        x0 = conv0(x)  # type: ignore
        x1 = norm0(x0)  # type: ignore
        x2 = relu0(x1)  # type: ignore
        x3 = pool0(x2)  # type: ignore
        feature_maps['block1'] = denseblock1(x3)  # type: ignore
        x4 = transition1(feature_maps['block1'])  # type: ignore
        feature_maps['block2'] = denseblock2(x4)  # type: ignore
        x5 = transition2(feature_maps['block2'])  # type: ignore
        feature_maps['block3'] = denseblock3(x5)  # type: ignore
        x6 = transition3(feature_maps['block3'])  # type: ignore
        feature_maps['block4'] = denseblock4(x6)  # type: ignore
        return feature_maps

    def _extract_resnet_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:  # type: ignore
        backbone = self.backbone
        conv1 = getattr(backbone, 'conv1')  # type: ignore
        bn1 = getattr(backbone, 'bn1')  # type: ignore
        relu = getattr(backbone, 'relu')  # type: ignore
        maxpool = getattr(backbone, 'maxpool')  # type: ignore
        layer1 = getattr(backbone, 'layer1')  # type: ignore
        layer2 = getattr(backbone, 'layer2')  # type: ignore
        layer3 = getattr(backbone, 'layer3')  # type: ignore
        layer4 = getattr(backbone, 'layer4')  # type: ignore

        feature_maps = {}
        x0 = conv1(x)  # type: ignore
        x1 = bn1(x0)  # type: ignore
        x2 = relu(x1)  # type: ignore
        x3 = maxpool(x2)  # type: ignore
        feature_maps['layer1'] = layer1(x3)  # type: ignore
        feature_maps['layer2'] = layer2(feature_maps['layer1'])  # type: ignore
        feature_maps['layer3'] = layer3(feature_maps['layer2'])  # type: ignore
        feature_maps['layer4'] = layer4(feature_maps['layer3'])  # type: ignore
        return feature_maps

    def freeze_backbone(self):
        """Freeze backbone parameters for progressive training."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class MedicalCNNBackbone(CausalBackbone):
    """
    Specialized backbone for medical imaging with domain-specific optimizations.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Medical imaging specific modifications
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.medical_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dims[0],
            num_heads=8,
            dropout=0.1
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Enhanced forward pass with medical imaging optimizations."""
        # Get base features
        base_output = super().forward(x)

        # Add medical-specific processing
        # This could include attention mechanisms, domain adaptation, etc.

        return base_output
