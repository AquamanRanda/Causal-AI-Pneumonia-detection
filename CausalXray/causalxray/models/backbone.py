"""
CNN Backbone architectures for CausalXray framework.

This module implements DenseNet-121 and ResNet-50 backbones with modifications
for causal reasoning integration and medical imaging applications.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Optional, Tuple


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
            # Remove original classifier
            self.backbone.classifier = nn.Identity()

        elif architecture == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feature_size = self.backbone.fc.in_features
            # Remove original classifier
            self.backbone.fc = nn.Identity()

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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
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

        Args:
            x: Input tensor

        Returns:
            Dictionary of feature maps at different network depths
        """
        feature_maps = {}

        if self.architecture == "densenet121":
            # Extract features at different densenet blocks
            x = self.backbone.features.conv0(x)
            x = self.backbone.features.norm0(x)
            x = self.backbone.features.relu0(x)
            x = self.backbone.features.pool0(x)

            feature_maps['block1'] = self.backbone.features.denseblock1(x)
            x = self.backbone.features.transition1(feature_maps['block1'])

            feature_maps['block2'] = self.backbone.features.denseblock2(x)
            x = self.backbone.features.transition2(feature_maps['block2'])

            feature_maps['block3'] = self.backbone.features.denseblock3(x)
            x = self.backbone.features.transition3(feature_maps['block3'])

            feature_maps['block4'] = self.backbone.features.denseblock4(x)

        elif self.architecture == "resnet50":
            # Extract features at different ResNet stages
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            feature_maps['layer1'] = self.backbone.layer1(x)
            feature_maps['layer2'] = self.backbone.layer2(feature_maps['layer1'])
            feature_maps['layer3'] = self.backbone.layer3(feature_maps['layer2'])
            feature_maps['layer4'] = self.backbone.layer4(feature_maps['layer3'])

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
