"""
Loss functions for CausalXray framework including classification, disentanglement,
and causal reasoning objectives.

This module implements specialized loss functions that combine traditional classification
losses with causal disentanglement and attribution objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any


class CausalLoss(nn.Module):
    """
    Multi-objective loss function for CausalXray framework combining:
    - Classification loss (focal loss for imbalanced data)
    - Causal disentanglement loss
    - Domain adaptation loss
    - Attribution consistency loss
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize causal loss function.

        Args:
            config: Loss configuration dictionary
        """
        super(CausalLoss, self).__init__()

        self.config = config

        # Classification loss
        self.focal_loss = FocalLoss(
            alpha=config.get('focal_alpha', 1.0),
            gamma=config.get('focal_gamma', 2.0),
            reduction='mean'
        )

        # Disentanglement loss
        self.disentanglement_loss = DisentanglementLoss(
            config.get('disentanglement', {})
        )

        # Domain adaptation loss
        self.domain_loss = DomainAdversarialLoss()

        # Attribution consistency loss
        self.attribution_loss = AttributionConsistencyLoss()

        # Loss weights
        self.weights = config.get('weights', {
            'classification': 1.0,
            'disentanglement': 0.3,
            'domain': 0.1,
            'attribution': 0.2
        })

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        confounders: Optional[Dict[str, torch.Tensor]] = None,
        domain_labels: Optional[torch.Tensor] = None,
        attribution_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective causal loss.

        Args:
            outputs: Model outputs dictionary
            targets: True class labels
            confounders: True confounder values
            domain_labels: True domain labels
            attribution_targets: Target attributions for consistency

        Returns:
            Dictionary of loss components
        """
        losses = {}

        # Classification loss
        classification_loss = self.focal_loss(outputs['logits'], targets)
        losses['classification_loss'] = classification_loss

        # Causal disentanglement loss
        if 'causal_outputs' in outputs and confounders is not None:
            disentanglement_loss = self.disentanglement_loss(
                outputs['causal_outputs'], confounders
            )
            losses['disentanglement_loss'] = disentanglement_loss
        else:
            losses['disentanglement_loss'] = torch.tensor(0.0, device=targets.device)

        # Domain adaptation loss
        if 'domain_predictions' in outputs and domain_labels is not None:
            domain_loss = self.domain_loss(outputs['domain_predictions'], domain_labels)
            losses['domain_loss'] = domain_loss
        else:
            losses['domain_loss'] = torch.tensor(0.0, device=targets.device)

        # Attribution consistency loss
        if 'attributions' in outputs and attribution_targets is not None:
            attribution_loss = self.attribution_loss(
                outputs['attributions'], attribution_targets
            )
            losses['attribution_loss'] = attribution_loss
        else:
            losses['attribution_loss'] = torch.tensor(0.0, device=targets.device)

        # Compute total loss
        total_loss = (
            self.weights['classification'] * losses['classification_loss'] +
            self.weights['disentanglement'] * losses['disentanglement_loss'] +
            self.weights['domain'] * losses['domain_loss'] +
            self.weights['attribution'] * losses['attribution_loss']
        )

        losses['total_loss'] = total_loss

        return losses

    def update_weights(self, new_weights: Dict[str, float]):
        """Update loss component weights dynamically."""
        self.weights.update(new_weights)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in medical datasets.

    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    """

    def __init__(
        self,
        alpha: Union[float, List[float]] = 1.0,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for rare class (default: 1.0)
            gamma: Focusing parameter (default: 2.0)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predicted logits of shape (batch_size, num_classes)
            targets: True labels of shape (batch_size,)

        Returns:
            Focal loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        # Compute alpha term
        if isinstance(self.alpha, torch.Tensor):
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha[targets]
        else:
            alpha_t = torch.tensor(self.alpha, device=targets.device) if isinstance(self.alpha, (float, int)) else torch.tensor(self.alpha, device=targets.device)[targets]
        # Compute focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DisentanglementLoss(nn.Module):
    """
    Loss function for causal disentanglement including:
    - Confounder prediction losses
    - Independence constraints
    - Variational regularization
    """

    def __init__(self, config: Dict[str, Any]):
        super(DisentanglementLoss, self).__init__()

        self.config = config
        self.beta_vae = config.get('beta_vae', 1.0)
        self.independence_weight = config.get('independence_weight', 0.1)

    def forward(
        self,
        causal_outputs: Dict[str, torch.Tensor],
        true_confounders: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute disentanglement loss.

        Args:
            causal_outputs: Outputs from causal heads
            true_confounders: True confounder values

        Returns:
            Total disentanglement loss
        """
        total_loss = torch.tensor(0.0, device=next(iter(true_confounders.values())).device if true_confounders else 'cpu')
        # Confounder prediction losses
        if 'confounders' in causal_outputs:
            confounders = causal_outputs['confounders']
            if not isinstance(confounders, dict):
                raise TypeError("causal_outputs['confounders'] must be a dict, got {}".format(type(confounders)))
            for confounder_name, predictions in confounders.items():
                if confounder_name in true_confounders:
                    true_values = true_confounders[confounder_name]
                    if predictions.size(1) == 1:  # Regression
                        confounder_loss = F.mse_loss(predictions.squeeze(), true_values.float())
                    else:  # Classification
                        confounder_loss = F.cross_entropy(predictions, true_values.long())
                    total_loss = total_loss + confounder_loss
        # Variational loss (KL divergence)
        if 'variational' in causal_outputs:
            variational = causal_outputs['variational']
            if not isinstance(variational, dict):
                raise TypeError("causal_outputs['variational'] must be a dict, got {}".format(type(variational)))
            mu = variational['mu']  # type: ignore
            log_var = variational['log_var']  # type: ignore
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            kl_loss = kl_loss / mu.size(0)  # Normalize by batch size
            total_loss = total_loss + self.beta_vae * kl_loss
        # Independence constraint
        if 'causal_features' in causal_outputs:
            independence_loss = self._compute_independence_loss(
                causal_outputs['causal_features']
            )
            total_loss = total_loss + self.independence_weight * independence_loss
        return total_loss

    def _compute_independence_loss(self, features: torch.Tensor) -> torch.Tensor:
        """Compute independence loss to encourage decorrelated features."""
        # Compute correlation matrix
        features_normalized = features - features.mean(dim=0, keepdim=True)
        correlation_matrix = torch.mm(features_normalized.t(), features_normalized)
        correlation_matrix = correlation_matrix / features.size(0)

        # Penalize off-diagonal elements
        identity = torch.eye(correlation_matrix.size(0), device=features.device)
        independence_loss = torch.sum(torch.abs(correlation_matrix - identity))

        return independence_loss


class DomainAdversarialLoss(nn.Module):
    """
    Domain adversarial loss for domain adaptation.
    """

    def __init__(self):
        super(DomainAdversarialLoss, self).__init__()

    def forward(
        self,
        domain_predictions: torch.Tensor,
        domain_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute domain adversarial loss.

        Args:
            domain_predictions: Predicted domain labels
            domain_labels: True domain labels

        Returns:
            Domain adversarial loss
        """
        return F.cross_entropy(domain_predictions, domain_labels)


class AttributionConsistencyLoss(nn.Module):
    """
    Loss function to encourage consistent causal attributions.
    """

    def __init__(self):
        super(AttributionConsistencyLoss, self).__init__()

    def forward(
        self,
        attributions: Dict[str, torch.Tensor],
        target_attributions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attribution consistency loss.

        Args:
            attributions: Model attributions
            target_attributions: Target attribution maps

        Returns:
            Attribution consistency loss
        """
        if 'intervention' in attributions:
            intervention_attr = attributions['intervention']
            consistency_loss = F.mse_loss(intervention_attr, target_attributions)
            return consistency_loss

        return torch.tensor(0.0, device=target_attributions.device)

    def inter_method_consistency(self, attributions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute inter-method attribution consistency loss.
        This loss encourages attributions from different methods to be similar (high cosine similarity).

        Args:
            attributions: Dictionary of attribution maps from different methods (method_name -> tensor)

        Returns:
            Consistency loss (average 1 - cosine similarity across all method pairs)
        """
        if len(attributions) < 2:
            return torch.tensor(0.0, device=next(iter(attributions.values())).device)
        consistency_loss = torch.tensor(0.0, device=next(iter(attributions.values())).device)
        count = 0
        methods = list(attributions.keys())
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                attr1 = attributions[methods[i]]
                attr2 = attributions[methods[j]]
                attr1_norm = F.normalize(attr1.flatten(1), p=2, dim=1)
                attr2_norm = F.normalize(attr2.flatten(1), p=2, dim=1)
                sim = F.cosine_similarity(attr1_norm, attr2_norm, dim=1)
                consistency_loss = consistency_loss.add((1 - sim).mean())
                count += 1
        if count == 0:
            return torch.tensor(0.0, device=next(iter(attributions.values())).device)
        return consistency_loss / count


class ContrastiveCausalLoss(nn.Module):
    """
    Contrastive loss for causal representation learning.
    """

    def __init__(self, temperature: float = 0.1, margin: float = 1.0):
        """
        Initialize contrastive causal loss.

        Args:
            temperature: Temperature parameter for contrastive learning
            margin: Margin for contrastive loss
        """
        super(ContrastiveCausalLoss, self).__init__()

        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        causal_features: torch.Tensor,
        labels: torch.Tensor,
        confounders: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute contrastive causal loss.

        Args:
            causal_features: Causal feature representations
            labels: Class labels
            confounders: Confounder information

        Returns:
            Contrastive loss value
        """
        batch_size = causal_features.size(0)

        # Normalize features
        causal_features = F.normalize(causal_features, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.mm(causal_features, causal_features.t()) / self.temperature

        # Create positive and negative pairs based on labels and confounders
        positive_mask = self._create_positive_mask(labels, confounders)
        negative_mask = ~positive_mask

        # Compute contrastive loss
        # Positive pairs should have high similarity
        positive_loss = -torch.log(torch.exp(similarity_matrix) * positive_mask.float()).sum()

        # Negative pairs should have low similarity
        negative_loss = torch.clamp(
            self.margin - similarity_matrix * negative_mask.float(), min=0
        ).sum()

        total_loss = (positive_loss + negative_loss) / (batch_size * batch_size)

        return total_loss

    def _create_positive_mask(
        self,
        labels: torch.Tensor,
        confounders: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Create mask for positive pairs (same class, similar confounders)."""
        batch_size = labels.size(0)

        # Same class mask
        label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)

        # Similar confounders mask (simplified)
        confounder_mask = torch.ones(batch_size, batch_size, dtype=torch.bool, device=labels.device)

        for confounder_name, confounder_values in confounders.items():
            if confounder_values.dtype in [torch.float32, torch.float64]:
                # Continuous confounders - use threshold
                diff = torch.abs(confounder_values.unsqueeze(0) - confounder_values.unsqueeze(1))
                threshold = torch.std(confounder_values) * 0.5
                confounder_mask &= (diff < threshold)
            else:
                # Categorical confounders - exact match
                confounder_mask &= (confounder_values.unsqueeze(0) == confounder_values.unsqueeze(1))

        # Combine masks
        positive_mask = label_mask & confounder_mask

        # Remove diagonal (self-similarity)
        identity = torch.eye(batch_size, dtype=torch.bool, device=labels.device)
        positive_mask = positive_mask & ~identity

        return positive_mask


class UncertaintyLoss(nn.Module):
    """
    Loss function for uncertainty quantification in causal models.
    """

    def __init__(self, uncertainty_weight: float = 0.1):
        super(UncertaintyLoss, self).__init__()
        self.uncertainty_weight = uncertainty_weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainty: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute uncertainty-aware loss.

        Args:
            predictions: Model predictions
            targets: True targets
            uncertainty: Predicted uncertainty (log variance)

        Returns:
            Uncertainty-aware loss
        """
        # Gaussian likelihood with learned uncertainty
        mse_loss = F.mse_loss(predictions, targets, reduction='none')

        # Uncertainty-weighted loss
        precision = torch.exp(-uncertainty)
        uncertainty_loss = precision * mse_loss + uncertainty

        # Regularization term to prevent uncertainty collapse
        uncertainty_reg = self.uncertainty_weight * torch.mean(uncertainty)

        return torch.mean(uncertainty_loss) + uncertainty_reg


class CausalRegularization(nn.Module):
    """
    Regularization terms for causal models to encourage proper causal structure.
    """

    def __init__(self, config: Dict[str, Any]):
        super(CausalRegularization, self).__init__()

        self.sparsity_weight = config.get('sparsity_weight', 0.01)
        self.smoothness_weight = config.get('smoothness_weight', 0.01)

    def forward(self, causal_graph_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute causal regularization loss.

        Args:
            causal_graph_weights: Weights representing causal graph structure

        Returns:
            Regularization loss
        """
        # Sparsity regularization (L1 norm)
        sparsity_loss = self.sparsity_weight * torch.sum(torch.abs(causal_graph_weights))

        # Smoothness regularization (encourages local connectivity)
        smoothness_loss = self.smoothness_weight * torch.sum(
            torch.abs(causal_graph_weights[1:] - causal_graph_weights[:-1])
        )

        return sparsity_loss + smoothness_loss
