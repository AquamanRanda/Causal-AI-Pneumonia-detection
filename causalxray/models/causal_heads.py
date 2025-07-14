"""
Causal disentanglement modules for CausalXray framework.

This module implements specialized neural network heads that predict confounding variables
and separate causally relevant features from spurious correlations according to Pearl's
causal hierarchy framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, cast
import numpy as np


class CausalHeads(nn.Module):
    """
    Multi-head causal disentanglement module that predicts confounding variables
    and separates causal from spurious correlations.
    """

    def __init__(
        self,
        input_dim: int,
        confounders: Dict[str, int],
        hidden_dims: List[int] = [512, 256],
        dropout_rate: float = 0.3,
        use_variational: bool = True
    ):
        """
        Initialize causal disentanglement heads.

        Args:
            input_dim: Dimension of input features from backbone
            confounders: Dictionary mapping confounder names to their dimensionality
                        e.g., {"age": 1, "sex": 2, "scanner_type": 5}
            hidden_dims: Hidden layer dimensions for each head
            dropout_rate: Dropout probability
            use_variational: Whether to use variational inference for latent variables
        """
        super(CausalHeads, self).__init__()

        self.input_dim = input_dim
        self.confounders = confounders
        self.hidden_dims = hidden_dims
        self.use_variational = use_variational

        # Build confounder prediction heads
        self.confounder_heads = nn.ModuleDict()
        for confounder_name, confounder_dim in confounders.items():
            self.confounder_heads[confounder_name] = self._build_prediction_head(
                input_dim, confounder_dim, hidden_dims, dropout_rate
            )

        # Causal feature extractor
        self.causal_extractor = self._build_causal_extractor(
            input_dim, hidden_dims, dropout_rate
        )

        # Variational components for uncertainty quantification
        self.variational_encoder: Optional[nn.ModuleDict] = None
        if use_variational:
            self.variational_encoder = cast(nn.ModuleDict, self._build_variational_encoder(
                input_dim, hidden_dims[-1]
            ))

    def _build_prediction_head(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int],
        dropout_rate: float
    ) -> nn.Module:
        """Build a prediction head for a specific confounder."""
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        return nn.Sequential(*layers)

    def _build_causal_extractor(
        self, 
        input_dim: int, 
        hidden_dims: List[int],
        dropout_rate: float
    ) -> nn.Module:
        """Build causal feature extraction network."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dims[1], hidden_dims[1]),
            nn.Tanh()  # Bounded output for causal features
        )

    def _build_variational_encoder(
        self, 
        input_dim: int, 
        latent_dim: int
    ) -> nn.Module:
        """Build variational encoder for uncertainty quantification."""
        return nn.ModuleDict({
            'mu': nn.Sequential(
                nn.Linear(input_dim, latent_dim),
                nn.ReLU(inplace=True),
                nn.Linear(latent_dim, latent_dim)
            ),
            'log_var': nn.Sequential(
                nn.Linear(input_dim, latent_dim),
                nn.ReLU(inplace=True),
                nn.Linear(latent_dim, latent_dim)
            )
        })

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through causal disentanglement heads.

        Args:
            features: Input features from backbone network

        Returns:
            Dictionary containing:
                - confounder predictions for each confounder
                - causal features
                - variational parameters (if enabled)
        """
        batch_size = features.size(0)
        outputs = {}

        # Predict confounders
        confounder_predictions: Dict[str, torch.Tensor] = {}
        for confounder_name, head in self.confounder_heads.items():
            confounder_predictions[confounder_name] = head(features)

        outputs['confounders'] = confounder_predictions

        # Extract causal features
        causal_features = self.causal_extractor(features)
        outputs['causal_features'] = causal_features

        # Variational inference
        if self.use_variational:
            assert self.variational_encoder is not None, "Variational encoder is not initialized."
            variational_dict = cast(Dict[str, torch.Tensor], outputs['variational'])
            mu = variational_dict['mu']  # type: ignore
            log_var = variational_dict['log_var']  # type: ignore

            # Reparameterization trick
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std

            outputs['variational'] = {
                'mu': mu,
                'log_var': log_var,
                'z': z
            }

        return outputs

    def compute_disentanglement_loss(
        self, 
        outputs: Dict[str, torch.Tensor],
        true_confounders: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:  # type: ignore
        """
        Compute disentanglement loss components.

        Args:
            outputs: Forward pass outputs
            true_confounders: Ground truth confounder values

        Returns:
            Dictionary of loss components
        """
        losses = {}
        total_loss = 0.0

        # Confounder prediction losses
        if not isinstance(outputs['confounders'], dict):
            raise TypeError("outputs['confounders'] must be a dict, got {}".format(type(outputs['confounders'])))
        for confounder_name, predictions in outputs['confounders'].items():
            if confounder_name in true_confounders:
                true_values = true_confounders[confounder_name]

                if predictions.size(1) == 1:  # Regression
                    confounder_loss = F.mse_loss(predictions.squeeze(), true_values)
                else:  # Classification
                    confounder_loss = F.cross_entropy(predictions, true_values.long())

                losses[f'{confounder_name}_loss'] = confounder_loss
                total_loss += confounder_loss

        # Variational loss (KL divergence)
        if self.use_variational and 'variational' in outputs:
            if not isinstance(outputs['variational'], dict):
                raise TypeError("outputs['variational'] must be a dict, got {}".format(type(outputs['variational'])))
            mu = outputs['variational']['mu']  # type: ignore
            log_var = outputs['variational']['log_var']  # type: ignore

            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            kl_loss = kl_loss / mu.size(0)  # Normalize by batch size

            losses['kl_loss'] = kl_loss
            total_loss += 0.1 * kl_loss  # Scale KL loss

        # Independence loss (encourage decorrelation between causal features)
        if 'causal_features' in outputs:
            causal_features = outputs['causal_features']
            correlation_matrix = torch.corrcoef(causal_features.T)

            # Penalize off-diagonal correlations
            identity = torch.eye(correlation_matrix.size(0), device=correlation_matrix.device)
            independence_loss = torch.mean(torch.abs(correlation_matrix - identity))

            losses['independence_loss'] = independence_loss
            total_loss += 0.05 * independence_loss

        losses['total_disentanglement_loss'] = total_loss
        return losses


class DomainAdaptationHead(nn.Module):
    """
    Specialized head for domain adaptation through adversarial training.
    """

    def __init__(
        self,
        input_dim: int,
        num_domains: int,
        hidden_dim: int = 256,
        dropout_rate: float = 0.3
    ):
        super().__init__()

        self.domain_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_domains)
        )

        # Gradient reversal layer
        self.gradient_reversal = GradientReversalLayer()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass with gradient reversal for domain adaptation."""
        reversed_features = self.gradient_reversal(features)
        return self.domain_classifier(reversed_features)


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient reversal layer for domain adaptation.
    """

    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class CausalGraphModule(nn.Module):
    """
    Module that explicitly models causal graph structure and interventions.
    """

    def __init__(
        self,
        variables: List[str],
        causal_graph: Dict[str, List[str]],
        variable_dims: Dict[str, int]
    ):
        """
        Initialize causal graph module.

        Args:
            variables: List of variable names
            causal_graph: Adjacency list representation of causal graph
            variable_dims: Dimensionality of each variable
        """
        super().__init__()

        self.variables = variables
        self.causal_graph = causal_graph
        self.variable_dims = variable_dims

        # Build structural equation networks
        self.structural_equations = nn.ModuleDict()
        for var in variables:
            parents = causal_graph.get(var, [])
            parent_dim = sum(variable_dims[p] for p in parents)

            if parent_dim > 0:
                self.structural_equations[var] = nn.Sequential(
                    nn.Linear(parent_dim, variable_dims[var] * 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(variable_dims[var] * 2, variable_dims[var])
                )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through structural equations.

        Args:
            inputs: Dictionary of input variables

        Returns:
            Dictionary of computed variables
        """
        outputs = inputs.copy()

        # Compute variables in topological order
        for var in self.variables:
            if var not in outputs:
                parents = self.causal_graph.get(var, [])
                if parents:
                    parent_values = [outputs[p] for p in parents]
                    parent_concat = torch.cat(parent_values, dim=1)
                    outputs[var] = self.structural_equations[var](parent_concat)

        return outputs

    def intervention(
        self,
        inputs: Dict[str, torch.Tensor],
        interventions: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Perform causal intervention (do-calculus).

        Args:
            inputs: Original inputs
            interventions: Variables to intervene on and their values

        Returns:
            Outputs under intervention
        """
        # Override intervened variables
        modified_inputs = inputs.copy()
        modified_inputs.update(interventions)

        # Recompute downstream variables
        return self.forward(modified_inputs)
