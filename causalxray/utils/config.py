"""
Visualization utilities for CausalXray framework including attribution heatmaps,
training curves, and model performance visualizations.

This module provides comprehensive visualization capabilities for understanding
model behavior, causal attributions, and cross-domain performance.
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import torch
from PIL import Image
import cv2
from typing import Dict, List, Optional, Tuple, Union, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd


class AttributionVisualizer:
    """
    Visualizer for causal attribution maps and explanations.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize attribution visualizer.

        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')

    def visualize_attribution_comparison(
        self,
        image: np.ndarray,
        attributions: Dict[str, np.ndarray],
        prediction: Optional[Dict[str, float]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comparison visualization of different attribution methods.

        Args:
            image: Original input image
            attributions: Dictionary mapping method names to attribution maps
            prediction: Model prediction information
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        n_methods = len(attributions)
        fig, axes = plt.subplots(2, n_methods + 1, figsize=(4 * (n_methods + 1), 8))

        if n_methods == 1:
            axes = axes.reshape(2, -1)

        # Original image
        axes[0, 0].imshow(self._prepare_image_for_display(image), cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # Add prediction information if available
        if prediction:
            pred_text = f"Prediction: {prediction.get('class', 'Unknown')}\n"
            pred_text += f"Confidence: {prediction.get('confidence', 0):.3f}"
            axes[1, 0].text(0.1, 0.5, pred_text, fontsize=12, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 0].axis('off')

        # Attribution maps
        for idx, (method_name, attribution) in enumerate(attributions.items()):
            col_idx = idx + 1

            # Original attribution
            im1 = axes[0, col_idx].imshow(attribution, cmap='RdBu_r', 
                                         vmin=-np.max(np.abs(attribution)), 
                                         vmax=np.max(np.abs(attribution)))
            axes[0, col_idx].set_title(f'{method_name}\nAttribution')
            axes[0, col_idx].axis('off')
            plt.colorbar(im1, ax=axes[0, col_idx], shrink=0.8)

            # Overlay on original image
            overlay = self._create_attribution_overlay(image, attribution)
            axes[1, col_idx].imshow(overlay)
            axes[1, col_idx].set_title(f'{method_name}\nOverlay')
            axes[1, col_idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def visualize_intervention_effects(
        self,
        original_image: np.ndarray,
        intervention_masks: List[np.ndarray],
        predictions: List[Dict[str, float]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize the effects of causal interventions.

        Args:
            original_image: Original input image
            intervention_masks: List of intervention masks applied
            predictions: List of prediction results for each intervention
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        n_interventions = len(intervention_masks)
        fig, axes = plt.subplots(2, n_interventions + 1, 
                                figsize=(4 * (n_interventions + 1), 8))

        if n_interventions == 0:
            axes = axes.reshape(2, -1)

        # Original image and prediction
        axes[0, 0].imshow(self._prepare_image_for_display(original_image), cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')

        if predictions:
            pred_text = f"Pneumonia: {predictions[0].get('pneumonia_prob', 0):.3f}"
            axes[1, 0].text(0.1, 0.5, pred_text, fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        axes[1, 0].axis('off')

        # Interventions
        for idx, (mask, pred) in enumerate(zip(intervention_masks, predictions[1:])):
            col_idx = idx + 1

            # Intervention mask
            axes[0, col_idx].imshow(mask, cmap='Reds', alpha=0.7)
            axes[0, col_idx].imshow(self._prepare_image_for_display(original_image), 
                                  cmap='gray', alpha=0.3)
            axes[0, col_idx].set_title(f'Intervention {idx + 1}')
            axes[0, col_idx].axis('off')

            # Prediction change
            pred_change = pred.get('pneumonia_prob', 0) - predictions[0].get('pneumonia_prob', 0)
            pred_text = f"Pneumonia: {pred.get('pneumonia_prob', 0):.3f}\n"
            pred_text += f"Change: {pred_change:+.3f}"

            color = "lightcoral" if pred_change > 0 else "lightblue"
            axes[1, col_idx].text(0.1, 0.5, pred_text, fontsize=12,
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
            axes[1, col_idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_attribution_heatmap(
        self,
        attribution: np.ndarray,
        image: Optional[np.ndarray] = None,
        colormap: str = 'RdBu_r',
        alpha: float = 0.6,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create detailed attribution heatmap.

        Args:
            attribution: Attribution map
            image: Original image for overlay
            colormap: Matplotlib colormap name
            alpha: Transparency for overlay
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2 if image is not None else 1, figsize=self.figsize)

        if image is not None and not isinstance(axes, np.ndarray):
            axes = [axes]
        elif image is None:
            axes = [axes]

        # Attribution heatmap
        vmax = np.max(np.abs(attribution))
        im = axes[0].imshow(attribution, cmap=colormap, vmin=-vmax, vmax=vmax)
        axes[0].set_title('Causal Attribution')
        axes[0].axis('off')

        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[0], shrink=0.8)
        cbar.set_label('Attribution Intensity')

        # Overlay if image provided
        if image is not None:
            overlay = self._create_attribution_overlay(image, attribution, alpha=alpha)
            axes[1].imshow(overlay)
            axes[1].set_title('Attribution Overlay')
            axes[1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def _prepare_image_for_display(self, image: np.ndarray) -> np.ndarray:
        """Prepare image for matplotlib display."""
        if len(image.shape) == 3:
            if image.shape[0] == 3:  # CHW format
                image = image.transpose(1, 2, 0)
            elif image.shape[2] == 1:  # HWC with single channel
                image = image.squeeze(2)

        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        return image

    def _create_attribution_overlay(
        self,
        image: np.ndarray,
        attribution: np.ndarray,
        alpha: float = 0.6,
        colormap: str = 'RdBu_r'
    ) -> np.ndarray:
        """Create overlay of attribution on original image."""
        # Prepare image
        img_display = self._prepare_image_for_display(image)
        if len(img_display.shape) == 2:
            img_display = np.stack([img_display] * 3, axis=2)

        # Normalize attribution
        attr_norm = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)

        # Apply colormap
        cmap = cm.get_cmap(colormap)
        attr_colored = cmap(attr_norm)[..., :3]  # Remove alpha channel

        # Create overlay
        overlay = alpha * attr_colored + (1 - alpha) * img_display
        overlay = np.clip(overlay, 0, 1)

        return overlay


class ResultsVisualizer:
    """
    Visualizer for training results, metrics, and model performance.
    """

    def __init__(self):
        """Initialize results visualizer."""
        plt.style.use('seaborn-v0_8')
        self.colors = plt.cm.Set1(np.linspace(0, 1, 10))

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        metrics: List[str] = ['loss', 'accuracy', 'auc'],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot training and validation curves.

        Args:
            history: Training history dictionary
            metrics: List of metrics to plot
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            train_key = f'train_{metric}'
            val_key = f'val_{metric}'

            if train_key in history:
                epochs = range(1, len(history[train_key]) + 1)
                axes[idx].plot(epochs, history[train_key], 'b-', label=f'Train {metric}')

            if val_key in history:
                epochs = range(1, len(history[val_key]) + 1)
                axes[idx].plot(epochs, history[val_key], 'r-', label=f'Val {metric}')

            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric.capitalize())
            axes[idx].set_title(f'{metric.capitalize()} vs Epoch')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_cross_domain_performance(
        self,
        domain_results: Dict[str, Dict[str, float]],
        metrics: List[str] = ['accuracy', 'auc', 'f1'],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot cross-domain performance comparison.

        Args:
            domain_results: Dictionary mapping domain names to metrics
            metrics: List of metrics to visualize
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        # Prepare data
        domains = list(domain_results.keys())
        data = []

        for domain in domains:
            for metric in metrics:
                if metric in domain_results[domain]:
                    data.append({
                        'Domain': domain,
                        'Metric': metric,
                        'Value': domain_results[domain][metric]
                    })

        df = pd.DataFrame(data)

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Bar plot
        sns.barplot(data=df, x='Domain', y='Value', hue='Metric', ax=axes[0])
        axes[0].set_title('Cross-Domain Performance')
        axes[0].set_ylabel('Score')
        axes[0].tick_params(axis='x', rotation=45)

        # Heatmap
        pivot_df = df.pivot(index='Metric', columns='Domain', values='Value')
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1])
        axes[1].set_title('Performance Heatmap')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_confusion_matrices(
        self,
        confusion_matrices: Dict[str, np.ndarray],
        class_names: List[str] = ['Normal', 'Pneumonia'],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot confusion matrices for different domains or methods.

        Args:
            confusion_matrices: Dictionary mapping names to confusion matrices
            class_names: List of class names
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        n_matrices = len(confusion_matrices)
        cols = min(3, n_matrices)
        rows = (n_matrices + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

        if n_matrices == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)

        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for idx, (name, cm) in enumerate(confusion_matrices.items()):
            if idx < len(axes_flat):
                # Normalize confusion matrix
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

                sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                           xticklabels=class_names, yticklabels=class_names,
                           ax=axes_flat[idx])
                axes_flat[idx].set_title(f'{name}')
                axes_flat[idx].set_ylabel('True Label')
                axes_flat[idx].set_xlabel('Predicted Label')

        # Hide empty subplots
        for idx in range(n_matrices, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_roc_curves(
        self,
        roc_data: Dict[str, Dict[str, np.ndarray]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot ROC curves for multiple models or domains.

        Args:
            roc_data: Dictionary mapping names to ROC curve data
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        for idx, (name, data) in enumerate(roc_data.items()):
            fpr = data['fpr']
            tpr = data['tpr']
            auc_score = np.trapz(tpr, fpr)

            ax.plot(fpr, tpr, color=self.colors[idx % len(self.colors)],
                   label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)

        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.8, linewidth=1)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_interactive_dashboard(
        self,
        training_history: Dict[str, List[float]],
        domain_results: Dict[str, Dict[str, float]],
        attribution_data: Optional[Dict] = None
    ) -> go.Figure:
        """
        Create interactive dashboard using Plotly.

        Args:
            training_history: Training history data
            domain_results: Cross-domain results
            attribution_data: Attribution visualization data

        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Validation Metrics', 
                          'Cross-Domain Performance', 'Model Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Training loss
        if 'train_total_loss' in training_history:
            epochs = list(range(1, len(training_history['train_total_loss']) + 1))
            fig.add_trace(
                go.Scatter(x=epochs, y=training_history['train_total_loss'],
                          mode='lines', name='Train Loss'),
                row=1, col=1
            )

        # Validation metrics
        for metric in ['val_accuracy', 'val_auc', 'val_f1']:
            if metric in training_history:
                epochs = list(range(1, len(training_history[metric]) + 1))
                fig.add_trace(
                    go.Scatter(x=epochs, y=training_history[metric],
                              mode='lines', name=metric.replace('val_', '').title()),
                    row=1, col=2
                )

        # Cross-domain performance
        domains = list(domain_results.keys())
        metrics = ['accuracy', 'auc', 'f1']

        for metric in metrics:
            values = [domain_results[domain].get(metric, 0) for domain in domains]
            fig.add_trace(
                go.Bar(x=domains, y=values, name=metric.title()),
                row=2, col=1
            )

        # Model comparison (placeholder)
        fig.add_trace(
            go.Scatter(x=[1, 2, 3], y=[0.85, 0.89, 0.92],
                      mode='markers+lines', name='CausalXray'),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            height=800,
            title_text="CausalXray Performance Dashboard",
            showlegend=True
        )

        return fig
