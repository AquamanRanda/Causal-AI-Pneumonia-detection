"""
Evaluation metrics for CausalXray framework including classification metrics,
causal reasoning evaluation, and domain generalization assessment.

This module implements comprehensive metrics for evaluating both traditional
classification performance and novel causal reasoning capabilities.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
import scipy.stats as stats
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings


class CausalMetrics:
    """
    Comprehensive metrics for evaluating CausalXray model performance.
    """

    def __init__(self, num_classes: int = 2):
        """
        Initialize metrics calculator.

        Args:
            num_classes: Number of classes for classification
        """
        self.num_classes = num_classes

    def compute_batch_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Compute metrics for a single batch.

        Args:
            predictions: Model predictions (probabilities)
            targets: True labels
            loss_dict: Dictionary of loss components

        Returns:
            Dictionary of batch metrics
        """
        # Convert to numpy
        if torch.is_tensor(predictions):
            predictions = predictions.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()

        # Get predicted classes
        predicted_classes = np.argmax(predictions, axis=1)

        # Basic classification metrics
        metrics = {
            'accuracy': accuracy_score(targets, predicted_classes),
            'precision': precision_score(targets, predicted_classes, average='weighted', zero_division=0),
            'recall': recall_score(targets, predicted_classes, average='weighted', zero_division=0),
            'f1': f1_score(targets, predicted_classes, average='weighted', zero_division=0)
        }

        # Add loss components if provided
        if loss_dict:
            for key, value in loss_dict.items():
                if torch.is_tensor(value):
                    metrics[key] = value.item()
                else:
                    metrics[key] = float(value)

        return metrics

    def compute_epoch_metrics(
        self,
        all_predictions: np.ndarray,
        all_targets: np.ndarray,
        return_curves: bool = False
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Compute comprehensive metrics for an entire epoch.

        Args:
            all_predictions: All predictions from epoch (probabilities)
            all_targets: All true labels from epoch
            return_curves: Whether to return ROC and PR curves

        Returns:
            Dictionary of epoch metrics
        """
        # Get predicted classes
        predicted_classes = np.argmax(all_predictions, axis=1)

        # Basic classification metrics
        metrics = {
            'accuracy': accuracy_score(all_targets, predicted_classes),
            'precision': precision_score(all_targets, predicted_classes, average='weighted', zero_division=0),
            'recall': recall_score(all_targets, predicted_classes, average='weighted', zero_division=0),
            'f1': f1_score(all_targets, predicted_classes, average='weighted', zero_division=0),
            'sensitivity': recall_score(all_targets, predicted_classes, pos_label=1, zero_division=0),
            'specificity': self._compute_specificity(all_targets, predicted_classes)
        }

        # AUC metrics
        if self.num_classes == 2:
            # Binary classification
            try:
                metrics['auc'] = roc_auc_score(all_targets, all_predictions[:, 1])
                metrics['auc_pr'] = self._compute_auc_pr(all_targets, all_predictions[:, 1])
            except ValueError as e:
                warnings.warn(f"Could not compute AUC: {e}")
                metrics['auc'] = 0.0
                metrics['auc_pr'] = 0.0
        else:
            # Multi-class classification
            try:
                metrics['auc'] = roc_auc_score(all_targets, all_predictions, multi_class='ovr')
            except ValueError as e:
                warnings.warn(f"Could not compute AUC: {e}")
                metrics['auc'] = 0.0

        # Calibration metrics
        calibration_metrics = self._compute_calibration_metrics(all_predictions, all_targets)
        metrics.update(calibration_metrics)

        # Class-specific metrics
        class_metrics = self._compute_class_specific_metrics(all_targets, predicted_classes)
        metrics.update(class_metrics)

        # Confusion matrix
        cm = confusion_matrix(all_targets, predicted_classes)
        metrics['confusion_matrix'] = cm

        # Return curves if requested
        if return_curves:
            curve_data = self._compute_curves(all_targets, all_predictions)
            metrics.update(curve_data)

        return metrics

    def _compute_specificity(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """Compute specificity (true negative rate)."""
        cm = confusion_matrix(targets, predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            # Multi-class case - average specificity
            specificities = []
            for class_idx in range(cm.shape[0]):
                tp = cm[class_idx, class_idx]
                fn = np.sum(cm[class_idx, :]) - tp
                fp = np.sum(cm[:, class_idx]) - tp
                tn = np.sum(cm) - tp - fn - fp
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                specificities.append(spec)
            return np.mean(specificities)

    def _compute_auc_pr(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """Compute area under precision-recall curve."""
        try:
            precision, recall, _ = precision_recall_curve(targets, predictions)
            return np.trapz(precision, recall)
        except Exception:
            return 0.0

    def _compute_calibration_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Compute calibration metrics."""
        if self.num_classes == 2:
            # Binary calibration
            try:
                prob_pred = predictions[:, 1]
                prob_true, prob_pred_binned = calibration_curve(
                    targets, prob_pred, n_bins=10, strategy='uniform'
                )

                # Expected Calibration Error (ECE)
                bin_boundaries = np.linspace(0, 1, 11)
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]

                ece = 0.0
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = (prob_pred > bin_lower) & (prob_pred <= bin_upper)
                    prop_in_bin = in_bin.mean()

                    if prop_in_bin > 0:
                        accuracy_in_bin = targets[in_bin].mean()
                        avg_confidence_in_bin = prob_pred[in_bin].mean()
                        ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

                # Brier Score
                brier_score = np.mean((prob_pred - targets) ** 2)

                return {
                    'ece': ece,
                    'brier_score': brier_score,
                    'calibration_slope': self._compute_calibration_slope(targets, prob_pred)
                }
            except Exception as e:
                warnings.warn(f"Could not compute calibration metrics: {e}")
                return {'ece': 0.0, 'brier_score': 0.0, 'calibration_slope': 0.0}
        else:
            return {'ece': 0.0, 'brier_score': 0.0, 'calibration_slope': 0.0}

    def _compute_calibration_slope(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """Compute calibration slope using logistic regression."""
        try:
            from sklearn.linear_model import LogisticRegression

            # Fit logistic regression
            logit_pred = np.log(predictions / (1 - predictions + 1e-8))
            lr = LogisticRegression()
            lr.fit(logit_pred.reshape(-1, 1), targets)

            return lr.coef_[0][0]
        except Exception:
            return 1.0  # Perfect calibration slope

    def _compute_class_specific_metrics(
        self,
        targets: np.ndarray,
        predictions: np.ndarray
    ) -> Dict[str, Dict]:
        """Compute per-class metrics."""
        class_metrics = {}

        # Get classification report
        try:
            report = classification_report(targets, predictions, output_dict=True, zero_division=0)

            for class_name, metrics in report.items():
                if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    class_metrics[f'class_{class_name}'] = metrics
        except Exception as e:
            warnings.warn(f"Could not compute class-specific metrics: {e}")

        return class_metrics

    def _compute_curves(
        self,
        targets: np.ndarray,
        predictions: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute ROC and PR curves."""
        curves = {}

        if self.num_classes == 2:
            try:
                # ROC curve
                fpr, tpr, roc_thresholds = roc_curve(targets, predictions[:, 1])
                curves['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds}

                # Precision-Recall curve
                precision, recall, pr_thresholds = precision_recall_curve(targets, predictions[:, 1])
                curves['pr_curve'] = {'precision': precision, 'recall': recall, 'thresholds': pr_thresholds}

            except Exception as e:
                warnings.warn(f"Could not compute curves: {e}")

        return curves


class DomainMetrics:
    """
    Metrics for evaluating domain generalization and adaptation performance.
    """

    def __init__(self):
        self.base_metrics = CausalMetrics()

    def compute_domain_shift_metrics(
        self,
        source_results: Dict[str, float],
        target_results: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute domain shift metrics comparing source and target performance.

        Args:
            source_results: Metrics on source domain
            target_results: Metrics on target domain

        Returns:
            Domain shift metrics
        """
        domain_metrics = {}

        # Performance degradation
        for metric_name in ['accuracy', 'auc', 'f1', 'precision', 'recall']:
            if metric_name in source_results and metric_name in target_results:
                source_val = source_results[metric_name]
                target_val = target_results[metric_name]

                # Absolute degradation
                domain_metrics[f'{metric_name}_degradation'] = source_val - target_val

                # Relative degradation
                if source_val > 0:
                    domain_metrics[f'{metric_name}_relative_degradation'] = (
                        (source_val - target_val) / source_val * 100
                    )

        # Overall domain gap (average of key metrics)
        key_metrics = ['accuracy', 'auc', 'f1']
        degradations = [
            domain_metrics.get(f'{metric}_degradation', 0)
            for metric in key_metrics
            if f'{metric}_degradation' in domain_metrics
        ]

        if degradations:
            domain_metrics['overall_domain_gap'] = np.mean(degradations)
            domain_metrics['max_domain_gap'] = np.max(degradations)

        return domain_metrics

    def compute_cross_domain_consistency(
        self,
        domain_results: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Compute consistency metrics across multiple domains.

        Args:
            domain_results: Dictionary mapping domain names to their metrics

        Returns:
            Cross-domain consistency metrics
        """
        if len(domain_results) < 2:
            return {}

        consistency_metrics = {}

        # Get all metric names
        all_metrics = set()
        for results in domain_results.values():
            all_metrics.update(results.keys())

        # Compute coefficient of variation for each metric
        for metric_name in all_metrics:
            values = []
            for domain_name, results in domain_results.items():
                if metric_name in results:
                    values.append(results[metric_name])

            if len(values) >= 2:
                mean_val = np.mean(values)
                std_val = np.std(values)

                # Coefficient of variation (lower is better)
                cv = std_val / (mean_val + 1e-8)
                consistency_metrics[f'{metric_name}_cv'] = cv

                # Range (max - min)
                consistency_metrics[f'{metric_name}_range'] = np.max(values) - np.min(values)

        # Overall consistency score (lower is better)
        key_metrics = ['accuracy', 'auc', 'f1']
        cv_values = [
            consistency_metrics.get(f'{metric}_cv', 0)
            for metric in key_metrics
            if f'{metric}_cv' in consistency_metrics
        ]

        if cv_values:
            consistency_metrics['overall_consistency'] = np.mean(cv_values)

        return consistency_metrics


class AttributionMetrics:
    """
    Metrics for evaluating causal attribution quality and consistency.
    """

    def __init__(self):
        pass

    def compute_attribution_metrics(
        self,
        attributions: Dict[str, np.ndarray],
        ground_truth: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute attribution quality metrics.

        Args:
            attributions: Dictionary of attribution maps from different methods
            ground_truth: Ground truth attribution map (if available)

        Returns:
            Attribution quality metrics
        """
        metrics = {}

        # Inter-method consistency
        if len(attributions) >= 2:
            method_names = list(attributions.keys())
            correlations = []

            for i in range(len(method_names)):
                for j in range(i + 1, len(method_names)):
                    attr1 = attributions[method_names[i]].flatten()
                    attr2 = attributions[method_names[j]].flatten()

                    # Pearson correlation
                    corr, _ = stats.pearsonr(attr1, attr2)
                    correlations.append(corr)

            metrics['inter_method_correlation'] = np.mean(correlations)
            metrics['min_inter_method_correlation'] = np.min(correlations)

        # Sparsity metrics
        for method_name, attribution in attributions.items():
            # Compute sparsity (percentage of near-zero values)
            threshold = 0.1 * np.max(np.abs(attribution))
            sparsity = np.mean(np.abs(attribution) < threshold)
            metrics[f'{method_name}_sparsity'] = sparsity

            # Compute concentration (how focused the attribution is)
            sorted_attr = np.sort(attribution.flatten())[::-1]
            cumsum_attr = np.cumsum(sorted_attr)
            total_attr = cumsum_attr[-1]

            # Find percentage of pixels containing 80% of attribution
            if total_attr > 0:
                idx_80 = np.where(cumsum_attr >= 0.8 * total_attr)[0]
                if len(idx_80) > 0:
                    concentration = idx_80[0] / len(sorted_attr)
                    metrics[f'{method_name}_concentration'] = concentration

        # Ground truth comparison (if available)
        if ground_truth is not None:
            for method_name, attribution in attributions.items():
                # Structural similarity
                ssim = self._compute_ssim(attribution, ground_truth)
                metrics[f'{method_name}_ssim'] = ssim

                # IoU for top attribution regions
                iou = self._compute_attribution_iou(attribution, ground_truth)
                metrics[f'{method_name}_iou'] = iou

        return metrics

    def compute_attribution_stability(
        self,
        attributions_list: List[np.ndarray],
        perturbation_levels: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Compute attribution stability across different inputs or perturbations.

        Args:
            attributions_list: List of attribution maps
            perturbation_levels: Perturbation levels applied (if applicable)

        Returns:
            Stability metrics
        """
        if len(attributions_list) < 2:
            return {}

        stability_metrics = {}

        # Pairwise correlations
        correlations = []
        for i in range(len(attributions_list)):
            for j in range(i + 1, len(attributions_list)):
                attr1 = attributions_list[i].flatten()
                attr2 = attributions_list[j].flatten()

                corr, _ = stats.pearsonr(attr1, attr2)
                correlations.append(corr)

        stability_metrics['mean_correlation'] = np.mean(correlations)
        stability_metrics['std_correlation'] = np.std(correlations)
        stability_metrics['min_correlation'] = np.min(correlations)

        # Consistency score (higher is better)
        stability_metrics['consistency_score'] = np.mean(correlations)

        return stability_metrics

    def _compute_ssim(self, attr1: np.ndarray, attr2: np.ndarray) -> float:
        """Compute structural similarity between attribution maps."""
        try:
            from skimage.metrics import structural_similarity as ssim

            # Normalize attributions to [0, 1]
            attr1_norm = (attr1 - attr1.min()) / (attr1.max() - attr1.min() + 1e-8)
            attr2_norm = (attr2 - attr2.min()) / (attr2.max() - attr2.min() + 1e-8)

            return ssim(attr1_norm, attr2_norm, data_range=1.0)
        except ImportError:
            # Fallback to correlation if scikit-image not available
            corr, _ = stats.pearsonr(attr1.flatten(), attr2.flatten())
            return corr

    def _compute_attribution_iou(
        self,
        attr1: np.ndarray,
        attr2: np.ndarray,
        threshold_percentile: float = 90
    ) -> float:
        """Compute IoU between top attribution regions."""
        # Threshold at top percentile
        thresh1 = np.percentile(attr1, threshold_percentile)
        thresh2 = np.percentile(attr2, threshold_percentile)

        # Binary masks
        mask1 = attr1 > thresh1
        mask2 = attr2 > thresh2

        # Compute IoU
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            return 0.0

        return intersection / union


class CausalEvaluationMetrics:
    """
    Specialized metrics for evaluating causal reasoning capabilities.
    """

    def __init__(self):
        pass

    def evaluate_intervention_effects(
        self,
        baseline_predictions: np.ndarray,
        intervention_predictions: np.ndarray,
        intervention_strength: float = 1.0
    ) -> Dict[str, float]:
        """
        Evaluate the effect of causal interventions.

        Args:
            baseline_predictions: Predictions without intervention
            intervention_predictions: Predictions with intervention
            intervention_strength: Strength of the intervention

        Returns:
            Intervention effect metrics
        """
        # Average Treatment Effect (ATE)
        ate = np.mean(intervention_predictions - baseline_predictions)

        # Standard error of ATE
        ate_se = np.std(intervention_predictions - baseline_predictions) / np.sqrt(len(baseline_predictions))

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(baseline_predictions) + np.var(intervention_predictions)) / 2)
        cohens_d = ate / (pooled_std + 1e-8)

        # Proportion of samples affected
        affected_proportion = np.mean(
            np.abs(intervention_predictions - baseline_predictions) > 0.1
        )

        return {
            'average_treatment_effect': ate,
            'ate_standard_error': ate_se,
            'effect_size_cohens_d': cohens_d,
            'affected_proportion': affected_proportion,
            'intervention_strength': intervention_strength
        }

    def evaluate_counterfactual_consistency(
        self,
        original_predictions: np.ndarray,
        counterfactual_predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate consistency of counterfactual predictions.

        Args:
            original_predictions: Original predictions
            counterfactual_predictions: Counterfactual predictions
            labels: True labels

        Returns:
            Counterfactual consistency metrics
        """
        # Consistency rate (predictions that change as expected)
        consistency_rate = np.mean(
            (original_predictions != counterfactual_predictions) == 
            (labels == 1)  # Assuming intervention should affect positive cases more
        )

        # Magnitude of changes
        change_magnitude = np.mean(np.abs(original_predictions - counterfactual_predictions))

        # Directional consistency (changes in expected direction)
        expected_direction = np.where(labels == 1, -1, 0)  # Expect decrease for positive cases
        actual_direction = np.sign(counterfactual_predictions - original_predictions)
        directional_consistency = np.mean(expected_direction == actual_direction)

        return {
            'counterfactual_consistency_rate': consistency_rate,
            'change_magnitude': change_magnitude,
            'directional_consistency': directional_consistency
        }
