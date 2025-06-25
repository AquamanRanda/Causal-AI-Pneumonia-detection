# CausalXray/tests/test_evaluation.py
"""Unit tests for evaluation metrics and evaluation pipeline in CausalXray."""

import pytest
import torch
import numpy as np
from causalxray.training.metrics import CausalMetrics


def test_causal_metrics_compute_epoch_metrics():
    metrics = CausalMetrics()
    # Create dummy probabilities and labels
    probabilities = np.array([[0.7, 0.3], [0.4, 0.6], [0.9, 0.1], [0.2, 0.8]])
    labels = np.array([0, 1, 0, 1])
    results = metrics.compute_epoch_metrics(probabilities, labels)
    # Check keys in results
    assert 'accuracy' in results
    assert 'auc' in results
    assert 'sensitivity' in results
    assert 'specificity' in results
    # Check values are floats
    for key in ['accuracy', 'auc', 'sensitivity', 'specificity']:
        assert isinstance(results[key], float)


def test_evaluation_pipeline_integration():
    # This test would simulate a small evaluation pipeline
    # For simplicity, we mock model outputs and dataset
    class DummyModel:
        def eval(self):
            pass
        def __call__(self, x):
            batch_size = x.shape[0]
            # Return dummy logits
            logits = torch.tensor([[0.6, 0.4]] * batch_size)
            return {'probabilities': logits}

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4
        def __getitem__(self, idx):
            return {'image': torch.randn(3, 224, 224), 'label': torch.tensor(idx % 2)}

    dummy_dataset = DummyDataset()
    dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=2)

    model = DummyModel()
    model.eval()

    metrics = CausalMetrics()

    all_probs = []
    all_labels = []

    for batch in dummy_loader:
        images = batch['image']
        labels = batch['label']
        outputs = model(images)
        all_probs.extend(outputs['probabilities'].detach().numpy())
        all_labels.extend(labels.numpy())

    results = metrics.compute_epoch_metrics(np.array(all_probs), np.array(all_labels))

    assert 'accuracy' in results
    assert results['accuracy'] >= 0.0 and results['accuracy'] <= 1.0

    return True


if __name__ == '__main__':
    test_causal_metrics_compute_epoch_metrics()
    test_evaluation_pipeline_integration()
    print('All tests passed successfully!')
