# CausalXray/tests/test_training.py
"""Unit tests for training components in CausalXray."""

import pytest
import torch
from torch.utils.data import DataLoader
from causalxray.training.trainer import CausalTrainer
from causalxray.models.causalxray import CausalXray

@pytest.fixture
def dummy_dataset():
    """Fixture providing a dummy dataset with confounders."""
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, length=10):
            self.length = length
        def __len__(self):
            return self.length
        def __getitem__(self, idx):
            # Return dummy image tensor and label
            image = torch.randn(3, 224, 224)
            label = torch.randint(0, 2, (1,)).item()
            confounders = {
                'age': torch.tensor(float(idx % 100)),
                'sex': torch.tensor(idx % 2),
                'scanner_type': torch.tensor(idx % 5)
            }
            return {'image': image, 'label': label, 'confounders': confounders}
    
    return DummyDataset()

@pytest.fixture
def model_config():
    """Fixture providing standard model configuration."""
    return {
        'backbone': {
            'architecture': 'densenet121',
            'pretrained': False,
            'num_classes': 2,
            'feature_dims': [1024]
        },
        'causal': {
            'confounders': {'age': 1, 'sex': 2, 'scanner_type': 5},
            'use_variational': True
        }
    }

@pytest.fixture
def trainer_config():
    """Fixture providing standard trainer configuration."""
    return {
        'batch_size': 2,
        'num_epochs': 1,
        'learning_rate': 1e-3,
        'progressive_training': True
    }

def test_trainer_initialization(dummy_dataset, model_config, trainer_config):
    """Test successful initialization of CausalTrainer."""
    dataloader = DataLoader(dummy_dataset, batch_size=trainer_config['batch_size'])
    model = CausalXray(**model_config)
    trainer = CausalTrainer(
        model=model,
        train_loader=dataloader,
        val_loader=dataloader,
        config=trainer_config,
        device='cpu'
    )
    assert isinstance(trainer, CausalTrainer)
    assert trainer.model is not None
    assert trainer.optimizer is not None

def test_training_step(dummy_dataset, model_config, trainer_config):
    """Test successful execution of training step."""
    dataloader = DataLoader(dummy_dataset, batch_size=trainer_config['batch_size'])
    model = CausalXray(**model_config)
    trainer = CausalTrainer(
        model=model,
        train_loader=dataloader,
        val_loader=dataloader,
        config=trainer_config,
        device='cpu'
    )
    model.train()
    
    for batch in dataloader:
        outputs = trainer.training_step(batch)
        assert 'loss' in outputs
        assert outputs['loss'].item() >= 0.0
        break

def test_validation_step(dummy_dataset, model_config, trainer_config):
    """Test successful execution of validation step."""
    dataloader = DataLoader(dummy_dataset, batch_size=trainer_config['batch_size'])
    model = CausalXray(**model_config)
    trainer = CausalTrainer(
        model=model,
        train_loader=dataloader,
        val_loader=dataloader,
        config=trainer_config,
        device='cpu'
    )
    model.eval()
    
    for batch in dataloader:
        outputs = trainer.validation_step(batch)
        assert 'val_loss' in outputs
        assert outputs['val_loss'].item() >= 0.0
        break

def test_full_training_loop(dummy_dataset, model_config, trainer_config):
    """Test successful execution of full training loop."""
    dataloader = DataLoader(dummy_dataset, batch_size=trainer_config['batch_size'])
    model = CausalXray(**model_config)
    trainer = CausalTrainer(
        model=model,
        train_loader=dataloader,
        val_loader=dataloader,
        config=trainer_config,
        device='cpu'
    )
    
    history = trainer.train(trainer_config['num_epochs'])
    assert isinstance(history, dict)
    assert 'training_loss' in history
    assert 'validation_loss' in history
    assert len(history['training_loss']) == trainer_config['num_epochs']
    assert len(history['validation_loss']) == trainer_config['num_epochs']

if __name__ == '__main__':
    pytest.main([__file__])
