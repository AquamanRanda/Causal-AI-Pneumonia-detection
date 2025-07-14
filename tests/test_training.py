# CausalXray/tests/test_training.py
"""Unit tests for training components in CausalXray."""

import torch
from torch.utils.data import DataLoader
from causalxray.training.trainer import CausalTrainer
from causalxray.models.backbone import CausalBackbone

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

def trainer_config():
    """Fixture providing standard trainer configuration."""
    return {
        'batch_size': 2,
        'num_epochs': 1,
        'learning_rate': 1e-3,
        'progressive_training': True
    }

def test_trainer_initialization():
    """Test successful initialization of CausalTrainer."""
    dataset = dummy_dataset()
    mconfig = model_config()
    tconfig = trainer_config()
    dataloader = DataLoader(dataset, batch_size=tconfig['batch_size'])
    model = CausalBackbone(
        architecture=mconfig['backbone'].get('architecture', 'densenet121'),
        pretrained=mconfig['backbone'].get('pretrained', False),
        num_classes=mconfig['backbone'].get('num_classes', 2),
        feature_dims=mconfig['backbone'].get('feature_dims', [1024])
    )
    trainer = CausalTrainer(
        model=model,
        train_loader=dataloader,
        val_loader=dataloader,
        config=tconfig,
        device='cpu'
    )
    assert isinstance(trainer, CausalTrainer)
    assert trainer.model is not None
    assert trainer.optimizer is not None

def test_full_training_loop():
    """Test successful execution of full training loop."""
    dataset = dummy_dataset()
    mconfig = model_config()
    tconfig = trainer_config()
    dataloader = DataLoader(dataset, batch_size=tconfig['batch_size'])
    model = CausalBackbone(
        architecture=mconfig['backbone'].get('architecture', 'densenet121'),
        pretrained=mconfig['backbone'].get('pretrained', False),
        num_classes=mconfig['backbone'].get('num_classes', 2),
        feature_dims=mconfig['backbone'].get('feature_dims', [1024])
    )
    trainer = CausalTrainer(
        model=model,
        train_loader=dataloader,
        val_loader=dataloader,
        config=tconfig,
        device='cpu'
    )
    history = trainer.train(tconfig['num_epochs'])
    assert isinstance(history, dict)
    # The keys in history depend on the trainer implementation; check for non-empty history
    assert len(history) > 0
