# CausalXray/tests/test_models.py
"""Unit tests for CausalXray models."""

import pytest
import torch
import torch.nn as nn
from causalxray.models import CausalBackbone, CausalHeads, CausalAttribution, CausalXray


class TestCausalBackbone:
    """Test CausalBackbone model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.backbone_config = {
            'architecture': 'densenet121',
            'pretrained': False,  # Faster for testing
            'num_classes': 2,
            'feature_dims': [512, 256],
            'dropout_rate': 0.3
        }
    
    def test_backbone_initialization(self):
        """Test backbone initialization."""
        backbone = CausalBackbone(**self.backbone_config)
        assert backbone.architecture == 'densenet121'
        assert backbone.num_classes == 2
        assert len(backbone.feature_dims) == 2
    
    def test_backbone_forward(self):
        """Test backbone forward pass."""
        backbone = CausalBackbone(**self.backbone_config)
        x = torch.randn(2, 3, 224, 224)
        
        outputs = backbone(x)
        
        assert 'features' in outputs
        assert 'causal_features' in outputs
        assert 'logits' in outputs
        assert 'probabilities' in outputs
        
        assert outputs['logits'].shape == (2, 2)
        assert outputs['probabilities'].shape == (2, 2)
        assert len(outputs['causal_features']) == 2
    
    def test_feature_maps_extraction(self):
        """Test feature map extraction."""
        backbone = CausalBackbone(**self.backbone_config)
        x = torch.randn(2, 3, 224, 224)
        
        feature_maps = backbone.get_feature_maps(x)
        assert isinstance(feature_maps, dict)
        assert len(feature_maps) > 0


class TestCausalHeads:
    """Test CausalHeads model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.causal_config = {
            'input_dim': 256,
            'confounders': {'age': 1, 'sex': 2, 'scanner_type': 3},
            'hidden_dims': [128, 64],
            'use_variational': True
        }
    
    def test_causal_heads_initialization(self):
        """Test causal heads initialization."""
        heads = CausalHeads(**self.causal_config)
        assert len(heads.confounder_heads) == 3
        assert 'age' in heads.confounder_heads
        assert 'sex' in heads.confounder_heads
        assert 'scanner_type' in heads.confounder_heads
    
    def test_causal_heads_forward(self):
        """Test causal heads forward pass."""
        heads = CausalHeads(**self.causal_config)
        x = torch.randn(4, 256)
        
        outputs = heads(x)
        
        assert 'confounders' in outputs
        assert 'causal_features' in outputs
        assert 'variational' in outputs
        
        assert 'age' in outputs['confounders']
        assert 'sex' in outputs['confounders']
        assert 'scanner_type' in outputs['confounders']
        
        assert outputs['confounders']['age'].shape == (4, 1)
        assert outputs['confounders']['sex'].shape == (4, 2)
        assert outputs['confounders']['scanner_type'].shape == (4, 3)


class TestCausalXray:
    """Test complete CausalXray model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.backbone_config = {
            'architecture': 'densenet121',
            'pretrained': False,
            'num_classes': 2,
            'feature_dims': [512, 256],
            'dropout_rate': 0.3
        }
        
        self.causal_config = {
            'confounders': {'age': 1, 'sex': 2},
            'hidden_dims': [128, 64],
            'use_variational': True,
            'use_domain_adaptation': False
        }
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = CausalXray(
            backbone_config=self.backbone_config,
            causal_config=self.causal_config
        )
        
        assert model.backbone is not None
        assert model.causal_heads is not None
        assert model.training_phase == 'full'
    
    def test_model_forward(self):
        """Test model forward pass."""
        model = CausalXray(
            backbone_config=self.backbone_config,
            causal_config=self.causal_config
        )
        
        x = torch.randn(2, 3, 224, 224)
        confounders = {
            'age': torch.randn(2),
            'sex': torch.randint(0, 2, (2,))
        }
        
        outputs = model(x, confounders=confounders)
        
        assert 'logits' in outputs
        assert 'probabilities' in outputs
        assert 'causal_outputs' in outputs
        
        assert outputs['logits'].shape == (2, 2)
        assert outputs['probabilities'].shape == (2, 2)
    
    def test_progressive_training_phases(self):
        """Test progressive training phase switching."""
        model = CausalXray(
            backbone_config=self.backbone_config,
            causal_config=self.causal_config,
            training_phase='backbone'
        )
        
        assert model.training_phase == 'backbone'
        
        model.set_training_phase('causal')
        assert model.training_phase == 'causal'
        
        model.set_training_phase('full')
        assert model.training_phase == 'full'
    
    def test_model_prediction(self):
        """Test model prediction method."""
        model = CausalXray(
            backbone_config=self.backbone_config,
            causal_config=self.causal_config
        )
        
        x = torch.randn(2, 3, 224, 224)
        
        predictions = model.predict(x, return_probabilities=True)
        
        assert 'predicted_class' in predictions
        assert 'probabilities' in predictions
        
        assert predictions['predicted_class'].shape == (2,)
        assert predictions['probabilities'].shape == (2, 2)


if __name__ == '__main__':
    pytest.main([__file__])
