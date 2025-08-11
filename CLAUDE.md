# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CausalXray is a structural causal framework for interpretable and robust pneumonia detection from chest X-ray images. It integrates structural causal models with convolutional neural networks to provide genuine causal explanations through intervention-based attribution methods.

## Core Architecture

### Framework Components
1. **CNN Backbone** (`causalxray/models/backbone.py`): DenseNet-121 or ResNet-50 feature extraction
2. **Causal Heads** (`causalxray/models/causal_heads.py`): Disentanglement modules for confounder prediction  
3. **Attribution Module** (`causalxray/models/attribution.py`): Intervention-based causal explanations using Pearl's do-calculus
4. **Progressive Training** (`causalxray/training/trainer.py`): Three-phase optimization strategy

### Data Pipeline
- **Datasets** (`causalxray/data/datasets.py`): NIHChestXray14, RSNAPneumonia, PediatricDataset
- **Preprocessing** (`causalxray/data/preprocessing.py`): Image normalization and augmentation
- **Transforms** (`causalxray/data/transforms.py`): CausalTransforms for data augmentation

### Training Strategy
The framework uses a three-phase progressive training approach:
1. **Phase 1**: Backbone feature learning (typically 50 epochs)
2. **Phase 2**: Causal head training (typically 50 epochs)  
3. **Phase 3**: Full model fine-tuning (typically 50 epochs)

## Common Development Commands

### Environment Setup
```bash
# Create conda environment
conda env create -f environment.yml
conda activate causalxray

# Or use pip
pip install -r requirements.txt
```

### Training Models
```bash
# Train with NIH dataset
python scripts/train.py \
  --config configs/causalxray_nih.yaml \
  --data_dir /path/to/nih/data \
  --output_dir ./outputs

# Train with RSNA dataset  
python scripts/train.py \
  --config configs/causalxray_rsna.yaml \
  --data_dir /path/to/rsna/data \
  --output_dir ./outputs

# Fast training for development
python scripts/train.py \
  --config configs/causalxray_nih_fast.yaml \
  --data_dir /path/to/data \
  --output_dir ./outputs
```

### Model Evaluation
```bash
# Evaluate trained model
python scripts/evaluate.py \
  --checkpoint ./outputs/best_model.pth \
  --data_dir /path/to/test/data \
  --dataset nih

# Cross-domain evaluation
python scripts/evaluate.py \
  --config configs/cross_domain.yaml \
  --checkpoint ./outputs/best_model.pth \
  --data_dir /path/to/test/data
```

### Inference and Attribution
```bash
# Single image inference
python scripts/inference.py \
  --checkpoint ./outputs/best_model.pth \
  --image /path/to/chest_xray.jpg

# Inference with causal attributions
python scripts/inference.py \
  --checkpoint ./outputs/best_model.pth \
  --image /path/to/chest_xray.jpg \
  --show_attributions

# Batch inference with attribution analysis
python scripts/inference_with_attributions.py \
  --checkpoint ./outputs/best_model.pth \
  --data_dir /path/to/images \
  --output_dir ./attributions
```

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Test specific components
python -m pytest tests/test_models.py -v
python -m pytest tests/test_training.py -v
python -m pytest tests/test_evaluation.py -v

# Test single file
python -m pytest tests/test_models.py::TestCausalBackbone -v
```

## Configuration Files

The `configs/` directory contains YAML configuration files for different training scenarios:

- `causalxray_nih.yaml`: NIH ChestX-ray14 dataset configuration
- `causalxray_rsna.yaml`: RSNA pneumonia dataset configuration  
- `causalxray_nih_fast.yaml`: Fast training configuration for development
- `baseline_cnn.yaml`: Baseline CNN without causal components
- `cross_domain.yaml`: Cross-domain evaluation configuration

## Key Configuration Parameters

### Model Configuration
- `backbone`: Architecture (densenet121, resnet50)
- `num_classes`: Number of output classes (typically 2 for binary pneumonia detection)
- `feature_dims`: Feature dimensions for causal heads

### Causal Configuration
- `confounders`: Age, sex, view_position, follow_up dimensions
- `use_variational`: Enable variational inference
- `use_causal_graph`: Enable causal graph structure
- `graph_variables`: Variables in the causal graph

### Training Configuration
- `progressive_training`: Enable three-phase training
- `phase_epochs`: Epochs for each training phase [backbone, causal, full]
- `loss_weights`: Weights for different loss components

## Development Notebooks

- `notebooks/complete_causalxray_setup.ipynb`: Complete setup and training pipeline
- `notebooks/data_analysis.ipynb`: Dataset analysis and visualization
- `notebooks/demo.ipynb`: Model demonstration
- `notebooks/inference_demo.ipynb`: Inference examples
- `notebooks/results_visualization.ipynb`: Results visualization

## Output Structure

Training outputs are saved in structured directories:
- `outputs/`: Training logs and model checkpoints
- `outputs/tensorboard/`: TensorBoard logs for monitoring
- `attributions/`: Generated attribution visualizations
- `causal_reports/`: Causal inference analysis reports

## Attribution Methods

The framework supports multiple attribution methods:
- **intervention**: Direct intervention-based attribution using do-calculus
- **counterfactual**: Counterfactual reasoning for causal explanations
- **gradcam**: Gradient-weighted class activation mapping for comparison

These can be configured in the YAML files under `attribution.attribution_methods`.