# CausalXray Inference Scripts

This directory contains scripts for using trained CausalXray models to perform inference on X-ray images with causal attributions.

## Quick Start

### 1. Single Image Analysis

Analyze a single X-ray image with causal attributions:

```bash
python inference_with_attributions.py \
    --model_path /path/to/your/trained/model.pth \
    --image_path /path/to/xray/image.jpg \
    --output_dir ./results
```

### 2. Batch Analysis

Analyze multiple X-ray images in a directory:

```bash
python inference_with_attributions.py \
    --model_path /path/to/your/trained/model.pth \
    --image_dir /path/to/xray/images/ \
    --output_dir ./batch_results
```

### 3. Quick Prediction (No Attributions)

For faster inference when you only need the prediction:

```bash
python inference_with_attributions.py \
    --model_path /path/to/your/trained/model.pth \
    --image_path /path/to/xray/image.jpg \
    --no_attributions
```

### 4. Target Class Attribution

Generate attributions for a specific target class:

```bash
python inference_with_attributions.py \
    --model_path /path/to/your/trained/model.pth \
    --image_path /path/to/xray/image.jpg \
    --target_class 1  # 0=Normal, 1=Pneumonia
```

## Scripts Overview

### `inference_with_attributions.py`

Main inference script with full functionality:

- **Single image inference** with causal attributions
- **Batch processing** of multiple images
- **Quick predictions** without attributions
- **Target class attributions** for interpretability
- **Automatic visualization** generation
- **Comprehensive logging** and error handling

**Features:**
- Loads trained model checkpoints
- Generates multiple attribution methods (intervention, counterfactual, GradCAM)
- Creates visualization plots
- Saves results in JSON format
- Provides detailed analysis reports

### `example_inference.py`

Example script demonstrating different usage patterns:

- **Single image analysis** example
- **Batch inference** example
- **Quick prediction** example
- **Target class attribution** example

## Usage Examples

### Command Line Usage

```bash
# Basic single image inference
python inference_with_attributions.py \
    --model_path experiments/your_experiment/checkpoints/best_model.pth \
    --image_path data/test_xray.jpg

# Batch inference with custom output directory
python inference_with_attributions.py \
    --model_path experiments/your_experiment/checkpoints/best_model.pth \
    --image_dir data/test_images/ \
    --output_dir ./my_results

# Quick prediction only
python inference_with_attributions.py \
    --model_path experiments/your_experiment/checkpoints/best_model.pth \
    --image_path data/test_xray.jpg \
    --no_attributions

# Generate attributions for Normal class
python inference_with_attributions.py \
    --model_path experiments/your_experiment/checkpoints/best_model.pth \
    --image_path data/test_xray.jpg \
    --target_class 0
```

### Python API Usage

```python
from scripts.inference_with_attributions import CausalXrayInference

# Initialize inference engine
inference_engine = CausalXrayInference(
    model_path="/path/to/model.pth",
    device=None  # Auto-detect device
)

# Single image inference with attributions
results = inference_engine.generate_attributions(
    image_path="/path/to/xray.jpg",
    save_visualization=True,
    output_dir="./results"
)

# Quick prediction only
results = inference_engine.predict("/path/to/xray.jpg")

# Batch inference
image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
batch_results = inference_engine.batch_inference(
    image_paths=image_paths,
    save_attributions=True,
    output_dir="./batch_results"
)
```

## Output Structure

The inference scripts generate the following outputs:

```
output_dir/
├── image_name_attributions.png      # Attribution comparison visualization
├── image_name_statistics.png        # Attribution statistics plots
├── batch_results.json              # Batch inference results (JSON)
└── interactive/                    # Interactive analysis results
    ├── uploaded_image_attributions.png
    └── uploaded_image_statistics.png
```

### Results Format

Each inference result contains:

```json
{
  "image_path": "/path/to/image.jpg",
  "predicted_class": "Pneumonia",
  "predicted_class_id": 1,
  "confidence": 0.9234,
  "probabilities": {
    "normal": 0.0766,
    "pneumonia": 0.9234
  },
  "attributions": {
    "intervention": [...],
    "counterfactual": [...],
    "gradcam": [...]
  },
  "attribution_methods": ["intervention", "counterfactual", "gradcam"]
}
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## Device Support

- **CUDA**: Automatic detection and usage if available
- **CPU**: Fallback when CUDA is not available
- **Manual selection**: Use `--device cuda` or `--device cpu`

## Attribution Methods

The inference engine generates multiple attribution methods:

1. **Intervention-based**: Causal attributions using intervention methods
2. **Counterfactual**: What-if analysis for different scenarios
3. **GradCAM**: Gradient-based class activation mapping
4. **Integrated Gradients**: Path-based attribution method

## Visualization Features

- **Comparison plots**: Side-by-side attribution method comparison
- **Statistics plots**: Attribution value distributions and correlations
- **Overlay visualizations**: Attribution heatmaps overlaid on original images
- **High-resolution outputs**: 300 DPI PNG files

## Error Handling

The scripts include comprehensive error handling:

- **Model loading errors**: Clear messages for missing or corrupted checkpoints
- **Image processing errors**: Support for various image formats and sizes
- **Device errors**: Automatic fallback to CPU if CUDA fails
- **Memory management**: Efficient processing of large images

## Performance Tips

1. **Use `--no_attributions`** for faster inference when only predictions are needed
2. **Batch processing** is more efficient than individual image processing
3. **GPU memory**: Large images may require more GPU memory
4. **Output directory**: Use SSD storage for faster I/O operations

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the model path is correct and the file exists
2. **CUDA out of memory**: Try using CPU or reducing image size
3. **Image format not supported**: Convert to JPEG or PNG format
4. **Permission errors**: Ensure write permissions for output directories

### Debug Mode

For detailed logging, modify the script to include:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration with Training

The inference scripts work seamlessly with trained models from the CausalXray framework:

1. **Load checkpoints** from training experiments
2. **Use same preprocessing** as training
3. **Maintain model architecture** compatibility
4. **Preserve configuration** from training runs

## Advanced Usage

### Custom Attribution Methods

```python
# Initialize with custom attribution methods
inference_engine = CausalXrayInference(
    model_path="/path/to/model.pth"
)

# Modify attribution methods
inference_engine.attribution_module.attribution_methods = ['intervention', 'gradcam']
```

### Custom Visualization

```python
# Generate custom visualizations
from causalxray.utils.visualization import AttributionVisualizer

visualizer = AttributionVisualizer()
fig = visualizer.visualize_attribution_comparison(
    original_image, attributions, prediction_info
)
fig.savefig('custom_visualization.png')
```

## Support

For issues and questions:

1. Check the error messages for specific guidance
2. Verify model checkpoint compatibility
3. Ensure all dependencies are installed
4. Check GPU memory availability

The inference scripts provide a complete solution for analyzing X-ray images with causal attributions, making the CausalXray framework accessible for real-world applications. 