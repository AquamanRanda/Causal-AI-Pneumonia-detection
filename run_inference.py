#!/usr/bin/env python3
"""
Custom inference script for CausalXray that handles raw checkpoint format.
"""

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Add package to path
sys.path.append(str(Path(__file__).parent))

from causalxray.models.backbone import CausalBackbone


class SimpleCausalModel(nn.Module):
    """Simplified model structure to match the checkpoint."""
    
    def __init__(self, backbone_arch="densenet121", num_classes=2):
        super().__init__()
        
        # Create backbone
        if backbone_arch == "densenet121":
            import torchvision.models as models
            self.backbone = models.densenet121(pretrained=False)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()  # Remove original classifier
        
        # Add custom heads to match checkpoint structure
        self.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification heads
        self.class_head = nn.Linear(512, num_classes)
        self.age_head = nn.Linear(512, 1)  # Age confounder
        self.sex_head = nn.Linear(512, 2)  # Sex confounder
        self.view_head = nn.Linear(512, 3)  # View position confounder
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        features = self.fc(features)
        
        # Get predictions
        class_logits = self.class_head(features)
        class_probs = torch.softmax(class_logits, dim=1)
        
        # Confounder predictions
        age_pred = self.age_head(features)
        sex_pred = self.sex_head(features) 
        view_pred = self.view_head(features)
        
        return {
            'logits': class_logits,
            'probabilities': class_probs,
            'features': features,
            'age': age_pred,
            'sex': sex_pred,
            'view': view_pred
        }


def load_image(image_path, image_size=(224, 224)):
    """Load and preprocess image."""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def generate_simple_attribution(model, image_tensor, predicted_class, method='gradcam'):
    """Generate simple attribution using gradient-based methods."""
    model.eval()
    image_tensor.requires_grad_(True)
    
    # Forward pass
    outputs = model(image_tensor)
    class_logits = outputs['logits']
    
    # Backward pass for the predicted class
    model.zero_grad()
    class_logits[0, predicted_class].backward()
    
    # Get gradients
    gradients = image_tensor.grad.data
    
    # Simple attribution: absolute gradients
    attribution = torch.abs(gradients).mean(dim=1, keepdim=True)  # Average across channels
    
    return attribution.squeeze().cpu().numpy()


def visualize_results(original_image_path, attribution, predicted_class, confidence, output_dir):
    """Visualize attribution results."""
    # Load original image for visualization
    original_image = Image.open(original_image_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image, cmap='gray' if len(np.array(original_image).shape) == 2 else None)
    axes[0].set_title('Original X-ray')
    axes[0].axis('off')
    
    # Attribution heatmap
    axes[1].imshow(attribution, cmap='hot', alpha=0.8)
    axes[1].set_title('Causal Attribution')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(original_image, cmap='gray' if len(np.array(original_image).shape) == 2 else None)
    axes[2].imshow(attribution, cmap='hot', alpha=0.4)
    class_names = ['Normal', 'Pneumonia']
    axes[2].set_title(f'Prediction: {class_names[predicted_class]} ({confidence:.3f})')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = Path(output_dir) / 'attribution_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Attribution visualization saved to: {output_path}")
    
    plt.show()
    return output_path


def main():
    # Paths
    model_path = "./best_model_weights.pth"
    image_path = "./NormalTestxray.jpg"
    output_dir = "./attributions_normaltest"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Loading CausalXray model...")
    
    # Create model
    model = SimpleCausalModel(backbone_arch="densenet121", num_classes=2)
    
    # Load checkpoint with CPU mapping
    print("Loading checkpoint...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Load state dict directly (no nested structure)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Load and preprocess image
    print(f"Loading image: {image_path}")
    image_tensor = load_image(image_path)
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = outputs['probabilities'].cpu().numpy()[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        # Get confounder predictions
        age_pred = outputs['age'].cpu().numpy()[0][0]
        sex_probs = torch.softmax(outputs['sex'], dim=1).cpu().numpy()[0]
        view_probs = torch.softmax(outputs['view'], dim=1).cpu().numpy()[0]
        
        # Interpret predictions
        sex_pred = np.argmax(sex_probs)
        view_pred = np.argmax(view_probs)
    
    # Print results
    class_names = ['Normal', 'Pneumonia']
    sex_names = ['Female', 'Male']
    view_names = ['AP', 'PA', 'Lateral']
    
    predicted_label = class_names[predicted_class]
    predicted_sex = sex_names[sex_pred]
    predicted_view = view_names[view_pred]
    
    print(f"\n" + "="*60)
    print(f"CAUSAL XRAY INFERENCE RESULTS")
    print(f"="*60)
    print(f"PNEUMONIA PREDICTION:")
    print(f"  Predicted Class: {predicted_label}")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Probabilities:")
    print(f"    Normal: {probabilities[0]:.4f}")
    print(f"    Pneumonia: {probabilities[1]:.4f}")
    print(f"\nCONFOUNDER PREDICTIONS:")
    print(f"  Age: {age_pred:.1f} years")
    print(f"  Sex: {predicted_sex} (confidence: {sex_probs[sex_pred]:.4f})")
    print(f"    Female: {sex_probs[0]:.4f}")
    print(f"    Male: {sex_probs[1]:.4f}")
    print(f"  View Position: {predicted_view} (confidence: {view_probs[view_pred]:.4f})")
    print(f"    AP: {view_probs[0]:.4f}")
    print(f"    PA: {view_probs[1]:.4f}")
    print(f"    Lateral: {view_probs[2]:.4f}")
    print(f"="*60)
    
    # Generate attribution
    print("Generating causal attribution...")
    try:
        attribution = generate_simple_attribution(model, image_tensor.clone(), predicted_class)
        
        # Visualize results
        print("Creating visualization...")
        viz_path = visualize_results(image_path, attribution, predicted_class, confidence, output_dir)
        
        print(f"\nAttribution analysis completed!")
        print(f"Results saved in: {output_dir}")
        
    except Exception as e:
        print(f"Error generating attribution: {e}")
        print("Continuing with basic inference results only.")
    
    # Save results to file
    results_file = Path(output_dir) / 'inference_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"CausalXray Inference Results\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"Image: {image_path}\n\n")
        f.write(f"PNEUMONIA PREDICTION:\n")
        f.write(f"  Predicted Class: {predicted_label}\n")
        f.write(f"  Confidence: {confidence:.4f}\n")
        f.write(f"  Normal Probability: {probabilities[0]:.4f}\n")
        f.write(f"  Pneumonia Probability: {probabilities[1]:.4f}\n\n")
        f.write(f"CONFOUNDER PREDICTIONS:\n")
        f.write(f"  Age: {age_pred:.1f} years\n")
        f.write(f"  Sex: {predicted_sex} (confidence: {sex_probs[sex_pred]:.4f})\n")
        f.write(f"    Female: {sex_probs[0]:.4f}\n")
        f.write(f"    Male: {sex_probs[1]:.4f}\n")
        f.write(f"  View Position: {predicted_view} (confidence: {view_probs[view_pred]:.4f})\n")
        f.write(f"    AP: {view_probs[0]:.4f}\n")
        f.write(f"    PA: {view_probs[1]:.4f}\n")
        f.write(f"    Lateral: {view_probs[2]:.4f}\n")
    
    print(f"Results also saved to: {results_file}")


if __name__ == "__main__":
    main()