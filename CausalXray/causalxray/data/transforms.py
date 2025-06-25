"""
Data augmentation and transforms for medical chest X-ray images.

This module implements medical imaging specific data augmentations that preserve
anatomical relationships while improving model robustness and generalization.
"""

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import cv2


class CausalTransforms:
    """
    Comprehensive transform pipeline for causal medical image analysis.
    """

    def __init__(
        self,
        mode: str = "train",
        image_size: Tuple[int, int] = (224, 224),
        augmentation_config: Optional[Dict] = None
    ):
        """
        Initialize transform pipeline.

        Args:
            mode: "train", "val", or "test"
            image_size: Target image dimensions
            augmentation_config: Configuration for augmentations
        """
        self.mode = mode
        self.image_size = image_size
        self.augmentation_config = augmentation_config or self._get_default_config()

        # Build transform pipeline
        self.transforms = self._build_transform_pipeline()

    def _get_default_config(self) -> Dict:
        """Get default augmentation configuration."""
        return {
            'rotation_degrees': 10,
            'translation_ratio': 0.1,
            'brightness_factor': 0.2,
            'contrast_factor': 0.2,
            'gaussian_blur_prob': 0.3,
            'gaussian_noise_prob': 0.3,
            'horizontal_flip_prob': 0.5,
            'elastic_transform_prob': 0.2,
            'cutout_prob': 0.3,
            'cutout_ratio': 0.1
        }

    def _build_transform_pipeline(self) -> Callable:
        """Build the appropriate transform pipeline based on mode."""
        if self.mode == "train":
            return self._build_train_transforms()
        else:
            return self._build_eval_transforms()

    def _build_train_transforms(self) -> Callable:
        """Build training transforms with augmentations."""
        transforms = [
            T.Resize(self.image_size),
            MedicalAugmentation(self.augmentation_config),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        return T.Compose(transforms)

    def _build_eval_transforms(self) -> Callable:
        """Build evaluation transforms without augmentations."""
        transforms = [
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        return T.Compose(transforms)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Apply transforms to input image."""
        return self.transforms(image)


class MedicalAugmentation:
    """
    Medical imaging specific augmentations that preserve anatomical relationships.
    """

    def __init__(self, config: Dict):
        self.config = config

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply medical augmentations to image."""
        # Apply augmentations with specified probabilities

        # Geometric transformations
        if random.random() < 0.7:  # Apply geometric transforms
            image = self._apply_geometric_transforms(image)

        # Photometric transformations
        if random.random() < 0.8:  # Apply photometric transforms
            image = self._apply_photometric_transforms(image)

        # Noise and blur
        if random.random() < self.config.get('gaussian_noise_prob', 0.3):
            image = self._add_gaussian_noise(image)

        if random.random() < self.config.get('gaussian_blur_prob', 0.3):
            image = self._apply_gaussian_blur(image)

        # Cutout/erasure
        if random.random() < self.config.get('cutout_prob', 0.3):
            image = self._apply_cutout(image)

        return image

    def _apply_geometric_transforms(self, image: Image.Image) -> Image.Image:
        """Apply geometric transformations while preserving anatomy."""
        # Rotation (small angles to preserve anatomy)
        if random.random() < 0.6:
            angle = random.uniform(
                -self.config.get('rotation_degrees', 10),
                self.config.get('rotation_degrees', 10)
            )
            image = TF.rotate(image, angle, fill=0)

        # Translation
        if random.random() < 0.4:
            max_translate = self.config.get('translation_ratio', 0.1)
            translate_x = random.uniform(-max_translate, max_translate) * image.width
            translate_y = random.uniform(-max_translate, max_translate) * image.height
            image = TF.affine(image, angle=0, translate=[translate_x, translate_y], 
                            scale=1.0, shear=0, fill=0)

        # Horizontal flip (anatomically plausible for chest X-rays)
        if random.random() < self.config.get('horizontal_flip_prob', 0.5):
            image = TF.hflip(image)

        # Elastic deformation (very subtle for medical images)
        if random.random() < self.config.get('elastic_transform_prob', 0.2):
            image = self._apply_elastic_transform(image)

        return image

    def _apply_photometric_transforms(self, image: Image.Image) -> Image.Image:
        """Apply photometric transformations."""
        # Brightness adjustment
        if random.random() < 0.7:
            brightness_factor = 1 + random.uniform(
                -self.config.get('brightness_factor', 0.2),
                self.config.get('brightness_factor', 0.2)
            )
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness_factor)

        # Contrast adjustment
        if random.random() < 0.7:
            contrast_factor = 1 + random.uniform(
                -self.config.get('contrast_factor', 0.2),
                self.config.get('contrast_factor', 0.2)
            )
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast_factor)

        # Gamma correction
        if random.random() < 0.5:
            gamma = random.uniform(0.8, 1.2)
            image = self._apply_gamma_correction(image, gamma)

        return image

    def _add_gaussian_noise(self, image: Image.Image) -> Image.Image:
        """Add Gaussian noise to simulate acquisition noise."""
        img_array = np.array(image).astype(np.float32) / 255.0

        noise_std = random.uniform(0.01, 0.05)
        noise = np.random.normal(0, noise_std, img_array.shape)

        noisy_img = img_array + noise
        noisy_img = np.clip(noisy_img, 0, 1)

        return Image.fromarray((noisy_img * 255).astype(np.uint8))

    def _apply_gaussian_blur(self, image: Image.Image) -> Image.Image:
        """Apply Gaussian blur to simulate motion or acquisition blur."""
        radius = random.uniform(0.5, 2.0)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def _apply_cutout(self, image: Image.Image) -> Image.Image:
        """Apply cutout augmentation to simulate partial occlusion."""
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        # Cutout dimensions
        cutout_ratio = self.config.get('cutout_ratio', 0.1)
        cutout_h = int(h * cutout_ratio * random.uniform(0.5, 1.5))
        cutout_w = int(w * cutout_ratio * random.uniform(0.5, 1.5))

        # Random position
        y = random.randint(0, max(1, h - cutout_h))
        x = random.randint(0, max(1, w - cutout_w))

        # Apply cutout
        img_array[y:y+cutout_h, x:x+cutout_w] = 0

        return Image.fromarray(img_array)

    def _apply_elastic_transform(self, image: Image.Image) -> Image.Image:
        """Apply subtle elastic deformation."""
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        # Create displacement fields
        dx = np.random.randn(h, w) * 2.0  # Small displacement
        dy = np.random.randn(h, w) * 2.0

        # Smooth the displacement fields
        dx = cv2.GaussianBlur(dx, (17, 17), 5)
        dy = cv2.GaussianBlur(dy, (17, 17), 5)

        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x_new = (x + dx).astype(np.float32)
        y_new = (y + dy).astype(np.float32)

        # Apply the deformation
        if len(img_array.shape) == 3:
            deformed = cv2.remap(img_array, x_new, y_new, cv2.INTER_LINEAR)
        else:
            deformed = cv2.remap(img_array, x_new, y_new, cv2.INTER_LINEAR)

        return Image.fromarray(deformed)

    def _apply_gamma_correction(self, image: Image.Image, gamma: float) -> Image.Image:
        """Apply gamma correction."""
        img_array = np.array(image).astype(np.float32) / 255.0
        corrected = np.power(img_array, gamma)
        corrected = np.clip(corrected, 0, 1)
        return Image.fromarray((corrected * 255).astype(np.uint8))


class TestTimeAugmentation:
    """
    Test-time augmentation for improved inference robustness.
    """

    def __init__(
        self,
        n_augmentations: int = 5,
        base_transform: Optional[Callable] = None
    ):
        """
        Initialize TTA pipeline.

        Args:
            n_augmentations: Number of augmented versions to create
            base_transform: Base transform to apply
        """
        self.n_augmentations = n_augmentations
        self.base_transform = base_transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Define TTA transforms
        self.tta_transforms = [
            T.Compose([T.RandomRotation(5), self.base_transform]),
            T.Compose([T.RandomHorizontalFlip(p=1.0), self.base_transform]),
            T.Compose([T.ColorJitter(brightness=0.1, contrast=0.1), self.base_transform]),
            T.Compose([T.RandomAffine(degrees=0, translate=(0.1, 0.1)), self.base_transform]),
            self.base_transform  # Original
        ]

    def __call__(self, image: Image.Image) -> List[torch.Tensor]:
        """
        Apply test-time augmentation.

        Args:
            image: Input image

        Returns:
            List of augmented image tensors
        """
        augmented_images = []

        for i in range(self.n_augmentations):
            transform = self.tta_transforms[i % len(self.tta_transforms)]
            augmented = transform(image)
            augmented_images.append(augmented)

        return augmented_images


class DomainSpecificTransforms:
    """
    Domain-specific transforms for handling different datasets and acquisition protocols.
    """

    def __init__(self, domain_config: Dict[str, Any]):
        """
        Initialize domain-specific transforms.

        Args:
            domain_config: Configuration specific to the domain/dataset
        """
        self.domain_config = domain_config

    def get_transforms(self, domain: str, mode: str = "train") -> Callable:
        """
        Get transforms specific to a domain.

        Args:
            domain: Domain identifier (e.g., "nih", "rsna", "pediatric")
            mode: "train", "val", or "test"

        Returns:
            Appropriate transform pipeline
        """
        if domain == "nih":
            return self._get_nih_transforms(mode)
        elif domain == "rsna":
            return self._get_rsna_transforms(mode)
        elif domain == "pediatric":
            return self._get_pediatric_transforms(mode)
        else:
            # Default transforms
            return CausalTransforms(mode=mode).transforms

    def _get_nih_transforms(self, mode: str) -> Callable:
        """Transforms optimized for NIH ChestX-ray14 dataset."""
        if mode == "train":
            return T.Compose([
                T.Resize((256, 256)),
                T.RandomCrop((224, 224)),
                T.RandomRotation(10),
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _get_rsna_transforms(self, mode: str) -> Callable:
        """Transforms optimized for RSNA dataset (DICOM images)."""
        if mode == "train":
            return T.Compose([
                T.Resize((256, 256)),
                T.RandomCrop((224, 224)),
                T.RandomRotation(8),  # Slightly less rotation for DICOM
                T.RandomHorizontalFlip(0.3),  # Less flipping
                T.ColorJitter(brightness=0.15, contrast=0.15),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _get_pediatric_transforms(self, mode: str) -> Callable:
        """Transforms optimized for pediatric chest X-rays."""
        if mode == "train":
            return T.Compose([
                T.Resize((256, 256)),
                T.RandomCrop((224, 224)),
                T.RandomRotation(12),  # Slightly more rotation for pediatric
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(brightness=0.25, contrast=0.25),  # More aggressive for pediatric
                MedicalAugmentation({
                    'gaussian_noise_prob': 0.4,
                    'gaussian_blur_prob': 0.3,
                    'cutout_prob': 0.2
                }),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


class CausalConsistentTransforms:
    """
    Transforms that maintain causal consistency for counterfactual analysis.
    """

    def __init__(self, base_transform: Callable):
        """
        Initialize causal consistent transforms.

        Args:
            base_transform: Base transformation pipeline
        """
        self.base_transform = base_transform

    def create_counterfactual_pair(
        self,
        image: Image.Image,
        intervention_mask: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a counterfactual image pair for causal analysis.

        Args:
            image: Original image
            intervention_mask: Mask indicating regions to modify

        Returns:
            Tuple of (original_tensor, counterfactual_tensor)
        """
        # Apply base transform to original
        original_tensor = self.base_transform(image)

        # Create counterfactual version
        if intervention_mask is not None:
            counterfactual_image = self._apply_intervention(image, intervention_mask)
        else:
            counterfactual_image = self._generate_random_intervention(image)

        counterfactual_tensor = self.base_transform(counterfactual_image)

        return original_tensor, counterfactual_tensor

    def _apply_intervention(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """Apply specific intervention based on mask."""
        img_array = np.array(image)

        # Replace masked regions with noise or mean values
        masked_regions = mask > 0.5
        img_array[masked_regions] = np.mean(img_array[~masked_regions])

        return Image.fromarray(img_array)

    def _generate_random_intervention(self, image: Image.Image) -> Image.Image:
        """Generate random intervention for counterfactual analysis."""
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        # Create random mask
        mask_size = random.randint(20, min(h, w) // 4)
        x = random.randint(0, w - mask_size)
        y = random.randint(0, h - mask_size)

        # Apply intervention
        img_array[y:y+mask_size, x:x+mask_size] = np.mean(img_array)

        return Image.fromarray(img_array)
