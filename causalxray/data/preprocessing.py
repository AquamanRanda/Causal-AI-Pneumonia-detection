"""
Image preprocessing utilities for medical chest X-ray images.

This module implements specialized preprocessing techniques for medical imaging
including adaptive histogram equalization, noise reduction, and standardization
protocols specific to chest radiography.
"""

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
import warnings

try:
    import cv2
except ImportError:
    cv2 = None
    warnings.warn('cv2 is not installed. Some preprocessing functions will not work.')

try:
    from skimage import exposure, filters, morphology
except ImportError:
    exposure = filters = morphology = None
    warnings.warn('scikit-image is not installed. Some preprocessing functions will not work.')

from typing import Tuple, Optional, Union, Dict, Any


class CausalPreprocessor:
    """
    Main preprocessing class for CausalXray framework with medical imaging optimizations.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        enhance_contrast: bool = True,
        reduce_noise: bool = True,
        preserve_aspect_ratio: bool = False
    ):
        """
        Initialize medical image preprocessor.

        Args:
            target_size: Target image dimensions (height, width)
            normalize: Whether to normalize pixel values
            enhance_contrast: Whether to apply contrast enhancement
            reduce_noise: Whether to apply noise reduction
            preserve_aspect_ratio: Whether to preserve aspect ratio during resize
        """
        self.target_size = target_size
        self.normalize = normalize
        self.enhance_contrast = enhance_contrast
        self.reduce_noise = reduce_noise
        self.preserve_aspect_ratio = preserve_aspect_ratio

        # Medical imaging specific parameters
        self.clahe_params = {
            'clip_limit': 3.0,
            'tile_grid_size': (8, 8)
        }

        # Normalization statistics (computed on medical imaging datasets)
        self.mean = [0.485, 0.456, 0.406]  # ImageNet means (good starting point)
        self.std = [0.229, 0.224, 0.225]   # ImageNet stds

        # Medical-specific normalization (can be updated based on dataset)
        self.medical_mean = [0.485]  # Single channel for grayscale
        self.medical_std = [0.229]

    def __call__(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Apply preprocessing pipeline to input image.

        Args:
            image: Input image (PIL Image or numpy array)

        Returns:
            Preprocessed image tensor
        """
        # Convert to PIL Image if necessary
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)

        # Apply preprocessing steps
        image = self._resize_image(image)

        if self.enhance_contrast:
            image = self._enhance_contrast(image)

        if self.reduce_noise:
            image = self._reduce_noise(image)

        # Convert to tensor and normalize
        tensor = self._to_tensor(image)

        if self.normalize:
            tensor = self._normalize_tensor(tensor)

        return tensor

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to target size with medical imaging considerations."""
        if self.preserve_aspect_ratio:
            # Resize maintaining aspect ratio, then pad
            image.thumbnail(self.target_size, Image.Resampling.LANCZOS)

            # Create new image with target size and paste resized image
            new_image = Image.new('RGB', self.target_size, (0, 0, 0))
            paste_x = (self.target_size[1] - image.width) // 2
            paste_y = (self.target_size[0] - image.height) // 2
            new_image.paste(image, (paste_x, paste_y))

            return new_image
        else:
            # Direct resize
            return image.resize((self.target_size[1], self.target_size[0]), Image.Resampling.LANCZOS)

    def _enhance_contrast(self, image: Image.Image) -> Image.Image:
        """Apply adaptive histogram equalization for contrast enhancement."""
        # Convert to numpy for CLAHE
        img_array = np.array(image)

        if len(img_array.shape) == 3:
            if cv2 is None:
                raise ImportError('cv2 is required for contrast enhancement but is not installed.')
            # RGB image - convert to LAB and apply CLAHE to L channel
            img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_params['clip_limit'],
                tileGridSize=self.clahe_params['tile_grid_size']
            )
            img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])

            # Convert back to RGB
            img_enhanced = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        else:
            if cv2 is None:
                raise ImportError('cv2 is required for contrast enhancement but is not installed.')
            # Grayscale image
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_params['clip_limit'],
                tileGridSize=self.clahe_params['tile_grid_size']
            )
            img_enhanced = clahe.apply(img_array)

        return Image.fromarray(img_enhanced)

    def _reduce_noise(self, image: Image.Image) -> Image.Image:
        """Apply noise reduction techniques suitable for medical images."""
        # Convert to numpy
        img_array = np.array(image)

        if cv2 is None:
            raise ImportError('cv2 is required for noise reduction but is not installed.')
        # Apply bilateral filtering for noise reduction while preserving edges
        if len(img_array.shape) == 3:
            # RGB image
            img_denoised = cv2.bilateralFilter(img_array, 9, 75, 75)
        else:
            # Grayscale
            img_denoised = cv2.bilateralFilter(img_array, 9, 75, 75)

        return Image.fromarray(img_denoised)

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to PyTorch tensor."""
        # Convert to numpy
        img_array = np.array(image)

        # Normalize to [0, 1] range
        img_array = img_array.astype(np.float32) / 255.0

        # Handle different image formats
        if len(img_array.shape) == 2:
            # Grayscale - add channel dimension and convert to RGB
            img_array = np.stack([img_array] * 3, axis=2)

        # Convert to CHW format
        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))

        return img_tensor

    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor using dataset statistics."""
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)

        return (tensor - mean) / std

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize tensor for visualization."""
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)

        return tensor * std + mean

    def update_normalization_stats(self, mean: list, std: list):
        """Update normalization statistics based on dataset."""
        self.mean = mean
        self.std = std


class MedicalImageProcessor:
    """
    Advanced medical image processing utilities.
    """

    @staticmethod
    def enhance_lung_regions(image: np.ndarray) -> np.ndarray:
        """
        Enhance lung regions in chest X-ray images.

        Args:
            image: Input chest X-ray image

        Returns:
            Enhanced image with improved lung visibility
        """
        if exposure is None:
            raise ImportError('scikit-image is required for enhance_lung_regions but is not installed.')
        # Apply gamma correction to enhance lung regions
        gamma_corrected = exposure.adjust_gamma(image, gamma=int(0.8))

        # Apply sigmoid correction for better contrast
        sigmoid_corrected = exposure.adjust_sigmoid(gamma_corrected, cutoff=0.5, gain=10)

        return sigmoid_corrected

    @staticmethod
    def remove_background_artifacts(image: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        Remove background artifacts and noise from medical images.

        Args:
            image: Input image
            threshold: Threshold for background removal

        Returns:
            Cleaned image
        """
        if filters is None or morphology is None:
            raise ImportError('scikit-image is required for remove_background_artifacts but is not installed.')
        # Apply Otsu thresholding to separate foreground from background
        thresh = filters.threshold_otsu(image)
        binary = image > thresh * threshold

        # Remove small artifacts using morphological operations
        cleaned = morphology.remove_small_objects(binary, min_size=100)

        # Apply to original image
        result = image.copy()
        result[~cleaned] = 0

        return result

    @staticmethod
    def normalize_chest_orientation(image: np.ndarray) -> np.ndarray:
        """
        Normalize chest X-ray orientation and positioning.

        Args:
            image: Input chest X-ray

        Returns:
            Normalized image
        """
        # This is a simplified implementation
        # In practice, this would involve more sophisticated anatomical landmark detection

        # Center the image based on intensity distribution
        center_of_mass = np.array(np.where(image > np.mean(image))).mean(axis=1)
        h, w = image.shape[:2]

        # Calculate translation to center
        target_center = np.array([h//2, w//2])
        translation = target_center - center_of_mass[:2]

        # Apply translation
        rows, cols = h, w
        # Ensure translation values are float and matrix is float32
        M = np.array([[1, 0, float(translation[1])], [0, 1, float(translation[0])]], dtype=np.float32)
        if cv2 is not None:
            normalized = cv2.warpAffine(image, M, (cols, rows))
        else:
            raise ImportError('cv2 is required for normalization but is not installed.')
        return normalized

    @staticmethod
    def detect_and_crop_lung_region(image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Detect and crop the lung region from chest X-ray.

        Args:
            image: Input chest X-ray

        Returns:
            Tuple of (cropped_image, crop_info)
        """
        if filters is None:
            raise ImportError('scikit-image is required for detect_and_crop_lung_region but is not installed.')
        # Apply thresholding to find lung regions
        thresh = filters.threshold_otsu(image)
        binary = image > thresh * 0.3

        if cv2 is None:
            raise ImportError('cv2 is required for detect_and_crop_lung_region but is not installed.')
        # Find contours
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)

        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find largest contour (likely to be lung region)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)

            # Crop image
            cropped = image[y:y+h, x:x+w]

            crop_info = {
                'x': x, 'y': y, 'width': w, 'height': h,
                'original_shape': image.shape
            }
        else:
            # No contours found, return original
            cropped = image
            crop_info = {
                'x': 0, 'y': 0, 
                'width': image.shape[1], 'height': image.shape[0],
                'original_shape': image.shape
            }

        return cropped, crop_info


class QualityAssessment:
    """
    Image quality assessment for medical images.
    """

    @staticmethod
    def compute_image_quality_metrics(image: np.ndarray) -> Dict[str, float]:
        """
        Compute various image quality metrics.

        Args:
            image: Input image

        Returns:
            Dictionary of quality metrics
        """
        metrics = {}

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if cv2 is None:
                raise ImportError('cv2 is required for compute_image_quality_metrics but is not installed.')
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Ensure gray is a numpy ndarray
        if not isinstance(gray, np.ndarray):
            gray = np.array(gray)
        signal = np.mean(gray)  # type: ignore
        noise = np.std(gray)  # type: ignore
        metrics['snr'] = signal / (noise + 1e-8)

        # Contrast measure
        metrics['contrast'] = np.std(gray)  # type: ignore

        # Sharpness (Laplacian variance)
        if cv2 is None:
            raise ImportError('cv2 is required for compute_image_quality_metrics but is not installed.')
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        metrics['sharpness'] = laplacian.var()

        # Brightness
        metrics['brightness'] = np.mean(gray)  # type: ignore

        # Dynamic range
        metrics['dynamic_range'] = np.max(gray) - np.min(gray)

        return metrics

    @staticmethod
    def is_acceptable_quality(image: np.ndarray, thresholds: Optional[Dict] = None) -> Tuple[bool, Dict]:
        """
        Assess if image meets quality requirements.

        Args:
            image: Input image
            thresholds: Quality thresholds

        Returns:
            Tuple of (is_acceptable, quality_report)
        """
        default_thresholds = {
            'min_contrast': 10,
            'min_sharpness': 100,
            'min_brightness': 0.1,
            'max_brightness': 0.9,
            'min_dynamic_range': 0.3
        }

        thresholds = thresholds or default_thresholds

        metrics = QualityAssessment.compute_image_quality_metrics(image)

        quality_checks = {
            'contrast_ok': metrics['contrast'] > thresholds['min_contrast'],
            'sharpness_ok': metrics['sharpness'] > thresholds['min_sharpness'],
            'brightness_ok': thresholds['min_brightness'] < metrics['brightness'] < thresholds['max_brightness'],
            'dynamic_range_ok': metrics['dynamic_range'] > thresholds['min_dynamic_range']
        }

        is_acceptable = all(quality_checks.values())

        quality_report = {
            'metrics': metrics,
            'checks': quality_checks,
            'overall_acceptable': is_acceptable
        }

        return is_acceptable, quality_report
