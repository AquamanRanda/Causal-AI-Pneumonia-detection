"""
Dataset classes for chest X-ray pneumonia detection.

This module implements PyTorch datasets for loading and processing chest X-ray images
from various sources including NIH ChestX-ray14, RSNA, and pediatric datasets with
support for causal confounder information.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import pydicom
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import json
import warnings

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ChestXrayDataset(Dataset):
    """
    Base class for chest X-ray datasets with causal confounder support.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        include_confounders: bool = True,
        confounder_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize chest X-ray dataset.

        Args:
            data_dir: Root directory containing the dataset
            split: Dataset split ("train", "val", "test")
            transform: Image transformations
            target_transform: Target transformations
            include_confounders: Whether to load confounder information
            confounder_config: Configuration for confounder handling
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.include_confounders = include_confounders
        self.confounder_config = confounder_config or {}

        # Will be set by subclasses
        self.images = []
        self.labels = []
        self.confounders = []
        self.metadata = []

        # Label mappings
        self.class_to_idx = {"normal": 0, "pneumonia": 1}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Load dataset
        self._load_dataset()

    def _load_dataset(self):
        """Load dataset - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _load_dataset")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict]]:
        """
        Get dataset item with image, label, and optional confounders.

        Args:
            idx: Index of item to retrieve

        Returns:
            Dictionary containing:
                - image: Processed image tensor
                - label: Class label
                - confounders: Dictionary of confounder values (if enabled)
                - metadata: Additional metadata
        """
        # Load image
        image_path = self.images[idx]
        image = self._load_image(image_path)

        # Get label
        label = self.labels[idx]
        if isinstance(label, str):
            label = self.class_to_idx.get(label.lower(), 0)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        # Prepare output
        item = {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'index': idx
        }

        # Add confounders if available
        if self.include_confounders and idx < len(self.confounders):
            confounders = self.confounders[idx]
            processed_confounders = self._process_confounders(confounders)
            item['confounders'] = processed_confounders

        # Add metadata
        if idx < len(self.metadata):
            item['metadata'] = self.metadata[idx]

        return item

    def _load_image(self, image_path: str) -> Image.Image:
        """Load image from file path."""
        try:
            if image_path.endswith('.dcm'):
                # Load DICOM image
                dicom = pydicom.dcmread(image_path)
                image_array = dicom.pixel_array

                # Normalize to 0-255 range
                image_array = image_array.astype(np.float32)
                image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)
                image_array = (image_array * 255).astype(np.uint8)

                # Convert to PIL Image
                image = Image.fromarray(image_array)
                if image.mode != 'RGB':
                    image = image.convert('RGB')

            else:
                # Load regular image
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')

        except Exception as e:
            warnings.warn(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (224, 224), color='black')

        return image

    def _process_confounders(self, confounders: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process confounder values into tensors."""
        processed = {}

        for name, value in confounders.items():
            if isinstance(value, (int, float)):
                processed[name] = torch.tensor(float(value), dtype=torch.float32)
            elif isinstance(value, str):
                # Handle categorical confounders
                if name in self.confounder_config:
                    categories = self.confounder_config[name].get('categories', [])
                    if value in categories:
                        idx = categories.index(value)
                        processed[name] = torch.tensor(idx, dtype=torch.long)
                    else:
                        processed[name] = torch.tensor(0, dtype=torch.long)  # Default category
                else:
                    # Try to convert to float
                    try:
                        processed[name] = torch.tensor(float(value), dtype=torch.float32)
                    except ValueError:
                        processed[name] = torch.tensor(0, dtype=torch.long)
            elif isinstance(value, (list, np.ndarray)):
                processed[name] = torch.tensor(value, dtype=torch.float32)
            else:
                # Default handling
                processed[name] = torch.tensor(0, dtype=torch.float32)

        return processed

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced datasets."""
        label_counts = np.bincount(self.labels)
        total_samples = len(self.labels)

        # Inverse frequency weighting
        weights = total_samples / (len(label_counts) * label_counts)
        return torch.tensor(weights, dtype=torch.float32)

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            'total_samples': len(self),
            'class_distribution': {},
            'split': self.split
        }

        # Class distribution
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            class_name = self.idx_to_class.get(label, f"class_{label}")
            stats['class_distribution'][class_name] = {
                'count': int(count),
                'percentage': float(count / len(self.labels) * 100)
            }

        # Confounder statistics
        if self.include_confounders and self.confounders:
            stats['confounders'] = {}
            for i, confounder_dict in enumerate(self.confounders[:100]):  # Sample first 100
                for name, value in confounder_dict.items():
                    if name not in stats['confounders']:
                        stats['confounders'][name] = {'values': [], 'type': None}

                    stats['confounders'][name]['values'].append(value)

                    if stats['confounders'][name]['type'] is None:
                        if isinstance(value, (int, float)):
                            stats['confounders'][name]['type'] = 'numeric'
                        else:
                            stats['confounders'][name]['type'] = 'categorical'

        return stats


class NIHChestXray14(ChestXrayDataset):
    """
    NIH ChestX-ray14 dataset with pneumonia labels and confounders.
    """

    def __init__(self, data_dir: str, **kwargs):
        self.labels_file = os.path.join(data_dir, "Data_Entry_2017_v2020.csv")
        super().__init__(data_dir, **kwargs)

    def _load_dataset(self):
        """Load NIH ChestX-ray14 dataset."""
        if not os.path.exists(self.labels_file):
            raise FileNotFoundError(f"Labels file not found: {self.labels_file}")

        # Load metadata
        df = pd.read_csv(self.labels_file)

        # Filter for frontal X-rays
        df = df[df['View Position'].isin(['PA', 'AP'])]

        # Create pneumonia labels
        df['pneumonia'] = df['Finding Labels'].str.contains('Pneumonia').astype(int)

        # Split dataset
        if self.split == "train":
            df = df[df['Patient ID'] % 10 < 7]  # 70% for training
        elif self.split == "val":
            df = df[df['Patient ID'] % 10 == 7]  # 10% for validation
        elif self.split == "test":
            df = df[df['Patient ID'] % 10 > 7]   # 20% for testing

        # Extract image paths and labels
        self.images = [os.path.join(self.data_dir, "images", fname) for fname in df['Image Index']]
        self.labels = df['pneumonia'].tolist()

        # Extract confounders
        if self.include_confounders:
            self.confounders = []
            for _, row in df.iterrows():
                confounders = {
                    'age': self._extract_age(row.get('Patient Age', 'Unknown')),
                    'sex': row.get('Patient Gender', 'Unknown'),
                    'view_position': row.get('View Position', 'Unknown'),
                    'follow_up': row.get('Follow-up #', 0)
                }
                self.confounders.append(confounders)

        # Store metadata
        self.metadata = df.to_dict('records')

        print(f"Loaded NIH ChestX-ray14 {self.split} split: {len(self.images)} images")
        print(f"Pneumonia prevalence: {np.mean(self.labels):.3f}")

    def _extract_age(self, age_str: str) -> float:
        """Extract numeric age from age string."""
        if isinstance(age_str, str) and age_str.endswith('Y'):
            try:
                return float(age_str[:-1])
            except ValueError:
                pass
        return 0.0  # Default age


class RSNAPneumonia(ChestXrayDataset):
    """
    RSNA Pneumonia Detection Challenge dataset.
    """

    def __init__(self, data_dir: str, **kwargs):
        self.labels_file = os.path.join(data_dir, "stage_2_detailed_class_info.csv")
        super().__init__(data_dir, **kwargs)

    def _load_dataset(self):
        """Load RSNA pneumonia dataset."""
        if not os.path.exists(self.labels_file):
            raise FileNotFoundError(f"Labels file not found: {self.labels_file}")

        df = pd.read_csv(self.labels_file)

        # Convert class labels to binary (pneumonia vs normal)
        df['pneumonia'] = (df['class'] != 'Normal').astype(int)

        # Split dataset (simple random split for RSNA)
        np.random.seed(42)
        indices = np.random.permutation(len(df))

        if self.split == "train":
            df = df.iloc[indices[:int(0.7 * len(df))]]
        elif self.split == "val":
            df = df.iloc[indices[int(0.7 * len(df)):int(0.8 * len(df))]]
        elif self.split == "test":
            df = df.iloc[indices[int(0.8 * len(df)):]]

        # Extract image paths and labels
        self.images = [os.path.join(self.data_dir, "stage_2_train_images", f"{pid}.dcm") 
                      for pid in df['patientId']]
        self.labels = df['pneumonia'].tolist()

        # Extract confounders (limited for RSNA dataset)
        if self.include_confounders:
            self.confounders = []
            for _, row in df.iterrows():
                confounders = {
                    'patient_id': hash(row['patientId']) % 1000,  # Anonymous patient identifier
                    'class_type': 1 if row['class'] == 'Lung Opacity' else 0
                }
                self.confounders.append(confounders)

        self.metadata = df.to_dict('records')

        print(f"Loaded RSNA {self.split} split: {len(self.images)} images")
        print(f"Pneumonia prevalence: {np.mean(self.labels):.3f}")


class PediatricDataset(ChestXrayDataset):
    """
    Pediatric chest X-ray dataset for children aged 1-5 years.
    """

    def __init__(self, data_dir: str, **kwargs):
        super().__init__(data_dir, **kwargs)

    def _load_dataset(self):
        """Load pediatric dataset."""
        # Look for organized directory structure
        normal_dir = os.path.join(self.data_dir, "NORMAL")
        pneumonia_dir = os.path.join(self.data_dir, "PNEUMONIA")

        if not (os.path.exists(normal_dir) and os.path.exists(pneumonia_dir)):
            raise FileNotFoundError("Expected NORMAL and PNEUMONIA directories not found")

        images = []
        labels = []

        # Load normal images
        for fname in os.listdir(normal_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(os.path.join(normal_dir, fname))
                labels.append(0)  # Normal

        # Load pneumonia images
        for fname in os.listdir(pneumonia_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(os.path.join(pneumonia_dir, fname))
                labels.append(1)  # Pneumonia

        # Split dataset
        combined = list(zip(images, labels))
        np.random.seed(42)
        np.random.shuffle(combined)

        total_size = len(combined)
        if self.split == "train":
            combined = combined[:int(0.7 * total_size)]
        elif self.split == "val":
            combined = combined[int(0.7 * total_size):int(0.8 * total_size)]
        elif self.split == "test":
            combined = combined[int(0.8 * total_size):]

        self.images, self.labels = zip(*combined) if combined else ([], [])
        self.images = list(self.images)
        self.labels = list(self.labels)

        # Generate synthetic confounders for pediatric dataset
        if self.include_confounders:
            self.confounders = []
            for i, _ in enumerate(self.images):
                # Simulate pediatric patient characteristics
                confounders = {
                    'age_months': np.random.randint(12, 60),  # 1-5 years in months
                    'sex': np.random.choice(['M', 'F']),
                    'weight_kg': np.random.normal(15, 3),  # Typical pediatric weight
                    'hospital_type': np.random.choice(['public', 'private'])
                }
                self.confounders.append(confounders)

        print(f"Loaded Pediatric {self.split} split: {len(self.images)} images")
        print(f"Pneumonia prevalence: {np.mean(self.labels):.3f}")


def create_dataloader(
    dataset: ChestXrayDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create PyTorch DataLoader for chest X-ray dataset.

    Args:
        dataset: ChestXrayDataset instance
        batch_size: Batch size for loading
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer

    Returns:
        PyTorch DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_fn
    )


def _collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for handling confounders."""
    # Standard collation for images and labels
    images = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    indices = torch.tensor([item['index'] for item in batch])

    result = {
        'image': images,
        'label': labels,
        'index': indices
    }

    # Handle confounders
    if 'confounders' in batch[0]:
        confounders = {}
        confounder_names = batch[0]['confounders'].keys()

        for name in confounder_names:
            values = [item['confounders'][name] for item in batch]
            if all(isinstance(v, torch.Tensor) for v in values):
                confounders[name] = torch.stack(values)
            else:
                # Handle mixed types
                confounders[name] = torch.tensor(values)

        result['confounders'] = confounders

    return result
