"""
Training framework for CausalXray model with progressive training strategy.

This module implements the training loop with support for multi-objective optimization,
progressive training phases, and comprehensive evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from collections import defaultdict
import json
from tqdm import tqdm

from .losses import CausalLoss
from .metrics import CausalMetrics
from ..utils.logging import setup_logger


class CausalTrainer:
    """
    Main trainer class for CausalXray model with causal reasoning capabilities.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: str = "cuda",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize CausalXray trainer.

        Args:
            model: CausalXray model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device for training ("cuda" or "cpu")
            logger: Logger instance
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = logger or setup_logger("CausalTrainer")

        # Training components
        self.criterion = CausalLoss(config.get('loss', {}))
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.metrics = CausalMetrics()

        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.training_history = defaultdict(list)

        # Progressive training configuration
        self.progressive_config = config.get('progressive_training', {})
        self.phase_epochs = self.progressive_config.get('phase_epochs', [50, 50, 50])

        # Logging and checkpointing
        self.log_dir = config.get('log_dir', './logs')
        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        self._setup_logging()

        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on configuration."""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam')

        if optimizer_type.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 1e-3),
                weight_decay=optimizer_config.get('weight_decay', 1e-4),
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_type.lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 1e-2),
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        elif optimizer_type.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 1e-3),
                weight_decay=optimizer_config.get('weight_decay', 1e-4),
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def _setup_scheduler(self) -> Optional[Any]:
        """Setup learning rate scheduler."""
        scheduler_config = self.config.get('scheduler', {})
        if not scheduler_config.get('enabled', False):
            return None

        scheduler_type = scheduler_config.get('type', 'cosine')

        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', 100),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10)
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def _setup_logging(self):
        """Setup tensorboard logging."""
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def train(self, num_epochs: int, resume_from: Optional[str] = None) -> Dict[str, List]:
        """
        Main training loop with progressive training strategy.

        Args:
            num_epochs: Total number of epochs to train
            resume_from: Path to checkpoint to resume from

        Returns:
            Training history
        """
        if resume_from:
            self._load_checkpoint(resume_from)

        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Progressive training phases: {self.phase_epochs}")

        start_time = time.time()

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # Determine training phase
            phase = self._get_training_phase(epoch)
            # type: ignore for model attributes
            if getattr(self.model, 'training_phase', None) != phase:
                if hasattr(self.model, 'set_training_phase'):
                    self.model.set_training_phase(phase)  # type: ignore
                self.logger.info(f"Switched to training phase: {phase}")

            # Training step
            train_metrics = self._train_epoch()

            # Validation step
            val_metrics = self._validate_epoch()

            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['auc'])
                else:
                    self.scheduler.step()

            # Log metrics
            self._log_metrics(train_metrics, val_metrics, epoch)

            # Save checkpoint
            if self._should_save_checkpoint(val_metrics):
                self._save_checkpoint(epoch, val_metrics)

            # Early stopping check
            if self._should_early_stop(val_metrics):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")

        self.writer.close()
        return dict(self.training_history)

    def _get_training_phase(self, epoch: int) -> str:
        """Determine current training phase based on epoch."""
        cumulative_epochs = np.cumsum([0] + self.phase_epochs)

        if epoch < cumulative_epochs[1]:
            return "backbone"
        elif epoch < cumulative_epochs[2]:
            return "causal"
        else:
            return "full"

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = defaultdict(float)
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Training Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            confounders = {k: v.to(self.device) for k, v in batch.get('confounders', {}).items()}

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, confounders=confounders)

            # Compute loss
            # type: ignore for model method
            loss_dict = self.model.compute_loss(
                outputs, labels, confounders, 
                loss_weights=self.config.get('loss_weights', {})
            )  # type: ignore

            total_loss = loss_dict['total_loss']

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip']
                )

            self.optimizer.step()

            # Update metrics
            batch_metrics = self.metrics.compute_batch_metrics(
                outputs['probabilities'], labels, loss_dict
            )

            for key, value in batch_metrics.items():
                epoch_metrics[key] += value

            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'acc': f"{batch_metrics.get('accuracy', 0):.4f}"
            })

        # Average metrics over epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        return dict(epoch_metrics)

    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_metrics = defaultdict(float)
        num_batches = 0

        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validation Epoch {self.current_epoch}")

            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                confounders = {k: v.to(self.device) for k, v in batch.get('confounders', {}).items()}

                # Forward pass
                outputs = self.model(images, confounders=confounders)

                # Compute loss
                # type: ignore for model method
                loss_dict = self.model.compute_loss(
                    outputs, labels, confounders,
                    loss_weights=self.config.get('loss_weights', {})
                )  # type: ignore

                # Compute metrics
                batch_metrics = self.metrics.compute_batch_metrics(
                    outputs['probabilities'], labels, loss_dict
                )

                for key, value in batch_metrics.items():
                    epoch_metrics[key] += value

                # Collect for epoch-level metrics
                all_predictions.extend(torch.argmax(outputs['probabilities'], dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(outputs['probabilities'].cpu().numpy())

                num_batches += 1

        # Average metrics over epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        # Compute epoch-level metrics
        epoch_level_metrics = self.metrics.compute_epoch_metrics(
            np.array(all_probabilities), 
            np.array(all_labels)
        )

        # Only update with float values
        epoch_metrics.update({k: float(v) if not isinstance(v, float) else v for k, v in epoch_level_metrics.items() if isinstance(v, (float, int, np.floating))})

        return dict(epoch_metrics)

    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """Log metrics to tensorboard and console."""
        # Console logging
        self.logger.info(f"Epoch {epoch}:")
        self.logger.info(f"  Train - Loss: {train_metrics.get('total_loss', 0):.4f}, "
                        f"Acc: {train_metrics.get('accuracy', 0):.4f}")
        self.logger.info(f"  Val   - Loss: {val_metrics.get('total_loss', 0):.4f}, "
                        f"Acc: {val_metrics.get('accuracy', 0):.4f}, "
                        f"AUC: {val_metrics.get('auc', 0):.4f}")

        # Tensorboard logging
        for key, value in train_metrics.items():
            self.writer.add_scalar(f"train/{key}", value, epoch)
            self.training_history[f"train_{key}"].append(value)

        for key, value in val_metrics.items():
            self.writer.add_scalar(f"val/{key}", value, epoch)
            self.training_history[f"val_{key}"].append(value)

        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar("learning_rate", current_lr, epoch)
        self.training_history["learning_rate"].append(current_lr)

    def _should_save_checkpoint(self, val_metrics: Dict) -> bool:
        """Check if current model should be saved."""
        current_metric = val_metrics.get('auc', 0)
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            return True
        return False

    def _save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'metrics': metrics,
            'config': self.config,
            'training_history': dict(self.training_history)
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save best model
        best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)

        # Save epoch checkpoint
        epoch_path = os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_path)

        self.logger.info(f"Checkpoint saved: {best_path}")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint.get('best_metric', 0)
        self.training_history = defaultdict(list, checkpoint.get('training_history', {}))

        self.logger.info(f"Resumed training from epoch {self.current_epoch}")

    def _should_early_stop(self, val_metrics: Dict) -> bool:
        """Check if early stopping criteria are met."""
        early_stop_config = self.config.get('early_stopping', {})
        if not early_stop_config.get('enabled', False):
            return False

        patience = early_stop_config.get('patience', 20)
        min_delta = early_stop_config.get('min_delta', 1e-4)

        metric_name = early_stop_config.get('metric', 'auc')
        metric_history = self.training_history.get(f"val_{metric_name}", [])

        if len(metric_history) < patience:
            return False

        # Check if no improvement in last 'patience' epochs
        recent_best = max(metric_history[-patience:])
        overall_best = max(metric_history[:-patience]) if len(metric_history) > patience else 0

        return recent_best - overall_best < min_delta


class ProgressiveTrainer(CausalTrainer):
    """
    Specialized trainer for progressive training with phase-specific optimizations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Phase-specific configurations
        self.phase_configs = {
            'backbone': {
                'loss_weights': {'classification': 1.0, 'disentanglement': 0.0, 'domain': 0.0},
                'lr_multiplier': 1.0
            },
            'causal': {
                'loss_weights': {'classification': 0.7, 'disentanglement': 0.3, 'domain': 0.1},
                'lr_multiplier': 0.5
            },
            'full': {
                'loss_weights': {'classification': 1.0, 'disentanglement': 0.3, 'domain': 0.1},
                'lr_multiplier': 0.1
            }
        }

    def _train_epoch(self) -> Dict[str, float]:
        """Enhanced training epoch with phase-specific optimizations."""
        current_phase = getattr(self.model, 'training_phase', 'backbone')
        phase_config = self.phase_configs.get(str(current_phase), {})

        # Adjust learning rate for phase
        lr_multiplier = phase_config.get('lr_multiplier', 1.0)
        base_lr = self.config.get('optimizer', {}).get('lr', 1e-3)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = base_lr * lr_multiplier

        # Update loss weights
        loss_weights = phase_config.get('loss_weights', {})

        # Call parent training method with phase-specific modifications
        return super()._train_epoch()

    def evaluate_cross_domain(
        self, 
        test_loaders: Dict[str, DataLoader]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model across different domains for domain generalization assessment.

        Args:
            test_loaders: Dictionary mapping domain names to test data loaders

        Returns:
            Dictionary of metrics for each domain
        """
        self.model.eval()
        domain_results = {}

        with torch.no_grad():
            for domain_name, test_loader in test_loaders.items():
                self.logger.info(f"Evaluating on {domain_name} domain...")

                all_predictions = []
                all_labels = []
                all_probabilities = []

                for batch in tqdm(test_loader, desc=f"Testing {domain_name}"):
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)

                    outputs = self.model(images)

                    all_predictions.extend(torch.argmax(outputs['probabilities'], dim=1).cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(outputs['probabilities'].cpu().numpy())

                # Compute domain metrics
                domain_metrics = self.metrics.compute_epoch_metrics(
                    np.array(all_probabilities),
                    np.array(all_labels)
                )

                domain_results[domain_name] = domain_metrics

                self.logger.info(f"{domain_name} - "
                               f"Accuracy: {domain_metrics['accuracy']:.4f}, "
                               f"AUC: {domain_metrics['auc']:.4f}")

        return domain_results
