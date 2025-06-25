"""
Logging utilities for CausalXray framework with support for experiment tracking,
progress monitoring, and comprehensive logging of training and evaluation metrics.

This module provides structured logging capabilities that integrate with
popular experiment tracking platforms like Weights & Biases and TensorBoard.
"""

import logging
import sys
import os
from typing import Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import json
import time
from collections import defaultdict


class CausalLogger:
    """
    Enhanced logger for CausalXray experiments with structured logging capabilities.
    """

    def __init__(
        self,
        name: str,
        log_dir: str,
        level: str = "INFO",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_config: Optional[Dict] = None
    ):
        """
        Initialize CausalLogger.

        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level
            use_tensorboard: Whether to use TensorBoard logging
            use_wandb: Whether to use Weights & Biases logging
            wandb_config: Configuration for W&B
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup basic logger
        self.logger = self._setup_logger(level)

        # Experiment tracking
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.wandb_config = wandb_config or {}

        # Initialize trackers
        self.tb_writer = None
        self.wandb_run = None

        if use_tensorboard:
            self._setup_tensorboard()

        if use_wandb:
            self._setup_wandb()

        # Metrics storage
        self.metrics_history = defaultdict(list)
        self.current_epoch = 0

        # Timing
        self.start_time = None
        self.epoch_start_time = None

    def _setup_logger(self, level: str) -> logging.Logger:
        """Setup basic Python logger."""
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, level.upper()))

        # Clear existing handlers
        logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler
        log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        return logger

    def _setup_tensorboard(self):
        """Setup TensorBoard logging."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = self.log_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)

            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            self.logger.info(f"TensorBoard logging enabled: {tb_dir}")

        except ImportError:
            self.logger.warning("TensorBoard not available. Install with: pip install tensorboard")
            self.use_tensorboard = False

    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        try:
            import wandb

            wandb_config = self.wandb_config.copy()
            if 'project' not in wandb_config:
                wandb_config['project'] = 'causalxray'
            if 'name' not in wandb_config:
                wandb_config['name'] = f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            self.wandb_run = wandb.init(**wandb_config)
            self.logger.info(f"W&B logging enabled: {wandb_config['project']}/{wandb_config['name']}")

        except ImportError:
            self.logger.warning("Weights & Biases not available. Install with: pip install wandb")
            self.use_wandb = False

    def start_training(self, total_epochs: int):
        """Mark the start of training."""
        self.start_time = time.time()
        self.logger.info(f"Starting training for {total_epochs} epochs")

        if self.wandb_run:
            self.wandb_run.config.update({"total_epochs": total_epochs})

    def start_epoch(self, epoch: int):
        """Mark the start of an epoch."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        self.logger.info(f"Starting epoch {epoch}")

    def end_epoch(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Mark the end of an epoch and log metrics."""
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0

        # Log to console
        self.logger.info(f"Epoch {self.current_epoch} completed in {epoch_time:.2f}s")
        self.logger.info(f"Train metrics: {self._format_metrics(train_metrics)}")
        self.logger.info(f"Val metrics: {self._format_metrics(val_metrics)}")

        # Store metrics
        for key, value in train_metrics.items():
            self.metrics_history[f"train_{key}"].append(value)

        for key, value in val_metrics.items():
            self.metrics_history[f"val_{key}"].append(value)

        self.metrics_history["epoch_time"].append(epoch_time)

        # Log to TensorBoard
        if self.tb_writer:
            for key, value in train_metrics.items():
                self.tb_writer.add_scalar(f"train/{key}", value, self.current_epoch)

            for key, value in val_metrics.items():
                self.tb_writer.add_scalar(f"val/{key}", value, self.current_epoch)

            self.tb_writer.add_scalar("epoch_time", epoch_time, self.current_epoch)

        # Log to W&B
        if self.wandb_run:
            wandb_metrics = {
                "epoch": self.current_epoch,
                "epoch_time": epoch_time
            }

            for key, value in train_metrics.items():
                wandb_metrics[f"train_{key}"] = value

            for key, value in val_metrics.items():
                wandb_metrics[f"val_{key}"] = value

            self.wandb_run.log(wandb_metrics)

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a single metric."""
        step = step or self.current_epoch

        self.logger.info(f"{name}: {value:.6f}")

        if self.tb_writer:
            self.tb_writer.add_scalar(name, value, step)

        if self.wandb_run:
            self.wandb_run.log({name: value, "step": step})

        self.metrics_history[name].append(value)

    def log_metrics(self, metrics: Dict[str, float], prefix: str = "", step: Optional[int] = None):
        """Log multiple metrics."""
        step = step or self.current_epoch

        for name, value in metrics.items():
            full_name = f"{prefix}_{name}" if prefix else name
            self.log_metric(full_name, value, step)

    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters."""
        self.logger.info(f"Hyperparameters: {json.dumps(hparams, indent=2)}")

        if self.tb_writer:
            # TensorBoard expects string values for hyperparameters
            hparams_str = {k: str(v) for k, v in hparams.items()}
            self.tb_writer.add_hparams(hparams_str, {})

        if self.wandb_run:
            self.wandb_run.config.update(hparams)

    def log_model_summary(self, model, input_size: tuple):
        """Log model architecture summary."""
        try:
            from torchinfo import summary

            model_summary = summary(model, input_size=input_size, verbose=0)
            self.logger.info(f"Model Summary:\n{model_summary}")

            # Log parameter count
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.log_metric("total_parameters", total_params)
            self.log_metric("trainable_parameters", trainable_params)

        except ImportError:
            self.logger.warning("torchinfo not available for model summary")

    def log_image(self, name: str, image, step: Optional[int] = None):
        """Log image to TensorBoard."""
        step = step or self.current_epoch

        if self.tb_writer:
            self.tb_writer.add_image(name, image, step)

        if self.wandb_run:
            import wandb
            self.wandb_run.log({name: wandb.Image(image), "step": step})

    def log_histogram(self, name: str, values, step: Optional[int] = None):
        """Log histogram to TensorBoard."""
        step = step or self.current_epoch

        if self.tb_writer:
            self.tb_writer.add_histogram(name, values, step)

    def log_confusion_matrix(self, cm, class_names: list, step: Optional[int] = None):
        """Log confusion matrix."""
        step = step or self.current_epoch

        if self.tb_writer:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')

            self.tb_writer.add_figure('confusion_matrix', fig, step)
            plt.close(fig)

        if self.wandb_run:
            import wandb
            self.wandb_run.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None, y_true=None, preds=None, class_names=class_names
                ),
                "step": step
            })

    def save_metrics_history(self, filepath: Optional[str] = None):
        """Save metrics history to JSON file."""
        if filepath is None:
            filepath = self.log_dir / "metrics_history.json"

        metrics_dict = dict(self.metrics_history)

        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        self.logger.info(f"Metrics history saved to: {filepath}")

    def end_training(self):
        """Mark the end of training."""
        if self.start_time:
            total_time = time.time() - self.start_time
            self.logger.info(f"Training completed in {total_time:.2f} seconds")

            if self.wandb_run:
                self.wandb_run.log({"total_training_time": total_time})

        # Save metrics
        self.save_metrics_history()

        # Close trackers
        if self.tb_writer:
            self.tb_writer.close()

        if self.wandb_run:
            self.wandb_run.finish()

    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics for console display."""
        return ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)


def setup_logger(
    name: str,
    log_dir: str = "./logs",
    level: str = "INFO",
    use_tensorboard: bool = True,
    use_wandb: bool = False,
    wandb_config: Optional[Dict] = None
) -> CausalLogger:
    """
    Setup logger for CausalXray experiments.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        use_tensorboard: Whether to use TensorBoard
        use_wandb: Whether to use W&B
        wandb_config: W&B configuration

    Returns:
        Configured CausalLogger instance
    """
    return CausalLogger(
        name=name,
        log_dir=log_dir,
        level=level,
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
        wandb_config=wandb_config
    )


class ProgressTracker:
    """
    Simple progress tracker for long-running operations.
    """

    def __init__(self, total_steps: int, description: str = "Progress"):
        """
        Initialize progress tracker.

        Args:
            total_steps: Total number of steps
            description: Description of the operation
        """
        self.total_steps = total_steps
        self.description = description
        self.current_step = 0
        self.start_time = time.time()

    def update(self, steps: int = 1):
        """Update progress by specified number of steps."""
        self.current_step += steps

    def get_progress_string(self) -> str:
        """Get formatted progress string."""
        elapsed_time = time.time() - self.start_time
        progress_percent = (self.current_step / self.total_steps) * 100

        if self.current_step > 0:
            eta = (elapsed_time / self.current_step) * (self.total_steps - self.current_step)
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "ETA: N/A"

        return (f"{self.description}: {self.current_step}/{self.total_steps} "
                f"({progress_percent:.1f}%) - {eta_str}")

    def print_progress(self):
        """Print current progress."""
        print(f"\r{self.get_progress_string()}", end="", flush=True)

    def finish(self):
        """Mark completion and print final stats."""
        elapsed_time = time.time() - self.start_time
        print(f"\n{self.description} completed in {elapsed_time:.2f}s")


def log_system_info(logger: logging.Logger):
    """Log system information for reproducibility."""
    import platform
    import torch

    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python: {platform.python_version()}")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"  CUDA Version: {torch.version.cuda}")
        logger.info(f"  GPU Count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
