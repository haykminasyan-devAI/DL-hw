"""
Custom callbacks for PyTorch Lightning training.
Implements various training callbacks for monitoring and control.
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (ModelCheckpoint, EarlyStopping, 
                                        LearningRateMonitor, Callback)
import os


class MetricsLogger(Callback):
    """Custom callback to log metrics during training."""
    
    def __init__(self):
        super().__init__()
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""
        if 'train_loss_epoch' in trainer.callback_metrics:
            self.metrics['train_loss'].append(
                trainer.callback_metrics['train_loss_epoch'].item()
            )
        if 'train_acc' in trainer.callback_metrics:
            self.metrics['train_acc'].append(
                trainer.callback_metrics['train_acc'].item()
            )
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation epoch ends."""
        if 'val_loss' in trainer.callback_metrics:
            self.metrics['val_loss'].append(
                trainer.callback_metrics['val_loss'].item()
            )
        if 'val_acc' in trainer.callback_metrics:
            self.metrics['val_acc'].append(
                trainer.callback_metrics['val_acc'].item()
            )


class PrintMetrics(Callback):
    """Print metrics at the end of each epoch."""
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Print training metrics."""
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        
        print(f"\nEpoch {epoch}:")
        if 'train_loss_epoch' in metrics:
            print(f"  Train Loss: {metrics['train_loss_epoch']:.4f}")
        if 'train_acc' in metrics:
            print(f"  Train Acc:  {metrics['train_acc']:.4f}")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Print validation metrics."""
        metrics = trainer.callback_metrics
        
        if 'val_loss' in metrics:
            print(f"  Val Loss:   {metrics['val_loss']:.4f}")
        if 'val_acc' in metrics:
            print(f"  Val Acc:    {metrics['val_acc']:.4f}")
        if 'val_acc_top5' in metrics:
            print(f"  Val Top-5:  {metrics['val_acc_top5']:.4f}")


def get_callbacks(checkpoint_dir='checkpoints', monitor='val_acc', 
                  patience=10, min_delta=0.001, mode='max'):
    """
    Create a list of standard callbacks for training.
    
    Args:
        checkpoint_dir: Directory to save model checkpoints
        monitor: Metric to monitor for checkpointing
        patience: Patience for early stopping
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' or 'max' for the monitored metric
        
    Returns:
        List of PyTorch Lightning callbacks
    """
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = []
    
    # Model checkpoint - save best model
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best-model-{epoch:02d}-{val_acc:.4f}',
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor=monitor,
        patience=patience,
        min_delta=min_delta,
        mode=mode,
        verbose=True
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Custom metrics logger
    metrics_logger = MetricsLogger()
    callbacks.append(metrics_logger)
    
    # Print metrics
    print_metrics = PrintMetrics()
    callbacks.append(print_metrics)
    
    return callbacks


def get_simple_callbacks(checkpoint_dir='checkpoints'):
    """
    Get a simple set of callbacks for basic training.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        List of basic callbacks
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = []
    
    # Save last checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='model-{epoch:02d}',
        save_last=True,
        every_n_epochs=1
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Print metrics
    print_metrics = PrintMetrics()
    callbacks.append(print_metrics)
    
    return callbacks

