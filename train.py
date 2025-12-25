"""
Training module using PyTorch Lightning.
Handles model training, validation, and optimization.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import torchmetrics


class SpeechCommandsModule(pl.LightningModule):
    """
    PyTorch Lightning module for speech commands classification.
    """
    
    def __init__(self, model, num_classes=35, learning_rate=0.001, 
                 optimizer='adam', scheduler='reduce_on_plateau',
                 weight_decay=0.0001):
        """
        Args:
            model: PyTorch model for classification
            num_classes: Number of output classes
            learning_rate: Initial learning rate
            optimizer: Optimizer type ('adam' or 'sgd')
            scheduler: Learning rate scheduler type
            weight_decay: Weight decay for regularization
        """
        super(SpeechCommandsModule, self).__init__()
        
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer
        self.scheduler_type = scheduler
        self.weight_decay = weight_decay
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        
        # Top-5 accuracy (useful for 35 classes)
        self.val_acc_top5 = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
        self.test_acc_top5 = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """
        Training step for a single batch.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of the batch
            
        Returns:
            Loss value
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        
        # Calculate accuracy
        self.train_acc(outputs, targets)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for a single batch.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of the batch
            
        Returns:
            Loss value
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        
        # Calculate accuracy
        self.val_acc(outputs, targets)
        self.val_acc_top5(outputs, targets)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc_top5', self.val_acc_top5, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        Test step for a single batch.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of the batch
            
        Returns:
            Loss value
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        
        # Calculate accuracy
        self.test_acc(outputs, targets)
        self.test_acc_top5(outputs, targets)
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        self.log('test_acc_top5', self.test_acc_top5, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        # Select optimizer
        if self.optimizer_type == 'adam':
            optimizer = Adam(self.parameters(), 
                           lr=self.learning_rate,
                           weight_decay=self.weight_decay)
        elif self.optimizer_type == 'sgd':
            optimizer = SGD(self.parameters(), 
                          lr=self.learning_rate,
                          momentum=0.9,
                          weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
        
        # Select scheduler
        if self.scheduler_type == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', 
                                         factor=0.5, patience=5, 
                                         verbose=True)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        elif self.scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        elif self.scheduler_type == 'step':
            scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            return optimizer
    
    def predict_step(self, batch, batch_idx):
        """
        Prediction step for inference.
        
        Args:
            batch: Input batch
            batch_idx: Index of the batch
            
        Returns:
            Predictions (class indices)
        """
        if isinstance(batch, tuple):
            inputs, _ = batch
        else:
            inputs = batch
            
        outputs = self(inputs)
        predictions = torch.argmax(outputs, dim=1)
        
        return predictions


def train_model(model, train_loader, val_loader, num_classes=35,
                max_epochs=50, learning_rate=0.001, optimizer='adam',
                scheduler='reduce_on_plateau', callbacks=None,
                gpus=None, precision=32):
    """
    Train the model using PyTorch Lightning.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_classes: Number of output classes
        max_epochs: Maximum number of epochs
        learning_rate: Initial learning rate
        optimizer: Optimizer type
        scheduler: Scheduler type
        callbacks: List of PyTorch Lightning callbacks
        gpus: Number of GPUs to use (None for CPU)
        precision: Training precision (16 or 32)
        
    Returns:
        Trained PyTorch Lightning module and trainer
    """
    
    # Create Lightning module
    pl_module = SpeechCommandsModule(
        model=model,
        num_classes=num_classes,
        learning_rate=learning_rate,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    # Determine device type
    if gpus is None or gpus == 0:
        accelerator = 'cpu'
        devices = 'auto'  # Let Lightning auto-detect CPU cores
    elif torch.cuda.is_available():
        accelerator = 'gpu'
        devices = gpus if isinstance(gpus, int) else len(gpus)
    elif torch.backends.mps.is_available():
        accelerator = 'mps'
        devices = 1
    else:
        accelerator = 'cpu'
        devices = 'auto'
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train the model
    trainer.fit(pl_module, train_loader, val_loader)
    
    return pl_module, trainer

