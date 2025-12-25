"""
Neural network architectures for speech command recognition.
Implements various CNN models for keyword spotting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_layers import (AttentionLayer, ResidualBlock, SeparableConv2d, 
                          SqueezeExcitation, TemporalAveragePooling)


class SimpleCNN(nn.Module):
    """
    Simple CNN model for speech commands classification.
    Good starting point for baseline results.
    """
    
    def __init__(self, num_classes=35, input_channels=1, dropout=0.5):
        """
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (1 for mono audio features)
            dropout: Dropout probability
        """
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, channels, freq, time)
            
        Returns:
            Logits (batch, num_classes)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class DeepCNN(nn.Module):
    """
    Deeper CNN model based on the baseline from the paper.
    Better performance but more computationally intensive.
    """
    
    def __init__(self, num_classes=35, input_channels=1, dropout=0.5):
        """
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels
            dropout: Dropout probability
        """
        super(DeepCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """Forward pass."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class ResNet(nn.Module):
    """
    ResNet-style architecture for speech commands.
    Uses residual connections for deeper networks.
    """
    
    def __init__(self, num_classes=35, input_channels=1, dropout=0.3):
        """
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels
            dropout: Dropout probability
        """
        super(ResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.layer1 = ResidualBlock(64, 64, stride=1, dropout=dropout)
        self.layer2 = ResidualBlock(64, 128, stride=2, dropout=dropout)
        self.layer3 = ResidualBlock(128, 256, stride=2, dropout=dropout)
        self.layer4 = ResidualBlock(256, 512, stride=2, dropout=dropout)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        """Forward pass."""
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class AttentionCNN(nn.Module):
    """
    CNN with attention mechanism for improved feature aggregation.
    """
    
    def __init__(self, num_classes=35, input_channels=1, dropout=0.5):
        """
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels
            dropout: Dropout probability
        """
        super(AttentionCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.dropout = nn.Dropout(dropout)
        
        # Attention layer
        self.attention = AttentionLayer(256)
        
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """Forward pass with attention."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout(x)
        
        # Apply attention
        x = self.attention(x)
        
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class EfficientCNN(nn.Module):
    """
    Efficient CNN using separable convolutions for mobile deployment.
    Smaller model size and fewer computations.
    """
    
    def __init__(self, num_classes=35, input_channels=1, dropout=0.3):
        """
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels
            dropout: Dropout probability
        """
        super(EfficientCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Separable convolutions for efficiency
        self.sep_conv1 = SeparableConv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.sep_conv2 = SeparableConv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.sep_conv3 = SeparableConv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """Forward pass."""
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = F.relu(self.bn2(self.sep_conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn3(self.sep_conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn4(self.sep_conv3(x)))
        x = self.pool3(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


def get_model(model_name='simple_cnn', num_classes=35, input_channels=1, dropout=0.5):
    """
    Factory function to get model by name.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        input_channels: Number of input channels
        dropout: Dropout probability
        
    Returns:
        PyTorch model
    """
    models = {
        'simple_cnn': SimpleCNN,
        'deep_cnn': DeepCNN,
        'resnet': ResNet,
        'attention_cnn': AttentionCNN,
        'efficient_cnn': EfficientCNN
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(models.keys())}")
    
    return models[model_name](num_classes=num_classes, 
                             input_channels=input_channels, 
                             dropout=dropout)

