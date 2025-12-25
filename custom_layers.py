"""
Custom neural network layers for speech recognition.
Includes attention mechanisms and other specialized layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Self-attention layer for feature aggregation."""
    
    def __init__(self, input_dim):
        """
        Args:
            input_dim: Dimension of input features
        """
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        """
        Apply attention mechanism.
        
        Args:
            x: Input tensor (batch, channels, height, width) or (batch, time, features)
            
        Returns:
            Attended features
        """
        # Flatten spatial dimensions if needed
        if len(x.shape) == 4:
            batch, channels, height, width = x.shape
            x = x.permute(0, 2, 3, 1)  # (batch, height, width, channels)
            x = x.reshape(batch, height * width, channels)
        
        # Compute attention scores
        attention_scores = self.attention_weights(x)  # (batch, time, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention
        attended = torch.sum(x * attention_weights, dim=1)  # (batch, channels)
        
        return attended


class TemporalAveragePooling(nn.Module):
    """Average pooling over the temporal dimension."""
    
    def __init__(self, dim=2):
        """
        Args:
            dim: Dimension to pool over (default 2 for time dimension in spectrograms)
        """
        super(TemporalAveragePooling, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        """
        Apply temporal average pooling.
        
        Args:
            x: Input tensor
            
        Returns:
            Pooled tensor
        """
        return torch.mean(x, dim=self.dim)


class TemporalMaxPooling(nn.Module):
    """Max pooling over the temporal dimension."""
    
    def __init__(self, dim=2):
        """
        Args:
            dim: Dimension to pool over
        """
        super(TemporalMaxPooling, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        """
        Apply temporal max pooling.
        
        Args:
            x: Input tensor
            
        Returns:
            Pooled tensor
        """
        return torch.max(x, dim=self.dim)[0]


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.3):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for convolution
            dropout: Dropout probability
        """
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        Forward pass through residual block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with residual connection
        """
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = F.relu(out)
        
        return out


class SeparableConv2d(nn.Module):
    """Depthwise separable convolution for efficient processing."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, padding=1, bias=False):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolutional kernel
            stride: Stride for convolution
            padding: Padding for convolution
            bias: Whether to use bias
        """
        super(SeparableConv2d, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, groups=in_channels,
                                  bias=bias)
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                  stride=1, padding=0, bias=bias)
    
    def forward(self, x):
        """
        Forward pass through separable convolution.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels, reduction=16):
        """
        Args:
            channels: Number of channels
            reduction: Reduction ratio for SE block
        """
        super(SqueezeExcitation, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass through SE block.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            
        Returns:
            Channel-attended tensor
        """
        batch, channels, _, _ = x.size()
        
        # Squeeze
        y = self.avg_pool(x).view(batch, channels)
        
        # Excitation
        y = self.fc(y).view(batch, channels, 1, 1)
        
        # Scale
        return x * y.expand_as(x)

