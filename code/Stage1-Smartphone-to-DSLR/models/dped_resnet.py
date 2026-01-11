"""
DPED: DSLR Photo Enhancement Dataset Model
12-layer Residual Convolutional Neural Network
Based on: https://aiff22.github.io/index.html

Architecture:
- 12 convolutional layers with residual connections
- Takes phone photo as input, outputs DSLR-quality photo
- Trained to learn translation function from phone to DSLR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with two convolutions"""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual  # Residual connection
        out = F.relu(out)
        
        return out


class DPEDResNet(nn.Module):
    """
    12-layer Residual CNN for DPED-style photo enhancement
    
    Architecture based on DPED paper:
    - Input: Phone photo (3 channels)
    - Output: DSLR-quality photo (3 channels)
    - 12 convolutional layers with residual connections
    
    Reference: https://aiff22.github.io/index.html
    """
    
    def __init__(self, in_channels=3, out_channels=3):
        super(DPEDResNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks (10 layers = 5 blocks × 2 layers each)
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        self.res4 = ResidualBlock(64)
        self.res5 = ResidualBlock(64)
        
        # Final convolution
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Output layer
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=9, padding=4)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input phone photo [B, 3, H, W] in range [0, 1]
        
        Returns:
            Enhanced DSLR-quality photo [B, 3, H, W] in range [0, 1]
        """
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        
        # Final convolutions
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        
        # Output is residual (enhancement), will be added to input in wrapper
        # Don't apply sigmoid here - let wrapper handle it
        
        return x
    
    def get_num_params(self):
        """Return the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DPEDResNetWithSkip(nn.Module):
    """
    DPED ResNet with skip connection from input to output
    This helps preserve image details while enhancing quality
    """
    
    def __init__(self, in_channels=3, out_channels=3):
        super(DPEDResNetWithSkip, self).__init__()
        
        self.dped_net = DPEDResNet(in_channels, out_channels)
        self.use_skip = True
        self.skip_weight = 0.1  # Weight for input skip connection
    
    def forward(self, x):
        """
        Forward pass with skip connection
        
        Args:
            x: Input phone photo [B, 3, H, W] in range [0, 1]
        
        Returns:
            Enhanced photo [B, 3, H, W] in range [0, 1]
        """
        input_image = x
        
        # Get network output (residual enhancement)
        residual = self.dped_net(x)
        
        # Add residual to input (residual learning)
        if self.use_skip:
            # Network predicts enhancement delta, add to input
            output = input_image + residual
            # Clamp to valid range and apply sigmoid for safety
            output = torch.sigmoid(output)
            output = torch.clamp(output, 0, 1)
        else:
            # Direct output
            output = torch.sigmoid(residual)
            output = torch.clamp(output, 0, 1)
        
        return output
    
    def get_num_params(self):
        """Return the number of parameters in the model"""
        return self.dped_net.get_num_params()

