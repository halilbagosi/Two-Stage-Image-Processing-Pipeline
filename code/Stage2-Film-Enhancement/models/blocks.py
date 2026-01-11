"""
Reusable building blocks for neural network architectures
Includes convolution blocks, attention mechanisms, and residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Basic convolution block with conv -> norm -> activation
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolution kernel
        stride: Stride for convolution
        padding: Padding mode ('same' or int)
        use_bn: Whether to use batch normalization
        activation: Activation function ('relu', 'leaky', 'none')
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, use_bn=True, activation='leaky'):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=padding, bias=not use_bn)
        
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'none':
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection: out = F(x) + x
    
    Args:
        channels: Number of channels
        use_bn: Whether to use batch normalization
    """
    
    def __init__(self, channels, use_bn=True):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = ConvBlock(channels, channels, kernel_size=3, padding=1, 
                              use_bn=use_bn, activation='leaky')
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_bn)
        
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(channels)
        
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn(out)
        out = out + residual
        out = self.activation(out)
        return out


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (squeeze-and-excitation)
    Adaptively recalibrates channel-wise feature responses
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for bottleneck
    """
    
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).reshape(b, c)
        # Excitation
        y = self.fc(y).reshape(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    Focuses on 'where' is important in the feature map
    
    Args:
        kernel_size: Size of convolution kernel for attention
    """
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    Combines channel and spatial attention
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for channel attention
        kernel_size: Kernel size for spatial attention
    """
    
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class DownBlock(nn.Module):
    """
    Downsampling block for U-Net encoder
    Consists of: Conv -> Conv -> Optional Attention -> MaxPool
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        use_attention: Whether to use CBAM attention
        use_bn: Whether to use batch normalization
    """
    
    def __init__(self, in_channels, out_channels, use_attention=True, use_bn=True):
        super(DownBlock, self).__init__()
        
        self.conv1 = ConvBlock(in_channels, out_channels, use_bn=use_bn)
        self.conv2 = ConvBlock(out_channels, out_channels, use_bn=use_bn)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAM(out_channels)
        
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # Convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Attention
        if self.use_attention:
            x = self.attention(x)
        
        # Save skip connection before pooling
        skip = x
        
        # Downsample
        x = self.pool(x)
        
        return x, skip


class UpBlock(nn.Module):
    """
    Upsampling block for U-Net decoder
    Consists of: Upsample -> Concat with skip -> Conv -> Conv -> Optional Attention
    
    Args:
        in_channels: Number of input channels (from previous layer)
        skip_channels: Number of channels from skip connection
        out_channels: Number of output channels
        use_attention: Whether to use CBAM attention
        use_bn: Whether to use batch normalization
    """
    
    def __init__(self, in_channels, skip_channels, out_channels, 
                 use_attention=True, use_bn=True):
        super(UpBlock, self).__init__()
        
        # Upsampling - reduce channels first
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # Convolutions after concatenation
        self.conv1 = ConvBlock(out_channels + skip_channels, out_channels, use_bn=use_bn)
        self.conv2 = ConvBlock(out_channels, out_channels, use_bn=use_bn)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAM(out_channels)
    
    def forward(self, x, skip):
        # Upsample
        x = self.up(x)
        
        # Handle size mismatch (if any)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Attention
        if self.use_attention:
            x = self.attention(x)
        
        return x


class Bottleneck(nn.Module):
    """
    Bottleneck block for U-Net
    Processing at the lowest resolution with multiple residual blocks
    
    Args:
        channels: Number of channels
        num_blocks: Number of residual blocks
        use_bn: Whether to use batch normalization
    """
    
    def __init__(self, channels, num_blocks=2, use_bn=True):
        super(Bottleneck, self).__init__()
        
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(channels, use_bn=use_bn))
        
        self.blocks = nn.Sequential(*blocks)
        self.attention = CBAM(channels)
    
    def forward(self, x):
        x = self.blocks(x)
        x = self.attention(x)
        return x


if __name__ == "__main__":
    # Test blocks
    print("Testing building blocks...")
    
    batch_size = 2
    channels = 64
    height, width = 56, 56
    
    x = torch.randn(batch_size, channels, height, width)
    
    # Test ConvBlock
    conv_block = ConvBlock(channels, channels)
    out = conv_block(x)
    print(f"✓ ConvBlock: {x.shape} -> {out.shape}")
    
    # Test ResidualBlock
    res_block = ResidualBlock(channels)
    out = res_block(x)
    print(f"✓ ResidualBlock: {x.shape} -> {out.shape}")
    
    # Test ChannelAttention
    ca = ChannelAttention(channels)
    out = ca(x)
    print(f"✓ ChannelAttention: {x.shape} -> {out.shape}")
    
    # Test SpatialAttention
    sa = SpatialAttention()
    out = sa(x)
    print(f"✓ SpatialAttention: {x.shape} -> {out.shape}")
    
    # Test CBAM
    cbam = CBAM(channels)
    out = cbam(x)
    print(f"✓ CBAM: {x.shape} -> {out.shape}")
    
    # Test DownBlock
    down = DownBlock(3, 64)
    x_in = torch.randn(batch_size, 3, 224, 224)
    out, skip = down(x_in)
    print(f"✓ DownBlock: {x_in.shape} -> out: {out.shape}, skip: {skip.shape}")
    
    # Test UpBlock
    up = UpBlock(128, 64, 64)
    x_up = torch.randn(batch_size, 128, 56, 56)
    skip_up = torch.randn(batch_size, 64, 112, 112)
    out = up(x_up, skip_up)
    print(f"✓ UpBlock: {x_up.shape} + skip {skip_up.shape} -> {out.shape}")
    
    # Test Bottleneck
    bottleneck = Bottleneck(512, num_blocks=3)
    x_bottle = torch.randn(batch_size, 512, 14, 14)
    out = bottleneck(x_bottle)
    print(f"✓ Bottleneck: {x_bottle.shape} -> {out.shape}")
    
    print("\n✓ All blocks tested successfully!")

