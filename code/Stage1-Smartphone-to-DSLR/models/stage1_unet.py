"""
Stage 1: Smartphone to DSLR Enhancement Model
U-Net architecture with attention mechanisms for image-to-image translation
Focuses on: shadows, highlights, saturation, tone, clarity, sharpness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvBlock, DownBlock, UpBlock, Bottleneck, CBAM


class SmartphoneToDSLRUNet(nn.Module):
    """
    U-Net model for transforming smartphone images to DSLR quality
    
    Architecture:
        - Encoder: 4 downsampling blocks (3 -> 64 -> 128 -> 256 -> 512)
        - Bottleneck: Residual blocks with attention at lowest resolution
        - Decoder: 4 upsampling blocks with skip connections (512 -> 256 -> 128 -> 64 -> 3)
        
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        out_channels: Number of output channels (default: 3 for RGB)
        base_channels: Base number of channels, doubles at each level (default: 64)
        use_attention: Whether to use CBAM attention (default: True)
        use_bn: Whether to use batch normalization (default: True)
    """
    
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, 
                 use_attention=True, use_bn=True):
        super(SmartphoneToDSLRUNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initial convolution
        self.init_conv = ConvBlock(in_channels, base_channels, use_bn=use_bn)
        
        # Encoder (downsampling path)
        self.down1 = DownBlock(base_channels, base_channels * 2, use_attention, use_bn)      # 64 -> 128
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, use_attention, use_bn)  # 128 -> 256
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, use_attention, use_bn)  # 256 -> 512
        
        # Bottleneck
        self.bottleneck = Bottleneck(base_channels * 8, num_blocks=3, use_bn=use_bn)  # 512
        
        # Decoder (upsampling path)
        self.up1 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4, use_attention, use_bn)  # 512 -> 256
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2, use_attention, use_bn)  # 256 -> 128
        self.up3 = UpBlock(base_channels * 2, base_channels, base_channels, use_attention, use_bn)           # 128 -> 64
        
        # Final convolution
        self.final_conv = nn.Sequential(
            ConvBlock(base_channels, base_channels, use_bn=use_bn, activation='leaky'),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
        # Skip connection from input to output (residual learning)
        self.use_skip = True
        
    def forward(self, x):
        # Save input for skip connection
        input_image = x
        
        # Initial convolution
        x = self.init_conv(x)
        skip0 = x
        
        # Encoder
        x, skip1 = self.down1(x)  # 224 -> 112
        x, skip2 = self.down2(x)  # 112 -> 56
        x, skip3 = self.down3(x)  # 56 -> 28
        
        # Bottleneck
        x = self.bottleneck(x)    # 28
        
        # Decoder
        x = self.up1(x, skip3)    # 28 -> 56
        x = self.up2(x, skip2)    # 56 -> 112
        x = self.up3(x, skip1)    # 112 -> 224
        
        # Final convolution
        x = self.final_conv(x)
        
        # Add skip connection from input (helps preserve details)
        if self.use_skip:
            # Weighted combination: 80% network output + 20% input
            x = 0.8 * x + 0.2 * input_image
            x = torch.clamp(x, 0, 1)
        
        return x
    
    def get_num_params(self):
        """Return the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LightweightUNet(nn.Module):
    """
    Lightweight version of U-Net for faster training and inference
    Uses fewer channels and fewer blocks
    
    Good for:
        - Quick experimentation
        - Limited memory (M1 Pro with 16GB)
        - Faster training iterations
    """
    
    def __init__(self, in_channels=3, out_channels=3, base_channels=32):
        super(LightweightUNet, self).__init__()
        
        # Simpler architecture with fewer channels
        # Encoder
        self.down1 = DownBlock(in_channels, base_channels, use_attention=True)              # 3 -> 32
        self.down2 = DownBlock(base_channels, base_channels * 2, use_attention=True)        # 32 -> 64
        self.down3 = DownBlock(base_channels * 2, base_channels * 4, use_attention=True)    # 64 -> 128
        
        # Bottleneck (fewer residual blocks)
        self.bottleneck = Bottleneck(base_channels * 4, num_blocks=2)  # 128
        
        # Decoder
        self.up1 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2, use_attention=True)  # 128, skip=128 -> 64
        self.up2 = UpBlock(base_channels * 2, base_channels * 2, base_channels, use_attention=True)      # 64, skip=64 -> 32
        
        # Final output
        self.final_conv = nn.Sequential(
            ConvBlock(base_channels, base_channels),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        input_image = x
        
        # Encoder
        x, skip1 = self.down1(x)  # 224 -> 112
        x, skip2 = self.down2(x)  # 112 -> 56
        x, skip3 = self.down3(x)  # 56 -> 28
        
        # Bottleneck
        x = self.bottleneck(x)    # 28
        
        # Decoder
        x = self.up1(x, skip3)    # 28 -> 56
        x = self.up2(x, skip2)    # 56 -> 112
        
        # Upsample to original size
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Skip connection
        x = 0.8 * x + 0.2 * input_image
        x = torch.clamp(x, 0, 1)
        
        return x
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_stage1_model(model_type='dped', device='mps'):
    """
    Factory function to build Stage 1 model
    
    Args:
        model_type: 'dped' (default, DPED ResNet), 'standard' (U-Net), or 'lightweight' (U-Net)
        device: Device to place model on ('mps', 'cuda', 'cpu')
    
    Returns:
        Model instance on specified device
    """
    if model_type == 'dped':
        # DPED: 12-layer Residual CNN (default, based on ICCV 2017 paper)
        from .dped_resnet import DPEDResNetWithSkip
        model = DPEDResNetWithSkip(
            in_channels=3,
            out_channels=3
        )
    elif model_type == 'standard':
        model = SmartphoneToDSLRUNet(
            in_channels=3,
            out_channels=3,
            base_channels=64,
            use_attention=True,
            use_bn=True
        )
    elif model_type == 'lightweight':
        model = LightweightUNet(
            in_channels=3,
            out_channels=3,
            base_channels=32
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: 'dped', 'standard', 'lightweight'")
    
    model = model.to(device)
    
    num_params = model.get_num_params()
    print(f"[Stage1] Built {model_type} model with {num_params:,} parameters")
    
    return model


if __name__ == "__main__":
    # Test the models
    print("=== Testing Stage 1 Models ===\n")
    
    batch_size = 2
    height, width = 224, 224
    x = torch.randn(batch_size, 3, height, width)
    
    # Test standard model
    print("Testing Standard U-Net...")
    model_standard = SmartphoneToDSLRUNet()
    with torch.no_grad():
        output = model_standard(x)
    print(f"Input: {x.shape}, Output: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"Parameters: {model_standard.get_num_params():,}\n")
    
    # Test lightweight model
    print("Testing Lightweight U-Net...")
    model_light = LightweightUNet()
    with torch.no_grad():
        output = model_light(x)
    print(f"Input: {x.shape}, Output: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"Parameters: {model_light.get_num_params():,}\n")
    
    # Memory estimation
    def estimate_memory(model, input_shape):
        """Rough estimate of memory usage"""
        import numpy as np
        
        # Parameters
        param_mem = sum(p.numel() * p.element_size() for p in model.parameters())
        
        # Forward pass (rough estimate)
        batch, c, h, w = input_shape
        forward_mem = param_mem * 2  # Rough approximation
        
        total_mb = (param_mem + forward_mem) / (1024 ** 2)
        return total_mb
    
    mem_standard = estimate_memory(model_standard, x.shape)
    mem_light = estimate_memory(model_light, x.shape)
    
    print(f"Estimated memory:")
    print(f"  Standard: ~{mem_standard:.1f} MB")
    print(f"  Lightweight: ~{mem_light:.1f} MB")
    
    print("\n✓ All models tested successfully!")
    print("\nRecommendation for M1 Pro 16GB:")
    print("  - Use 'lightweight' for initial experiments (faster)")
    print("  - Use 'standard' for final training (better quality)")

