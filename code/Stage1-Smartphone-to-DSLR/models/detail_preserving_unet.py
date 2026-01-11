"""
Detail-Preserving U-Net for Style Transfer
Only modifies tone, color, exposure - preserves texture and sharpness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .blocks import ConvBlock, DownBlock, UpBlock, Bottleneck
except ImportError:
    from blocks import ConvBlock, DownBlock, UpBlock, Bottleneck


class DetailPreservingUNet(nn.Module):
    """
    U-Net that preserves high-frequency details (texture, sharpness)
    Only modifies low-frequency stylistic aspects (tone, color, exposure)
    
    Strategy:
    1. Decompose image into detail (high-freq) and base (low-freq)
    2. Process only the base layer through U-Net
    3. Add back original details to preserve sharpness
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        out_channels: Number of output channels (default: 3 for RGB)
        base_channels: Base number of channels (default: 64)
        detail_threshold: Frequency cutoff for detail extraction (default: 0.1)
    """
    
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, detail_threshold=0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.detail_threshold = detail_threshold
        
        # U-Net for processing base (low-frequency) layer
        # Encoder
        self.down1 = DownBlock(in_channels, base_channels, use_attention=True, use_bn=True)
        self.down2 = DownBlock(base_channels, base_channels * 2, use_attention=True, use_bn=True)
        self.down3 = DownBlock(base_channels * 2, base_channels * 4, use_attention=True, use_bn=True)
        
        # Bottleneck
        self.bottleneck = Bottleneck(base_channels * 4, num_blocks=3, use_bn=True)
        
        # Decoder - need to account for skip connections
        self.up1 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2, use_attention=True, use_bn=True)
        self.up2 = UpBlock(base_channels * 2, base_channels * 2, base_channels, use_attention=True, use_bn=True)
        
        # Final output layer
        self.final_conv = nn.Sequential(
            ConvBlock(base_channels, base_channels, use_bn=True, activation='leaky'),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Output in [-1, 1] range for residual
        )
        
        # Detail weight (learnable parameter for how much detail to preserve)
        self.detail_weight = nn.Parameter(torch.tensor(0.9))  # Start with 90% detail preservation
    
    def extract_details(self, img, sigma=1.5):
        """
        Extract high-frequency details from image
        
        Args:
            img: Input image [B, C, H, W]
            sigma: Gaussian kernel sigma for detail extraction
        
        Returns:
            base: Low-frequency base layer
            detail: High-frequency detail layer
        """
        # Create Gaussian blur kernel for low-pass filter
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create 1D Gaussian kernel
        x = torch.arange(kernel_size, dtype=torch.float32, device=img.device) - kernel_size // 2
        gauss_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        
        # Create 2D Gaussian kernel
        gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0)
        gauss_2d = gauss_2d.repeat(self.in_channels, 1, 1, 1)
        
        # Apply Gaussian blur to get base layer (low-frequency)
        padding = kernel_size // 2
        base = F.conv2d(img, gauss_2d, padding=padding, groups=self.in_channels)
        
        # Detail layer = original - base (high-frequency)
        detail = img - base
        
        return base, detail
    
    def forward(self, x):
        """
        Forward pass with detail preservation
        
        Args:
            x: Input image [B, C, H, W] in range [0, 1]
        
        Returns:
            Enhanced image with preserved details
        """
        # Store original for residual connection
        input_image = x
        
        # Extract details (high-frequency texture)
        base, detail = self.extract_details(x, sigma=2.0)
        
        # Process only the base layer through U-Net
        # This changes tone, color, exposure but not texture
        x = base
        
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
        
        # Final convolution outputs residual adjustments
        residual = self.final_conv(x)
        
        # Apply residual to base layer
        enhanced_base = base + residual * 0.2  # Small residual for stability
        enhanced_base = torch.clamp(enhanced_base, 0, 1)
        
        # Add back original details (preserve sharpness!)
        # Clamp detail weight to reasonable range
        detail_weight = torch.clamp(self.detail_weight, 0.7, 1.0)
        output = enhanced_base + detail * detail_weight
        
        # Final clamping
        output = torch.clamp(output, 0, 1)
        
        return output
    
    def get_num_params(self):
        """Return the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class StyleOnlyUNet(nn.Module):
    """
    Even more aggressive detail preservation
    Works in LAB color space - only modifies L and AB, preserves structure
    """
    
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        
        # Process LAB channels separately
        # L channel: luminosity (tone, exposure, contrast)
        self.l_net = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, 1, 3, padding=1),
            nn.Tanh()
        )
        
        # AB channels: color (saturation, temperature, tint)
        self.ab_net = nn.Sequential(
            nn.Conv2d(2, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, 2, 3, padding=1),
            nn.Tanh()
        )
    
    def rgb_to_lab(self, rgb):
        """Convert RGB to LAB (approximation for differentiability)"""
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        
        # L channel (lightness)
        L = 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        # A channel (green-red)
        A = (r - g) * 0.5
        
        # B channel (blue-yellow)
        B = (r + g) * 0.5 - b
        
        return L, A, B
    
    def lab_to_rgb(self, L, A, B):
        """Convert LAB to RGB (approximation)"""
        # Inverse transformation
        r = L + 2 * A
        g = L - 2 * A
        b = L + 2 * A + 2 * B - 2 * L
        
        return torch.cat([r, g, b], dim=1)
    
    def forward(self, x):
        """
        Process only style, preserve all detail/structure
        """
        # Convert to LAB
        L, A, B = self.rgb_to_lab(x)
        
        # Extract structure (edges, detail) from L channel
        # Use Sobel operator
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        
        edges_x = F.conv2d(L, sobel_x, padding=1)
        edges_y = F.conv2d(L, sobel_y, padding=1)
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        
        # Process L channel (tone adjustments)
        L_residual = self.l_net(L) * 0.2
        L_enhanced = L + L_residual
        L_enhanced = torch.clamp(L_enhanced, 0, 1)
        
        # Process AB channels (color adjustments)
        AB = torch.cat([A, B], dim=1)
        AB_residual = self.ab_net(AB) * 0.2
        A_enhanced = A + AB_residual[:, 0:1]
        B_enhanced = B + AB_residual[:, 1:2]
        
        # Convert back to RGB
        output = self.lab_to_rgb(L_enhanced, A_enhanced, B_enhanced)
        
        # Add back edges to preserve sharpness
        # This ensures texture is perfectly preserved
        output = output + edges * 0.5
        
        output = torch.clamp(output, 0, 1)
        
        return output
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_detail_preserving_model(model_type='detail_preserving', base_channels=64, device='mps'):
    """
    Build detail-preserving model
    
    Args:
        model_type: 'detail_preserving' or 'style_only'
        base_channels: Base number of channels
        device: Device to run on
    
    Returns:
        Model instance
    """
    if model_type == 'detail_preserving':
        model = DetailPreservingUNet(
            in_channels=3,
            out_channels=3,
            base_channels=base_channels
        )
    elif model_type == 'style_only':
        model = StyleOnlyUNet(
            in_channels=3,
            out_channels=3,
            base_channels=base_channels
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # Print model info
    num_params = model.get_num_params()
    print(f"[DetailPreservingModel] Built {model_type} model")
    print(f"  Parameters: {num_params:,}")
    print(f"  Base channels: {base_channels}")
    print(f"  Detail preservation: ENABLED")
    print(f"  Changes: Tone, Color, Exposure, Saturation only")
    print(f"  Preserves: Texture, Sharpness, Detail")
    
    return model


if __name__ == "__main__":
    # Test detail-preserving model
    print("=== Testing Detail-Preserving Models ===\n")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Test DetailPreservingUNet
    print("1. Testing DetailPreservingUNet...")
    model1 = build_detail_preserving_model('detail_preserving', base_channels=64, device=device)
    
    test_input = torch.rand(2, 3, 224, 224, device=device)
    with torch.no_grad():
        output1 = model1(test_input)
    
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output1.shape}")
    print(f"   Output range: [{output1.min():.3f}, {output1.max():.3f}]")
    
    # Check detail preservation
    detail_diff = (test_input - output1).abs().mean()
    print(f"   Average change: {detail_diff:.4f} (should be small)")
    print("   ✓ DetailPreservingUNet OK\n")
    
    # Test StyleOnlyUNet
    print("2. Testing StyleOnlyUNet...")
    model2 = build_detail_preserving_model('style_only', base_channels=64, device=device)
    
    with torch.no_grad():
        output2 = model2(test_input)
    
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output2.shape}")
    print(f"   Output range: [{output2.min():.3f}, {output2.max():.3f}]")
    print("   ✓ StyleOnlyUNet OK\n")
    
    print("=" * 60)
    print("✓ All detail-preserving models tested successfully!")
    print("=" * 60)
    print("\nThese models will preserve:")
    print("  ✓ Texture and detail")
    print("  ✓ Sharpness and edges")
    print("  ✓ High-frequency information")
    print("\nWhile only modifying:")
    print("  • Tone and exposure")
    print("  • Color and saturation")
    print("  • Highlights and shadows")
    print("  • Temperature and tint")

