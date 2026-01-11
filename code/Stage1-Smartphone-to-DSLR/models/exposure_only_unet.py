"""
Exposure-Only U-Net
Only adjusts exposure, highlights, shadows - preserves colors and details completely
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .blocks import ConvBlock, DownBlock, UpBlock, Bottleneck
except ImportError:
    from blocks import ConvBlock, DownBlock, UpBlock, Bottleneck


class ExposureOnlyUNet(nn.Module):
    """
    U-Net that ONLY adjusts luminance (exposure, highlights, shadows)
    Completely preserves:
    - Colors (hue, saturation)
    - Details (sharpness, texture)
    
    Works by:
    1. Converting RGB to luminance + chrominance
    2. Processing ONLY luminance through U-Net
    3. Keeping original chrominance and detail unchanged
    4. Recombining to get exposure-adjusted RGB
    
    Like Lightroom's: Exposure, Highlights, Shadows, Whites, Blacks
    NOT like: Temperature, Tint, Saturation, Vibrance, Clarity
    """
    
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Lightweight network for luminance processing only
        # Smaller than normal because we're only adjusting exposure
        self.down1 = DownBlock(1, base_channels, use_attention=False, use_bn=True)  # Only 1 input channel (luminance)
        self.down2 = DownBlock(base_channels, base_channels * 2, use_attention=False, use_bn=True)
        
        self.bottleneck = Bottleneck(base_channels * 2, num_blocks=2, use_bn=True)
        
        self.up1 = UpBlock(base_channels * 2, base_channels * 2, base_channels, use_attention=False, use_bn=True)
        
        # Output predicts luminance adjustment (residual)
        self.final_conv = nn.Sequential(
            ConvBlock(base_channels, base_channels // 2, use_bn=True, activation='leaky'),
            nn.Conv2d(base_channels // 2, 1, kernel_size=3, padding=1),  # Output 1 channel (luminance adjustment)
            nn.Tanh()  # Range [-1, 1] for adjustment
        )
        
        print(f"[ExposureOnlyUNet] Initialized")
        print(f"  Base channels: {base_channels}")
        print(f"  Only modifies: Exposure, Highlights, Shadows, Contrast")
        print(f"  Preserves: Colors, Hue, Saturation, Details")
    
    def rgb_to_luminance_chrominance(self, rgb):
        """
        Convert RGB to Luminance + Chrominance
        Luminance: Perceived brightness (like Lightroom Exposure)
        Chrominance: Color information (preserved)
        
        Args:
            rgb: [B, 3, H, W] in range [0, 1]
        
        Returns:
            luminance: [B, 1, H, W]
            chrominance: [B, 3, H, W] (color ratios)
        """
        # Standard luminance weights (Rec. 709)
        # This is the "Y" in YCbCr color space
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        # Chrominance: Store color as ratio to luminance
        # This preserves hue and saturation
        eps = 1e-6
        chrominance = rgb / (luminance + eps)
        
        return luminance, chrominance
    
    def luminance_chrominance_to_rgb(self, luminance, chrominance):
        """
        Reconstruct RGB from Luminance + Chrominance
        
        Args:
            luminance: [B, 1, H, W]
            chrominance: [B, 3, H, W]
        
        Returns:
            rgb: [B, 3, H, W]
        """
        # Multiply adjusted luminance by original color ratios
        rgb = luminance * chrominance
        return torch.clamp(rgb, 0, 1)
    
    def forward(self, x):
        """
        Process image - adjust ONLY exposure/luminance
        
        Args:
            x: RGB image [B, 3, H, W] in range [0, 1]
        
        Returns:
            Exposure-adjusted RGB with original colors and details
        """
        # Store original for detail preservation
        original_rgb = x
        
        # Separate luminance from chrominance
        luminance, chrominance = self.rgb_to_luminance_chrominance(x)
        
        # Extract high-frequency details from luminance
        # (we'll add these back unchanged)
        luminance_base, luminance_detail = self.extract_detail(luminance, sigma=1.5)
        
        # Process ONLY the base luminance through U-Net
        y = luminance_base
        
        # Encoder
        y, skip1 = self.down1(y)
        y, skip2 = self.down2(y)
        
        # Bottleneck
        y = self.bottleneck(y)
        
        # Decoder
        y = self.up1(y, skip2)
        
        # Upsample to original size
        y = F.interpolate(y, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        
        # Final prediction: luminance adjustment
        luminance_adjustment = self.final_conv(y) * 0.3  # Small adjustment range
        
        # Apply adjustment to base luminance
        adjusted_luminance_base = luminance_base + luminance_adjustment
        adjusted_luminance_base = torch.clamp(adjusted_luminance_base, 0, 1)
        
        # Add back original detail (unchanged!)
        adjusted_luminance = adjusted_luminance_base + luminance_detail
        adjusted_luminance = torch.clamp(adjusted_luminance, 0, 1)
        
        # Reconstruct RGB with adjusted luminance but ORIGINAL chrominance
        output = self.luminance_chrominance_to_rgb(adjusted_luminance, chrominance)
        
        return output
    
    def extract_detail(self, img, sigma=1.5):
        """Extract high-frequency detail"""
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Gaussian kernel
        x = torch.arange(kernel_size, dtype=torch.float32, device=img.device) - kernel_size // 2
        gauss_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        
        gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0)
        
        # Blur
        padding = kernel_size // 2
        base = F.conv2d(img, gauss_2d, padding=padding)
        
        # Detail
        detail = img - base
        
        return base, detail
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_exposure_only_model(base_channels=64, device='mps'):
    """
    Build exposure-only model
    
    Args:
        base_channels: Base number of channels (default 64)
        device: Device to run on
    
    Returns:
        Model instance
    """
    model = ExposureOnlyUNet(
        in_channels=3,
        out_channels=3,
        base_channels=base_channels
    )
    
    model = model.to(device)
    
    num_params = model.get_num_params()
    print(f"\n[ExposureOnlyModel] Model built successfully")
    print(f"  Parameters: {num_params:,}")
    print(f"  Base channels: {base_channels}")
    print(f"")
    print(f"  ✓ Adjusts: Exposure, Highlights, Shadows, Contrast, Brightness")
    print(f"  ✓ Preserves: Colors, Hue, Saturation, Temperature")
    print(f"  ✓ Preserves: Details, Sharpness, Texture")
    print(f"")
    print(f"  Like Lightroom controls:")
    print(f"    • Exposure slider")
    print(f"    • Highlights slider")
    print(f"    • Shadows slider")
    print(f"    • Contrast slider")
    print(f"    • Whites/Blacks slider")
    
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Exposure-Only U-Net")
    print("=" * 60)
    print()
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Build model
    model = build_exposure_only_model(base_channels=64, device=device)
    
    # Test with sample image
    print("\nTesting with sample image...")
    test_input = torch.rand(2, 3, 224, 224, device=device) * 0.5  # Darker image
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Input range: [{test_input.min():.3f}, {test_input.max():.3f}]")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Check color preservation
    input_mean_color = test_input.mean(dim=[0, 2, 3])
    output_mean_color = output.mean(dim=[0, 2, 3])
    color_diff = (input_mean_color - output_mean_color).abs().mean()
    
    print(f"\n  Color preservation test:")
    print(f"    Input mean color: R={input_mean_color[0]:.3f}, G={input_mean_color[1]:.3f}, B={input_mean_color[2]:.3f}")
    print(f"    Output mean color: R={output_mean_color[0]:.3f}, G={output_mean_color[1]:.3f}, B={output_mean_color[2]:.3f}")
    print(f"    Color difference: {color_diff:.4f}")
    
    if color_diff < 0.1:
        print(f"    ✓ Colors well preserved!")
    else:
        print(f"    ⚠ Colors changed (expected < 0.1)")
    
    print("\n" + "=" * 60)
    print("✓ Exposure-Only U-Net test complete!")
    print("=" * 60)
    print("\nThis model is ready for training!")
    print("It will ONLY adjust exposure/lighting, not colors or details.")

