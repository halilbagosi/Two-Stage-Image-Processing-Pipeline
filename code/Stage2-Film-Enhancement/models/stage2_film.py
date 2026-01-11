"""
Stage 2: DSLR to Cinematic Film Enhancement Model
Lightweight architecture focused on color grading and tone mapping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvBlock, ResidualBlock, ChannelAttention


class FilmLUTNetwork(nn.Module):
    """
    3D LUT-based Film Enhancement Network
    
    Uses a learned 3D Look-Up Table for color grading
    Very efficient and produces high-quality cinematic looks
    
    Architecture:
        - Lightweight encoder to predict LUT parameters
        - 3D LUT for efficient color transformation
        - Refinement network for final adjustments
    
    Args:
        lut_dim: Dimension of 3D LUT (default: 33)
        num_luts: Number of different LUTs to blend (default: 1)
    """
    
    def __init__(self, lut_dim=33, num_luts=1):
        super(FilmLUTNetwork, self).__init__()
        
        self.lut_dim = lut_dim
        self.num_luts = num_luts
        
        # Lightweight feature extractor
        self.feature_extractor = nn.Sequential(
            ConvBlock(3, 16, kernel_size=3),
            nn.MaxPool2d(2),
            ConvBlock(16, 32, kernel_size=3),
            nn.MaxPool2d(2),
            ConvBlock(32, 64, kernel_size=3),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # LUT weight predictor (if using multiple LUTs)
        if num_luts > 1:
            self.lut_weight_predictor = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, num_luts),
                nn.Softmax(dim=1)
            )
        
        # Initialize 3D LUT(s)
        # Each LUT maps RGB input -> RGB output
        self.register_buffer('lut_identity', self._generate_identity_lut())
        
        # Learnable LUT offsets (learn deviations from identity)
        self.lut_offsets = nn.Parameter(
            torch.zeros(num_luts, lut_dim, lut_dim, lut_dim, 3)
        )
        
        # Small refinement network for final touch-ups
        self.refinement = nn.Sequential(
            ConvBlock(3, 16, kernel_size=3),
            ResidualBlock(16),
            ResidualBlock(16),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()  # Output in [-1, 1] for residual
        )
    
    def _generate_identity_lut(self):
        """Generate identity 3D LUT"""
        dim = self.lut_dim
        identity = torch.zeros(dim, dim, dim, 3)
        
        for r in range(dim):
            for g in range(dim):
                for b in range(dim):
                    identity[r, g, b, 0] = r / (dim - 1)
                    identity[r, g, b, 1] = g / (dim - 1)
                    identity[r, g, b, 2] = b / (dim - 1)
        
        return identity
    
    def apply_lut(self, img, lut):
        """
        Apply 3D LUT to image using trilinear interpolation
        
        Args:
            img: Input image [B, 3, H, W] in range [0, 1]
            lut: 3D LUT [lut_dim, lut_dim, lut_dim, 3]
        
        Returns:
            Transformed image [B, 3, H, W]
        """
        B, C, H, W = img.shape
        
        # Ensure LUT is contiguous for proper backpropagation
        lut = lut.contiguous()
        
        # Flatten spatial dimensions - ensure contiguous before reshape
        img_flat = img.permute(0, 2, 3, 1).contiguous().reshape(-1, 3)  # [B*H*W, 3]
        
        # Scale to LUT coordinates
        lut_coords = img_flat * (self.lut_dim - 1)
        
        # Trilinear interpolation
        lut_coords_floor = torch.floor(lut_coords).long()
        lut_coords_ceil = torch.ceil(lut_coords).long()
        lut_coords_frac = lut_coords - lut_coords_floor.float()
        
        # Clamp coordinates
        lut_coords_floor = torch.clamp(lut_coords_floor, 0, self.lut_dim - 1)
        lut_coords_ceil = torch.clamp(lut_coords_ceil, 0, self.lut_dim - 1)
        
        # Get 8 corner values using linear indexing to avoid view issues
        r0, g0, b0 = lut_coords_floor[:, 0], lut_coords_floor[:, 1], lut_coords_floor[:, 2]
        r1, g1, b1 = lut_coords_ceil[:, 0], lut_coords_ceil[:, 1], lut_coords_ceil[:, 2]
        
        # Flatten LUT for linear indexing: [lut_dim^3, 3]
        lut_flat = lut.reshape(-1, 3)  # [lut_dim^3, 3]
        
        # Convert 3D indices to linear indices: idx = r * dim^2 + g * dim + b
        dim = self.lut_dim
        idx000 = (r0 * dim * dim + g0 * dim + b0).long()
        idx001 = (r0 * dim * dim + g0 * dim + b1).long()
        idx010 = (r0 * dim * dim + g1 * dim + b0).long()
        idx011 = (r0 * dim * dim + g1 * dim + b1).long()
        idx100 = (r1 * dim * dim + g0 * dim + b0).long()
        idx101 = (r1 * dim * dim + g0 * dim + b1).long()
        idx110 = (r1 * dim * dim + g1 * dim + b0).long()
        idx111 = (r1 * dim * dim + g1 * dim + b1).long()
        
        # Use gather for efficient lookup
        c000 = lut_flat[idx000]
        c001 = lut_flat[idx001]
        c010 = lut_flat[idx010]
        c011 = lut_flat[idx011]
        c100 = lut_flat[idx100]
        c101 = lut_flat[idx101]
        c110 = lut_flat[idx110]
        c111 = lut_flat[idx111]
        
        # Interpolation weights
        wr = lut_coords_frac[:, 0:1]
        wg = lut_coords_frac[:, 1:2]
        wb = lut_coords_frac[:, 2:3]
        
        # Trilinear interpolation
        c00 = c000 * (1 - wb) + c001 * wb
        c01 = c010 * (1 - wb) + c011 * wb
        c10 = c100 * (1 - wb) + c101 * wb
        c11 = c110 * (1 - wb) + c111 * wb
        
        c0 = c00 * (1 - wg) + c01 * wg
        c1 = c10 * (1 - wg) + c11 * wg
        
        result = c0 * (1 - wr) + c1 * wr
        
        # Reshape back - ensure contiguous
        result = result.contiguous().reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
        
        return result
    
    def forward(self, x):
        """
        Apply film look to DSLR-style image
        
        Args:
            x: Input DSLR-style image [B, 3, H, W] in range [0, 1]
        
        Returns:
            Cinematic film look image [B, 3, H, W] in range [0, 1]
        """
        # Extract features for LUT weight prediction
        features = self.feature_extractor(x)
        features = features.reshape(features.size(0), -1)
        
        # Compute effective LUT
        if self.num_luts > 1:
            # Blend multiple LUTs based on input
            weights = self.lut_weight_predictor(features)  # [B, num_luts]
            
            # Weighted combination of LUTs
            lut = self.lut_identity.clone()
            for i in range(self.num_luts):
                lut = lut + weights[:, i:i+1].reshape(-1, 1, 1, 1) * self.lut_offsets[i]
            lut = lut.contiguous()
        else:
            # Single LUT - ensure contiguous
            lut = (self.lut_identity + self.lut_offsets[0]).contiguous()
        
        # Apply LUT
        lut_output = self.apply_lut(x, lut)
        
        # Refinement pass
        residual = self.refinement(lut_output)
        residual = residual * 0.1  # Small residual
        
        # Final output
        output = torch.clamp(lut_output + residual, 0, 1)
        
        return output
    
    def get_num_params(self):
        """Return the number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LightweightFilmNet(nn.Module):
    """
    Lightweight CNN-based film enhancement
    
    Simple encoder-decoder for fast film look application
    Good alternative to LUT-based approach
    
    Args:
        base_channels: Base number of channels (default: 32)
    """
    
    def __init__(self, base_channels=32):
        super(LightweightFilmNet, self).__init__()
        
        # Encoder
        self.enc1 = ConvBlock(3, base_channels, kernel_size=3)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 2),
            ResidualBlock(base_channels * 2),
            ChannelAttention(base_channels * 2)
        )
        
        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ConvBlock(base_channels * 2, base_channels, kernel_size=3)
        
        # Output
        self.output = nn.Sequential(
            ConvBlock(base_channels, base_channels, kernel_size=3),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Transform DSLR to film look
        
        Args:
            x: Input image [B, 3, H, W] in range [0, 1]
        
        Returns:
            Film look image [B, 3, H, W] in range [0, 1]
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.pool(e1)
        e2 = self.enc2(e2)
        
        # Bottleneck
        b = self.bottleneck(e2)
        
        # Decoder
        d1 = self.up(b)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=True)
        
        d1 = self.dec1(d1)
        
        # Output with skip connection
        out = self.output(d1)
        out = 0.7 * out + 0.3 * x  # Preserve some original
        out = torch.clamp(out, 0, 1)
        
        return out
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_stage2_model(model_type='lut', device='mps'):
    """
    Factory function to build Stage 2 model
    
    Args:
        model_type: 'lut' (LUT-based, recommended) or 'cnn' (CNN-based)
        device: Device to place model on
    
    Returns:
        Model instance on specified device
    """
    if model_type == 'lut':
        model = FilmLUTNetwork(lut_dim=33, num_luts=1)
    elif model_type == 'cnn':
        model = LightweightFilmNet(base_channels=32)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    num_params = model.get_num_params()
    print(f"[Stage2] Built {model_type} model with {num_params:,} parameters")
    
    return model


if __name__ == "__main__":
    # Test Stage 2 models
    print("=== Testing Stage 2 Models ===\n")
    
    batch_size = 2
    x = torch.rand(batch_size, 3, 224, 224)
    
    # Test LUT model
    print("Testing LUT-based Film Enhancement...")
    model_lut = FilmLUTNetwork(lut_dim=33, num_luts=1)
    with torch.no_grad():
        output = model_lut(x)
    print(f"Input: {x.shape}, Output: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"Parameters: {model_lut.get_num_params():,}\n")
    
    # Test CNN model
    print("Testing CNN-based Film Enhancement...")
    model_cnn = LightweightFilmNet(base_channels=32)
    with torch.no_grad():
        output = model_cnn(x)
    print(f"Input: {x.shape}, Output: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"Parameters: {model_cnn.get_num_params():,}\n")
    
    print("✓ All Stage 2 models tested successfully!")
    print("\nRecommendation:")
    print("  - Use 'lut' for best quality and efficiency")
    print("  - Use 'cnn' for more flexibility")

