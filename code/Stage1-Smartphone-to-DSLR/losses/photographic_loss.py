"""
Photographic Quality Losses
Losses designed to preserve natural photographic appearance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorHistogramLoss(nn.Module):
    """
    Preserves color distribution between images
    Ensures overall color palette matches target
    
    Args:
        num_bins: Number of histogram bins per channel
    """
    
    def __init__(self, num_bins=64):
        super().__init__()
        self.num_bins = num_bins
    
    def forward(self, pred, target):
        """
        Compute color histogram loss
        
        Args:
            pred: Predicted image [B, 3, H, W] in range [0, 1]
            target: Target image [B, 3, H, W] in range [0, 1]
        
        Returns:
            Color histogram loss
        """
        loss = 0.0
        batch_size = pred.size(0)
        
        for b in range(batch_size):
            for c in range(3):  # RGB channels
                # Compute histograms
                pred_hist = torch.histc(pred[b, c], bins=self.num_bins, min=0.0, max=1.0)
                target_hist = torch.histc(target[b, c], bins=self.num_bins, min=0.0, max=1.0)
                
                # Normalize to probability distributions
                pred_hist = pred_hist / (pred_hist.sum() + 1e-8)
                target_hist = target_hist / (target_hist.sum() + 1e-8)
                
                # L1 distance between distributions
                loss += torch.abs(pred_hist - target_hist).sum()
        
        return loss / (batch_size * 3)


class GradientLoss(nn.Module):
    """
    Preserves image gradients (edges and details)
    Helps maintain sharpness and structure
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        """
        Compute gradient loss
        
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
        
        Returns:
            Gradient loss
        """
        # Compute horizontal gradients
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        
        # Compute vertical gradients
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        # L1 loss on gradients
        loss_dx = F.l1_loss(pred_dx, target_dx)
        loss_dy = F.l1_loss(pred_dy, target_dy)
        
        return (loss_dx + loss_dy) / 2


class ToneMappingLoss(nn.Module):
    """
    Preserves proper tone mapping across luminance ranges
    Ensures shadows, midtones, and highlights are handled correctly
    """
    
    def __init__(self):
        super().__init__()
    
    def rgb_to_luminance(self, rgb):
        """Convert RGB to luminance (perceived brightness)"""
        # Standard luminance weights
        return 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    
    def forward(self, pred, target):
        """
        Compute tone mapping loss
        
        Separately weights errors in:
        - Shadows (0-0.3)
        - Midtones (0.3-0.7)
        - Highlights (0.7-1.0)
        """
        # Convert to luminance
        pred_lum = self.rgb_to_luminance(pred)
        target_lum = self.rgb_to_luminance(target)
        
        # Create masks for different tonal regions
        shadows_mask = (target_lum < 0.3).float()
        midtones_mask = ((target_lum >= 0.3) & (target_lum < 0.7)).float()
        highlights_mask = (target_lum >= 0.7).float()
        
        # Compute weighted losses
        shadows_loss = F.l1_loss(pred_lum * shadows_mask, target_lum * shadows_mask)
        midtones_loss = F.l1_loss(pred_lum * midtones_mask, target_lum * midtones_mask)
        highlights_loss = F.l1_loss(pred_lum * highlights_mask, target_lum * highlights_mask)
        
        # Weight: midtones most important, then shadows, then highlights
        return 0.5 * midtones_loss + 0.3 * shadows_loss + 0.2 * highlights_loss


class ColorConstancyLoss(nn.Module):
    """
    Encourages neutral colors to remain neutral
    Prevents color casts in grays
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        """
        Compute color constancy loss
        
        Penalizes color shifts in regions that should be achromatic
        """
        # Identify achromatic regions (low saturation in target)
        target_mean = target.mean(dim=1, keepdim=True)
        target_std = target.std(dim=1, keepdim=True)
        
        # Regions with low color variation are likely achromatic
        achromatic_mask = (target_std < 0.05).float()
        
        # In these regions, RGB channels should be similar
        pred_mean = pred.mean(dim=1, keepdim=True)
        pred_diff = pred - pred_mean
        
        # Penalize color variance in achromatic regions
        loss = (pred_diff.abs() * achromatic_mask).mean()
        
        return loss


class PhotographicLoss(nn.Module):
    """
    Combined loss for photographic quality
    
    Includes:
    - L1 Loss: Basic pixel accuracy
    - Perceptual Loss: High-level feature matching
    - Color Histogram: Preserves color distribution
    - Gradient Loss: Preserves edges and details
    - Tone Mapping: Proper shadows/midtones/highlights
    - Color Constancy: Prevents color casts
    
    Args:
        device: Device for computation
        perceptual_weight: Weight for perceptual loss
        color_hist_weight: Weight for color histogram loss
        gradient_weight: Weight for gradient loss
        tone_weight: Weight for tone mapping loss
        constancy_weight: Weight for color constancy loss
    """
    
    def __init__(self, 
                 device='mps',
                 perceptual_weight=0.3,
                 color_hist_weight=0.15,
                 gradient_weight=0.1,
                 tone_weight=0.1,
                 constancy_weight=0.05):
        super().__init__()
        
        # Import perceptual loss
        from .perceptual_loss import PerceptualLossVGG16
        
        # Initialize all loss functions
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLossVGG16(device=device)
        self.color_hist_loss = ColorHistogramLoss(num_bins=64)
        self.gradient_loss = GradientLoss()
        self.tone_loss = ToneMappingLoss()
        self.constancy_loss = ColorConstancyLoss()
        
        # Loss weights
        self.w_l1 = 1.0
        self.w_perceptual = perceptual_weight
        self.w_color_hist = color_hist_weight
        self.w_gradient = gradient_weight
        self.w_tone = tone_weight
        self.w_constancy = constancy_weight
        
        print(f"Photographic Loss initialized with weights:")
        print(f"  L1: {self.w_l1:.3f}")
        print(f"  Perceptual: {self.w_perceptual:.3f}")
        print(f"  Color Histogram: {self.w_color_hist:.3f}")
        print(f"  Gradient: {self.w_gradient:.3f}")
        print(f"  Tone Mapping: {self.w_tone:.3f}")
        print(f"  Color Constancy: {self.w_constancy:.3f}")
    
    def forward(self, pred, target):
        """
        Compute combined photographic loss
        
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        # Compute individual losses
        loss_l1 = self.l1_loss(pred, target)
        loss_perceptual = self.perceptual_loss(pred, target)
        loss_color_hist = self.color_hist_loss(pred, target)
        loss_gradient = self.gradient_loss(pred, target)
        loss_tone = self.tone_loss(pred, target)
        loss_constancy = self.constancy_loss(pred, target)
        
        # Combine with weights
        total_loss = (
            self.w_l1 * loss_l1 +
            self.w_perceptual * loss_perceptual +
            self.w_color_hist * loss_color_hist +
            self.w_gradient * loss_gradient +
            self.w_tone * loss_tone +
            self.w_constancy * loss_constancy
        )
        
        # Return loss and components for logging
        loss_dict = {
            'total': total_loss.item(),
            'l1': loss_l1.item(),
            'perceptual': loss_perceptual.item(),
            'color_hist': loss_color_hist.item(),
            'gradient': loss_gradient.item(),
            'tone': loss_tone.item(),
            'constancy': loss_constancy.item()
        }
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # Test photographic losses
    print("=== Testing Photographic Losses ===\n")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Create test images
    pred = torch.rand(2, 3, 224, 224, device=device)
    target = torch.rand(2, 3, 224, 224, device=device)
    
    # Test Color Histogram Loss
    print("1. Testing Color Histogram Loss...")
    color_hist_loss = ColorHistogramLoss()
    loss = color_hist_loss(pred, target)
    print(f"   Loss: {loss.item():.6f}\n")
    
    # Test Gradient Loss
    print("2. Testing Gradient Loss...")
    gradient_loss = GradientLoss()
    loss = gradient_loss(pred, target)
    print(f"   Loss: {loss.item():.6f}\n")
    
    # Test Tone Mapping Loss
    print("3. Testing Tone Mapping Loss...")
    tone_loss = ToneMappingLoss()
    loss = tone_loss(pred, target)
    print(f"   Loss: {loss.item():.6f}\n")
    
    # Test Color Constancy Loss
    print("4. Testing Color Constancy Loss...")
    constancy_loss = ColorConstancyLoss()
    loss = constancy_loss(pred, target)
    print(f"   Loss: {loss.item():.6f}\n")
    
    # Test Combined Photographic Loss
    print("5. Testing Combined Photographic Loss...")
    photo_loss = PhotographicLoss(device=device)
    total_loss, loss_dict = photo_loss(pred, target)
    print(f"   Total Loss: {total_loss.item():.6f}")
    print(f"   Components:")
    for name, value in loss_dict.items():
        if name != 'total':
            print(f"     {name}: {value:.6f}")
    
    print("\n✓ All tests passed!")









