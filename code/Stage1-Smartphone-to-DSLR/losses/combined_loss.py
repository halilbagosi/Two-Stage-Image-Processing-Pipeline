"""
Combined Loss Functions for Image Enhancement
Combines multiple loss terms: L1, Perceptual, SSIM, Color
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .perceptual_loss import PerceptualLossVGG16


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) Loss
    Measures perceived quality based on luminance, contrast, and structure
    
    Args:
        window_size: Size of Gaussian window (default: 11)
        channel: Number of image channels (default: 3)
    """
    
    def __init__(self, window_size=11, channel=3):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        
        # Create Gaussian window
        sigma = 1.5
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size//2)**2 / (2.0 * sigma**2)))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        
        # Create 2D window
        _2D_window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        window = _2D_window.unsqueeze(0).unsqueeze(0).expand(channel, 1, window_size, window_size)
        
        self.register_buffer('window', window)
        self.padding = window_size // 2
    
    def forward(self, pred, target):
        """
        Compute SSIM loss
        
        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
        
        Returns:
            SSIM loss (1 - SSIM)
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Ensure window is on same device as input
        window = self.window.to(pred.device)
        
        # Mean
        mu1 = F.conv2d(pred, window, padding=self.padding, groups=self.channel)
        mu2 = F.conv2d(target, window, padding=self.padding, groups=self.channel)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Variance and covariance
        sigma1_sq = F.conv2d(pred * pred, window, padding=self.padding, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.padding, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.padding, groups=self.channel) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # Return 1 - SSIM as loss (so minimizing improves SSIM)
        return 1 - ssim_map.mean()


class ColorLoss(nn.Module):
    """
    Color Consistency Loss
    Ensures color statistics match between predicted and target images
    Compares color histograms in LAB color space
    """
    
    def __init__(self):
        super(ColorLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
    
    def rgb_to_lab(self, rgb):
        """
        Convert RGB to LAB color space (simplified)
        This is an approximation for differentiable operations
        """
        # Normalize to [0, 1] if not already
        rgb = torch.clamp(rgb, 0, 1)
        
        # Simple RGB to LAB approximation
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        
        # L channel (lightness)
        L = 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        # A channel (green-red)
        A = (r - g) * 0.5
        
        # B channel (blue-yellow)
        B = (r + g) * 0.5 - b
        
        return torch.cat([L, A, B], dim=1)
    
    def forward(self, pred, target):
        """
        Compute color loss in LAB space
        
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
        
        Returns:
            Color loss
        """
        # Convert to LAB
        pred_lab = self.rgb_to_lab(pred)
        target_lab = self.rgb_to_lab(target)
        
        # Compute mean and std for each channel
        pred_mean = pred_lab.mean(dim=[2, 3])
        target_mean = target_lab.mean(dim=[2, 3])
        
        pred_std = pred_lab.std(dim=[2, 3])
        target_std = target_lab.std(dim=[2, 3])
        
        # Loss on mean and std
        mean_loss = self.l1_loss(pred_mean, target_mean)
        std_loss = self.l1_loss(pred_std, target_std)
        
        return mean_loss + std_loss


class CombinedLoss(nn.Module):
    """
    Combined loss for image enhancement
    Combines multiple loss terms with configurable weights
    
    Loss = w_l1 * L1 + w_perceptual * Perceptual + w_ssim * SSIM + w_color * Color
    
    Args:
        w_l1: Weight for L1 loss (default: 1.0)
        w_perceptual: Weight for perceptual loss (default: 0.1)
        w_ssim: Weight for SSIM loss (default: 0.5)
        w_color: Weight for color loss (default: 0.1)
        device: Device for perceptual loss model (default: 'mps')
    """
    
    def __init__(self, w_l1=1.0, w_perceptual=0.1, w_ssim=0.5, w_color=0.1, device='mps'):
        super(CombinedLoss, self).__init__()
        
        self.w_l1 = w_l1
        self.w_perceptual = w_perceptual
        self.w_ssim = w_ssim
        self.w_color = w_color
        
        # Individual loss functions
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLossVGG16(device=device)
        self.ssim_loss = SSIMLoss()
        self.color_loss = ColorLoss()
        
        print(f"[CombinedLoss] Initialized with weights:")
        print(f"  L1: {w_l1}, Perceptual: {w_perceptual}, SSIM: {w_ssim}, Color: {w_color}")
    
    def forward(self, pred, target):
        """
        Compute combined loss
        
        Args:
            pred: Predicted image [B, 3, H, W] in range [0, 1]
            target: Target image [B, 3, H, W] in range [0, 1]
        
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary of individual loss components
        """
        # Compute individual losses
        l1 = self.l1_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        color = self.color_loss(pred, target)
        
        # Combined loss
        total_loss = (
            self.w_l1 * l1 +
            self.w_perceptual * perceptual +
            self.w_ssim * ssim +
            self.w_color * color
        )
        
        # Return loss dict for logging
        loss_dict = {
            'total': total_loss.item(),
            'l1': l1.item(),
            'perceptual': perceptual.item(),
            'ssim': ssim.item(),
            'color': color.item()
        }
        
        return total_loss, loss_dict


class SimplifiedLoss(nn.Module):
    """
    Simplified loss for faster training (good for initial experiments)
    Only uses L1 and SSIM (no perceptual loss = faster)
    """
    
    def __init__(self, w_l1=1.0, w_ssim=0.5):
        super(SimplifiedLoss, self).__init__()
        
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        
        print(f"[SimplifiedLoss] L1: {w_l1}, SSIM: {w_ssim}")
    
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        
        total_loss = self.w_l1 * l1 + self.w_ssim * ssim
        
        loss_dict = {
            'total': total_loss.item(),
            'l1': l1.item(),
            'ssim': ssim.item()
        }
        
        return total_loss, loss_dict


def build_loss_function(loss_type='combined', device='mps', **kwargs):
    """
    Factory function to build loss function
    
    Args:
        loss_type: 'dped' (default, DPED loss), 'combined', 'simplified', 'photographic', or 'style_preserving'
        device: Device for perceptual loss model
        **kwargs: Additional arguments for loss function
    
    Returns:
        Loss function instance
    """
    if loss_type == 'combined':
        return CombinedLoss(device=device, **kwargs)
    elif loss_type == 'simplified':
        return SimplifiedLoss(**kwargs)
    elif loss_type == 'photographic':
        from .photographic_loss import PhotographicLoss
        return PhotographicLoss(device=device, **kwargs)
    elif loss_type == 'style_preserving':
        from .style_preserving_loss import StylePreservingLoss
        return StylePreservingLoss(device=device, **kwargs)
    elif loss_type == 'dped':
        from .dped_loss import DPEDLoss
        return DPEDLoss(device=device, **kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from: 'combined', 'simplified', 'photographic', 'style_preserving', 'dped'")


if __name__ == "__main__":
    # Test loss functions
    print("=== Testing Loss Functions ===\n")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Create dummy images
    batch_size = 2
    pred = torch.rand(batch_size, 3, 224, 224, device=device)
    target = torch.rand(batch_size, 3, 224, 224, device=device)
    
    # Test SSIM loss
    print("Testing SSIM Loss...")
    ssim_loss = SSIMLoss()
    ssim_loss = ssim_loss.to(device)
    loss = ssim_loss(pred, target)
    print(f"SSIM Loss: {loss.item():.6f}\n")
    
    # Test Color loss
    print("Testing Color Loss...")
    color_loss = ColorLoss()
    color_loss = color_loss.to(device)
    loss = color_loss(pred, target)
    print(f"Color Loss: {loss.item():.6f}\n")
    
    # Test Combined loss
    print("Testing Combined Loss...")
    combined_loss = CombinedLoss(device=device)
    total_loss, loss_dict = combined_loss(pred, target)
    print(f"Total Loss: {total_loss.item():.6f}")
    print("Loss components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.6f}")
    print()
    
    # Test Simplified loss
    print("Testing Simplified Loss...")
    simple_loss = SimplifiedLoss()
    simple_loss = simple_loss.to(device)
    total_loss, loss_dict = simple_loss(pred, target)
    print(f"Total Loss: {total_loss.item():.6f}")
    print("Loss components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.6f}")
    
    print("\n✓ All loss functions tested successfully!")

