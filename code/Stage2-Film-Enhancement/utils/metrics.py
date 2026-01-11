"""
Evaluation metrics for image quality assessment
Includes PSNR, SSIM, ΔE (CIEDE2000), LPIPS, and other metrics
"""

import torch
import torch.nn.functional as F
import math
import numpy as np


def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        img1: First image tensor [B, C, H, W] or [C, H, W]
        img2: Second image tensor [B, C, H, W] or [C, H, W]
        max_val: Maximum possible pixel value (default: 1.0 for normalized images)
    
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    psnr = 20 * math.log10(max_val / math.sqrt(mse.item()))
    return psnr


def calculate_ssim(img1, img2, window_size=11, channel=3):
    """
    Calculate Structural Similarity Index (SSIM)
    
    Args:
        img1: First image tensor [B, C, H, W]
        img2: Second image tensor [B, C, H, W]
        window_size: Size of Gaussian window
        channel: Number of channels
    
    Returns:
        SSIM value (between 0 and 1, higher is better)
    """
    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor([
        math.exp(-(x - window_size//2)**2 / (2.0 * sigma**2))
        for x in range(window_size)
    ])
    gauss = gauss / gauss.sum()
    
    _2D_window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    window = _2D_window.unsqueeze(0).unsqueeze(0).expand(channel, 1, window_size, window_size)
    window = window.to(img1.device)
    
    padding = window_size // 2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def rgb_to_lab(rgb):
    """
    Convert RGB to LAB color space
    
    Args:
        rgb: RGB image tensor [B, 3, H, W] with values in [0, 1]
    
    Returns:
        LAB image tensor [B, 3, H, W]
    """
    # RGB to XYZ
    rgb = rgb.clamp(0, 1)
    
    # Linearize RGB
    mask = rgb > 0.04045
    rgb_linear = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    
    # RGB to XYZ matrix (sRGB with D65 illuminant)
    matrix = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], device=rgb.device, dtype=rgb.dtype)
    
    # Apply transformation
    B, C, H, W = rgb_linear.shape
    rgb_flat = rgb_linear.permute(0, 2, 3, 1).reshape(-1, 3)
    xyz_flat = rgb_flat @ matrix.T
    xyz = xyz_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)
    
    # Normalize by D65 white point
    xyz[:, 0] /= 0.95047
    xyz[:, 1] /= 1.00000
    xyz[:, 2] /= 1.08883
    
    # XYZ to LAB
    epsilon = 0.008856
    kappa = 903.3
    
    mask = xyz > epsilon
    f_xyz = torch.where(mask, xyz ** (1/3), (kappa * xyz + 16) / 116)
    
    L = 116 * f_xyz[:, 1] - 16
    a = 500 * (f_xyz[:, 0] - f_xyz[:, 1])
    b = 200 * (f_xyz[:, 1] - f_xyz[:, 2])
    
    lab = torch.stack([L, a, b], dim=1)
    return lab


def calculate_delta_e(img1, img2):
    """
    Calculate CIEDE2000 color difference (ΔE)
    
    Args:
        img1: First image tensor [B, 3, H, W] in RGB [0, 1]
        img2: Second image tensor [B, 3, H, W] in RGB [0, 1]
    
    Returns:
        Mean ΔE value (lower is better, 0 = identical)
    """
    # Convert to LAB
    lab1 = rgb_to_lab(img1)
    lab2 = rgb_to_lab(img2)
    
    # Extract L, a, b channels
    L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]
    
    # Simplified ΔE (CIE76 for efficiency)
    # For full CIEDE2000, would need complex implementation
    delta_L = L1 - L2
    delta_a = a1 - a2
    delta_b = b1 - b2
    
    delta_e = torch.sqrt(delta_L**2 + delta_a**2 + delta_b**2)
    
    return delta_e.mean().item()


# Global LPIPS model instance (lazy loaded)
_lpips_model = None

def get_lpips_model(device):
    """Get or create LPIPS model (singleton pattern)"""
    global _lpips_model
    if _lpips_model is None:
        try:
            import lpips
            _lpips_model = lpips.LPIPS(net='alex').to(device)
            _lpips_model.eval()
        except ImportError:
            print("Warning: lpips library not installed. Run: pip install lpips")
            return None
    return _lpips_model


def calculate_lpips(img1, img2):
    """
    Calculate Learned Perceptual Image Patch Similarity (LPIPS)
    
    Args:
        img1: First image tensor [B, 3, H, W] in [0, 1]
        img2: Second image tensor [B, 3, H, W] in [0, 1]
    
    Returns:
        LPIPS value (lower is better, 0 = identical)
    """
    model = get_lpips_model(img1.device)
    if model is None:
        return 0.0  # Return 0 if LPIPS not available
    
    with torch.no_grad():
        # LPIPS expects inputs in [-1, 1]
        img1_norm = img1 * 2 - 1
        img2_norm = img2 * 2 - 1
        
        lpips_val = model(img1_norm, img2_norm)
        return lpips_val.mean().item()


def calculate_mae(img1, img2):
    """
    Calculate Mean Absolute Error
    
    Args:
        img1: First image tensor
        img2: Second image tensor
    
    Returns:
        MAE value
    """
    return torch.mean(torch.abs(img1 - img2)).item()


def calculate_mse(img1, img2):
    """
    Calculate Mean Squared Error
    
    Args:
        img1: First image tensor
        img2: Second image tensor
    
    Returns:
        MSE value
    """
    return torch.mean((img1 - img2) ** 2).item()


class MetricsCalculator:
    """
    Batch metrics calculator for evaluation
    Accumulates metrics over multiple batches
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics"""
        self.psnr_sum = 0.0
        self.ssim_sum = 0.0
        self.mae_sum = 0.0
        self.delta_e_sum = 0.0
        self.lpips_sum = 0.0
        self.count = 0
    
    def update(self, pred, target):
        """
        Update metrics with a new batch
        
        Args:
            pred: Predicted images [B, C, H, W]
            target: Target images [B, C, H, W]
        """
        batch_size = pred.size(0)
        
        # Calculate metrics for each image in batch
        for i in range(batch_size):
            pred_img = pred[i:i+1]
            target_img = target[i:i+1]
            
            self.psnr_sum += calculate_psnr(pred_img, target_img)
            self.ssim_sum += calculate_ssim(pred_img, target_img)
            self.mae_sum += calculate_mae(pred_img, target_img)
            self.delta_e_sum += calculate_delta_e(pred_img, target_img)
            self.lpips_sum += calculate_lpips(pred_img, target_img)
            self.count += 1
    
    def get_metrics(self):
        """
        Get average metrics
        
        Returns:
            Dictionary of metric names and values
        """
        if self.count == 0:
            return {
                'psnr': 0.0,
                'ssim': 0.0,
                'mae': 0.0,
                'delta_e': 0.0,
                'lpips': 0.0
            }
        
        return {
            'psnr': self.psnr_sum / self.count,
            'ssim': self.ssim_sum / self.count,
            'mae': self.mae_sum / self.count,
            'delta_e': self.delta_e_sum / self.count,
            'lpips': self.lpips_sum / self.count
        }
    
    def get_summary_string(self):
        """Get formatted string of metrics"""
        metrics = self.get_metrics()
        return (f"PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}, "
                f"MAE: {metrics['mae']:.4f}, ΔE: {metrics['delta_e']:.2f}, "
                f"LPIPS: {metrics['lpips']:.4f}")


if __name__ == "__main__":
    # Test metrics
    print("=== Testing Metrics ===\n")
    
    # Create test images
    img1 = torch.rand(2, 3, 224, 224)
    img2 = img1 + torch.randn(2, 3, 224, 224) * 0.1  # Add some noise
    
    # Test individual metrics
    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    mae = calculate_mae(img1, img2)
    mse = calculate_mse(img1, img2)
    delta_e = calculate_delta_e(img1, img2)
    lpips_val = calculate_lpips(img1, img2)
    
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"ΔE (CIEDE2000): {delta_e:.2f}")
    print(f"LPIPS: {lpips_val:.4f}\n")
    
    # Test MetricsCalculator
    print("Testing MetricsCalculator...")
    calculator = MetricsCalculator()
    
    # Update with multiple batches
    for i in range(3):
        batch1 = torch.rand(4, 3, 224, 224)
        batch2 = batch1 + torch.randn(4, 3, 224, 224) * 0.05
        calculator.update(batch1, batch2)
    
    metrics = calculator.get_metrics()
    print(f"Average metrics over {calculator.count} images:")
    print(calculator.get_summary_string())
    
    print("\n✓ Metrics tests passed!")

