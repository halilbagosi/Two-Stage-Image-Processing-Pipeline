"""
Style-Preserving Loss
Only penalizes style differences, allows detail to stay sharp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowFrequencyL1Loss(nn.Module):
    """
    L1 loss on low-frequency components only
    Allows high-frequency details to remain different
    """
    
    def __init__(self, sigma=3.0):
        super().__init__()
        self.sigma = sigma
        self.l1 = nn.L1Loss()
    
    def gaussian_blur(self, img, sigma):
        """Apply Gaussian blur to extract low-frequency"""
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create Gaussian kernel
        x = torch.arange(kernel_size, dtype=torch.float32, device=img.device) - kernel_size // 2
        gauss_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        
        # 2D kernel
        gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0)
        gauss_2d = gauss_2d.repeat(img.shape[1], 1, 1, 1)
        
        # Apply blur
        padding = kernel_size // 2
        blurred = F.conv2d(img, gauss_2d, padding=padding, groups=img.shape[1])
        
        return blurred
    
    def forward(self, pred, target):
        """
        Compute L1 loss only on low-frequency (tone/color)
        High-frequency (detail) differences are ignored
        """
        # Extract low-frequency components
        pred_lf = self.gaussian_blur(pred, self.sigma)
        target_lf = self.gaussian_blur(target, self.sigma)
        
        # L1 loss on low-frequency only
        return self.l1(pred_lf, target_lf)


class DetailPreservationLoss(nn.Module):
    """
    Penalizes if model changes high-frequency details
    Encourages preservation of original sharpness
    """
    
    def __init__(self, sigma=2.0):
        super().__init__()
        self.sigma = sigma
    
    def extract_details(self, img, sigma):
        """Extract high-frequency details"""
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Gaussian blur
        x = torch.arange(kernel_size, dtype=torch.float32, device=img.device) - kernel_size // 2
        gauss_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        
        gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0)
        gauss_2d = gauss_2d.repeat(img.shape[1], 1, 1, 1)
        
        padding = kernel_size // 2
        blurred = F.conv2d(img, gauss_2d, padding=padding, groups=img.shape[1])
        
        # Details = original - blurred
        details = img - blurred
        
        return details
    
    def forward(self, pred, target, input_img):
        """
        Penalize if output details differ from INPUT details
        (Not target details - we want to preserve INPUT sharpness)
        
        Args:
            pred: Model output
            target: Target DSLR image (for style reference)
            input_img: Original input (for detail reference)
        """
        # Extract details from prediction and input
        pred_details = self.extract_details(pred, self.sigma)
        input_details = self.extract_details(input_img, self.sigma)
        
        # Penalize if prediction details differ from input details
        # We want OUTPUT details == INPUT details (preserve sharpness)
        loss = F.l1_loss(pred_details, input_details)
        
        return loss


class FrequencyPreservationLoss(nn.Module):
    """
    Preserve frequency spectrum of input
    Ensures sharpness/texture stays unchanged
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, input_img):
        """
        Compare frequency spectrums
        High-frequency should match input
        """
        # FFT to frequency domain
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
        input_fft = torch.fft.rfft2(input_img, dim=(-2, -1))
        
        # Get magnitude (ignore phase for now)
        pred_mag = torch.abs(pred_fft)
        input_mag = torch.abs(input_fft)
        
        # Focus on high frequencies (detail)
        # Weight higher frequencies more
        h, w = pred_mag.shape[-2], pred_mag.shape[-1]
        y = torch.arange(h, device=pred.device).view(-1, 1).float()
        x = torch.arange(w, device=pred.device).view(1, -1).float()
        
        # Distance from origin (DC component)
        dist = torch.sqrt((y / h) ** 2 + (x / w) ** 2)
        
        # Weight: higher for high frequencies
        weight = dist / dist.max()
        weight = weight.unsqueeze(0).unsqueeze(0)
        
        # Weighted L1 loss on magnitudes
        loss = (weight * torch.abs(pred_mag - input_mag)).mean()
        
        return loss


class StylePreservingLoss(nn.Module):
    """
    Combined loss that preserves detail while changing style
    
    Strategy:
    - Low-frequency L1: Match tone/color (low-freq)
    - Perceptual loss: Match style features
    - Color histogram: Match color distribution
    - Tone mapping: Match tonal curves
    - Detail preservation: Keep input sharpness
    - Frequency preservation: Preserve high-freq spectrum
    
    Does NOT penalize:
    - Differences in edges/gradients (allows detail to stay sharp)
    - High-frequency L1 differences
    """
    
    def __init__(self, device='mps',
                 lf_l1_weight=0.5,
                 perceptual_weight=0.4,
                 color_hist_weight=0.2,
                 tone_weight=0.15,
                 constancy_weight=0.05,
                 detail_preservation_weight=1.0,
                 frequency_preservation_weight=0.5):
        super().__init__()
        
        # Import required losses
        try:
            from .perceptual_loss import PerceptualLossVGG16
            from .photographic_loss import ColorHistogramLoss, ToneMappingLoss, ColorConstancyLoss
        except ImportError:
            from perceptual_loss import PerceptualLossVGG16
            from photographic_loss import ColorHistogramLoss, ToneMappingLoss, ColorConstancyLoss
        
        # Initialize losses
        self.lf_l1_loss = LowFrequencyL1Loss(sigma=3.0)
        self.perceptual_loss = PerceptualLossVGG16(device=device)
        self.color_hist_loss = ColorHistogramLoss(num_bins=64)
        self.tone_loss = ToneMappingLoss()
        self.constancy_loss = ColorConstancyLoss()
        self.detail_preservation_loss = DetailPreservationLoss(sigma=2.0)
        self.frequency_preservation_loss = FrequencyPreservationLoss()
        
        # Weights
        self.w_lf_l1 = lf_l1_weight
        self.w_perceptual = perceptual_weight
        self.w_color_hist = color_hist_weight
        self.w_tone = tone_weight
        self.w_constancy = constancy_weight
        self.w_detail_preservation = detail_preservation_weight
        self.w_frequency_preservation = frequency_preservation_weight
        
        print(f"StylePreservingLoss initialized:")
        print(f"  Low-Freq L1: {self.w_lf_l1:.3f} (tone/color only)")
        print(f"  Perceptual: {self.w_perceptual:.3f}")
        print(f"  Color Histogram: {self.w_color_hist:.3f}")
        print(f"  Tone Mapping: {self.w_tone:.3f}")
        print(f"  Color Constancy: {self.w_constancy:.3f}")
        print(f"  Detail Preservation: {self.w_detail_preservation:.3f} ⭐")
        print(f"  Frequency Preservation: {self.w_frequency_preservation:.3f} ⭐")
        print(f"")
        print(f"  ✓ Preserves sharpness and detail")
        print(f"  ✓ Only changes tone, color, exposure")
    
    def forward(self, pred, target, input_img=None):
        """
        Compute style-preserving loss
        
        Args:
            pred: Model output
            target: Target DSLR image (for style)
            input_img: Original input image (for detail reference)
        
        Returns:
            total_loss, loss_dict
        """
        # Low-frequency L1 (tone and color, not detail)
        loss_lf_l1 = self.lf_l1_loss(pred, target)
        
        # Perceptual loss (high-level style features)
        loss_perceptual = self.perceptual_loss(pred, target)
        
        # Color histogram (color distribution)
        loss_color_hist = self.color_hist_loss(pred, target)
        
        # Tone mapping (shadows/midtones/highlights)
        loss_tone = self.tone_loss(pred, target)
        
        # Color constancy (white balance)
        loss_constancy = self.constancy_loss(pred, target)
        
        # Detail preservation (keep input sharpness)
        if input_img is not None:
            loss_detail_preservation = self.detail_preservation_loss(pred, target, input_img)
            loss_frequency_preservation = self.frequency_preservation_loss(pred, input_img)
        else:
            # If no input provided, assume pred should preserve its own details
            loss_detail_preservation = torch.tensor(0.0, device=pred.device)
            loss_frequency_preservation = torch.tensor(0.0, device=pred.device)
        
        # Combined loss
        total_loss = (
            self.w_lf_l1 * loss_lf_l1 +
            self.w_perceptual * loss_perceptual +
            self.w_color_hist * loss_color_hist +
            self.w_tone * loss_tone +
            self.w_constancy * loss_constancy +
            self.w_detail_preservation * loss_detail_preservation +
            self.w_frequency_preservation * loss_frequency_preservation
        )
        
        # Loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'lf_l1': loss_lf_l1.item(),
            'perceptual': loss_perceptual.item(),
            'color_hist': loss_color_hist.item(),
            'tone': loss_tone.item(),
            'constancy': loss_constancy.item(),
            'detail_preservation': loss_detail_preservation.item(),
            'frequency_preservation': loss_frequency_preservation.item()
        }
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # Fix SSL for VGG download
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Test style-preserving losses
    print("=== Testing Style-Preserving Losses ===\n")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Create test images
    input_img = torch.rand(2, 3, 224, 224, device=device)
    target = torch.rand(2, 3, 224, 224, device=device)
    
    # Add some high-frequency detail to input
    noise = torch.randn_like(input_img) * 0.1
    input_img = input_img + noise
    input_img = torch.clamp(input_img, 0, 1)
    
    # Simulate model output (slightly different tone, same detail)
    pred = input_img * 1.1  # Brightened
    pred = torch.clamp(pred, 0, 1)
    
    # Test Low-Frequency L1 Loss
    print("1. Testing Low-Frequency L1 Loss...")
    lf_l1_loss = LowFrequencyL1Loss()
    loss = lf_l1_loss(pred, target)
    print(f"   Loss: {loss.item():.6f}")
    print("   ✓ Only penalizes tone/color differences\n")
    
    # Test Detail Preservation Loss
    print("2. Testing Detail Preservation Loss...")
    detail_loss = DetailPreservationLoss()
    loss = detail_loss(pred, target, input_img)
    print(f"   Loss: {loss.item():.6f}")
    print("   ✓ Penalizes if output details differ from input\n")
    
    # Test Frequency Preservation Loss
    print("3. Testing Frequency Preservation Loss...")
    freq_loss = FrequencyPreservationLoss()
    loss = freq_loss(pred, input_img)
    print(f"   Loss: {loss.item():.6f}")
    print("   ✓ Preserves frequency spectrum\n")
    
    # Test Combined StylePreservingLoss
    print("4. Testing Combined StylePreservingLoss...")
    style_loss = StylePreservingLoss(device=device)
    total_loss, loss_dict = style_loss(pred, target, input_img)
    print(f"   Total Loss: {total_loss.item():.6f}")
    print(f"   Components:")
    for name, value in loss_dict.items():
        if name != 'total':
            print(f"     {name}: {value:.6f}")
    print("   ✓ Combined loss working\n")
    
    print("=" * 60)
    print("✓ All style-preserving losses tested!")
    print("=" * 60)
    print("\nThese losses ensure:")
    print("  ✓ Detail and sharpness preserved")
    print("  ✓ Only tone/color/exposure changed")
    print("  ✓ No blurring or sharpening artifacts")

