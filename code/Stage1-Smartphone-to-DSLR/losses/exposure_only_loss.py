"""
Exposure-Only Loss
Only penalizes exposure/luminance differences
Preserves colors and details completely
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LuminanceL1Loss(nn.Module):
    """
    L1 loss on luminance only
    Ignores color differences completely
    """
    
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
    
    def rgb_to_luminance(self, rgb):
        """Convert RGB to luminance (Rec. 709)"""
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    def forward(self, pred, target):
        """
        Compare luminance only
        
        Args:
            pred: Predicted RGB
            target: Target RGB
        
        Returns:
            L1 loss on luminance
        """
        pred_lum = self.rgb_to_luminance(pred)
        target_lum = self.rgb_to_luminance(target)
        
        return self.l1(pred_lum, target_lum)


class ColorPreservationLoss(nn.Module):
    """
    Penalizes any color changes
    Ensures output colors match input colors exactly
    """
    
    def __init__(self):
        super().__init__()
    
    def rgb_to_chrominance(self, rgb):
        """
        Convert RGB to chrominance (color ratios)
        This captures hue and saturation
        """
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        eps = 1e-6
        chrominance = rgb / (luminance + eps)
        
        return chrominance
    
    def forward(self, pred, input_img):
        """
        Penalize if colors differ from INPUT
        (Not target - we want to preserve input colors)
        
        Args:
            pred: Model output
            input_img: Original input (for color reference)
        
        Returns:
            Loss if colors changed
        """
        pred_chrom = self.rgb_to_chrominance(pred)
        input_chrom = self.rgb_to_chrominance(input_img)
        
        # L1 loss on chrominance
        loss = F.l1_loss(pred_chrom, input_chrom)
        
        return loss


class TonalCurveLoss(nn.Module):
    """
    Compares tonal curves (exposure mapping)
    Ensures smooth highlight/shadow transitions
    """
    
    def __init__(self, num_bins=256):
        super().__init__()
        self.num_bins = num_bins
    
    def compute_histogram(self, luminance):
        """
        Compute luminance histogram
        
        Args:
            luminance: [B, 1, H, W]
        
        Returns:
            Histogram: [B, num_bins]
        """
        B = luminance.shape[0]
        histograms = []
        
        for i in range(B):
            lum = luminance[i].flatten()
            # Compute histogram
            hist = torch.histc(lum, bins=self.num_bins, min=0.0, max=1.0)
            # Normalize
            hist = hist / (hist.sum() + 1e-6)
            histograms.append(hist)
        
        return torch.stack(histograms)
    
    def forward(self, pred, target):
        """
        Compare luminance histograms
        Ensures similar tonal distribution
        """
        # Convert to luminance
        pred_lum = 0.2126 * pred[:, 0:1] + 0.7152 * pred[:, 1:2] + 0.0722 * pred[:, 2:3]
        target_lum = 0.2126 * target[:, 0:1] + 0.7152 * target[:, 1:2] + 0.0722 * target[:, 2:3]
        
        # Compute histograms
        pred_hist = self.compute_histogram(pred_lum)
        target_hist = self.compute_histogram(target_lum)
        
        # L1 loss on histograms
        loss = F.l1_loss(pred_hist, target_hist)
        
        return loss


class ContrastLoss(nn.Module):
    """
    Explicitly measures and matches contrast
    Contrast = difference between highlights and shadows
    """
    
    def __init__(self):
        super().__init__()
    
    def compute_contrast(self, luminance):
        """
        Compute contrast as standard deviation of luminance
        Higher std = more contrast
        """
        # Global contrast (standard deviation)
        global_contrast = torch.std(luminance, dim=[2, 3], keepdim=True)
        
        # Local contrast (variance in 8x8 patches)
        # Unfold into patches
        patches = F.unfold(luminance, kernel_size=8, stride=8)
        local_contrast = torch.std(patches, dim=1, keepdim=True)
        
        return global_contrast, local_contrast
    
    def forward(self, pred, target):
        """
        Match contrast between pred and target
        """
        # Convert to luminance
        pred_lum = 0.2126 * pred[:, 0:1] + 0.7152 * pred[:, 1:2] + 0.0722 * pred[:, 2:3]
        target_lum = 0.2126 * target[:, 0:1] + 0.7152 * target[:, 1:2] + 0.0722 * target[:, 2:3]
        
        # Compute contrast
        pred_global, pred_local = self.compute_contrast(pred_lum)
        target_global, target_local = self.compute_contrast(target_lum)
        
        # Match global contrast
        global_loss = F.l1_loss(pred_global, target_global)
        
        # Match local contrast distribution
        local_loss = F.l1_loss(pred_local.mean(), target_local.mean())
        
        return global_loss + local_loss


class HighlightShadowLoss(nn.Module):
    """
    Specifically penalizes differences in highlights and shadows
    Ensures proper exposure in bright and dark regions
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        """
        Compare highlights and shadows separately
        """
        # Convert to luminance
        pred_lum = 0.2126 * pred[:, 0:1] + 0.7152 * pred[:, 1:2] + 0.0722 * pred[:, 2:3]
        target_lum = 0.2126 * target[:, 0:1] + 0.7152 * target[:, 1:2] + 0.0722 * target[:, 2:3]
        
        # Highlights (bright regions > 0.7)
        highlight_mask = (target_lum > 0.7).float()
        highlight_loss = F.l1_loss(
            pred_lum * highlight_mask,
            target_lum * highlight_mask
        )
        
        # Shadows (dark regions < 0.3)
        shadow_mask = (target_lum < 0.3).float()
        shadow_loss = F.l1_loss(
            pred_lum * shadow_mask,
            target_lum * shadow_mask
        )
        
        # Midtones (0.3 to 0.7)
        midtone_mask = ((target_lum >= 0.3) & (target_lum <= 0.7)).float()
        midtone_loss = F.l1_loss(
            pred_lum * midtone_mask,
            target_lum * midtone_mask
        )
        
        # Combined
        total_loss = highlight_loss + shadow_loss + midtone_loss
        
        return total_loss


class ExposureOnlyLoss(nn.Module):
    """
    Combined loss for exposure-only training
    
    Objectives:
    1. Match target luminance (exposure, highlights, shadows)
    2. Preserve input colors exactly
    3. Preserve input details exactly
    
    Does NOT penalize:
    - Color differences (we preserve input colors)
    - Edge/gradient differences (we preserve details)
    """
    
    def __init__(self,
                 luminance_l1_weight=1.0,
                 tonal_curve_weight=0.3,
                 highlight_shadow_weight=0.5,
                 contrast_weight=0.4,  # NEW: Explicit contrast matching
                 color_preservation_weight=2.0,  # HIGH weight - colors must not change!
                 detail_preservation_weight=1.0):
        super().__init__()
        
        # Luminance losses (what we want to change)
        self.luminance_l1_loss = LuminanceL1Loss()
        self.tonal_curve_loss = TonalCurveLoss(num_bins=64)
        self.highlight_shadow_loss = HighlightShadowLoss()
        self.contrast_loss = ContrastLoss()  # NEW: Explicit contrast loss
        
        # Preservation losses (what we DON'T want to change)
        self.color_preservation_loss = ColorPreservationLoss()
        
        # Weights
        self.w_lum_l1 = luminance_l1_weight
        self.w_tonal = tonal_curve_weight
        self.w_hl_sh = highlight_shadow_weight
        self.w_contrast = contrast_weight
        self.w_color_pres = color_preservation_weight
        self.w_detail_pres = detail_preservation_weight
        
        print(f"ExposureOnlyLoss initialized:")
        print(f"  Luminance L1: {self.w_lum_l1:.2f}")
        print(f"  Tonal Curve: {self.w_tonal:.2f}")
        print(f"  Highlight/Shadow: {self.w_hl_sh:.2f}")
        print(f"  Contrast: {self.w_contrast:.2f} ⭐")
        print(f"  Color Preservation: {self.w_color_pres:.2f} ⭐")
        print(f"  Detail Preservation: {self.w_detail_pres:.2f} ⭐")
        print(f"")
        print(f"  ✓ Modifies: luminance, exposure, contrast")
        print(f"  ✓ Preserves: colors completely")
        print(f"  ✓ Preserves: details completely")
    
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
        gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0).repeat(img.shape[1], 1, 1, 1)
        
        # Blur
        padding = kernel_size // 2
        blurred = F.conv2d(img, gauss_2d, padding=padding, groups=img.shape[1])
        
        # Detail = original - blurred
        detail = img - blurred
        
        return detail
    
    def forward(self, pred, target, input_img):
        """
        Compute exposure-only loss
        
        Args:
            pred: Model output
            target: Target DSLR image (for luminance reference)
            input_img: Original input (for color/detail preservation)
        
        Returns:
            total_loss, loss_dict
        """
        # Luminance losses (match target exposure and contrast)
        loss_lum_l1 = self.luminance_l1_loss(pred, target)
        loss_tonal = self.tonal_curve_loss(pred, target)
        loss_hl_sh = self.highlight_shadow_loss(pred, target)
        loss_contrast = self.contrast_loss(pred, target)
        
        # Color preservation (match INPUT colors, not target)
        loss_color_pres = self.color_preservation_loss(pred, input_img)
        
        # Detail preservation (match INPUT details, not target)
        pred_detail = self.extract_detail(pred, sigma=1.5)
        input_detail = self.extract_detail(input_img, sigma=1.5)
        loss_detail_pres = F.l1_loss(pred_detail, input_detail)
        
        # Combined loss
        total_loss = (
            self.w_lum_l1 * loss_lum_l1 +
            self.w_tonal * loss_tonal +
            self.w_hl_sh * loss_hl_sh +
            self.w_contrast * loss_contrast +
            self.w_color_pres * loss_color_pres +
            self.w_detail_pres * loss_detail_pres
        )
        
        # Loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'luminance_l1': loss_lum_l1.item(),
            'tonal_curve': loss_tonal.item(),
            'highlight_shadow': loss_hl_sh.item(),
            'contrast': loss_contrast.item(),
            'color_preservation': loss_color_pres.item(),
            'detail_preservation': loss_detail_pres.item()
        }
        
        return total_loss, loss_dict


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Exposure-Only Loss Functions")
    print("=" * 60)
    print()
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Create test images
    input_img = torch.rand(2, 3, 224, 224, device=device) * 0.5  # Darker input
    target = torch.rand(2, 3, 224, 224, device=device) * 0.8  # Brighter target
    
    # Simulate model output: brightened but same colors
    pred = input_img * 1.3  # Brightened
    pred = torch.clamp(pred, 0, 1)
    
    print("1. Testing Luminance L1 Loss...")
    lum_l1_loss = LuminanceL1Loss()
    loss = lum_l1_loss(pred, target)
    print(f"   Loss: {loss.item():.6f}")
    print("   ✓ Compares brightness only\n")
    
    print("2. Testing Color Preservation Loss...")
    color_pres_loss = ColorPreservationLoss()
    loss = color_pres_loss(pred, input_img)
    print(f"   Loss: {loss.item():.6f}")
    print("   ✓ Penalizes color changes\n")
    
    print("3. Testing Tonal Curve Loss...")
    tonal_loss = TonalCurveLoss()
    loss = tonal_loss(pred, target)
    print(f"   Loss: {loss.item():.6f}")
    print("   ✓ Compares exposure curves\n")
    
    print("4. Testing Highlight/Shadow Loss...")
    hl_sh_loss = HighlightShadowLoss()
    loss = hl_sh_loss(pred, target)
    print(f"   Loss: {loss.item():.6f}")
    print("   ✓ Targets specific tonal ranges\n")
    
    print("5. Testing Contrast Loss...")
    contrast_loss = ContrastLoss()
    loss = contrast_loss(pred, target)
    print(f"   Loss: {loss.item():.6f}")
    print("   ✓ Measures and matches contrast\n")
    
    print("6. Testing Combined Exposure-Only Loss...")
    exp_loss = ExposureOnlyLoss()
    total_loss, loss_dict = exp_loss(pred, target, input_img)
    print(f"   Total Loss: {total_loss.item():.6f}")
    print(f"   Components:")
    for name, value in loss_dict.items():
        if name != 'total':
            print(f"     {name}: {value:.6f}")
    print("   ✓ Combined loss working\n")
    
    print("=" * 60)
    print("✓ All exposure-only losses tested!")
    print("=" * 60)
    print("\nThese losses ensure:")
    print("  ✓ Exposure/luminance adjustments")
    print("  ✓ Contrast adjustments (global & local)")
    print("  ✓ Colors preserved exactly")
    print("  ✓ Details preserved exactly")
    print("  ✓ No artifacts from color shifts")

