"""
Film Look Transformation Functions
Generate cinematic film targets from DSLR images
"""

import torch
import torch.nn.functional as F
import numpy as np


class CinematicFilmTransform:
    """
    Procedural cinematic film look generator
    Applies professional color grading and tone curves
    """
    
    def __init__(self, 
                 contrast_boost=1.15,
                 saturation=1.1,
                 shadow_lift=0.05,
                 highlight_roll=0.95,
                 teal_orange=True,
                 warmth=1.05):
        
        self.contrast_boost = contrast_boost
        self.saturation = saturation
        self.shadow_lift = shadow_lift
        self.highlight_roll = highlight_roll
        self.teal_orange = teal_orange
        self.warmth = warmth
    
    def apply_s_curve(self, img):
        """Apply S-curve for cinematic contrast"""
        # S-curve using cubic hermite interpolation
        midpoint = 0.5
        
        # Create S-curve
        shadows = torch.where(img < midpoint,
                             torch.pow(img / midpoint, self.contrast_boost) * midpoint,
                             img)
        
        highlights = torch.where(shadows >= midpoint,
                                1 - torch.pow((1 - shadows) / (1 - midpoint), 
                                            self.contrast_boost) * (1 - midpoint),
                                shadows)
        
        return highlights
    
    def lift_shadows(self, img):
        """Lift shadows (film-like black level)"""
        return img + self.shadow_lift * (1 - img)
    
    def roll_highlights(self, img):
        """Roll off highlights (prevent clipping)"""
        return torch.where(img > self.highlight_roll,
                          self.highlight_roll + (img - self.highlight_roll) * 0.3,
                          img)
    
    def adjust_saturation(self, img):
        """Adjust color saturation"""
        # Convert to HSV-like space
        max_rgb = torch.max(img, dim=0, keepdim=True)[0]
        min_rgb = torch.min(img, dim=0, keepdim=True)[0]
        
        luminance = (max_rgb + min_rgb) / 2
        saturation = max_rgb - min_rgb
        
        # Adjust saturation
        new_saturation = saturation * self.saturation
        new_saturation = torch.clamp(new_saturation, 0, 1)
        
        # Reconstruct RGB
        result = luminance + (img - luminance) * (new_saturation / (saturation + 1e-6))
        return torch.clamp(result, 0, 1)
    
    def teal_orange_grade(self, img):
        """Apply teal/orange cinematic color grading"""
        if not self.teal_orange:
            return img
        
        r, g, b = img[0], img[1], img[2]
        
        # Shift blues/cyans toward teal
        teal_shift = torch.where(b > r,
                                b + (b - r) * 0.1,
                                b)
        
        # Shift yellows/reds toward orange
        orange_shift_r = torch.where(r > b,
                                    r + (r - b) * 0.05,
                                    r)
        
        result = torch.stack([orange_shift_r, g, teal_shift], dim=0)
        return torch.clamp(result, 0, 1)
    
    def add_warmth(self, img):
        """Add warmth to the image"""
        if self.warmth == 1.0:
            return img
        
        r, g, b = img[0], img[1], img[2]
        
        # Increase red channel slightly
        r = r * self.warmth
        # Decrease blue channel slightly  
        b = b / self.warmth
        
        result = torch.stack([r, g, b], dim=0)
        return torch.clamp(result, 0, 1)
    
    def __call__(self, img):
        """
        Apply full cinematic film transformation
        
        Args:
            img: Input DSLR image tensor [C, H, W] in range [0, 1]
        
        Returns:
            Film look image tensor [C, H, W] in range [0, 1]
        """
        # Apply transformations in order
        result = img.clone()
        
        # 1. S-curve contrast
        result = self.apply_s_curve(result)
        
        # 2. Lift shadows
        result = self.lift_shadows(result)
        
        # 3. Roll highlights
        result = self.roll_highlights(result)
        
        # 4. Teal/orange grading
        result = self.teal_orange_grade(result)
        
        # 5. Add warmth
        result = self.add_warmth(result)
        
        # 6. Adjust saturation
        result = self.adjust_saturation(result)
        
        return torch.clamp(result, 0, 1)


class VintageFilmTransform:
    """Vintage film look with faded colors"""
    
    def __init__(self, fade_amount=0.15, warmth=1.1):
        self.fade_amount = fade_amount
        self.warmth = warmth
    
    def __call__(self, img):
        # Fade blacks
        result = img + self.fade_amount
        
        # Reduce contrast
        result = (result - 0.5) * 0.8 + 0.5
        
        # Add warmth
        r, g, b = result[0], result[1], result[2]
        r = r * self.warmth
        b = b / (self.warmth * 1.1)
        
        result = torch.stack([r, g, b], dim=0)
        return torch.clamp(result, 0, 1)


class ModernFilmTransform:
    """Modern high-contrast film look"""
    
    def __init__(self, contrast=1.3, saturation=1.15):
        self.contrast = contrast
        self.saturation = saturation
    
    def __call__(self, img):
        # High contrast
        result = (img - 0.5) * self.contrast + 0.5
        
        # High saturation
        luminance = result.mean(dim=0, keepdim=True)
        result = luminance + (result - luminance) * self.saturation
        
        return torch.clamp(result, 0, 1)


def get_film_transform(style='cinematic', **kwargs):
    """
    Factory function for film transforms
    
    Args:
        style: 'cinematic', 'vintage', or 'modern'
        **kwargs: Style-specific parameters
    
    Returns:
        Film transform function
    """
    # Filter out unsupported parameters for each style
    if style == 'cinematic':
        # Only pass parameters that CinematicFilmTransform accepts
        supported_params = ['contrast_boost', 'saturation', 'shadow_lift', 
                          'highlight_roll', 'teal_orange', 'warmth']
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}
        return CinematicFilmTransform(**filtered_kwargs)
    elif style == 'vintage':
        # Only pass parameters that VintageFilmTransform accepts
        supported_params = ['fade_amount', 'warmth']
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}
        return VintageFilmTransform(**filtered_kwargs)
    elif style == 'modern':
        # Only pass parameters that ModernFilmTransform accepts
        supported_params = ['contrast', 'saturation']
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}
        return ModernFilmTransform(**filtered_kwargs)
    else:
        raise ValueError(f"Unknown film style: {style}")


if __name__ == "__main__":
    # Test film transforms
    print("=== Testing Film Transforms ===\n")
    
    # Create test image
    test_img = torch.rand(3, 224, 224)
    
    # Test cinematic
    cinematic = CinematicFilmTransform()
    result = cinematic(test_img)
    print(f"Cinematic: {test_img.shape} -> {result.shape}")
    print(f"Range: [{result.min():.3f}, {result.max():.3f}]\n")
    
    # Test vintage
    vintage = VintageFilmTransform()
    result = vintage(test_img)
    print(f"Vintage: {test_img.shape} -> {result.shape}")
    print(f"Range: [{result.min():.3f}, {result.max():.3f}]\n")
    
    # Test modern
    modern = ModernFilmTransform()
    result = modern(test_img)
    print(f"Modern: {test_img.shape} -> {result.shape}")
    print(f"Range: [{result.min():.3f}, {result.max():.3f}]\n")
    
    print("✓ All film transforms tested successfully!")

