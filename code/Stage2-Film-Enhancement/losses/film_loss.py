"""
Film Enhancement Loss Functions
Focuses on perceptual quality and style consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FilmStyleLoss(nn.Module):
    """
    Style loss for film aesthetics
    Matches color distribution and tonal characteristics
    """
    
    def __init__(self):
        super(FilmStyleLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
    
    def compute_gram_matrix(self, features):
        """Compute Gram matrix for style"""
        b, c, h, w = features.size()
        features = features.reshape(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def compute_color_histogram(self, img, bins=64):
        """Compute color histogram features"""
        b, c, h, w = img.size()
        histograms = []
        
        for channel in range(c):
            channel_data = img[:, channel, :, :].flatten(1)
            hist = torch.histc(channel_data, bins=bins, min=0, max=1)
            histograms.append(hist)
        
        return torch.stack(histograms, dim=0)
    
    def forward(self, pred, target):
        """
        Compute style loss between predicted and target
        
        Args:
            pred: Predicted film look [B, 3, H, W]
            target: Target film look [B, 3, H, W]
        
        Returns:
            Style loss value
        """
        # Color distribution loss
        pred_mean = pred.mean(dim=[2, 3])
        target_mean = target.mean(dim=[2, 3])
        mean_loss = self.l1_loss(pred_mean, target_mean)
        
        pred_std = pred.std(dim=[2, 3])
        target_std = target.std(dim=[2, 3])
        std_loss = self.l1_loss(pred_std, target_std)
        
        return mean_loss + std_loss


class FilmEnhancementLoss(nn.Module):
    """
    Combined loss for film enhancement
    
    Args:
        w_l1: Weight for L1 loss
        w_style: Weight for style loss
        w_color: Weight for color loss
    """
    
    def __init__(self, w_l1=0.5, w_style=0.5, w_color=0.3):
        super(FilmEnhancementLoss, self).__init__()
        
        self.w_l1 = w_l1
        self.w_style = w_style
        self.w_color = w_color
        
        self.l1_loss = nn.L1Loss()
        self.style_loss = FilmStyleLoss()
        self.color_loss = ColorConsistencyLoss()
        
        print(f"[FilmLoss] L1: {w_l1}, Style: {w_style}, Color: {w_color}")
    
    def forward(self, pred, target):
        """
        Compute combined film enhancement loss
        
        Args:
            pred: Predicted film look [B, 3, H, W]
            target: Target film look [B, 3, H, W]
        
        Returns:
            total_loss, loss_dict
        """
        # Individual losses
        l1 = self.l1_loss(pred, target)
        style = self.style_loss(pred, target)
        color = self.color_loss(pred, target)
        
        # Combined loss
        total_loss = (
            self.w_l1 * l1 +
            self.w_style * style +
            self.w_color * color
        )
        
        # Loss dict for logging
        loss_dict = {
            'total': total_loss.item(),
            'l1': l1.item(),
            'style': style.item(),
            'color': color.item()
        }
        
        return total_loss, loss_dict


class ColorConsistencyLoss(nn.Module):
    """Color consistency in LAB space"""
    
    def __init__(self):
        super(ColorConsistencyLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
    
    def rgb_to_lab_approx(self, rgb):
        """Approximate RGB to LAB conversion"""
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        
        L = 0.2126 * r + 0.7152 * g + 0.0722 * b
        A = (r - g) * 0.5
        B = (r + g) * 0.5 - b
        
        return torch.cat([L, A, B], dim=1)
    
    def forward(self, pred, target):
        pred_lab = self.rgb_to_lab_approx(pred)
        target_lab = self.rgb_to_lab_approx(target)
        
        return self.l1_loss(pred_lab, target_lab)


if __name__ == "__main__":
    # Test film losses
    print("=== Testing Film Losses ===\n")
    
    pred = torch.rand(2, 3, 224, 224)
    target = torch.rand(2, 3, 224, 224)
    
    # Test style loss
    style_loss = FilmStyleLoss()
    loss = style_loss(pred, target)
    print(f"Style Loss: {loss.item():.6f}")
    
    # Test color loss
    color_loss = ColorConsistencyLoss()
    loss = color_loss(pred, target)
    print(f"Color Loss: {loss.item():.6f}")
    
    # Test combined loss
    film_loss = FilmEnhancementLoss()
    total_loss, loss_dict = film_loss(pred, target)
    print(f"\nFilm Enhancement Loss:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.6f}")
    
    print("\n✓ Film losses tested successfully!")

