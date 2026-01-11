# Loss functions for training
from .perceptual_loss import VGGPerceptualLoss, PerceptualLossVGG16
from .combined_loss import SSIMLoss, ColorLoss, CombinedLoss, SimplifiedLoss, build_loss_function

__all__ = [
    'VGGPerceptualLoss',
    'PerceptualLossVGG16',
    'SSIMLoss',
    'ColorLoss',
    'CombinedLoss',
    'SimplifiedLoss',
    'build_loss_function'
]
