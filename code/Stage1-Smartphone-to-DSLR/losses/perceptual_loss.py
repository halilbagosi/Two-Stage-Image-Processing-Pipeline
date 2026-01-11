"""
Perceptual Loss using pre-trained VGG network
Compares high-level features between generated and target images
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss based on VGG19 features
    Compares features from intermediate layers of pre-trained VGG
    
    Args:
        layers: List of VGG layer indices to extract features from
                Default: [3, 8, 17, 26] corresponds to relu1_2, relu2_2, relu3_4, relu4_4
        weights: Weights for each layer (default: equal weights)
        use_gpu: Whether to use GPU (MPS for M1 Pro)
    """
    
    def __init__(self, layers=None, weights=None, device='mps'):
        super(VGGPerceptualLoss, self).__init__()
        
        if layers is None:
            # Default: use features from relu1_2, relu2_2, relu3_4, relu4_4
            layers = [3, 8, 17, 26]
        
        if weights is None:
            # Equal weights for all layers
            weights = [1.0] * len(layers)
        
        assert len(layers) == len(weights), "Layers and weights must have same length"
        
        self.layers = layers
        self.weights = weights
        self.device = device
        
        # Load pre-trained VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # Extract only the layers we need
        self.feature_extractor = nn.ModuleList()
        prev_layer = 0
        for layer_idx in layers:
            self.feature_extractor.append(vgg[prev_layer:layer_idx+1])
            prev_layer = layer_idx + 1
        
        # Freeze VGG parameters (no training)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.feature_extractor.eval()
        self.feature_extractor.to(device)
        
        # Normalization for VGG (ImageNet statistics)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.criterion = nn.L1Loss()
    
    def normalize(self, x):
        """Normalize image for VGG (ImageNet normalization)"""
        # Ensure mean and std are on same device as input
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std
    
    def forward(self, pred, target):
        """
        Compute perceptual loss
        
        Args:
            pred: Predicted image [B, 3, H, W] in range [0, 1]
            target: Target image [B, 3, H, W] in range [0, 1]
        
        Returns:
            Perceptual loss value
        """
        # Normalize inputs
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)
        
        # Extract features and compute loss
        loss = 0.0
        pred_features = pred_norm
        target_features = target_norm
        
        for i, (extractor, weight) in enumerate(zip(self.feature_extractor, self.weights)):
            # Extract features
            pred_features = extractor(pred_features)
            target_features = extractor(target_features)
            
            # Compute loss for this layer
            layer_loss = self.criterion(pred_features, target_features)
            loss += weight * layer_loss
        
        return loss


class PerceptualLossVGG16(nn.Module):
    """
    Lighter perceptual loss using VGG16 (fewer parameters than VGG19)
    Good for M1 Pro with limited memory
    """
    
    def __init__(self, layers=None, weights=None, device='mps'):
        super(PerceptualLossVGG16, self).__init__()
        
        if layers is None:
            # Use features from relu2_2 and relu4_3
            layers = [8, 22]
        
        if weights is None:
            weights = [1.0, 1.0]
        
        self.layers = layers
        self.weights = weights
        self.device = device
        
        # Load pre-trained VGG16
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        
        self.feature_extractor = nn.ModuleList()
        prev_layer = 0
        for layer_idx in layers:
            self.feature_extractor.append(vgg[prev_layer:layer_idx+1])
            prev_layer = layer_idx + 1
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.feature_extractor.eval()
        self.feature_extractor.to(device)
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.criterion = nn.L1Loss()
    
    def normalize(self, x):
        # Ensure mean and std are on same device as input
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std
    
    def forward(self, pred, target):
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)
        
        loss = 0.0
        pred_features = pred_norm
        target_features = target_norm
        
        for extractor, weight in zip(self.feature_extractor, self.weights):
            pred_features = extractor(pred_features)
            target_features = extractor(target_features)
            layer_loss = self.criterion(pred_features, target_features)
            loss += weight * layer_loss
        
        return loss


if __name__ == "__main__":
    # Test perceptual loss
    print("=== Testing Perceptual Loss ===\n")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dummy images
    batch_size = 2
    pred = torch.rand(batch_size, 3, 224, 224, device=device)
    target = torch.rand(batch_size, 3, 224, 224, device=device)
    
    # Test VGG19 loss
    print("Testing VGG19 Perceptual Loss...")
    vgg19_loss = VGGPerceptualLoss(device=device)
    loss19 = vgg19_loss(pred, target)
    print(f"VGG19 Loss: {loss19.item():.6f}")
    
    # Test VGG16 loss
    print("\nTesting VGG16 Perceptual Loss...")
    vgg16_loss = PerceptualLossVGG16(device=device)
    loss16 = vgg16_loss(pred, target)
    print(f"VGG16 Loss: {loss16.item():.6f}")
    
    # Test gradients
    print("\nTesting gradients...")
    pred.requires_grad = True
    loss = vgg16_loss(pred, target)
    loss.backward()
    print(f"Gradient shape: {pred.grad.shape}")
    print(f"Gradient range: [{pred.grad.min():.6f}, {pred.grad.max():.6f}]")
    
    print("\n✓ Perceptual loss tests passed!")

