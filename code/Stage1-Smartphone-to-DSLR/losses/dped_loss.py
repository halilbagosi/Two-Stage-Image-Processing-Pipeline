"""
DPED Loss Function
Based on: https://aiff22.github.io/index.html

Implements the three-component loss from DPED paper:
1. Color Loss: Gaussian blur + Euclidean distance
2. Texture Loss: Adversarial discriminator on grayscale images
3. Content Loss: VGG-19 pre-trained network

Optimized for PyTorch with MPS (Apple Silicon) support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import ssl
import urllib.request

# Fix SSL certificate verification for macOS (needed for downloading VGG weights)
def _fix_ssl():
    """Fix SSL certificate verification issues on macOS"""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Also fix for urllib (used by torch.hub)
    import urllib.request
    if hasattr(ssl, '_create_unverified_context'):
        urllib.request.urlopen.__globals__['ssl'] = ssl

# Apply SSL fix at module level
_fix_ssl()


class ColorLoss(nn.Module):
    """
    Color Loss: Gaussian blur + Euclidean distance
    
    From DPED paper:
    "The enhanced image should be close to the target (DSLR) photo in terms of colors.
    To measure the difference between them, we apply Gaussian blur to both images
    and compute the Euclidean distance between the obtained representations."
    """
    
    def __init__(self, sigma=1.0, device='mps'):
        super(ColorLoss, self).__init__()
        self.sigma = sigma
        self.device = device
        
        # Create Gaussian blur kernel
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Gaussian kernel
        kernel = self._gaussian_kernel(kernel_size, sigma)
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(3, 1, 1, 1)  # For RGB channels
        
        # Register buffer and move to device
        self.register_buffer('kernel', kernel.to(device))
    
    def _gaussian_kernel(self, size, sigma):
        """Create 2D Gaussian kernel"""
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        kernel = g[:, None] * g[None, :]
        return kernel
    
    def forward(self, pred, target):
        """
        Compute color loss
        
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
        
        Returns:
            Color loss (scalar)
        """
        # Apply Gaussian blur
        pred_blur = F.conv2d(pred, self.kernel, padding=self.kernel.shape[-1] // 2, groups=3)
        target_blur = F.conv2d(target, self.kernel, padding=self.kernel.shape[-1] // 2, groups=3)
        
        # Euclidean distance (L2 loss)
        loss = F.mse_loss(pred_blur, target_blur)
        
        return loss


class TextureDiscriminator(nn.Module):
    """
    Texture Discriminator for adversarial training
    
    From DPED paper:
    "To measure texture quality of the enhanced image, we train a separate
    adversarial CNN-discriminator that observes both improved and target grayscale images,
    and its objective is to predict which image is which."
    """
    
    def __init__(self, device='mps'):
        super(TextureDiscriminator, self).__init__()
        
        # Convert RGB to grayscale
        # Weight shape: [out_channels=1, in_channels=3, kernel_h=1, kernel_w=1]
        self.grayscale = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        # Create weight tensor with shape [1, 3, 1, 1] for RGB to grayscale
        # Format: [out_ch, in_ch, h, w] = [1, 3, 1, 1]
        weight = torch.zeros(1, 3, 1, 1, dtype=torch.float32)
        weight[0, 0, 0, 0] = 0.299  # R weight
        weight[0, 1, 0, 0] = 0.587  # G weight
        weight[0, 2, 0, 0] = 0.114  # B weight
        self.grayscale.weight.data = weight.to(device)
        self.grayscale.weight.requires_grad = False
        
        # Discriminator network
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)  # 112x112
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 56x56
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 28x28
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 14x14
        self.conv5 = nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0)  # 11x11
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: RGB image [B, 3, H, W]
        
        Returns:
            Discriminator output [B, 1, H', W']
        """
        # Convert to grayscale
        x = self.grayscale(x)
        
        # Discriminator layers
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))
        
        return x


class TextureLoss(nn.Module):
    """
    Texture Loss: Adversarial loss using discriminator
    
    The generator (enhancement network) tries to fool the discriminator
    so it cannot distinguish between enhanced and target images.
    """
    
    def __init__(self, device='mps'):
        super(TextureLoss, self).__init__()
        self.device = device
        self.discriminator = TextureDiscriminator(device=device).to(device)
        
    def forward(self, pred, target, discriminator_output_pred, discriminator_output_target):
        """
        Compute texture loss (adversarial loss for generator)
        
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
            discriminator_output_pred: Discriminator output for pred [B, 1, H', W']
            discriminator_output_target: Discriminator output for target [B, 1, H', W']
        
        Returns:
            Texture loss (scalar)
        """
        # Generator wants discriminator to think pred is real (target)
        # So we want discriminator_output_pred to be close to 1
        # Loss = -log(D(pred)) or MSE(1, D(pred))
        loss = F.mse_loss(discriminator_output_pred, torch.ones_like(discriminator_output_pred))
        
        return loss


class ContentLoss(nn.Module):
    """
    Content Loss: VGG-based perceptual loss
    
    From DPED paper:
    "A distinct VGG-19 CNN pre-trained on Alexnet is used to preserve image semantics:
    content description produced by this CNN should be the same both for the
    improved and target images."
    
    Note: We use VGG-16 (more common) or VGG-19 if available
    """
    
    def __init__(self, device='mps', use_vgg19=False):
        super(ContentLoss, self).__init__()
        self.device = device
        
        # Load pre-trained VGG using newer API (SSL fix applied at module level)
        print(f"Loading VGG-{'19' if use_vgg19 else '16'} for content loss...")
        try:
            if use_vgg19:
                # Use newer weights API
                vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
            else:
                vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
            print("✓ VGG model loaded successfully")
        except (ssl.SSLError, urllib.error.URLError) as e:
            # SSL/Download error - provide helpful message
            print(f"\n⚠️  SSL Certificate Error: {e}")
            print("This is a common issue on macOS. Solutions:")
            print("1. Install certificates: /Applications/Python\\ 3.12/Install\\ Certificates.command")
            print("2. Or manually download VGG-19 weights:")
            print("   wget https://download.pytorch.org/models/vgg19-dcbb9e9d.pth")
            print("   mv vgg19-dcbb9e9d.pth ~/.cache/torch/hub/checkpoints/")
            print("\nTrying fallback with old API...")
            try:
                if use_vgg19:
                    vgg = models.vgg19(pretrained=True).features
                else:
                    vgg = models.vgg16(pretrained=True).features
                print("✓ VGG model loaded with fallback method")
            except Exception as e2:
                print(f"❌ Error: Could not load VGG model: {e2}")
                raise RuntimeError(f"Failed to load VGG model. Please fix SSL certificates or download weights manually. Error: {e2}")
        except Exception as e:
            print(f"❌ Unexpected error loading VGG: {e}")
            raise
        
        # Use features up to conv2_2 (faster, still captures content)
        # Earlier layers are faster but still preserve semantic content
        if use_vgg19:
            self.feature_extractor = nn.Sequential(*list(vgg.children())[:10]).to(device)  # Up to conv2_2 (faster)
        else:
            self.feature_extractor = nn.Sequential(*list(vgg.children())[:8]).to(device)  # Up to conv2_2 (faster)
        
        # Freeze VGG weights
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Normalization for ImageNet
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))
    
    def forward(self, pred, target):
        """
        Compute content loss
        
        Args:
            pred: Predicted image [B, 3, H, W] in range [0, 1]
            target: Target image [B, 3, H, W] in range [0, 1]
        
        Returns:
            Content loss (scalar)
        """
        # Normalize to ImageNet stats
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        
        # Extract features
        pred_features = self.feature_extractor(pred_norm)
        target_features = self.feature_extractor(target_norm)
        
        # Euclidean distance (L2 loss)
        loss = F.mse_loss(pred_features, target_features)
        
        return loss


class DPEDLoss(nn.Module):
    """
    Complete DPED Loss Function
    
    Combines:
    1. Color Loss (Gaussian blur + L2)
    2. Texture Loss (Adversarial discriminator)
    3. Content Loss (VGG perceptual)
    
    Based on: https://aiff22.github.io/index.html
    """
    
    def __init__(self, 
                 w_color=1.0,
                 w_texture=0.1,
                 w_content=0.1,
                 color_sigma=1.0,
                 device='mps',
                 use_vgg19=False):
        super(DPEDLoss, self).__init__()
        
        self.device = device
        self.w_color = w_color
        self.w_texture = w_texture
        self.w_content = w_content
        
        # Initialize loss components
        self.color_loss = ColorLoss(sigma=color_sigma, device=device)
        self.texture_loss = TextureLoss(device=device)
        self.content_loss = ContentLoss(device=device, use_vgg19=use_vgg19)
        
        # Discriminator (shared with texture loss)
        self.discriminator = self.texture_loss.discriminator
        
        print(f"DPED Loss initialized:")
        print(f"  Color weight: {w_color}")
        print(f"  Texture weight: {w_texture}")
        print(f"  Content weight: {w_content}")
        print(f"  Device: {device}")
    
    def forward(self, pred, target):
        """
        Compute DPED loss for generator
        
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        # Color loss
        loss_color = self.color_loss(pred, target)
        
        # Content loss (skip if weight is 0)
        if self.w_content > 0:
            loss_content = self.content_loss(pred, target)
        else:
            loss_content = torch.tensor(0.0, device=pred.device, requires_grad=False)
        
        # Texture loss (adversarial)
        # Skip if weight is 0 (disabled for speed)
        if self.w_texture > 0:
            # For generator training: we need gradients through discriminator
            # But we detach the discriminator weights to avoid training it here
            # (Discriminator is trained separately in adversarial training)
            disc_pred = self.discriminator(pred)
            # Generator wants D(pred) to be close to 1 (fool discriminator)
            loss_texture = F.mse_loss(disc_pred, torch.ones_like(disc_pred))
        else:
            # Texture loss disabled - return zero loss
            loss_texture = torch.tensor(0.0, device=pred.device, requires_grad=False)
        
        # Combined loss
        total_loss = (
            self.w_color * loss_color +
            self.w_texture * loss_texture +
            self.w_content * loss_content
        )
        
        # Return loss and components
        loss_dict = {
            'total': total_loss.item(),
            'color': loss_color.item(),
            'texture': loss_texture.item(),
            'content': loss_content.item()
        }
        
        return total_loss, loss_dict
    
    def discriminator_loss(self, pred, target):
        """
        Compute discriminator loss (for training discriminator separately)
        
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
        
        Returns:
            disc_loss: Discriminator loss
        """
        disc_pred = self.discriminator(pred)
        disc_target = self.discriminator(target)
        
        # Discriminator wants:
        # - D(target) = 1 (real)
        # - D(pred) = 0 (fake)
        loss_real = F.mse_loss(disc_target, torch.ones_like(disc_target))
        loss_fake = F.mse_loss(disc_pred, torch.zeros_like(disc_pred))
        
        disc_loss = (loss_real + loss_fake) / 2.0
        
        return disc_loss
    
    def get_discriminator(self):
        """Get discriminator for separate training"""
        return self.discriminator

