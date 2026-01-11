"""
Stage 2 Dataset: DSLR to Film Enhancement
Loads DSLR images and generates film-look targets
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .film_transforms import get_film_transform


class FilmEnhancementDataset(Dataset):
    """
    Dataset for Stage 2: DSLR → Film enhancement
    
    Takes DSLR images and applies procedural film looks as targets
    
    Args:
        dslr_dir: Path to DSLR images
        split: 'train' or 'test'
        film_style: 'cinematic', 'vintage', or 'modern'
        img_size: Target image size
        augment: Whether to apply data augmentation
        film_params: Dict of film-specific parameters
    """
    
    def __init__(self, dslr_dir, split='train', film_style='cinematic',
                 img_size=224, augment=True, film_params=None):
        self.dslr_dir = dslr_dir
        self.split = split
        self.film_style = film_style
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        
        # Get list of DSLR images
        self.dslr_files = sorted([f for f in os.listdir(dslr_dir) 
                                  if f.endswith(('.jpg', '.png'))])
        
        print(f"[Stage2Dataset] Loaded {len(self.dslr_files)} DSLR images for {split}")
        print(f"[Stage2Dataset] Film style: {film_style}")
        
        # Image transforms
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((img_size, img_size),
                                       interpolation=transforms.InterpolationMode.BICUBIC,
                                       antialias=True)
        
        # Film look generator
        if film_params is None:
            film_params = {}
        self.film_transform = get_film_transform(film_style, **film_params)
        
        # Augmentation
        if self.augment:
            self.aug_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.05, contrast=0.05, 
                                      saturation=0.05, hue=0.02)
            ])
    
    def __len__(self):
        return len(self.dslr_files)
    
    def __getitem__(self, idx):
        # Load DSLR image
        dslr_file = self.dslr_files[idx]
        dslr_path = os.path.join(self.dslr_dir, dslr_file)
        
        dslr_img = Image.open(dslr_path).convert('RGB')
        dslr_img = self.resize(dslr_img)
        
        # Apply augmentation to PIL image
        if self.augment:
            dslr_img = self.aug_transforms(dslr_img)
        
        # Convert to tensor
        dslr_tensor = self.to_tensor(dslr_img)
        
        # Generate film target
        film_tensor = self.film_transform(dslr_tensor)
        
        return dslr_tensor, film_tensor


def get_stage2_dataloader(dataset, batch_size=16, shuffle=True, num_workers=4):
    """
    Create DataLoader for Stage 2 (optimized for Apple Silicon)
    
    Args:
        dataset: FilmEnhancementDataset instance
        batch_size: Batch size (can be larger than Stage 1)
        shuffle: Whether to shuffle
        num_workers: Number of workers
    
    Returns:
        DataLoader instance
    """
    # pin_memory=False for MPS (Apple Silicon) to avoid warning
    # MPS doesn't support pinned memory, so we disable it
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,  # Disabled for MPS compatibility
        persistent_workers=True if num_workers > 0 else False
    )


if __name__ == "__main__":
    # Test Stage 2 dataset
    print("\n=== Testing Stage 2 Dataset ===\n")
    
    import sys
    sys.path.append('../..')
    
    base_path = '../../datasets/dped/iphone'
    
    # Test dataset
    dataset = FilmEnhancementDataset(
        dslr_dir=f'{base_path}/test/canon',
        split='test',
        film_style='cinematic',
        augment=False
    )
    
    dslr, film = dataset[0]
    print(f"DSLR image shape: {dslr.shape}, range: [{dslr.min():.3f}, {dslr.max():.3f}]")
    print(f"Film image shape: {film.shape}, range: [{film.min():.3f}, {film.max():.3f}]")
    
    # Test dataloader
    dataloader = get_stage2_dataloader(dataset, batch_size=4, num_workers=2)
    dslr_batch, film_batch = next(iter(dataloader))
    print(f"\nBatch shapes: DSLR {dslr_batch.shape}, Film {film_batch.shape}")
    
    print("\n✓ Stage 2 dataset test passed!")

