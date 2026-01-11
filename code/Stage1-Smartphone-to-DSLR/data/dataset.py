"""
Dataset Loader for Smartphone-to-DSLR Enhancement
Handles loading paired smartphone and DSLR images
Optimized for DPED (DSLR Photo Enhancement Dataset)
Based on: https://aiff22.github.io/index.html
"""

import os
import random
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Allow loading truncated/corrupt PNGs without crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DPEDDataset(Dataset):
    """
    Dataset loader for paired smartphone and DSLR images
    Optimized for DPED dataset (iPhone 3GS, BlackBerry Passport, Sony Xperia Z → Canon 70D)
    Works with any paired dataset following the same structure
    
    Args:
        phone_dir: Path to smartphone images
        dslr_dir: Path to DSLR images
        split: 'train' or 'test'
        img_size: Target image size (default: 224)
        augment: Whether to apply data augmentation (default: True for training)
    """
    
    def __init__(self, phone_dir, dslr_dir, split='train', img_size=224, augment=True, max_samples=None):
        self.phone_dir = phone_dir
        self.dslr_dir = dslr_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        
        # Get list of image files and match by filename stem
        phone_files = [f for f in os.listdir(phone_dir) if f.endswith(('.jpg', '.png'))]
        dslr_files = [f for f in os.listdir(dslr_dir) if f.endswith(('.jpg', '.png'))]
        
        phone_stems = {os.path.splitext(f)[0]: f for f in phone_files}
        dslr_stems = {os.path.splitext(f)[0]: f for f in dslr_files}
        common = sorted(set(phone_stems.keys()) & set(dslr_stems.keys()))
        
        self.phone_files = [phone_stems[k] for k in common]
        self.dslr_files = [dslr_stems[k] for k in common]
        
        if len(self.phone_files) == 0:
            raise ValueError(f"No matching pairs found between {phone_dir} and {dslr_dir}")
        
        # Limit to max_samples if specified (for faster training)
        if max_samples is not None and max_samples > 0 and len(self.phone_files) > max_samples:
            print(f"[Dataset] Limiting to {max_samples} samples (from {len(self.phone_files)}) for faster training")
            self.phone_files = self.phone_files[:max_samples]
            self.dslr_files = self.dslr_files[:max_samples]
        
        if len(self.phone_files) != len(phone_files) or len(self.dslr_files) != len(dslr_files):
            if max_samples is None:
                print(f"[Dataset] Warning: matched {len(self.phone_files)} pairs (phone: {len(phone_files)}, dslr: {len(dslr_files)})")
        
        # Verify paired data
        assert len(self.phone_files) == len(self.dslr_files), \
            f"Mismatch: {len(self.phone_files)} phone images, {len(self.dslr_files)} DSLR images"
        
        print(f"[Dataset] Loaded {len(self.phone_files)} image pairs for {split}")
        
        # Basic transforms (always applied)
        self.to_tensor = transforms.ToTensor()
        
        # Resize both phone and DSLR to the target size to ensure alignment
        self.phone_resize = transforms.Resize((img_size, img_size),
                                              interpolation=transforms.InterpolationMode.BICUBIC,
                                              antialias=True)
        self.dslr_resize = transforms.Resize((img_size, img_size), 
                                             interpolation=transforms.InterpolationMode.BICUBIC,
                                             antialias=True)
        
        # Augmentation transforms (only for training)
        if self.augment:
            self.aug_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # Random rotation by 90 degrees
                transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.25),
            ])
    
    def __len__(self):
        return len(self.phone_files)
    
    def __getitem__(self, idx):
        # Retry a few times if an image is truncated/corrupt
        for _ in range(5):
            phone_file = self.phone_files[idx]
            dslr_file = self.dslr_files[idx]
            
            # Verify they match (should have same base name)
            phone_base = os.path.splitext(phone_file)[0]
            dslr_base = os.path.splitext(dslr_file)[0]
            if phone_base != dslr_base:
                idx = random.randint(0, len(self.phone_files) - 1)
                continue
            
            phone_path = os.path.join(self.phone_dir, phone_file)
            dslr_path = os.path.join(self.dslr_dir, dslr_file)
            
            try:
                phone_img = Image.open(phone_path).convert('RGB')
                dslr_img = Image.open(dslr_path).convert('RGB')
            except OSError:
                idx = random.randint(0, len(self.phone_files) - 1)
                continue
            
            # Resize both to the target size
            phone_img = self.phone_resize(phone_img)
            dslr_img = self.dslr_resize(dslr_img)
            
            if self.augment:
                seed = random.randint(0, 2**32)
                random.seed(seed); torch.manual_seed(seed)
                phone_img = self.aug_transforms(phone_img)
                random.seed(seed); torch.manual_seed(seed)
                dslr_img = self.aug_transforms(dslr_img)
            
            phone_tensor = self.to_tensor(phone_img)
            dslr_tensor = self.to_tensor(dslr_img)
            return phone_tensor, dslr_tensor
        raise OSError(f"Failed to load sample after retries (index {idx})")


class DPEDDatasetStage2(Dataset):
    """
    Dataset loader for Stage 2: DSLR → Film Enhancement
    
    For now, this will use the DSLR images as both input and target
    Later, we can add film-style references or apply film presets as targets
    
    Args:
        dslr_dir: Path to DSLR images (Canon 70D from DPED dataset)
        film_dir: Path to film-style references (optional, if None uses DSLR as base)
        split: 'train' or 'test'
        img_size: Target image size (default: 224)
        augment: Whether to apply data augmentation
    """
    
    def __init__(self, dslr_dir, film_dir=None, split='train', img_size=224, augment=True):
        self.dslr_dir = dslr_dir
        self.film_dir = film_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        
        # Get list of image files
        self.dslr_files = sorted([f for f in os.listdir(dslr_dir) if f.endswith(('.jpg', '.png'))])
        
        if film_dir:
            self.film_files = sorted([f for f in os.listdir(film_dir) if f.endswith(('.jpg', '.png'))])
            assert len(self.dslr_files) == len(self.film_files), \
                f"Mismatch: {len(self.dslr_files)} DSLR, {len(self.film_files)} film images"
        else:
            self.film_files = None
            print(f"[Stage2] Film dir not provided, will use film presets during training")
        
        print(f"[DPEDDatasetStage2] Loaded {len(self.dslr_files)} images for {split}")
        
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((img_size, img_size), 
                                       interpolation=transforms.InterpolationMode.BICUBIC,
                                       antialias=True)
    
    def __len__(self):
        return len(self.dslr_files)
    
    def __getitem__(self, idx):
        # Load DSLR image
        dslr_file = self.dslr_files[idx]
        dslr_path = os.path.join(self.dslr_dir, dslr_file)
        dslr_img = Image.open(dslr_path).convert('RGB')
        dslr_img = self.resize(dslr_img)
        dslr_tensor = self.to_tensor(dslr_img)
        
        # Load film reference if available
        if self.film_files:
            film_file = self.film_files[idx]
            film_path = os.path.join(self.film_dir, film_file)
            film_img = Image.open(film_path).convert('RGB')
            film_img = self.resize(film_img)
            film_tensor = self.to_tensor(film_img)
        else:
            # Use DSLR as target for now (will apply film transforms during training)
            film_tensor = dslr_tensor.clone()
        
        return dslr_tensor, film_tensor


def get_dataloader(dataset, batch_size=8, shuffle=True, num_workers=4):
    """
    Create a DataLoader with appropriate settings for Apple Silicon
    
    Args:
        dataset: DPEDDataset instance
        batch_size: Batch size (default: 8, adjust based on memory)
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes (default: 4)
    
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
    # Test the dataset loader
    import sys
    sys.path.append('../..')
    
    base_path = '../../datasets/DPED'
    
    # Test Stage 1 dataset
    print("\n=== Testing Stage 1 Dataset (DPED) ===")
    train_dataset = DPEDDataset(
        phone_dir=f'{base_path}/train/phone',
        dslr_dir=f'{base_path}/train/dslr',
        split='train',
        augment=True
    )
    
    phone, dslr = train_dataset[0]
    print(f"Phone image shape: {phone.shape}, range: [{phone.min():.3f}, {phone.max():.3f}]")
    print(f"DSLR image shape: {dslr.shape}, range: [{dslr.min():.3f}, {dslr.max():.3f}]")
    
    # Test dataloader
    dataloader = get_dataloader(train_dataset, batch_size=4, num_workers=2)
    phone_batch, dslr_batch = next(iter(dataloader))
    print(f"\nBatch shapes: Phone {phone_batch.shape}, DSLR {dslr_batch.shape}")
    
    print("\n✓ Dataset test passed!")

