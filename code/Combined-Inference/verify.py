"""
Quick verification script to check if checkpoints are valid
Run this after getting checkpoints from your friend
"""

import sys
from pathlib import Path
import torch

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'Stage1-Smartphone-to-DSLR'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'Stage2-Film-Enhancement'))

from models.stage1_unet import LightweightUNet
from models.stage2_film import build_stage2_model

print("=" * 60)
print("Checkpoint Verification Script")
print("=" * 60 + "\n")

# Checkpoint paths
STAGE1_CHECKPOINT = '../Stage1-Smartphone-to-DSLR/checkpoints/stage1_production/best_model.pth'
STAGE2_CHECKPOINT = '../Stage2-Film-Enhancement/checkpoints/stage2_m4_optimized/best_model.pth'

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Test Stage 1
print("1. Checking Stage 1 checkpoint...")
stage1_path = Path(STAGE1_CHECKPOINT)

if not stage1_path.exists():
    print(f"   ✗ NOT FOUND: {stage1_path}")
    print(f"   Please place Stage 1 checkpoint at:")
    print(f"   {stage1_path.absolute()}\n")
else:
    try:
        # Load model
        model = LightweightUNet(in_channels=3, out_channels=3, base_channels=32)
        checkpoint = torch.load(stage1_path, map_location=device)
        
        # Check checkpoint structure
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   ✓ Checkpoint loaded successfully")
            print(f"   ✓ File size: {stage1_path.stat().st_size / (1024*1024):.1f} MB")
            
            # Check if training info is available
            if 'epoch' in checkpoint:
                print(f"   ✓ Trained for {checkpoint['epoch']} epochs")
            if 'best_psnr' in checkpoint:
                print(f"   ✓ Best PSNR: {checkpoint['best_psnr']:.2f} dB")
            
            # Test forward pass
            model = model.to(device)
            model.eval()
            test_input = torch.randn(1, 3, 224, 224, device=device)
            with torch.no_grad():
                output = model(test_input)
            print(f"   ✓ Test forward pass successful")
            print(f"   ✓ Output shape: {output.shape}")
            print(f"   ✓ Output range: [{output.min():.3f}, {output.max():.3f}]")
        else:
            print(f"   ✗ Invalid checkpoint format (missing 'model_state_dict')")
    except Exception as e:
        print(f"   ✗ Error loading checkpoint: {e}")

print()

# Test Stage 2
print("2. Checking Stage 2 checkpoint...")
stage2_path = Path(STAGE2_CHECKPOINT)

if not stage2_path.exists():
    print(f"   ✗ NOT FOUND: {stage2_path}")
    print(f"   Please place Stage 2 checkpoint at:")
    print(f"   {stage2_path.absolute()}\n")
else:
    try:
        # Load model
        model = build_stage2_model(model_type='lut', device=device)
        checkpoint = torch.load(stage2_path, map_location=device)
        
        # Check checkpoint structure
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   ✓ Checkpoint loaded successfully")
            print(f"   ✓ File size: {stage2_path.stat().st_size / (1024*1024):.1f} MB")
            
            # Check if training info is available
            if 'epoch' in checkpoint:
                print(f"   ✓ Trained for {checkpoint['epoch']} epochs")
            if 'best_psnr' in checkpoint:
                print(f"   ✓ Best PSNR: {checkpoint['best_psnr']:.2f} dB")
            
            # Test forward pass
            model.eval()
            test_input = torch.randn(1, 3, 224, 224, device=device)
            with torch.no_grad():
                output = model(test_input)
            print(f"   ✓ Test forward pass successful")
            print(f"   ✓ Output shape: {output.shape}")
            print(f"   ✓ Output range: [{output.min():.3f}, {output.max():.3f}]")
        else:
            print(f"   ✗ Invalid checkpoint format (missing 'model_state_dict')")
    except Exception as e:
        print(f"   ✗ Error loading checkpoint: {e}")

print()
print("=" * 60)
print("Verification complete!")
print("=" * 60)
print("\nIf both checkpoints are valid, you can now run:")
print("  python3 inference.py --stage1 <path> --stage2 <path> --input <image> --output ./results")


