"""
Tiled Inference Pipeline - Proper Full-Resolution Processing
Processes large images by splitting into tiles, processing each tile, and reconstructing
"""

import argparse
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import sys
import numpy as np

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HAS_HEIF = True
except ImportError:
    HAS_HEIF = False

import importlib.util

# ... (imports)

# Add parent directories to path for imports
# Note: Stage 2 is inserted LAST (index 0) so it takes precedence for shared names
sys.path.insert(0, str(Path(__file__).parent.parent / 'Stage1-Smartphone-to-DSLR'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'Stage2-Film-Enhancement'))

# Dynamically load Stage 1 models to avoid "models" package conflict
try:
    s1_unet_path = Path(__file__).parent.parent / 'Stage1-Smartphone-to-DSLR/models/stage1_unet.py'
    spec = importlib.util.spec_from_file_location("models.stage1_unet", str(s1_unet_path))
    s1_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s1_module)
    LightweightUNet = s1_module.LightweightUNet
    SmartphoneToDSLRUNet = s1_module.SmartphoneToDSLRUNet
except Exception as e:
    print(f"Warning: Could not import Stage 1 UNet models: {e}")
    LightweightUNet = None
    SmartphoneToDSLRUNet = None

# Dynamically load Stage 2 models
try:
    s2_film_path = Path(__file__).parent.parent / 'Stage2-Film-Enhancement/models/stage2_film.py'
    spec = importlib.util.spec_from_file_location("models.stage2_film", str(s2_film_path))
    s2_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s2_module)
    build_stage2_model = s2_module.build_stage2_model
except Exception as e:
    print(f"Warning: Could not import Stage 2 models: {e}")
    build_stage2_model = None

# Dynamically load ExposureOnlyUNet from Stage 1
try:
    exposure_model_path = Path(__file__).parent.parent / 'Stage1-Smartphone-to-DSLR/models/exposure_only_unet.py'
    spec = importlib.util.spec_from_file_location("models.exposure_only_unet", str(exposure_model_path))
    exposure_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exposure_module)
    ExposureOnlyUNet = exposure_module.ExposureOnlyUNet
except Exception as e:
    print(f"Warning: Could not import ExposureOnlyUNet: {e}")
    ExposureOnlyUNet = None

# Dynamically load DPEDResNetWithSkip from Stage 1
try:
    dped_model_path = Path(__file__).parent.parent / 'Stage1-Smartphone-to-DSLR/models/dped_resnet.py'
    spec = importlib.util.spec_from_file_location("models.dped_resnet", str(dped_model_path))
    dped_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dped_module)
    DPEDResNetWithSkip = dped_module.DPEDResNetWithSkip
except Exception as e:
    print(f"Warning: Could not import DPEDResNetWithSkip: {e}")
    DPEDResNetWithSkip = None


class TiledInferencePipeline:
    """
    Tile-based inference pipeline for full-resolution image processing
    
    Args:
        stage1_checkpoint: Path to Stage 1 model checkpoint
        stage2_checkpoint: Path to Stage 2 model checkpoint
        device: Device to run inference on
        tile_size: Size of tiles (model was trained on 224x224)
        overlap: Overlap between tiles to reduce edge artifacts
    """
    
    def __init__(self, 
                 stage1_checkpoint=None,
                 stage2_checkpoint=None,
                 device=None,
                 tile_size=224,
                 overlap=168):
        
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        
        print(f"Using device: {self.device}")
        print(f"Tile size: {tile_size}x{tile_size}")
        print(f"Overlap: {overlap} pixels")
        print(f"Stride: {self.stride} pixels\n")
        
        # Load Stage 1 model (optional)
        self.stage1_model = None
        if stage1_checkpoint:
            print("Loading Stage 1 model (Smartphone→DSLR)...")
            checkpoint = torch.load(stage1_checkpoint, map_location=self.device)
            state_dict = checkpoint['model_state_dict']
            
            # Strategy: Try loading models in order of likelihood provided by user
            # User specified DPED as the likely model
            
            model_loaded = False
            
            # 1. Try DPED ResNet (Standard/Default per dped.yaml)
            if not model_loaded and DPEDResNetWithSkip is not None:
                try:
                    model = DPEDResNetWithSkip(in_channels=3, out_channels=3)
                    model.load_state_dict(state_dict, strict=True)
                    self.stage1_model = model
                    model_loaded = True
                    print("  ✓ Detected and loaded: DPEDResNetWithSkip (Default)")
                except RuntimeError:
                    pass
            
            # 2. Try LightweightUNet
            if not model_loaded:
                try:
                    model = LightweightUNet(in_channels=3, out_channels=3, base_channels=32)
                    model.load_state_dict(state_dict, strict=True)
                    self.stage1_model = model
                    model_loaded = True
                    print("  ✓ Detected and loaded: LightweightUNet (Fast)")
                except RuntimeError:
                    pass

            # 3. Try ExposureOnlyUNet
            if not model_loaded and ExposureOnlyUNet is not None:
                try:
                    model = ExposureOnlyUNet(in_channels=3, out_channels=3, base_channels=64)
                    model.load_state_dict(state_dict, strict=True)
                    self.stage1_model = model
                    model_loaded = True
                    print("  ✓ Detected and loaded: ExposureOnlyUNet")
                except RuntimeError:
                    pass
            
            if model_loaded:
                self.stage1_model = self.stage1_model.to(self.device)
                self.stage1_model.eval()
                print(f"  ✓ Loaded from: {stage1_checkpoint}\n")
            else:
                raise ValueError("Could not load Stage 1 model. Tried: DPEDResNetWithSkip, LightweightUNet, ExposureOnlyUNet. Check architecture match.")
        else:
            print("Stage 1 model not provided. Will skip DSLR enhancement.\n")
        
        # Load Stage 2 model (optional)
        self.stage2_model = None
        if stage2_checkpoint:
            print("Loading Stage 2 model (DSLR→Film)...")
            self.stage2_model = build_stage2_model(model_type='lut', device=self.device)
            checkpoint = torch.load(stage2_checkpoint, map_location=self.device)
            self.stage2_model.load_state_dict(checkpoint['model_state_dict'])
            self.stage2_model.eval()
            print(f"  ✓ Loaded from: {stage2_checkpoint}\n")
        else:
            print("Stage 2 model not provided.")
        
        # Check that at least one model is provided
        if not stage1_checkpoint and not stage2_checkpoint:
            raise ValueError("At least one model (Stage 1 or Stage 2) must be provided!")
        
        # Image transforms
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
    
    def process(self, image_path, output_dir=None):
        """
        Process image using tiled inference with streaming accumulation (Low RAM)
        """
        import gc
        print(f"\nProcessing: {image_path}")
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        print(f"  Original size: {w}x{h}")
        
        # Prepare output accumulators on CPU
        # We need float32 for accumulation precision
        dslr_accum = torch.zeros(1, 3, h, w, dtype=torch.float32)
        weight_map = torch.zeros(1, 1, h, w, dtype=torch.float32)
        
        film_accum = None
        if self.stage2_model is not None:
             film_accum = torch.zeros(1, 3, h, w, dtype=torch.float32)
        
        # Prepare blending weight on DEVICE (small, reused)
        blend_weight = self.create_blend_weight()  # [1, 1, T, T] on Device
        
        # Calculate grid
        n_tiles_h = max(1, (h - self.overlap) // self.stride + (1 if (h - self.overlap) % self.stride != 0 else 0))
        n_tiles_w = max(1, (w - self.overlap) // self.stride + (1 if (w - self.overlap) % self.stride != 0 else 0))
        total_tiles = n_tiles_h * n_tiles_w
        
        print(f"  Processing {total_tiles} tiles (Streaming)...")
        
        tile_idx = 0
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                tile_idx += 1
                if tile_idx % 10 == 0 or tile_idx == 1:
                    print(f"    Tile {tile_idx}/{total_tiles}")
                    gc.collect() # Force GC periodically
                
                # Calculate coordinates
                y = min(i * self.stride, h - self.tile_size)
                x = min(j * self.stride, w - self.tile_size)
                
                # 1. Extract Tile (From PIL -> Tensor on Device)
                # Crop directly from PIL image to avoid holding full input tensor in RAM
                tile_pil = img.crop((x, y, x + self.tile_size, y + self.tile_size))
                tile_device = self.to_tensor(tile_pil).unsqueeze(0).to(self.device)
                
                # Pad if edge tile (PIL crop handles size, but we check just in case/consistency)
                if tile_device.shape[2] < self.tile_size or tile_device.shape[3] < self.tile_size:
                    curr_h, curr_w = tile_device.shape[2:]
                    pad_h = self.tile_size - curr_h
                    pad_w = self.tile_size - curr_w
                    tile_device = torch.nn.functional.pad(tile_device, (0, pad_w, 0, pad_h), mode='reflect')
                else:
                    pad_h = 0
                    pad_w = 0

                # 2. Stage 1 Inference
                if self.stage1_model is not None:
                    with torch.no_grad():
                        s1_out = self.stage1_model(tile_device)
                        s1_out = torch.clamp(s1_out, 0, 1)
                else:
                    s1_out = tile_device
                
                # 3. Stage 2 Inference (if needed)
                s2_out = None
                if self.stage2_model is not None:
                    with torch.no_grad():
                        s2_out = self.stage2_model(s1_out)
                        s2_out = torch.clamp(s2_out, 0, 1)
                
                # 4. Accumulate to CPU
                # We need to crop back the padding from the output and the weight
                valid_h = self.tile_size - pad_h
                valid_w = self.tile_size - pad_w
                
                # Get valid regions (on Device)
                s1_valid = s1_out[:, :, :valid_h, :valid_w]
                weight_valid = blend_weight[:, :, :valid_h, :valid_w]
                
                if s2_out is not None:
                    s2_valid = s2_out[:, :, :valid_h, :valid_w]
                
                # Transfer to CPU and Add
                # Using valid_h/valid_w ensures we put it in the right spot on canvas
                # Canvas coords: y:y+valid_h, x:x+valid_w
                
                # Move to CPU for accumulation
                dslr_accum[:, :, y:y+valid_h, x:x+valid_w] += s1_valid.cpu() * weight_valid.cpu()
                weight_map[:, :, y:y+valid_h, x:x+valid_w] += weight_valid.cpu()
                
                if s2_out is not None:
                    film_accum[:, :, y:y+valid_h, x:x+valid_w] += s2_valid.cpu() * weight_valid.cpu()
                
                # Explicit cleanup to be sure
                del tile_device, s1_out, s2_out, s1_valid, weight_valid, tile_pil
        
        # Normalize
        print("  Finalizing reconstruction...")
        weight_map = weight_map + 1e-8  # epsilon
        dslr_final = dslr_accum / weight_map
        
        film_final = None
        if film_accum is not None:
            film_final = film_accum / weight_map
            
        results = {
            'input': img, # store pil image
        }
        
        if self.stage1_model is not None:
            results['dslr'] = dslr_final
        if film_final is not None:
            results['film'] = film_final
        
        # Save outputs
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            image_name = Path(image_path).stem
            
            # Save input
            img.save(output_dir / f'{image_name}_0_input.png')
            print(f"  Saved input")
            
            # Save DSLR
            if 'dslr' in results:
                save_image(dslr_final, output_dir / f'{image_name}_1_dslr.png')
                print(f"  Saved DSLR")
            
            # Save Film
            if 'film' in results:
                save_image(film_final, output_dir / f'{image_name}_2_film.png')
                print(f"  Saved Film")

            # Save comparison
            if 'dslr' in results and 'film' in results:
                # Full pipeline: Input + DSLR + Film
                max_width = 3000
                
                # Convert PIL to tensor for comparison concatenation
                img_tensor_full = self.to_tensor(img).unsqueeze(0).to(self.device)
                
                if img_tensor_full.shape[3] > max_width:
                    scale = max_width / img_tensor_full.shape[3]
                    new_h = int(img_tensor_full.shape[2] * scale)
                    comparison_input = torch.nn.functional.interpolate(img_tensor_full, size=(new_h, max_width), mode='bilinear')
                    comparison_dslr = torch.nn.functional.interpolate(dslr_final.to(self.device), size=(new_h, max_width), mode='bilinear')
                    comparison_film = torch.nn.functional.interpolate(film_final.to(self.device), size=(new_h, max_width), mode='bilinear')
                else:
                    comparison_input = img_tensor_full
                    comparison_dslr = dslr_final.to(self.device)
                    comparison_film = film_final.to(self.device)
                
                comparison = torch.cat([comparison_input, comparison_dslr, comparison_film], dim=3)
                save_image(comparison, output_dir / f'{image_name}_comparison.png')
                print(f"  Saved comparison")
                del img_tensor_full, comparison

            elif 'film' in results:
                # Stage 2 only: Input + Film
                max_width = 3000
                
                img_tensor_full = self.to_tensor(img).unsqueeze(0).to(self.device)
                
                if img_tensor_full.shape[3] > max_width:
                    scale = max_width / img_tensor_full.shape[3]
                    new_h = int(img_tensor_full.shape[2] * scale)
                    comparison_input = torch.nn.functional.interpolate(img_tensor_full, size=(new_h, max_width), mode='bilinear')
                    comparison_film = torch.nn.functional.interpolate(film_final.to(self.device), size=(new_h, max_width), mode='bilinear')
                else:
                    comparison_input = img_tensor_full
                    comparison_film = film_final.to(self.device)
                
                comparison = torch.cat([comparison_input, comparison_film], dim=3)
                save_image(comparison, output_dir / f'{image_name}_comparison.png')
                print(f"  Saved comparison")
                del img_tensor_full, comparison 
                
        return results
    
    def create_blend_weight(self):
        """
        Create ultra-smooth Gaussian blending weight map
        Uses raised cosine (Hann window) for seamless blending
        """
        # Create 2D Hann window (raised cosine) - even smoother than regular cosine
        y = torch.linspace(0, 1, self.tile_size)
        x = torch.linspace(0, 1, self.tile_size)
        
        # Hann window: 0.5 * (1 - cos(2*pi*x))
        window_y = 0.5 * (1 - torch.cos(2 * 3.14159 * y))
        window_x = 0.5 * (1 - torch.cos(2 * 3.14159 * x))
        
        # Create 2D window
        Y, X = torch.meshgrid(window_y, window_x, indexing='ij')
        weight = Y * X
        
        # Apply Gaussian smoothing to weights for ultra-smooth transitions
        # This is critical for smooth areas like skies
        sigma = self.tile_size / 8
        kernel_size = int(self.tile_size / 4) | 1  # Make odd
        if kernel_size > 3:
            weight = self.gaussian_smooth(weight, kernel_size, sigma)
        
        # Normalize to avoid very small values
        weight = weight / weight.max()
        weight = weight + 0.01  # Small epsilon
        
        return weight.unsqueeze(0).unsqueeze(0).to(self.device)
    
    def gaussian_smooth(self, tensor, kernel_size, sigma):
        """Apply Gaussian smoothing to blend weights"""
        import torch.nn.functional as F
        
        # Create Gaussian kernel
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        
        # 2D Gaussian kernel
        kernel_1d = gauss.unsqueeze(0)
        kernel_2d = kernel_1d.t() @ kernel_1d
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
        
        # Apply smoothing
        padding = kernel_size // 2
        smoothed = F.conv2d(
            tensor.unsqueeze(0).unsqueeze(0),
            kernel_2d,
            padding=padding
        )
        
        return smoothed.squeeze(0).squeeze(0)
    
    def batch_process(self, input_dir, output_dir, extensions=('.jpg', '.png', '.jpeg', '.DNG', '.dng', '.heic', '.HEIC', '.heif', '.HEIF', '.PNG', '.JPG', '.JPEG')):
        """
        Process all images in a directory
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(input_dir.glob(f'*{ext}'))
        
        print(f"\nFound {len(image_files)} images to process")
        print("=" * 60)
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] {image_path.name}")
            try:
                self.process(str(image_path), output_dir=output_dir)
            except Exception as e:
                print(f"  ✗ Error processing {image_path.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print("\n" + "=" * 60)
        print(f"✓ Batch processing complete!")
        print(f"  Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Tiled Inference for Full-Resolution Enhancement')
    
    # Model checkpoints
    parser.add_argument('--stage1', type=str, default=None,
                      help='Path to Stage 1 checkpoint (optional)')
    parser.add_argument('--stage2', type=str, default=None,
                      help='Path to Stage 2 checkpoint (optional)')
    
    # Input/Output
    parser.add_argument('--input', type=str, required=True,
                      help='Input image or directory')
    parser.add_argument('--output', type=str, required=True,
                      help='Output directory')
    
    # Tiling parameters
    parser.add_argument('--tile-size', type=int, default=224,
                      help='Tile size (default: 224)')
    parser.add_argument('--overlap', type=int, default=168,
                      help='Overlap between tiles in pixels. More overlap = smoother blending (default: 168, 75%% overlap)')
    
    # Device
    parser.add_argument('--device', type=str, default=None, choices=['mps', 'cuda', 'cpu'],
                      help='Device to use for inference (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Validate that at least one model is provided
    if not args.stage1 and not args.stage2:
        parser.error("At least one model (--stage1 or --stage2) must be provided!")
    
    # Create pipeline
    pipeline = TiledInferencePipeline(
        stage1_checkpoint=args.stage1,
        stage2_checkpoint=args.stage2,
        device=args.device,
        tile_size=args.tile_size,
        overlap=args.overlap
    )
    
    # Process
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        pipeline.process(str(input_path), output_dir=args.output)
    elif input_path.is_dir():
        # Batch processing
        pipeline.batch_process(str(input_path), args.output)
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return
    
    print("\n" + "=" * 60)
    print("✨ Full-resolution enhancement complete! ✨")
    print("=" * 60)


if __name__ == '__main__':
    main()

