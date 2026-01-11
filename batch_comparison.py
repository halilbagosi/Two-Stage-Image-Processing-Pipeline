import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as transforms

# --- Configuration ---
ROOT_DIR = Path("/Users/halilbagosi/Downloads/Smartphone-DSLR-Image-Pipeline-V2-main")
TEST_IMAGES_DIR = ROOT_DIR / "code/Combined-Inference/test_images"
OUTPUT_DIR = ROOT_DIR / "visual_metrics"
OUTPUT_DIR.mkdir(exist_ok=True)

# Internal imports - Must be absolute to work from any CWD
sys.path.append(str(ROOT_DIR / "code" / "Combined-Inference"))
from inference import TiledInferencePipeline

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

CONFIGS = {
    "DPED_ResNet": {
        "s1": ROOT_DIR / "code/Stage1-Smartphone-to-DSLR/checkpoints/stage1_dped/best_model.pth",
        "s2": None
    },
    "ZRR_UNet": {
        "s1": ROOT_DIR / "code/Stage1-Smartphone-to-DSLR/checkpoints/stage1_ZRR/best_model.pth",
        "s2": None
    },
    "Failed_ZRR": {
        "s1": ROOT_DIR / "code/Stage1-Smartphone-to-DSLR/checkpoints/stage1_production/best_model.pth",
        "s2": None
    },
    "DPED_plus_FilmNet": {
        "s1": ROOT_DIR / "code/Stage1-Smartphone-to-DSLR/checkpoints/stage1_dped/best_model.pth",
        "s2": ROOT_DIR / "code/Stage2-Film-Enhancement/checkpoints/stage2_m4_optimized/best_model.pth"
    }
}

# Performance settings (Quality Defaults - User requested to stay at 224)
TILE_SIZE = 224
OVERLAP = 168

# Style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def generate_comparison(input_img, output_img, config_name, image_name):
    """Creates a side-by-side comparison: Input | Output."""
    w, h = input_img.size
    # Create a new image with double width
    combined = Image.new('RGB', (w * 2 + 10, h), (255, 255, 255))
    combined.paste(input_img, (0, 0))
    combined.paste(output_img, (w + 10, 0))
    
    # Save the comparison
    combined.save(OUTPUT_DIR / f"comparison_{image_name}_{config_name}.png")

def generate_histogram(img, config_name, image_name):
    """Generates RGB histogram for the processed image."""
    data = np.array(img)
    colors = ('red', 'green', 'blue')
    plt.figure(figsize=(10, 5))
    
    for i, color in enumerate(colors):
        hist, bin_edges = np.histogram(data[:, :, i], bins=256, range=(0, 255))
        plt.plot(bin_edges[0:-1], hist, color=color, alpha=0.7, label=f'{color.upper()}')
        plt.fill_between(bin_edges[0:-1], hist, color=color, alpha=0.1)

    plt.title(f'Histogram: {image_name} ({config_name})', fontsize=12, fontweight='bold')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Pixel Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(OUTPUT_DIR / f"histogram_{image_name}_{config_name}.png", bbox_inches='tight')
    plt.close()

def main():
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Initializing Batch Comparison on {device}...")
    
    image_files = sorted([f for f in TEST_IMAGES_DIR.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".heic", ".dng"]])
    
    if not image_files:
        print(f"⚠️ No test images found in {TEST_IMAGES_DIR}")
        return

    # Process each configuration
    for config_name, paths in CONFIGS.items():
        print(f"\n📁 Processing Configuration: {config_name}")
        
        # Load pipeline for this config
        try:
            pipeline = TiledInferencePipeline(
                stage1_checkpoint=str(paths["s1"]) if paths["s1"] else None,
                stage2_checkpoint=str(paths["s2"]) if paths["s2"] else None,
                device=device,
                tile_size=TILE_SIZE,
                overlap=OVERLAP
            )
        except Exception as e:
            print(f"❌ Failed to load pipeline for {config_name}: {e}")
            continue

        # Process all images
        for img_path in tqdm(image_files, desc=f"Enhancing images ({config_name})"):
            img_name = img_path.stem
            
            try:
                # Load input
                input_pil = Image.open(img_path).convert('RGB')
                
                # Run inference
                # Note: pipeline.process returns a dict with tensors
                results = pipeline.process(str(img_path), output_dir=None)
                
                # Convert result to PIL
                to_pil = transforms.ToPILImage()
                # Determine final result (film if exists, otherwise dslr)
                if 'film' in results:
                    output_tensor = results['film']
                elif 'dslr' in results:
                    output_tensor = results['dslr']
                else:
                    output_tensor = results['input']
                
                output_pil = to_pil(output_tensor.squeeze(0).cpu())
                
                # Generate visual items
                generate_comparison(input_pil, output_pil, config_name, img_name)
                generate_histogram(output_pil, config_name, img_name)
                
            except Exception as e:
                print(f"❌ Error processing {img_path.name} in {config_name}: {e}")

    print(f"\n✅ All results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
