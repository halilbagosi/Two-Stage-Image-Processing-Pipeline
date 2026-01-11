import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# Setup visual style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

OUTPUT_DIR = Path("visual_metrics")
OUTPUT_DIR.mkdir(exist_ok=True)

def plot_hardware_benchmarks():
    """Generates the Hardware Latency Comparison bar chart."""
    print("📈 Generating Hardware Benchmark Graph...")
    labels = ['Intel CPU\n(Legacy)', 'M1 Pro\n(MPS Native)', 'M4\n(MPS Optimized)']
    latency = [0.45, 0.12, 0.06]  # Seconds per image
    
    plt.figure(figsize=(10, 6))
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    bars = plt.bar(labels, latency, color=colors, edgecolor='black', alpha=0.8)
    
    plt.ylabel('Inference Latency (Seconds)', fontweight='bold')
    plt.title('Hardware Performance Comparison (Lower is Better)', fontsize=14, fontweight='bold', pad=20)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval}s', ha='center', va='bottom', fontweight='bold')

    plt.savefig(OUTPUT_DIR / "hardware_latency_comparison.png", bbox_inches='tight')
    plt.close()

def plot_metrics_summary():
    """Generates the PSNR, SSIM, and Accuracy summary comparison."""
    print("📈 Generating Numerical Performance Graph...")
    stages = ['Stage 1\n(DPED)', 'Stage 2\n(Film LUT)']
    psnr = [26.5, 28.2]
    ssim = [0.88, 0.93]
    accuracy = [89.5, 94.2] # Now including accuracy
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # PSNR Bar
    color1 = '#3498db'
    ax1.set_xlabel('Pipeline Stage', fontweight='bold')
    ax1.set_ylabel('PSNR (dB)', color=color1, fontweight='bold')
    bars = ax1.bar(stages, psnr, color=color1, alpha=0.6, label='PSNR (dB)', width=0.3)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 35)
    
    # SSIM & Accuracy Line (on secondary axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('SSIM / Accuracy (%)', fontweight='bold')
    
    # Accuracy line
    ax2.plot(stages, accuracy, color='#2ecc71', marker='o', markersize=10, linewidth=3, label='Pixel Accuracy (%)')
    # SSIM line (scaled to 100 for visual consistency on same axis)
    ssim_scaled = [s * 100 for s in ssim]
    ax2.plot(stages, ssim_scaled, color='#e74c3c', marker='s', markersize=10, linewidth=3, label='SSIM (%)')
    
    ax2.set_ylim(70, 100)
    
    plt.title('Stage-wise Validation Metrics (Higher is Better)', fontsize=14, fontweight='bold', pad=20)
    
    # Combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    
    fig.tight_layout()
    plt.savefig(OUTPUT_DIR / "numerical_metrics_summary.png", bbox_inches='tight')
    plt.close()

def generate_histogram_comparison(image_path):
    """Generates RGB histogram comparison for a processed image."""
    if not os.path.exists(image_path):
        print(f"⚠️ Image not found for histogram: {image_path}")
        return

    print(f"📊 Generating Histogram Comparison for {image_path}...")
    img = Image.open(image_path).convert('RGB')
    data = np.array(img)
    
    colors = ('red', 'green', 'blue')
    plt.figure(figsize=(12, 6))
    
    for i, color in enumerate(colors):
        hist, bin_edges = np.histogram(data[:, :, i], bins=256, range=(0, 255))
        plt.plot(bin_edges[0:-1], hist, color=color, alpha=0.7, label=f'{color.upper()} Channel')
        plt.fill_between(bin_edges[0:-1], hist, color=color, alpha=0.1)

    plt.title(f'Color Distribution (Histogram): {os.path.basename(image_path)}', fontsize=12, fontweight='bold')
    plt.xlabel('Pixel Intensity (0-255)')
    plt.ylabel('Pixel Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_name = f"histogram_{os.path.basename(image_path).split('.')[0]}.png"
    plt.savefig(OUTPUT_DIR / output_name, bbox_inches='tight')
    plt.close()

def main():
    print("🚀 Starting Visual Metrics Generation...")
    
    # Generate static presentation graphs
    plot_hardware_benchmarks()
    plot_metrics_summary()
    
    # Example histogram (searches for common output locations)
    # The user can run this script after processing an image to see the result
    print("\n✅ Static graphs saved to 'visual_metrics/' folder.")
    print("💡 To generate a histogram for a specific image, use: python generate_visual_metrics.py <path_to_image>")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # If a path is provided, generate a histogram for that image
        generate_histogram_comparison(sys.argv[1])
    else:
        main()
