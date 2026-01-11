"""
Simplified: Recompute validation metrics on saved checkpoints
Computes PSNR, SSIM, MAE, ΔE, and LPIPS for specified checkpoints
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from tqdm import tqdm

# Add paths
sys.path.append(str(Path(__file__).parent))

from utils.metrics import MetricsCalculator
from models.stage1_unet import build_stage1_model
from models.dped_resnet import DPEDResNetWithSkip
from models.exposure_only_unet import ExposureOnlyUNet

# Try to import dataset, but have fallback
try:
    from data.dataset import DPEDDataset, get_dataloader
    HAS_DATASET = True
except:
    HAS_DATASET = False
    print("Warning: Could not import dataset module. Will compute metrics on sample data.")

def load_model_from_checkpoint(checkpoint_path, device):
    """Load model from checkpoint, auto-detecting architecture"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Try different architectures
    for ModelClass in [DPEDResNetWithSkip, ExposureOnlyUNet]:
        try:
            if ModelClass == DPEDResNetWithSkip:
                model = ModelClass(in_channels=3, out_channels=3)
            else:
                model = ModelClass(in_channels=3, out_channels=3, base_channels=64)
            
            model.load_state_dict(state_dict, strict=True)
            model = model.to(device)
            model.eval()
            return model, checkpoint
        except:
            continue
    
    raise ValueError("Could not load model with any known architecture")

def recompute_metrics_simple(checkpoint_dir, val_phone_dir=None, val_dslr_dir=None, epochs_to_eval=None, max_batches=None, output_name='recomputed_metrics.txt'):
    """
    Recompute metrics for saved checkpoints
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        val_phone_dir: Validation phone images directory
        val_dslr_dir: Validation DSLR images directory  
        epochs_to_eval: List of epoch numbers to evaluate
        max_batches: Maximum number of batches to evaluate (None for all)
        output_name: Name of the output text file
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}\n")
    
    # Find checkpoints
    checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
    
    if epochs_to_eval is not None:
        epochs_to_eval = set(epochs_to_eval)
        checkpoints = [cp for cp in checkpoints 
                      if int(cp.stem.split('_')[-1]) in epochs_to_eval]
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoints)} checkpoints to evaluate")
    
    # Load validation data if available
    val_loader = None
    if HAS_DATASET and val_phone_dir and val_dslr_dir:
        print(f"Loading validation dataset from:")
        print(f"  Phone: {val_phone_dir}")
        print(f"  DSLR: {val_dslr_dir}\n")
        
        from data.dataset import DPEDDataset, get_dataloader
        val_dataset = DPEDDataset(
            phone_dir=val_phone_dir,
            dslr_dir=val_dslr_dir,
            img_size=224,
            split='test'
        )
        val_loader = get_dataloader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    else:
        print("No validation data provided. Using synthetic data for demonstration.\n")
    
    # Results storage
    results = []
    
    # Evaluate each checkpoint
    for checkpoint_path in tqdm(checkpoints, desc="Evaluating checkpoints"):
        epoch_num = int(checkpoint_path.stem.split('_')[-1])
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch_num}")
        print(f"{'='*60}")
        
        try:
            # Load model
            model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
            
            # Compute metrics
            metrics_calc = MetricsCalculator()
            
            with torch.no_grad():
                if val_loader:
                    # Real validation data
                    for batch_idx, (phone, dslr) in enumerate(val_loader):
                        phone = phone.to(device)
                        dslr = dslr.to(device)
                        
                        pred = model(phone)
                        pred = torch.clamp(pred, 0, 1)
                        
                        metrics_calc.update(pred, dslr)
                        
                        if max_batches is not None and batch_idx >= max_batches:
                            break
                else:
                    # Synthetic data (for demonstration)
                    print("  Using synthetic data...")
                    for _ in range(10):
                        phone = torch.rand(2, 3, 224, 224).to(device)
                        dslr = torch.rand(2, 3, 224, 224).to(device)
                        
                        pred = model(phone)
                        pred = torch.clamp(pred, 0, 1)
                        
                        metrics_calc.update(pred, dslr)
            
            # Get final metrics
            metrics = metrics_calc.get_metrics()
            metrics['epoch'] = epoch_num
            metrics['val_loss'] = checkpoint.get('val_loss', 0.0)
            
            results.append(metrics)
            
            print(f"Results: {metrics_calc.get_summary_string()}")
            
        except Exception as e:
            print(f"Error evaluating epoch {epoch_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        print("\nNo results generated.")
        return
    
    # Save results
    output_file = checkpoint_dir / output_name
    with open(output_file, 'w') as f:
        f.write("Epoch\tVal_Loss\tPSNR\tSSIM\tMAE\tDelta_E\tLPIPS\tAccuracy\n")
        for result in results:
            f.write(f"{result['epoch']}\t"
                   f"{result['val_loss']:.6f}\t"
                   f"{result['psnr']:.2f}\t"
                   f"{result['ssim']:.4f}\t"
                   f"{result['mae']:.4f}\t"
                   f"{result['delta_e']:.2f}\t"
                   f"{result['lpips']:.4f}\t"
                   f"{result['pixel_accuracy']:.2f}%\n")
    
    print(f"\n{'='*60}")
    print(f"✓ Results saved to: {output_file}")
    print(f"{'='*60}\n")
    
    # Find best epoch
    best_psnr_idx = max(range(len(results)), key=lambda i: results[i]['psnr'])
    best_ssim_idx = max(range(len(results)), key=lambda i: results[i]['ssim'])
    best_lpips_idx = min(range(len(results)), key=lambda i: results[i]['lpips'])
    best_delta_e_idx = min(range(len(results)), key=lambda i: results[i]['delta_e'])
    
    print("=== Best Models ===")
    print(f"Best PSNR:    Epoch {results[best_psnr_idx]['epoch']:3d} (PSNR={results[best_psnr_idx]['psnr']:.2f})")
    print(f"Best SSIM:    Epoch {results[best_ssim_idx]['epoch']:3d} (SSIM={results[best_ssim_idx]['ssim']:.4f})")
    print(f"Best LPIPS:   Epoch {results[best_lpips_idx]['epoch']:3d} (LPIPS={results[best_lpips_idx]['lpips']:.4f})")
    print(f"Best Delta-E: Epoch {results[best_delta_e_idx]['epoch']:3d} (ΔE={results[best_delta_e_idx]['delta_e']:.2f})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Recompute metrics on saved checkpoints')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                       help='Directory containing checkpoints')
    parser.add_argument('--val-phone-dir', type=str, default=None,
                       help='Validation phone images directory')
    parser.add_argument('--val-dslr-dir', type=str, default=None,
                       help='Validation DSLR images directory')
    parser.add_argument('--epochs', type=int, nargs='+', default=None,
                       help='Specific epochs to evaluate (e.g., 5 10 15 20 25)')
    parser.add_argument('--max-batches', type=int, default=None,
                       help='Maximum batches to evaluate (default: all)')
    parser.add_argument('--output', type=str, default='recomputed_metrics.txt',
                       help='Output filename')
    
    args = parser.parse_args()
    
    recompute_metrics_simple(
        args.checkpoint_dir,
        args.val_phone_dir,
        args.val_dslr_dir,
        args.epochs,
        max_batches=args.max_batches,
        output_name=args.output
    )


if __name__ == '__main__':
    main()
