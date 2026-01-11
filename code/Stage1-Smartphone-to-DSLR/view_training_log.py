#!/usr/bin/env python3
"""
View and analyze training logs
Displays training metrics from the log file in a formatted way
"""

import argparse
import json
from pathlib import Path
from datetime import timedelta


def view_text_log(log_file):
    """Display the text log file"""
    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        return
    
    print("\n" + "=" * 80)
    print("TRAINING LOG")
    print("=" * 80 + "\n")
    
    with open(log_file, 'r') as f:
        content = f.read()
        print(content)


def view_metrics_summary(metrics_file):
    """Display a summary of training metrics"""
    if not metrics_file.exists():
        print(f"❌ Metrics file not found: {metrics_file}")
        return
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    print("\n" + "=" * 80)
    print("TRAINING METRICS SUMMARY")
    print("=" * 80 + "\n")
    
    # Training info
    print("📋 Training Information:")
    info = data['training_info']
    print(f"  • Started: {info['start_time']}")
    print(f"  • Ended: {info['end_time']}")
    print(f"  • Duration: {info['total_duration']}")
    print(f"  • Total Epochs: {info['total_epochs']}")
    print(f"  • Model: {info['model_type']}")
    print(f"  • Device: {info['device']}")
    print(f"  • Batch Size: {info['batch_size']}")
    print(f"  • Learning Rate: {info['learning_rate']}")
    
    # Performance
    print("\n⚡ Performance:")
    perf = data['performance']
    print(f"  • Average Epoch Time: {perf['avg_epoch_time_seconds']:.1f} seconds")
    print(f"  • Average Speed: {perf['avg_batches_per_second']:.2f} batches/second")
    print(f"  • Total Batches: {perf['total_batches']:,}")
    print(f"  • Total Samples: {perf['total_samples_processed']:,}")
    
    # Best model
    if data['best_model']['epoch']:
        print("\n🏆 Best Model:")
        best = data['best_model']
        print(f"  • Epoch: {best['epoch']}")
        print(f"  • Train Loss: {best['train_loss']:.4f}")
        print(f"  • Val Loss: {best['val_loss']:.4f}")
        print(f"  • PSNR: {best['psnr']:.2f} dB")
        print(f"  • SSIM: {best['ssim']:.4f}")
    
    # Final metrics
    print("\n📊 Final Metrics:")
    final = data['final_metrics']
    print(f"  • Train Loss: {final['final_train_loss']:.4f}")
    if final['final_val_loss']:
        print(f"  • Val Loss: {final['final_val_loss']:.4f}")
        print(f"  • PSNR: {final['final_psnr']:.2f} dB")
        print(f"  • SSIM: {final['final_ssim']:.4f}")
    
    print("\n" + "=" * 80 + "\n")


def view_epoch_details(metrics_file, last_n=10):
    """Display detailed metrics for the last N epochs"""
    if not metrics_file.exists():
        print(f"❌ Metrics file not found: {metrics_file}")
        return
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    epochs = data['epoch_details'][-last_n:] if len(data['epoch_details']) > last_n else data['epoch_details']
    
    print("\n" + "=" * 100)
    print(f"LAST {len(epochs)} EPOCHS - DETAILED METRICS")
    print("=" * 100 + "\n")
    
    # Header
    print(f"{'Epoch':<8}{'Train Loss':<15}{'Val Loss':<15}{'PSNR (dB)':<12}{'SSIM':<12}{'Time (s)':<12}{'Speed':<15}")
    print("-" * 100)
    
    # Rows
    for epoch in epochs:
        epoch_num = epoch['epoch']
        train_loss = epoch['train_loss']
        val_loss = epoch.get('val_loss', '-')
        psnr = epoch.get('psnr', '-')
        ssim = epoch.get('ssim', '-')
        time_s = epoch['epoch_time']
        speed = epoch['batches_per_sec']
        
        val_loss_str = f"{val_loss:.4f}" if val_loss != '-' else '-'
        psnr_str = f"{psnr:.2f}" if psnr != '-' else '-'
        ssim_str = f"{ssim:.4f}" if ssim != '-' else '-'
        
        print(f"{epoch_num:<8}{train_loss:<15.4f}{val_loss_str:<15}{psnr_str:<12}{ssim_str:<12}{time_s:<12.1f}{speed:<15.2f}")
    
    print("\n" + "=" * 100 + "\n")


def plot_training_curves(metrics_file):
    """Plot training curves (requires matplotlib)"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("❌ Matplotlib not installed. Install it to plot curves:")
        print("   pip install matplotlib")
        return
    
    if not metrics_file.exists():
        print(f"❌ Metrics file not found: {metrics_file}")
        return
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    epochs_data = data['epoch_details']
    
    # Extract data
    epochs = [e['epoch'] for e in epochs_data]
    train_losses = [e['train_loss'] for e in epochs_data]
    val_losses = [e.get('val_loss') for e in epochs_data if 'val_loss' in e]
    val_epochs = [e['epoch'] for e in epochs_data if 'val_loss' in e]
    psnrs = [e.get('psnr') for e in epochs_data if 'psnr' in e]
    ssims = [e.get('ssim') for e in epochs_data if 'ssim' in e]
    speeds = [e['batches_per_sec'] for e in epochs_data]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')
    
    # Loss plot
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_losses, label='Train Loss', linewidth=2)
    if val_losses:
        ax1.plot(val_epochs, val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # PSNR plot
    ax2 = axes[0, 1]
    if psnrs:
        ax2.plot(val_epochs, psnrs, label='PSNR', color='green', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('PSNR (dB)')
        ax2.set_title('Peak Signal-to-Noise Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No PSNR data', ha='center', va='center', transform=ax2.transAxes)
    
    # SSIM plot
    ax3 = axes[1, 0]
    if ssims:
        ax3.plot(val_epochs, ssims, label='SSIM', color='orange', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('SSIM')
        ax3.set_title('Structural Similarity Index')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No SSIM data', ha='center', va='center', transform=ax3.transAxes)
    
    # Speed plot
    ax4 = axes[1, 1]
    ax4.plot(epochs, speeds, label='Training Speed', color='purple', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Batches/Second')
    ax4.set_title('Training Speed')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = metrics_file.parent / 'training_curves.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_file}")
    
    # Show plot
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='View training logs and metrics')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/stage1_exposure_only',
                      help='Path to checkpoint directory')
    parser.add_argument('--summary', action='store_true',
                      help='Show summary only')
    parser.add_argument('--details', action='store_true',
                      help='Show detailed epoch metrics')
    parser.add_argument('--last', type=int, default=10,
                      help='Number of last epochs to show in details (default: 10)')
    parser.add_argument('--plot', action='store_true',
                      help='Plot training curves (requires matplotlib)')
    parser.add_argument('--all', action='store_true',
                      help='Show everything')
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    log_file = checkpoint_dir / 'training_log.txt'
    metrics_file = checkpoint_dir / 'training_metrics.json'
    
    # Check if files exist
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        print("\nMake sure training has been run at least once.")
        return
    
    # Show based on arguments
    if args.all:
        view_text_log(log_file)
        view_metrics_summary(metrics_file)
        view_epoch_details(metrics_file, args.last)
        if args.plot:
            plot_training_curves(metrics_file)
    elif args.summary:
        view_metrics_summary(metrics_file)
    elif args.details:
        view_epoch_details(metrics_file, args.last)
    elif args.plot:
        plot_training_curves(metrics_file)
    else:
        # Default: show summary and recent epochs
        view_metrics_summary(metrics_file)
        view_epoch_details(metrics_file, args.last)


if __name__ == '__main__':
    main()









