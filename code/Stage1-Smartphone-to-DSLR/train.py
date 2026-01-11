"""
Training script for Stage 1: Smartphone to DSLR Enhancement
"""

import os
import sys
import time
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from data.dataset import DPEDDataset, get_dataloader
from models.stage1_unet import build_stage1_model
from losses.combined_loss import build_loss_function
from utils.metrics import MetricsCalculator


class Trainer:
    """
    Trainer class for Stage 1 model
    Handles training loop, validation, checkpointing, and logging
    """
    
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print("=" * 60)
        print("Stage 1: Smartphone to DSLR Enhancement - Training")
        print("=" * 60)
        
        # Support both 'epochs' and 'num_epochs'
        self.num_epochs = self.config['training'].get('epochs') or self.config['training'].get('num_epochs', 100)
        
        # Setup device
        self.device = self._setup_device()
        
        # Auto-optimize settings for detected hardware
        self._auto_optimize_settings()
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create results directory for validation images
        self.results_dir = self.checkpoint_dir / 'validation_images'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Build model
        self.model = self._build_model()
        
        # Build loss function
        self.criterion = self._build_loss()
        
        # Build optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Load datasets
        self.train_loader, self.val_loader = self._load_datasets()
        
        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Training metrics for logging
        self.epoch_metrics = []  # Store detailed metrics per epoch
        self.training_start_time = None
        
        # Setup training log file
        self.log_file = self.checkpoint_dir / 'training_log.txt'
        self.metrics_file = self.checkpoint_dir / 'training_metrics.json'
        
        # Resume from checkpoint if specified
        if self.config.get('checkpoint', {}).get('resume', False):
            self._resume_from_checkpoint()
        
        print("\n" + "=" * 60)
        print("Setup complete! Ready to train.")
        print("=" * 60 + "\n")
    
    def _setup_device(self):
        """Setup computation device"""
        device_type = self.config['device']['type']
        
        if device_type == 'mps' and torch.backends.mps.is_available():
            device = torch.device('mps')
            
            # Detect Apple Silicon chip
            import subprocess
            try:
                chip_info = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
                if 'M4' in chip_info:
                    chip = 'M4'
                elif 'M3' in chip_info:
                    chip = 'M3'
                elif 'M2' in chip_info:
                    chip = 'M2'
                elif 'M1' in chip_info:
                    chip = 'M1'
                else:
                    chip = 'Apple Silicon'
                print(f"✓ Using device: MPS ({chip})")
            except:
                print(f"✓ Using device: MPS (Apple Silicon)")
                
        elif device_type == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✓ Using device: CUDA")
        else:
            device = torch.device('cpu')
            print(f"✓ Using device: CPU")
        
        return device
    
    def _auto_optimize_settings(self):
        """Auto-optimize batch size and workers based on hardware"""
        import subprocess
        
        # Only optimize if using MPS (Apple Silicon)
        if self.device.type != 'mps':
            return
        
        # Check if auto_optimize is enabled (default: True)
        if not self.config.get('training', {}).get('auto_optimize', True):
            return
        
        try:
            chip_info = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
            
            # Detect chip and set optimal settings
            if 'M4' in chip_info or 'M3 Max' in chip_info or 'M3 Ultra' in chip_info:
                optimal_batch = 16
                optimal_workers = 8
                chip_name = 'M4/M3 Max'
            elif 'M3' in chip_info or 'M2 Max' in chip_info or 'M2 Ultra' in chip_info:
                optimal_batch = 14
                optimal_workers = 8
                chip_name = 'M3/M2 Max'
            elif 'M2' in chip_info or 'M1 Max' in chip_info or 'M1 Ultra' in chip_info:
                optimal_batch = 12
                optimal_workers = 6
                chip_name = 'M2/M1 Max'
            elif 'M1 Pro' in chip_info:
                optimal_batch = 12
                optimal_workers = 6
                chip_name = 'M1 Pro'
            elif 'M1' in chip_info:
                optimal_batch = 10
                optimal_workers = 6
                chip_name = 'M1'
            else:
                return  # Don't optimize if chip unknown
            
            # Get current settings
            current_batch = self.config['training']['batch_size']
            current_workers = self.config['training']['num_workers']
            
            # Apply optimizations if different
            if current_batch != optimal_batch or current_workers != optimal_workers:
                print(f"\n🔧 Auto-optimizing for {chip_name}:")
                if current_batch != optimal_batch:
                    print(f"   Batch size: {current_batch} → {optimal_batch}")
                    self.config['training']['batch_size'] = optimal_batch
                if current_workers != optimal_workers:
                    print(f"   Workers: {current_workers} → {optimal_workers}")
                    self.config['training']['num_workers'] = optimal_workers
                print()
        except:
            pass  # Silently fail if detection doesn't work
    
    def _build_model(self):
        """Build and initialize model"""
        model = build_stage1_model(
            model_type=self.config['model']['type'],
            device=self.device
        )
        return model
    
    def _build_loss(self):
        """Build loss function"""
        # Support both old and new config structures
        if 'loss' in self.config and 'type' in self.config['loss']:
            # New config structure (DPED)
            self.loss_type = self.config['loss']['type']
            loss_config = self.config['loss']
        else:
            # Old config structure
            self.loss_type = self.config['training'].get('loss_type', 'dped')
            loss_config = self.config['training'].get('loss_weights', {})
        
        loss_type = self.loss_type
        
        if loss_type == 'dped':
            # DPED loss: Color + Texture (Adversarial) + Content (VGG)
            criterion = build_loss_function(
                loss_type='dped',
                device=self.device,
                w_color=loss_config.get('w_color', 1.0),
                w_texture=loss_config.get('w_texture', 0.1),
                w_content=loss_config.get('w_content', 0.1),
                color_sigma=loss_config.get('color_sigma', 1.0),
                use_vgg19=loss_config.get('use_vgg19', False)
            )
        elif loss_type == 'combined':
            criterion = build_loss_function(
                loss_type='combined',
                device=self.device,
                w_l1=loss_config.get('l1', 1.0),
                w_perceptual=loss_config.get('perceptual', 0.1),
                w_ssim=loss_config.get('ssim', 0.1),
                w_color=loss_config.get('color', 0.1)
            )
        else:  # simplified
            criterion = build_loss_function(
                loss_type='simplified',
                w_l1=loss_config.get('l1', 1.0),
                w_ssim=loss_config.get('ssim', 0.1)
            )
        
        return criterion
    
    def _build_optimizer(self):
        """Build optimizer"""
        opt_config = self.config['training']
        optimizer_type = opt_config['optimizer'].lower()
        
        if optimizer_type == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                betas=opt_config.get('betas', (0.9, 0.999)),
                weight_decay=opt_config.get('weight_decay', 0)
            )
            print(f"✓ Optimizer: Adam, LR: {opt_config['lr']}")
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                betas=opt_config.get('betas', (0.9, 0.999)),
                weight_decay=opt_config.get('weight_decay', 0.0001)  # Use config value, not hardcoded
            )
            print(f"✓ Optimizer: AdamW, LR: {opt_config['lr']}, Weight Decay: {opt_config.get('weight_decay', 0.0001)}")
        elif optimizer_type == 'sgd':
            # SGD with momentum - often better for image enhancement tasks
            momentum = opt_config.get('momentum', 0.9)  # Standard momentum value
            nesterov = opt_config.get('nesterov', True)  # Nesterov momentum (often better)
            weight_decay = opt_config.get('weight_decay', 0.0001)
            
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=opt_config['lr'],
                momentum=momentum,
                nesterov=nesterov,
                weight_decay=weight_decay
            )
            print(f"✓ Optimizer: SGD, LR: {opt_config['lr']}, Momentum: {momentum}, Nesterov: {nesterov}")
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}. Choose from: 'adam', 'adamw', 'sgd'")
        
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler"""
        sched_type = self.config['training']['scheduler']
        
        if sched_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=self.config['training'].get('lr_min', 1e-7)
            )
            print(f"✓ Scheduler: Cosine Annealing")
        elif sched_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
            print(f"✓ Scheduler: Step LR")
        else:
            scheduler = None
            print(f"✓ Scheduler: None")
        
        return scheduler
    
    def _load_datasets(self):
        """Load training and validation datasets"""
        dataset_config = self.config['dataset']
        training_config = self.config['training']
        
        # Training dataset
        train_dataset = DPEDDataset(
            phone_dir=dataset_config['train_phone_dir'],
            dslr_dir=dataset_config['train_dslr_dir'],
            split='train',
            img_size=dataset_config['img_size'],
            augment=dataset_config['augment'],
            max_samples=dataset_config.get('max_samples', None)  # Support subset for faster training
        )
        
        train_loader = get_dataloader(
            train_dataset,
            batch_size=training_config['batch_size'],
            shuffle=True,
            num_workers=training_config['num_workers']
        )
        
        # Validation dataset (support both 'test' and 'val' naming)
        val_phone_dir = dataset_config.get('test_phone_dir') or dataset_config.get('val_phone_dir')
        val_dslr_dir = dataset_config.get('test_dslr_dir') or dataset_config.get('val_dslr_dir')
        
        val_dataset = DPEDDataset(
            phone_dir=val_phone_dir,
            dslr_dir=val_dslr_dir,
            split='test',
            img_size=dataset_config['img_size'],
            augment=False
        )
        
        val_loader = get_dataloader(
            val_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            num_workers=training_config['num_workers']
        )
        
        print(f"✓ Training samples: {len(train_dataset)}")
        print(f"✓ Validation samples: {len(val_dataset)}")
        print(f"✓ Batch size: {training_config['batch_size']}")
        
        return train_loader, val_loader
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_losses_dict = {}
        
        # Track timing
        epoch_start_time = time.time()
        batch_times = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for batch_idx, (phone_imgs, dslr_imgs) in enumerate(pbar):
            batch_start_time = time.time()
            
            # Move to device
            phone_imgs = phone_imgs.to(self.device)
            dslr_imgs = dslr_imgs.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_imgs = self.model(phone_imgs)
            
            # Compute loss
            loss, loss_dict = self.criterion(pred_imgs, dslr_imgs)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track batch time
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # Accumulate losses
            epoch_loss += loss.item()
            for k, v in loss_dict.items():
                epoch_losses_dict[k] = epoch_losses_dict.get(k, 0) + v
            
            # Calculate running averages for progress bar
            avg_batch_time = sum(batch_times[-10:]) / min(10, len(batch_times))  # Last 10 batches
            iter_per_sec = 1.0 / avg_batch_time if avg_batch_time > 0 else 0
            elapsed_time = time.time() - epoch_start_time
            remaining_batches = len(self.train_loader) - (batch_idx + 1)
            eta_seconds = remaining_batches * avg_batch_time
            
            # Update progress bar with detailed info
            log_freq = self.config.get('logging', {}).get('log_every') or self.config.get('logging', {}).get('log_frequency', 50)
            if batch_idx % log_freq == 0 or batch_idx == len(self.train_loader) - 1:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'iter/s': f"{iter_per_sec:.2f}",
                    'time/batch': f"{avg_batch_time:.3f}s",
                    'ETA': f"{eta_seconds/60:.1f}m"
                })
        
        # Calculate epoch statistics
        epoch_time = time.time() - epoch_start_time
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        batches_per_sec = len(self.train_loader) / epoch_time if epoch_time > 0 else 0
        
        # Average losses
        avg_loss = epoch_loss / len(self.train_loader)
        avg_losses_dict = {k: v / len(self.train_loader) for k, v in epoch_losses_dict.items()}
        
        # Add timing info to loss dict
        avg_losses_dict['epoch_time'] = epoch_time
        avg_losses_dict['batches_per_sec'] = batches_per_sec
        avg_losses_dict['avg_batch_time'] = avg_batch_time
        avg_losses_dict['iterations_per_sec'] = batches_per_sec
        
        return avg_loss, avg_losses_dict
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        val_losses_dict = {}
        metrics_calc = MetricsCalculator()
        
        with torch.no_grad():
            for phone_imgs, dslr_imgs in tqdm(self.val_loader, desc="Validating"):
                phone_imgs = phone_imgs.to(self.device)
                dslr_imgs = dslr_imgs.to(self.device)
                
                # Forward pass
                pred_imgs = self.model(phone_imgs)
                
                # Compute loss
                loss, loss_dict = self.criterion(pred_imgs, dslr_imgs)
                
                val_loss += loss.item()
                for k, v in loss_dict.items():
                    val_losses_dict[k] = val_losses_dict.get(k, 0) + v
                
                # Update metrics
                metrics_calc.update(pred_imgs, dslr_imgs)
        
        # Average losses and metrics
        avg_val_loss = val_loss / len(self.val_loader)
        avg_val_losses_dict = {k: v / len(self.val_loader) for k, v in val_losses_dict.items()}
        metrics = metrics_calc.get_metrics()
        
        # Save sample images
        if self.config['validation']['save_images']:
            self._save_validation_images(epoch, phone_imgs, pred_imgs, dslr_imgs)
        
        return avg_val_loss, avg_val_losses_dict, metrics
    
    def _save_validation_images(self, epoch, phone, pred, dslr):
        """Save sample validation images"""
        n_save = min(self.config['validation']['num_save_images'], phone.size(0))
        
        comparison = torch.cat([
            phone[:n_save],
            pred[:n_save],
            dslr[:n_save]
        ], dim=0)
        
        save_path = self.results_dir / f'epoch_{epoch+1:03d}.png'
        save_image(comparison, save_path, nrow=n_save, normalize=False)
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        # Always save last model (for resuming)
        last_save_path = self.checkpoint_dir / 'last_model.pth'
        torch.save(checkpoint, last_save_path)
        
        # Save regular checkpoint (every N epochs)
        save_freq = self.config['checkpoint'].get('save_every') or self.config.get('training', {}).get('save_frequency', 5)
        if (epoch + 1) % save_freq == 0:
            save_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1:03d}.pth'
            torch.save(checkpoint, save_path)
            print(f"✓ Saved checkpoint: {save_path} (epoch {epoch+1})")
        
        # Save best checkpoint
        if is_best and self.config['checkpoint']['save_best']:
            save_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, save_path)
            print(f"✓ Saved best model: {save_path}")
    
    def _resume_from_checkpoint(self):
        """Resume training from checkpoint"""
        resume_config = self.config['checkpoint'].get('resume')
        
        # If resume is True, look for last_model.pth
        if resume_config is True:
            checkpoint_path = self.checkpoint_dir / 'last_model.pth'
        elif isinstance(resume_config, str):
            checkpoint_path = Path(resume_config)
        else:
            return
        
        if not checkpoint_path.exists():
            print(f"ℹ️  No checkpoint found to resume from: {checkpoint_path}")
            print("   Starting training from scratch...")
            return
        
        print(f"\n{'='*60}")
        print(f"Resuming from checkpoint: {checkpoint_path}")
        print(f"{'='*60}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        # Load training history if available
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        
        print(f"✓ Resumed successfully!")
        print(f"  Starting from epoch {self.start_epoch}/{self.num_epochs}")
        print(f"  Best validation loss so far: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
    
    def _init_log_file(self):
        """Initialize training log file (only if it doesn't exist or starting from scratch)"""
        # Only initialize if file doesn't exist or we're starting from epoch 0
        if not self.log_file.exists() or self.start_epoch == 0:
            with open(self.log_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("Training Log - Stage 1: DPED Model\n")
                f.write("=" * 80 + "\n")
                f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {self.config['model']['type']}\n")
                f.write(f"Total Epochs: {self.num_epochs}\n")
                f.write(f"Batch Size: {self.config['training']['batch_size']}\n")
                f.write(f"Learning Rate: {self.config['training']['lr']}\n")
                f.write(f"Optimizer: {self.config['training']['optimizer']}\n")
                f.write(f"Device: {self.device}\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"{'Epoch':<8}{'Train Loss':<15}{'Val Loss':<15}{'PSNR':<12}{'SSIM':<12}{'Time(s)':<12}{'Time(min)':<12}{'Iter/s':<12}{'Batch(s)':<12}\n")
                f.write("-" * 100 + "\n")
        else:
            # Resuming - append a separator to show where resume happened
            with open(self.log_file, 'a') as f:
                f.write("\n" + "=" * 100 + "\n")
                f.write(f"RESUMED TRAINING - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Resuming from epoch {self.start_epoch}\n")
                f.write("=" * 100 + "\n\n")
    
    def _log_epoch(self, metrics):
        """Log epoch metrics to file"""
        with open(self.log_file, 'a') as f:
            epoch = metrics['epoch']
            train_loss = metrics['train_loss']
            val_loss = metrics.get('val_loss', '-')
            psnr = metrics.get('psnr', '-')
            ssim = metrics.get('ssim', '-')
            epoch_time = metrics['epoch_time']
            epoch_time_min = epoch_time / 60
            iter_per_sec = metrics.get('iterations_per_sec', metrics.get('batches_per_sec', 0))
            avg_batch_time = metrics.get('avg_batch_time', 0)
            
            # Format values
            val_loss_str = f"{val_loss:.4f}" if val_loss != '-' else '-' * 10
            psnr_str = f"{psnr:.2f}" if psnr != '-' else '-' * 8
            ssim_str = f"{ssim:.4f}" if ssim != '-' else '-' * 8
            
            line = f"{epoch:<8}{train_loss:<15.4f}{val_loss_str:<15}{psnr_str:<12}{ssim_str:<12}{epoch_time:<12.1f}{epoch_time_min:<12.2f}{iter_per_sec:<12.2f}{avg_batch_time:<12.3f}\n"
            f.write(line)
    
    def _save_training_summary(self):
        """Save comprehensive training summary"""
        import json
        from datetime import timedelta
        
        total_time = time.time() - self.training_start_time
        completed_epochs = len(self.epoch_metrics)
        
        # Calculate averages
        avg_epoch_time = sum(m['epoch_time'] for m in self.epoch_metrics) / completed_epochs
        avg_speed = sum(m['batches_per_sec'] for m in self.epoch_metrics) / completed_epochs
        
        # Get validation metrics (only from epochs with validation)
        val_metrics = [m for m in self.epoch_metrics if 'val_loss' in m]
        
        # Find best epoch
        best_epoch = None
        if val_metrics:
            best_epoch = min(val_metrics, key=lambda x: x['val_loss'])
        
        # Summary data
        summary = {
            'training_info': {
                'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.training_start_time)),
                'end_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_duration': str(timedelta(seconds=int(total_time))),
                'total_epochs': completed_epochs,
                'model_type': self.config['model']['type'],
                'device': str(self.device),
                'batch_size': self.config['training']['batch_size'],
                'learning_rate': self.config['training']['lr']
            },
            'performance': {
                'avg_epoch_time_seconds': round(avg_epoch_time, 2),
                'avg_batches_per_second': round(avg_speed, 2),
                'total_batches': len(self.train_loader) * completed_epochs,
                'total_samples_processed': len(self.train_loader.dataset) * completed_epochs
            },
            'best_model': {
                'epoch': best_epoch['epoch'] if best_epoch else None,
                'train_loss': round(best_epoch['train_loss'], 4) if best_epoch else None,
                'val_loss': round(best_epoch['val_loss'], 4) if best_epoch else None,
                'psnr': round(best_epoch['psnr'], 2) if best_epoch else None,
                'ssim': round(best_epoch['ssim'], 4) if best_epoch else None
            },
            'final_metrics': {
                'final_train_loss': round(self.epoch_metrics[-1]['train_loss'], 4),
                'final_val_loss': round(val_metrics[-1]['val_loss'], 4) if val_metrics else None,
                'final_psnr': round(val_metrics[-1]['psnr'], 2) if val_metrics else None,
                'final_ssim': round(val_metrics[-1]['ssim'], 4) if val_metrics else None
            },
            'epoch_details': self.epoch_metrics
        }
        
        # Save JSON
        with open(self.metrics_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Append summary to log file
        with open(self.log_file, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("TRAINING SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total Duration: {summary['training_info']['total_duration']}\n")
            f.write(f"Total Epochs: {completed_epochs}\n")
            f.write(f"Average Epoch Time: {avg_epoch_time:.1f} seconds\n")
            f.write(f"Average Speed: {avg_speed:.2f} batches/second\n")
            f.write(f"Total Samples Processed: {summary['performance']['total_samples_processed']:,}\n")
            f.write("\n")
            
            if best_epoch:
                f.write("Best Model:\n")
                f.write(f"  Epoch: {best_epoch['epoch']}\n")
                f.write(f"  Train Loss: {best_epoch['train_loss']:.4f}\n")
                f.write(f"  Val Loss: {best_epoch['val_loss']:.4f}\n")
                f.write(f"  PSNR: {best_epoch['psnr']:.2f} dB\n")
                f.write(f"  SSIM: {best_epoch['ssim']:.4f}\n")
                f.write("\n")
            
            f.write("Final Metrics:\n")
            f.write(f"  Train Loss: {summary['final_metrics']['final_train_loss']:.4f}\n")
            if summary['final_metrics']['final_val_loss']:
                f.write(f"  Val Loss: {summary['final_metrics']['final_val_loss']:.4f}\n")
                f.write(f"  PSNR: {summary['final_metrics']['final_psnr']:.2f} dB\n")
                f.write(f"  SSIM: {summary['final_metrics']['final_ssim']:.4f}\n")
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
    
    def train(self):
        """Main training loop"""
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60 + "\n")
        
        # Record training start time
        self.training_start_time = time.time()
        
        # Initialize log file
        self._init_log_file()
        
        total_epochs = self.num_epochs
        
        for epoch in range(self.start_epoch, total_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_loss, train_losses = self.train_epoch(epoch)
            
            # Extract timing info from train_losses (set by train_epoch)
            epoch_time = train_losses.get('epoch_time', time.time() - epoch_start_time)
            batches_per_sec = train_losses.get('batches_per_sec', len(self.train_loader) / epoch_time if epoch_time > 0 else 0)
            iterations_per_sec = train_losses.get('iterations_per_sec', batches_per_sec)
            avg_batch_time = train_losses.get('avg_batch_time', 0)
            
            # Initialize epoch metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'epoch_time': epoch_time,
                'batches_per_sec': batches_per_sec,
                'iterations_per_sec': iterations_per_sec,
                'avg_batch_time': avg_batch_time,
                'learning_rate': self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr']
            }
            
            # Validation
            val_freq = self.config.get('validation', {}).get('val_every') or self.config.get('training', {}).get('val_frequency', 5)
            if (epoch + 1) % val_freq == 0:
                val_loss, val_losses, metrics = self.validate(epoch)
                
                epoch_metrics.update({
                    'val_loss': val_loss,
                    'psnr': metrics['psnr'],
                    'ssim': metrics['ssim'],
                    'mae': metrics['mae']
                })
                
                # Print epoch summary
                epoch_time_min = epoch_time / 60
                print(f"\n{'='*60}")
                print(f"Epoch {epoch+1}/{total_epochs} Summary:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Metrics: {metrics['psnr']:.2f} dB PSNR, {metrics['ssim']:.4f} SSIM")
                print(f"  Time: {epoch_time:.1f}s ({epoch_time_min:.2f} min)")
                print(f"  Speed: {batches_per_sec:.2f} batches/sec ({iterations_per_sec:.2f} iter/sec)")
                print(f"  Avg Batch Time: {avg_batch_time:.3f}s")
                print(f"{'='*60}")
                
                # Save checkpoint
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    epoch_metrics['is_best'] = True
                
                self.save_checkpoint(epoch, val_loss, is_best)
            else:
                # No validation this epoch - still save checkpoint if needed
                save_freq = self.config['checkpoint'].get('save_every') or self.config.get('training', {}).get('save_frequency', 1)
                if (epoch + 1) % save_freq == 0:
                    # Save checkpoint even without validation (use train_loss as placeholder)
                    self.save_checkpoint(epoch, train_loss, is_best=False)
                else:
                    # Always save last_model.pth every epoch for resume capability
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                        'val_loss': train_loss,  # Use train_loss as placeholder
                        'train_losses': self.train_losses,
                        'val_losses': self.val_losses,
                        'config': self.config
                    }
                    last_save_path = self.checkpoint_dir / 'last_model.pth'
                    torch.save(checkpoint, last_save_path)
                
                epoch_time_min = epoch_time / 60
                print(f"\nEpoch {epoch+1}/{total_epochs}: Train Loss: {train_loss:.4f}")
                print(f"  Time: {epoch_time:.1f}s ({epoch_time_min:.2f} min) | Speed: {iterations_per_sec:.2f} iter/sec | Batch: {avg_batch_time:.3f}s")
            
            # Store metrics
            self.epoch_metrics.append(epoch_metrics)
            
            # Log to file
            self._log_epoch(epoch_metrics)
            
            # CRITICAL: Always save last_model.pth at end of every epoch for resume capability
            # This ensures we can resume even if training is interrupted
            try:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'val_loss': epoch_metrics.get('val_loss', epoch_metrics.get('train_loss', float('inf'))),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'config': self.config
                }
                last_save_path = self.checkpoint_dir / 'last_model.pth'
                torch.save(checkpoint, last_save_path)
            except Exception as e:
                print(f"⚠️  Warning: Failed to save last_model.pth: {e}")
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                if (epoch + 1) % 10 == 0:
                    print(f"  Learning rate: {current_lr:.6f}")
        
        # Save final summary
        self._save_training_summary()
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Training log saved to: {self.log_file}")
        print(f"Metrics saved to: {self.metrics_file}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Train Stage 1: Smartphone to DSLR')
    parser.add_argument('--config', type=str, default='configs/stage1_config.yaml',
                      help='Path to configuration file')
    args = parser.parse_args()
    
    # Create and run trainer
    trainer = Trainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()

