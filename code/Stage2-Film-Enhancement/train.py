"""
Training script for Stage 2: DSLR to Film Enhancement
"""

import os
import sys
import time
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.optim as optim
from torchvision.utils import save_image

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from data.stage2_dataset import FilmEnhancementDataset, get_stage2_dataloader
from models.stage2_film import build_stage2_model
from losses.film_loss import FilmEnhancementLoss
from utils.metrics import MetricsCalculator


class Stage2Trainer:
    """
    Trainer for Stage 2: Film Enhancement
    """
    
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print("=" * 60)
        print("Stage 2: DSLR to Film Enhancement - Training")
        print("=" * 60)
        
        # Setup device
        self.device = self._setup_device()
        
        # Create directories
        self.checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        # Resume if specified
        if self.config['checkpoint']['resume']:
            self._resume_from_checkpoint()
        
        print("\n" + "=" * 60)
        print("Setup complete! Ready to train Stage 2.")
        print("=" * 60 + "\n")
    
    def _setup_device(self):
        """Setup computation device"""
        device_type = self.config['device']['type']
        
        if device_type == 'mps' and torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"✓ Using device: MPS (Apple Silicon)")
        elif device_type == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✓ Using device: CUDA")
        else:
            device = torch.device('cpu')
            print(f"✓ Using device: CPU")
        
        return device
    
    def _build_model(self):
        """Build Stage 2 model"""
        model_config = self.config['model']
        model = build_stage2_model(
            model_type=model_config['type'],
            device=self.device
        )
        return model
    
    def _build_loss(self):
        """Build film enhancement loss"""
        loss_weights = self.config['training']['loss_weights']
        criterion = FilmEnhancementLoss(
            w_l1=loss_weights['l1'],
            w_style=loss_weights['style'],
            w_color=loss_weights['color']
        )
        return criterion.to(self.device)
    
    def _build_optimizer(self):
        """Build optimizer"""
        opt_config = self.config['training']
        
        if opt_config['optimizer'] == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                betas=opt_config['betas'],
                weight_decay=opt_config['weight_decay']
            )
        elif opt_config['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                betas=opt_config['betas'],
                weight_decay=opt_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['optimizer']}")
        
        print(f"✓ Optimizer: {opt_config['optimizer'].upper()}, LR: {opt_config['lr']}")
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler"""
        sched_type = self.config['training']['scheduler']
        
        if sched_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=self.config['training']['lr_min']
            )
            print(f"✓ Scheduler: Cosine Annealing")
        elif sched_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=20,
                gamma=0.5
            )
            print(f"✓ Scheduler: Step LR")
        else:
            scheduler = None
            print(f"✓ Scheduler: None")
        
        return scheduler
    
    def _load_datasets(self):
        """Load datasets"""
        dataset_config = self.config['dataset']
        training_config = self.config['training']
        film_config = self.config.get('film', {})
        
        # Training dataset
        train_dataset = FilmEnhancementDataset(
            dslr_dir=dataset_config['dslr_dir'],
            split='train',
            film_style=dataset_config['film_style'],
            img_size=dataset_config['img_size'],
            augment=dataset_config['augment'],
            film_params=film_config
        )
        
        train_loader = get_stage2_dataloader(
            train_dataset,
            batch_size=training_config['batch_size'],
            shuffle=True,
            num_workers=training_config['num_workers']
        )
        
        # Validation dataset
        val_dataset = FilmEnhancementDataset(
            dslr_dir=dataset_config['test_dslr_dir'],
            split='test',
            film_style=dataset_config['film_style'],
            img_size=dataset_config['img_size'],
            augment=False,
            film_params=film_config
        )
        
        val_loader = get_stage2_dataloader(
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
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['epochs']}")
        
        for batch_idx, (dslr_imgs, film_imgs) in enumerate(pbar):
            dslr_imgs = dslr_imgs.to(self.device)
            film_imgs = film_imgs.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_imgs = self.model(dslr_imgs)
            
            # Compute loss
            loss, loss_dict = self.criterion(pred_imgs, film_imgs)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            epoch_loss += loss.item()
            for k, v in loss_dict.items():
                epoch_losses_dict[k] = epoch_losses_dict.get(k, 0) + v
            
            # Update progress bar
            if batch_idx % self.config['logging']['log_every'] == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Average losses
        avg_loss = epoch_loss / len(self.train_loader)
        avg_losses_dict = {k: v / len(self.train_loader) for k, v in epoch_losses_dict.items()}
        
        return avg_loss, avg_losses_dict
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        val_losses_dict = {}
        metrics_calc = MetricsCalculator()
        
        with torch.no_grad():
            for dslr_imgs, film_imgs in tqdm(self.val_loader, desc="Validating"):
                dslr_imgs = dslr_imgs.to(self.device)
                film_imgs = film_imgs.to(self.device)
                
                # Forward pass
                pred_imgs = self.model(dslr_imgs)
                
                # Compute loss
                loss, loss_dict = self.criterion(pred_imgs, film_imgs)
                
                val_loss += loss.item()
                for k, v in loss_dict.items():
                    val_losses_dict[k] = val_losses_dict.get(k, 0) + v
                
                # Update metrics
                metrics_calc.update(pred_imgs, film_imgs)
        
        # Average
        avg_val_loss = val_loss / len(self.val_loader)
        avg_val_losses_dict = {k: v / len(self.val_loader) for k, v in val_losses_dict.items()}
        metrics = metrics_calc.get_metrics()
        
        # Save sample images
        if self.config['validation']['save_images']:
            self._save_validation_images(epoch, dslr_imgs, pred_imgs, film_imgs)
        
        return avg_val_loss, avg_val_losses_dict, metrics
    
    def _save_validation_images(self, epoch, dslr, pred, film):
        """Save sample validation images"""
        n_save = min(self.config['validation']['num_save_images'], dslr.size(0))
        
        comparison = torch.cat([
            dslr[:n_save],
            pred[:n_save],
            film[:n_save]
        ], dim=0)
        
        save_path = self.results_dir / f'epoch_{epoch+1:03d}.png'
        save_image(comparison, save_path, nrow=n_save, normalize=False)
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        if (epoch + 1) % self.config['checkpoint']['save_every'] == 0:
            save_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1:03d}.pth'
            torch.save(checkpoint, save_path)
            print(f"Saved checkpoint: {save_path}")
        
        # Save best checkpoint
        if is_best and self.config['checkpoint']['save_best']:
            save_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, save_path)
            print(f"✓ Saved best model: {save_path}")
    
    def _resume_from_checkpoint(self):
        """Resume from checkpoint"""
        checkpoint_path = self.config['checkpoint']['resume']
        
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['val_loss']
        
        print(f"✓ Resumed from checkpoint: {checkpoint_path}")
        print(f"  Starting from epoch {self.start_epoch}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "=" * 60)
        print("Starting Stage 2 training...")
        print("=" * 60 + "\n")
        
        total_epochs = self.config['training']['epochs']
        
        for epoch in range(self.start_epoch, total_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_loss, train_losses = self.train_epoch(epoch)
            
            # Validation
            if (epoch + 1) % self.config['validation']['val_every'] == 0:
                val_loss, val_losses, metrics = self.validate(epoch)
                
                # Print summary
                print(f"\nEpoch {epoch+1}/{total_epochs} Summary:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Metrics: {metrics['psnr']:.2f} dB PSNR, {metrics['ssim']:.4f} SSIM")
                print(f"  Time: {time.time() - epoch_start_time:.1f}s")
                
                # Save checkpoint
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                self.save_checkpoint(epoch, val_loss, is_best)
            
            # Update LR
            if self.scheduler:
                self.scheduler.step()
        
        print("\n" + "=" * 60)
        print("Stage 2 training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Train Stage 2: Film Enhancement')
    parser.add_argument('--config', type=str, default='configs/stage2_config.yaml',
                      help='Path to configuration file')
    args = parser.parse_args()
    
    # Create and run trainer
    trainer = Stage2Trainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()

