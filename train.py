"""
TwinLiteNet-MFA Training Script

Main training script for dual-task segmentation on BDD100K dataset.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --model small --epochs 100
"""

import os
import sys
import yaml
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import TwinLiteNetPlus
from loss import TotalLoss
from dataset import BDD100KDataset, create_dataloader
from utils import (
    SegmentationMetric,
    ModelEMA,
    poly_lr_scheduler,
    count_parameters,
    format_number_M,
    print_model_summary,
    save_checkpoint,
    AverageMeter
)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, scaler, epoch, config, ema=None):
    """
    Train for one epoch
    
    Args:
        model: TwinLiteNetPlus model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scaler: GradScaler for mixed precision
        epoch: Current epoch number
        config: Configuration dictionary
        ema: Exponential Moving Average (optional)
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    
    loss_meter = AverageMeter()
    focal_meter = AverageMeter()
    tversky_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    
    for batch_idx, (_, images, targets) in enumerate(pbar):
        # Move to GPU
        if torch.cuda.is_available():
            images = images.cuda().float() / 255.0
        
        optimizer.zero_grad()
        
        # Mixed precision training
        use_amp = torch.cuda.is_available()
        with torch.amp.autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp):
            outputs = model(images)
            focal_loss, tversky_loss, total_loss = criterion(outputs, targets)
        
        # Backward pass
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update EMA
        if ema is not None:
            ema.update(model)
        
        # Update metrics
        batch_size = images.size(0)
        loss_meter.update(total_loss.item(), batch_size)
        focal_meter.update(focal_loss.item(), batch_size)
        tversky_meter.update(tversky_loss.item(), batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'focal': f'{focal_meter.avg:.4f}',
            'tversky': f'{tversky_meter.avg:.4f}'
        })
    
    return loss_meter.avg, focal_meter.avg, tversky_meter.avg, ema


@torch.no_grad()
def validate(model, val_loader, config):
    """
    Validate the model
    
    Args:
        model: TwinLiteNetPlus model
        val_loader: Validation data loader
        config: Configuration dictionary
        
    Returns:
        Tuple of (drivable_area_metrics, lane_line_metrics)
    """
    model.eval()
    
    DA = SegmentationMetric(2)
    LL = SegmentationMetric(2)
    
    pbar = tqdm(val_loader, desc='Validation')
    
    for _, images, targets in pbar:
        # Move to GPU
        if torch.cuda.is_available():
            images = images.cuda().float() / 255.0
        
        # Forward pass
        outputs = model(images)
        
        # Drivable area evaluation
        out_da = outputs[0]
        target_da = targets[0]
        _, da_predict = torch.max(out_da, 1)
        da_predict = da_predict[:, 12:-12]  # Crop to remove padding
        _, da_gt = torch.max(target_da, 1)
        da_gt = da_gt[:, 12:-12]
        
        DA.addBatch(da_predict.cpu().numpy().flatten(), da_gt.cpu().numpy().flatten())
        
        # Lane line evaluation
        out_ll = outputs[1]
        target_ll = targets[1]
        _, ll_predict = torch.max(out_ll, 1)
        ll_predict = ll_predict[:, 12:-12]
        _, ll_gt = torch.max(target_ll, 1)
        ll_gt = ll_gt[:, 12:-12]
        
        LL.addBatch(ll_predict.cpu().numpy().flatten(), ll_gt.cpu().numpy().flatten())
    
    # Compute metrics
    da_acc = DA.pixelAccuracy()
    da_iou = DA.IntersectionOverUnion()
    da_miou = DA.meanIntersectionOverUnion()
    
    ll_acc = LL.lineAccuracy()
    ll_iou = LL.IntersectionOverUnion()
    ll_miou = LL.meanIntersectionOverUnion()
    
    return (da_acc, da_iou, da_miou), (ll_acc, ll_iou, ll_miou)


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train(config):
    """
    Main training function
    
    Args:
        config: Configuration dictionary
    """
    # Setup
    print("=" * 80)
    print("TwinLiteNet-MFA Training")
    print("=" * 80)
    
    # Set random seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # CUDA setup
    cuda_available = torch.cuda.is_available()
    device = 'cuda' if cuda_available else 'cpu'
    print(f"Using device: {device}")
    
    if cuda_available:
        cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create save directory
    save_dir = config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {save_dir}")
    
    # Model
    model_config = config['model']['architecture']
    print(f"\nInitializing model: {model_config}")
    model = TwinLiteNetPlus(model_config)
    
    if cuda_available:
        model = model.cuda()
    
    # Print model summary
    print_model_summary(model, device=device)
    
    # Data loaders
    print("\nLoading datasets...")
    train_loader = create_dataloader(config, valid=False)
    val_loader = create_dataloader(config, valid=True)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Loss and optimizer
    criterion = TotalLoss()
    
    lr = config['training']['learning_rate']
    weight_decay = config['training'].get('weight_decay', 0.0001)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay
    )
    
    # EMA
    use_ema = config['training'].get('use_ema', False)
    ema = ModelEMA(model) if use_ema else None
    if use_ema:
        print("Using Exponential Moving Average (EMA)")
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_miou = 0.0
    
    resume_path = config['training'].get('resume', '')
    if resume_path and os.path.isfile(resume_path):
        print(f"\nResuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
            if 'best_miou' in checkpoint:
                best_miou = checkpoint['best_miou']
            if use_ema and 'ema_state_dict' in checkpoint:
                ema.ema.load_state_dict(checkpoint['ema_state_dict'])
            print(f"Resumed from epoch {start_epoch}, best mIoU: {best_miou:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model weights only")
        
        if cuda_available:
            model = model.cuda()
    
    # Training loop
    max_epochs = config['training']['max_epochs']
    
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80 + "\n")
    
    for epoch in range(start_epoch, max_epochs):
        # Update learning rate
        current_lr = poly_lr_scheduler(optimizer, lr, epoch, max_epochs)
        print(f"\nEpoch [{epoch+1}/{max_epochs}] - LR: {current_lr:.6f}")
        
        # Train
        train_loss, focal_loss, tversky_loss, ema = train_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch, config, ema
        )
        
        print(f"Training - Loss: {train_loss:.4f}, Focal: {focal_loss:.4f}, Tversky: {tversky_loss:.4f}")
        
        # Validate
        val_model = ema.ema if use_ema else model
        da_metrics, ll_metrics = validate(val_model, val_loader, config)
        
        print(f"\nValidation Results:")
        print(f"  Drivable Area - Acc: {da_metrics[0]:.4f}, IoU: {da_metrics[1]:.4f}, mIoU: {da_metrics[2]:.4f}")
        print(f"  Lane Line     - Acc: {ll_metrics[0]:.4f}, IoU: {ll_metrics[1]:.4f}, mIoU: {ll_metrics[2]:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': current_lr,
            'da_acc': da_metrics[0],
            'da_iou': da_metrics[1],
            'da_miou': da_metrics[2],
            'll_acc': ll_metrics[0],
            'll_iou': ll_metrics[1],
            'll_miou': ll_metrics[2],
            'best_miou': best_miou
        }
        
        if use_ema:
            checkpoint['ema_state_dict'] = ema.ema.state_dict()
            checkpoint['updates'] = ema.updates
        
        # Save latest checkpoint
        save_checkpoint(checkpoint, os.path.join(save_dir, 'checkpoint_latest.pth'))
        
        # Save best model
        current_miou = (da_metrics[2] + ll_metrics[2]) / 2
        if current_miou > best_miou:
            best_miou = current_miou
            checkpoint['best_miou'] = best_miou
            save_checkpoint(checkpoint, os.path.join(save_dir, 'checkpoint_best.pth'))
            print(f"  ✓ New best model saved! (mIoU: {best_miou:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % config['training'].get('save_interval', 10) == 0:
            save_checkpoint(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"Best mIoU: {best_miou:.4f}")
    print("=" * 80)


# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_default_config():
    """Create default configuration"""
    import platform
    default_workers = 2 if platform.system() == 'Windows' else 4
    
    return {
        'model': {
            'architecture': 'nano'  # nano, small, medium, large
        },
        'dataset': {
            'root': './bdd100k',
        },
        'augmentation': {
            'degrees': 10,
            'translate': 0.1,
            'scale': 0.1,
            'shear': 10,
            'hgain': 0.015,
            'sgain': 0.7,
            'vgain': 0.4,
            'prob_perspective': 0.5,
            'prob_flip': 0.5,
            'prob_hsv': 0.5
        },
        'training': {
            'max_epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.0005,
            'weight_decay': 0.0001,
            'num_workers': default_workers,
            'use_ema': True,
            'save_dir': './checkpoints',
            'save_interval': 10,
            'resume': ''
        },
        'seed': 42
    }


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TwinLiteNet-MFA Training')
    parser.add_argument('--config', type=str, default='',
                       help='Path to config YAML file')
    parser.add_argument('--model', type=str, default='',
                       choices=['', 'nano', 'small', 'medium', 'large'],
                       help='Model architecture (overrides config)')
    parser.add_argument('--batch-size', type=int, default=0,
                       help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=0,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=0,
                       help='Learning rate (overrides config)')
    parser.add_argument('--data-root', type=str, default='',
                       help='Dataset root directory (overrides config)')
    parser.add_argument('--save-dir', type=str, default='',
                       help='Directory to save checkpoints (overrides config)')
    parser.add_argument('--resume', type=str, default='',
                       help='Resume from checkpoint')
    parser.add_argument('--no-ema', action='store_true',
                       help='Disable EMA')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config and os.path.isfile(args.config):
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
    else:
        print("Using default configuration")
        config = create_default_config()
    
    # Override with command line arguments
    if args.model:
        config['model']['architecture'] = args.model
    if args.batch_size > 0:
        config['training']['batch_size'] = args.batch_size
    if args.epochs > 0:
        config['training']['max_epochs'] = args.epochs
    if args.lr > 0:
        config['training']['learning_rate'] = args.lr
    if args.data_root:
        config['dataset']['root'] = args.data_root
    if args.save_dir:
        config['training']['save_dir'] = args.save_dir
    if args.resume:
        config['training']['resume'] = args.resume
    if args.no_ema:
        config['training']['use_ema'] = False
    
    # Print configuration
    print("\nConfiguration:")
    print("=" * 80)
    print(yaml.dump(config, default_flow_style=False))
    print("=" * 80 + "\n")
    
    # Start training
    train(config)
