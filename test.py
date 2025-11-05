"""
TwinLiteNet-MFA Testing Script

Evaluate trained models on BDD100K validation dataset.

Usage:
    # Test with checkpoint
    python test.py --checkpoint checkpoints/checkpoint_best.pth --model nano
    
    # Test with custom dataset path
    python test.py --checkpoint model.pth --model small --data-root /path/to/bdd100k
    
    # Test with specific batch size
    python test.py --checkpoint model.pth --model nano --batch-size 8
"""

import os
import sys
import argparse
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import TwinLiteNetPlus
from dataset import BDD100KDataset
from utils import SegmentationMetric, count_parameters, format_number_M


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

@torch.no_grad()
def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate model on test dataset
    
    Args:
        model: TwinLiteNetPlus model
        test_loader: Test data loader
        device: Device to run on
        
    Returns:
        Dictionary containing all metrics
    """
    model.eval()
    
    # Initialize metrics
    DA = SegmentationMetric(2)
    LL = SegmentationMetric(2)
    
    # Timing
    inference_times = []
    
    print("\nRunning evaluation...")
    pbar = tqdm(test_loader, desc='Testing')
    
    for batch_idx, (image_names, images, targets) in enumerate(pbar):
        # Move to device
        if device == 'cuda' and torch.cuda.is_available():
            images = images.cuda().float() / 255.0
        else:
            images = images.float() / 255.0
        
        # Measure inference time
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        
        # Forward pass
        outputs = model(images)
        
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # Drivable area evaluation
        out_da = outputs[0]
        target_da = targets[0]
        _, da_predict = torch.max(out_da, 1)
        da_predict = da_predict[:, 12:-12]  # Crop padding
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
        
        # Update progress
        pbar.set_postfix({
            'DA_IoU': f'{DA.IntersectionOverUnion():.4f}',
            'LL_IoU': f'{LL.IntersectionOverUnion():.4f}'
        })
    
    # Calculate metrics
    results = {
        'drivable_area': {
            'pixel_accuracy': DA.pixelAccuracy(),
            'iou': DA.IntersectionOverUnion(),
            'miou': DA.meanIntersectionOverUnion()
        },
        'lane_line': {
            'line_accuracy': LL.lineAccuracy(),
            'iou': LL.IntersectionOverUnion(),
            'miou': LL.meanIntersectionOverUnion()
        },
        'timing': {
            'mean_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'fps': 1.0 / np.mean(inference_times)
        }
    }
    
    return results


# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model(checkpoint_path, model_type='nano', device='cuda'):
    """
    Load model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_type: Model architecture (nano, small, medium, large)
        device: Device to load on
        
    Returns:
        Loaded model
    """
    print(f"\nLoading model...")
    print(f"  Architecture: {model_type}")
    print(f"  Checkpoint: {checkpoint_path}")
    
    # Create model
    model = TwinLiteNetPlus(model_type)
    
    # Load checkpoint
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            epoch = checkpoint.get('epoch', 'Unknown')
            print(f"  Loaded from epoch: {epoch}")
            
            # Print training metrics if available
            if 'da_miou' in checkpoint and 'll_miou' in checkpoint:
                print(f"  Training DA mIoU: {checkpoint['da_miou']:.4f}")
                print(f"  Training LL mIoU: {checkpoint['ll_miou']:.4f}")
        elif 'ema_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['ema_state_dict'])
            print(f"  Loaded EMA weights")
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)
            print(f"  Loaded model weights")
    else:
        raise ValueError(f"Invalid checkpoint format: {type(checkpoint)}")
    
    # Move to device
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        print(f"  Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        print(f"  Device: CPU")
    
    # Print model info
    num_params = count_parameters(model)
    print(f"  Parameters: {num_params:,} ({format_number_M(num_params)})")
    
    return model


# ============================================================================
# PRINT RESULTS
# ============================================================================

def print_results(results, model_type):
    """
    Print evaluation results in a formatted table
    
    Args:
        results: Dictionary containing metrics
        model_type: Model architecture name
    """
    print("\n" + "="*80)
    print(f"EVALUATION RESULTS - {model_type.upper()} MODEL")
    print("="*80)
    
    # Drivable Area Results
    print("\n📍 DRIVABLE AREA SEGMENTATION")
    print("-" * 80)
    da = results['drivable_area']
    print(f"  Pixel Accuracy:  {da['pixel_accuracy']:.4f}")
    print(f"  IoU:             {da['iou']:.4f}")
    print(f"  mIoU:            {da['miou']:.4f}")
    
    # Lane Line Results
    print("\n🛣️  LANE LINE DETECTION")
    print("-" * 80)
    ll = results['lane_line']
    print(f"  Line Accuracy:   {ll['line_accuracy']:.4f}")
    print(f"  IoU:             {ll['iou']:.4f}")
    print(f"  mIoU:            {ll['miou']:.4f}")
    
    # Overall Performance
    print("\n📊 OVERALL PERFORMANCE")
    print("-" * 80)
    overall_miou = (da['miou'] + ll['miou']) / 2
    overall_iou = (da['iou'] + ll['iou']) / 2
    print(f"  Overall mIoU:    {overall_miou:.4f}")
    print(f"  Overall IoU:     {overall_iou:.4f}")
    
    # Timing Results
    print("\n⏱️  INFERENCE SPEED")
    print("-" * 80)
    timing = results['timing']
    print(f"  Mean Time:       {timing['mean_inference_time']*1000:.2f} ms")
    print(f"  Std Time:        {timing['std_inference_time']*1000:.2f} ms")
    print(f"  FPS:             {timing['fps']:.2f}")
    
    print("\n" + "="*80)


def save_results_to_file(results, model_type, output_path='test_results.txt'):
    """
    Save results to a text file
    
    Args:
        results: Dictionary containing metrics
        model_type: Model architecture name
        output_path: Path to save results
    """
    with open(output_path, 'w') as f:
        f.write(f"TwinLiteNet-MFA Evaluation Results\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Model: {model_type}\n\n")
        
        f.write(f"Drivable Area Segmentation:\n")
        da = results['drivable_area']
        f.write(f"  Pixel Accuracy: {da['pixel_accuracy']:.4f}\n")
        f.write(f"  IoU: {da['iou']:.4f}\n")
        f.write(f"  mIoU: {da['miou']:.4f}\n\n")
        
        f.write(f"Lane Line Detection:\n")
        ll = results['lane_line']
        f.write(f"  Line Accuracy: {ll['line_accuracy']:.4f}\n")
        f.write(f"  IoU: {ll['iou']:.4f}\n")
        f.write(f"  mIoU: {ll['miou']:.4f}\n\n")
        
        f.write(f"Overall Performance:\n")
        overall_miou = (da['miou'] + ll['miou']) / 2
        overall_iou = (da['iou'] + ll['iou']) / 2
        f.write(f"  Overall mIoU: {overall_miou:.4f}\n")
        f.write(f"  Overall IoU: {overall_iou:.4f}\n\n")
        
        f.write(f"Inference Speed:\n")
        timing = results['timing']
        f.write(f"  Mean Time: {timing['mean_inference_time']*1000:.2f} ms\n")
        f.write(f"  Std Time: {timing['std_inference_time']*1000:.2f} ms\n")
        f.write(f"  FPS: {timing['fps']:.2f}\n")
    
    print(f"\n✓ Results saved to: {output_path}")


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def test(args):
    """
    Main testing function
    
    Args:
        args: Command line arguments
    """
    print("="*80)
    print("TwinLiteNet-MFA Model Evaluation")
    print("="*80)
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    
    # Load model
    model = load_model(args.checkpoint, args.model, device)
    
    # Create test configuration
    config = {
        'dataset': {
            'root': args.data_root
        },
        'augmentation': {
            'degrees': 0,
            'translate': 0,
            'scale': 0,
            'shear': 0,
            'prob_perspective': 0,
            'prob_flip': 0,
            'prob_hsv': 0
        },
        'training': {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers
        }
    }
    
    # Create test dataset and loader
    print(f"\nLoading test dataset...")
    print(f"  Dataset root: {args.data_root}")
    
    test_dataset = BDD100KDataset(config, valid=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total batches: {len(test_loader)}")
    
    # Evaluate
    results = evaluate_model(model, test_loader, device)
    
    # Print results
    print_results(results, args.model)
    
    # Save results
    if args.save_results:
        save_results_to_file(results, args.model, args.output)
    
    print("\n✓ Evaluation complete!")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TwinLiteNet-MFA Model Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model', type=str, required=True,
                       choices=['nano', 'small', 'medium', 'large'],
                       help='Model architecture')
    
    # Dataset arguments
    parser.add_argument('--data-root', type=str, default='./bdd100k',
                       help='Path to BDD100K dataset root')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for testing')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Device arguments
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU even if GPU is available')
    
    # Output arguments
    parser.add_argument('--save-results', action='store_true',
                       help='Save results to text file')
    parser.add_argument('--output', type=str, default='test_results.txt',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Validate checkpoint path
    if not os.path.isfile(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Validate dataset path
    if not os.path.isdir(args.data_root):
        print(f"Error: Dataset root not found: {args.data_root}")
        sys.exit(1)
    
    # Run test
    test(args)


if __name__ == '__main__':
    main()
