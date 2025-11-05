"""
TwinLiteNet-MFA Utilities

This file contains:
- SegmentationMetric: Evaluation metrics for segmentation
- ModelEMA: Exponential Moving Average for model parameters
- Learning rate schedulers
- Model information utilities
"""

import math
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# EVALUATION METRICS
# ============================================================================

class SegmentationMetric:
    """
    Metrics for semantic segmentation evaluation
    
    Computes:
    - Pixel Accuracy
    - IoU (Intersection over Union)
    - mIoU (mean IoU)
    - Line Accuracy (for lane detection)
    """
    def __init__(self, numClass):
        """
        Args:
            numClass: Number of classes (typically 2 for binary segmentation)
        """
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        """
        Pixel Accuracy = (TP + TN) / (TP + TN + FP + FN)
        """
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def IntersectionOverUnion(self):
        """
        IoU for the positive class (class 1)
        IoU = TP / (TP + FP + FN)
        """
        intersection = np.diag(self.confusionMatrix)
        union = (np.sum(self.confusionMatrix, axis=1) + 
                np.sum(self.confusionMatrix, axis=0) - 
                np.diag(self.confusionMatrix))
        IoU = intersection / union
        IoU[np.isnan(IoU)] = 0
        return IoU[1]  # Return IoU for class 1

    def meanIntersectionOverUnion(self):
        """
        Mean IoU across all classes
        mIoU = mean(IoU for each class)
        """
        intersection = np.diag(self.confusionMatrix)
        union = (np.sum(self.confusionMatrix, axis=1) + 
                np.sum(self.confusionMatrix, axis=0) - 
                np.diag(self.confusionMatrix))
        IoU = intersection / union
        IoU[np.isnan(IoU)] = 0
        mIoU = np.nanmean(IoU)
        return mIoU

    def lineAccuracy(self):
        """
        Line Accuracy for lane detection
        Average of sensitivity and specificity
        
        Sensitivity = TP / (TP + FN)
        Specificity = TN / (TN + FP)
        """
        tp = self.confusionMatrix[1, 1]
        fp = self.confusionMatrix[0, 1]
        fn = self.confusionMatrix[1, 0]
        tn = self.confusionMatrix[0, 0]
        
        sensitivity = tp / (tp + fn + 1e-12)
        specificity = tn / (tn + fp + 1e-12)
        
        return (sensitivity + specificity) / 2

    def genConfusionMatrix(self, imgPredict, imgLabel):
        """
        Generate confusion matrix from predictions and labels
        """
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def addBatch(self, imgPredict, imgLabel):
        """
        Add a batch of predictions to the confusion matrix
        
        Args:
            imgPredict: Predicted labels
            imgLabel: Ground truth labels
        """
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        """Reset the confusion matrix"""
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


# ============================================================================
# EXPONENTIAL MOVING AVERAGE
# ============================================================================

class ModelEMA:
    """
    Exponential Moving Average for model parameters
    
    Maintains shadow weights that are updated as an exponential moving average
    of the model parameters during training. Often improves generalization.
    
    Args:
        model: PyTorch model
        decay: Decay rate (default: 0.9999)
    """
    def __init__(self, model, decay=0.9999):
        self.ema = deepcopy(model).eval()
        self.updates = 0
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        
        # Disable gradient computation for EMA model
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """
        Update EMA weights
        
        Args:
            model: Current model with updated weights
        """
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()


# ============================================================================
# LEARNING RATE SCHEDULERS
# ============================================================================

def poly_lr_scheduler(optimizer, init_lr, curr_epoch, max_epochs, power=1.5):
    """
    Polynomial learning rate decay
    
    lr = init_lr * (1 - curr_epoch / max_epochs) ^ power
    
    Args:
        optimizer: PyTorch optimizer
        init_lr: Initial learning rate
        curr_epoch: Current epoch
        max_epochs: Maximum number of epochs
        power: Polynomial power (default: 1.5)
        
    Returns:
        Current learning rate
    """
    lr = init_lr * (1 - curr_epoch / max_epochs) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def cosine_lr_scheduler(optimizer, init_lr, curr_epoch, max_epochs, min_lr=0):
    """
    Cosine annealing learning rate decay
    
    Args:
        optimizer: PyTorch optimizer
        init_lr: Initial learning rate
        curr_epoch: Current epoch
        max_epochs: Maximum number of epochs
        min_lr: Minimum learning rate (default: 0)
        
    Returns:
        Current learning rate
    """
    lr = min_lr + (init_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * curr_epoch / max_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def step_lr_scheduler(optimizer, init_lr, curr_epoch, step_size=30, gamma=0.1):
    """
    Step learning rate decay
    
    Args:
        optimizer: PyTorch optimizer
        init_lr: Initial learning rate
        curr_epoch: Current epoch
        step_size: Period of learning rate decay
        gamma: Multiplicative factor of learning rate decay
        
    Returns:
        Current learning rate
    """
    lr = init_lr * (gamma ** (curr_epoch // step_size))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# ============================================================================
# MODEL UTILITIES
# ============================================================================

def count_parameters(model):
    """
    Count trainable parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number_M(num):
    """
    Format number in millions (M)
    
    Args:
        num: Number to format
        
    Returns:
        Formatted string (e.g., "2.45M")
    """
    return f"{num / 1e6:.2f}M"


def format_number_K(num):
    """
    Format number in thousands (K)
    
    Args:
        num: Number to format
        
    Returns:
        Formatted string (e.g., "245.67K")
    """
    return f"{num / 1e3:.2f}K"


def get_model_size_mb(model):
    """
    Get model size in MB
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def calculate_flops(model, input_size=(1, 3, 384, 640), device='cuda'):
    """
    Calculate FLOPs using thop library
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        device: Device to run on
        
    Returns:
        (flops, params) as formatted strings, or (None, None) if thop unavailable
    """
    try:
        from thop import profile, clever_format
        
        model_copy = deepcopy(model)
        model_copy.eval()
        
        if device == 'cuda' and torch.cuda.is_available():
            model_copy = model_copy.cuda()
            input_tensor = torch.randn(input_size).cuda()
        else:
            input_tensor = torch.randn(input_size)
        
        flops, params = profile(model_copy, inputs=(input_tensor,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        
        del model_copy
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return flops, params
    except ImportError:
        print("Warning: 'thop' not installed. Install with: pip install thop")
        return None, None
    except Exception as e:
        print(f"Warning: FLOPs calculation failed: {e}")
        return None, None


def print_model_summary(model, input_size=(1, 3, 384, 640), device='cuda'):
    """
    Print comprehensive model summary
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        device: Device to run on
    """
    print("\n" + "="*80)
    print("MODEL SUMMARY")
    print("="*80)
    
    # Parameter count
    num_params = count_parameters(model)
    print(f"Total Parameters: {num_params:,} ({format_number_M(num_params)})")
    
    # Model size
    size_mb = get_model_size_mb(model)
    print(f"Model Size: {size_mb:.2f} MB")
    
    # FLOPs
    flops, params_thop = calculate_flops(model, input_size, device)
    if flops is not None:
        print(f"FLOPs: {flops}")
        print(f"Parameters (thop): {params_thop}")
    
    print("="*80 + "\n")


# ============================================================================
# CHECKPOINT UTILITIES
# ============================================================================

def save_checkpoint(state, filename='checkpoint.pth'):
    """
    Save checkpoint
    
    Args:
        state: Dictionary containing model state, optimizer state, etc.
        filename: Path to save checkpoint
    """
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(model, checkpoint_path, optimizer=None, device='cuda'):
    """
    Load checkpoint
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        optimizer: PyTorch optimizer (optional)
        device: Device to load on
        
    Returns:
        Loaded state dictionary with epoch, metrics, etc.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            if optimizer is not None and 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint
        else:
            model.load_state_dict(checkpoint)
            print(f"Model weights loaded: {checkpoint_path}")
            return {}
    else:
        raise ValueError(f"Invalid checkpoint format: {type(checkpoint)}")


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    print("Testing utilities...")
    
    # Test metrics
    print("\n1. Testing SegmentationMetric:")
    metric = SegmentationMetric(2)
    
    # Dummy predictions and labels
    pred = np.array([0, 0, 1, 1, 1, 0])
    label = np.array([0, 1, 1, 1, 0, 0])
    
    metric.addBatch(pred, label)
    print(f"  Pixel Accuracy: {metric.pixelAccuracy():.4f}")
    print(f"  IoU: {metric.IntersectionOverUnion():.4f}")
    print(f"  mIoU: {metric.meanIntersectionOverUnion():.4f}")
    print(f"  Line Accuracy: {metric.lineAccuracy():.4f}")
    
    # Test parameter counting
    print("\n2. Testing parameter counting:")
    from model import TwinLiteNetPlus
    
    for config in ['nano', 'small', 'medium', 'large']:
        model = TwinLiteNetPlus(config)
        num_params = count_parameters(model)
        print(f"  {config}: {format_number_M(num_params)}")
    
    # Test learning rate schedulers
    print("\n3. Testing learning rate schedulers:")
    model = TwinLiteNetPlus('nano')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("  Poly LR Schedule:")
    for epoch in range(0, 100, 20):
        lr = poly_lr_scheduler(optimizer, 0.001, epoch, 100)
        print(f"    Epoch {epoch}: LR = {lr:.6f}")
    
    print("\n✓ All utilities working correctly!")
