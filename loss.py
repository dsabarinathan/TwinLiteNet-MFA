"""
TwinLiteNet-MFA Loss Functions

This file contains the loss functions used for training:
- FocalLoss: For handling class imbalance
- TverskyLoss: Generalization of Dice loss
- TotalLoss: Combined loss for dual-task learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    Args:
        alpha: Weighting factor in range (0,1) to balance positive/negative examples
        gamma: Exponent of the modulating factor (1 - pt)^gamma
        reduction: Specifies the reduction to apply to the output
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (logits) of shape (N, C, H, W)
            targets: Ground truth of shape (N, C, H, W)
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - Generalization of Dice loss
    
    Paper: "Tversky loss function for image segmentation using 3D fully 
           convolutional deep networks" (Salehi et al., 2017)
    
    Args:
        alpha: Weight of false positives
        beta: Weight of false negatives
        smooth: Smoothing constant to avoid division by zero
        
    Note:
        - When alpha = beta = 0.5, Tversky loss becomes Dice loss
        - When alpha = beta = 1.0, Tversky loss becomes Tanimoto loss
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (logits) of shape (N, C, H, W)
            targets: Ground truth of shape (N, C, H, W)
        """
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    
    Args:
        smooth: Smoothing constant to avoid division by zero
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (logits) of shape (N, C, H, W)
            targets: Ground truth of shape (N, C, H, W)
        """
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class TotalLoss(nn.Module):
    """
    Combined loss for dual-task segmentation
    
    Combines Focal Loss and Tversky Loss for both:
    - Drivable area segmentation
    - Lane line detection
    
    Total Loss = Focal Loss (DA + LL) + Tversky Loss (DA + LL)
    """
    def __init__(self, 
                 focal_alpha=0.25, 
                 focal_gamma=2.0,
                 tversky_alpha=0.3,
                 tversky_beta=0.7,
                 focal_weight=1.0,
                 tversky_weight=1.0):
        super(TotalLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight

    def forward(self, outputs, targets):
        """
        Args:
            outputs: Tuple of (drivable_area_output, lane_line_output)
            targets: Tuple of (drivable_area_target, lane_line_target)
            
        Returns:
            focal_loss: Combined focal loss
            tversky_loss: Combined tversky loss
            total_loss: Weighted sum of focal and tversky losses
        """
        da_output, ll_output = outputs
        da_target, ll_target = targets
        
        # Move targets to CUDA if available
        if torch.cuda.is_available():
            da_target = da_target.cuda().float()
            ll_target = ll_target.cuda().float()
        
        # Compute losses for drivable area
        focal_loss_da = self.focal_loss(da_output, da_target)
        tversky_loss_da = self.tversky_loss(da_output, da_target)
        
        # Compute losses for lane lines
        focal_loss_ll = self.focal_loss(ll_output, ll_target)
        tversky_loss_ll = self.tversky_loss(ll_output, ll_target)
        
        # Combine losses
        focal_loss = focal_loss_da + focal_loss_ll
        tversky_loss = tversky_loss_da + tversky_loss_ll
        total_loss = self.focal_weight * focal_loss + self.tversky_weight * tversky_loss
        
        return focal_loss, tversky_loss, total_loss


class WeightedTotalLoss(nn.Module):
    """
    Weighted combined loss with separate weights for each task
    
    Args:
        da_weight: Weight for drivable area task
        ll_weight: Weight for lane line task
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
        tversky_alpha: Alpha parameter for tversky loss
        tversky_beta: Beta parameter for tversky loss
    """
    def __init__(self,
                 da_weight=1.0,
                 ll_weight=1.0,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 tversky_alpha=0.3,
                 tversky_beta=0.7):
        super(WeightedTotalLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.da_weight = da_weight
        self.ll_weight = ll_weight

    def forward(self, outputs, targets):
        """
        Args:
            outputs: Tuple of (drivable_area_output, lane_line_output)
            targets: Tuple of (drivable_area_target, lane_line_target)
            
        Returns:
            focal_loss: Weighted focal loss
            tversky_loss: Weighted tversky loss
            total_loss: Sum of weighted focal and tversky losses
        """
        da_output, ll_output = outputs
        da_target, ll_target = targets
        
        if torch.cuda.is_available():
            da_target = da_target.cuda().float()
            ll_target = ll_target.cuda().float()
        
        # Compute losses
        focal_loss_da = self.focal_loss(da_output, da_target)
        tversky_loss_da = self.tversky_loss(da_output, da_target)
        focal_loss_ll = self.focal_loss(ll_output, ll_target)
        tversky_loss_ll = self.tversky_loss(ll_output, ll_target)
        
        # Apply task weights
        focal_loss = self.da_weight * focal_loss_da + self.ll_weight * focal_loss_ll
        tversky_loss = self.da_weight * tversky_loss_da + self.ll_weight * tversky_loss_ll
        total_loss = focal_loss + tversky_loss
        
        return focal_loss, tversky_loss, total_loss


if __name__ == '__main__':
    # Test loss functions
    print("Testing loss functions...")
    
    # Create dummy data
    batch_size = 2
    height, width = 384, 640
    
    # Predictions (logits)
    pred_da = torch.randn(batch_size, 2, height, width)
    pred_ll = torch.randn(batch_size, 2, height, width)
    
    # Ground truth (binary masks)
    target_da = torch.randint(0, 2, (batch_size, 2, height, width)).float()
    target_ll = torch.randint(0, 2, (batch_size, 2, height, width)).float()
    
    # Test TotalLoss
    criterion = TotalLoss()
    focal, tversky, total = criterion((pred_da, pred_ll), (target_da, target_ll))
    
    print(f"\nTotalLoss:")
    print(f"  Focal Loss: {focal.item():.4f}")
    print(f"  Tversky Loss: {tversky.item():.4f}")
    print(f"  Total Loss: {total.item():.4f}")
    
    # Test WeightedTotalLoss
    criterion_weighted = WeightedTotalLoss(da_weight=2.0, ll_weight=1.0)
    focal_w, tversky_w, total_w = criterion_weighted((pred_da, pred_ll), (target_da, target_ll))
    
    print(f"\nWeightedTotalLoss (DA:2.0, LL:1.0):")
    print(f"  Focal Loss: {focal_w.item():.4f}")
    print(f"  Tversky Loss: {tversky_w.item():.4f}")
    print(f"  Total Loss: {total_w.item():.4f}")
    
    print("\n✓ All loss functions working correctly!")
