"""
TwinLiteNet-MFA Dataset and Augmentation

This file contains:
- BDD100KDataset: Dataset loader for BDD100K
- Data augmentation functions
- Image preprocessing utilities
"""

import os
import math
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize and pad image while maintaining aspect ratio
    
    Args:
        im: Input image (numpy array)
        new_shape: Target shape (height, width)
        color: Padding color (B, G, R)
        
    Returns:
        Resized and padded image
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    """
    HSV color space augmentation
    
    Args:
        img: Input image in BGR format
        hgain: Hue gain
        sgain: Saturation gain
        vgain: Value gain
    """
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype
    
    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
    
    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)


def random_perspective(combination, degrees=10, translate=0.1, scale=0.1, shear=10):
    """
    Apply random perspective transformation
    
    Args:
        combination: Tuple of (image, drivable_mask, line_mask)
        degrees: Rotation range in degrees
        translate: Translation range as fraction of image size
        scale: Scaling range
        shear: Shear range in degrees
        
    Returns:
        Transformed (image, drivable_mask, line_mask)
    """
    img, drivable, line = combination
    height, width = img.shape[0], img.shape[1]
    
    # Center
    C = np.eye(3)
    C[0, 2] = -width / 2
    C[1, 2] = -height / 2
    
    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    
    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    
    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height
    
    # Combined transformation
    M = T @ S @ R @ C
    
    # Apply transformation
    img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
    drivable = cv2.warpAffine(drivable, M[:2], dsize=(width, height), borderValue=0)
    line = cv2.warpAffine(line, M[:2], dsize=(width, height), borderValue=0)
    
    return (img, drivable, line)


# ============================================================================
# DATASET
# ============================================================================

class BDD100KDataset(Dataset):
    """
    BDD100K Dataset for dual-task segmentation
    
    Dataset structure:
    bdd100k/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── drivable_area_annotations/
    │   ├── train/
    │   └── val/
    └── lane_line_annotations/
        ├── train/
        └── val/
    
    Args:
        config: Configuration dictionary with dataset and augmentation parameters
        valid: If True, load validation set; otherwise load training set
    """
    def __init__(self, config, valid=False):
        self.config = config
        self.valid = valid
        self.Tensor = transforms.ToTensor()
        
        # Load augmentation parameters
        aug_cfg = config.get('augmentation', {})
        self.degrees = aug_cfg.get('degrees', 10)
        self.translate = aug_cfg.get('translate', 0.1)
        self.scale = aug_cfg.get('scale', 0.1)
        self.shear = aug_cfg.get('shear', 10)
        self.hgain = aug_cfg.get('hgain', 0.015)
        self.sgain = aug_cfg.get('sgain', 0.7)
        self.vgain = aug_cfg.get('vgain', 0.4)
        self.prob_perspective = aug_cfg.get('prob_perspective', 0.5)
        self.prob_flip = aug_cfg.get('prob_flip', 0.5)
        self.prob_hsv = aug_cfg.get('prob_hsv', 0.5)
        
        # Dataset paths
        data_cfg = config.get('dataset', {})
        self.dataset_root = data_cfg.get('root', './bdd100k')
        self.split = 'val' if valid else 'train'
        
        # Image directory
        self.image_dir = os.path.join(self.dataset_root, 'images', self.split)
        
        # Get image list
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        self.names = [f for f in os.listdir(self.image_dir) 
                     if f.endswith(('.jpg', '.png'))]
        
        if len(self.names) == 0:
            raise ValueError(f"No images found in {self.image_dir}")
        
        print(f"{'Validation' if valid else 'Training'} dataset: {len(self.names)} images")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Returns:
            image_name: Path to the image
            image: Preprocessed image tensor (3, H, W)
            targets: Tuple of (drivable_area_mask, lane_line_mask)
        """
        W_, H_ = 640, 384
        image_name = os.path.join(self.image_dir, self.names[idx])
        
        # Load image
        image = cv2.imread(image_name)
        if image is None:
            raise ValueError(f"Failed to load image: {image_name}")
        
        # Get base filename without extension
        base_name = os.path.splitext(self.names[idx])[0]
        
        # Load labels
        da_label_path = os.path.join(self.dataset_root, 'drivable_area_annotations', 
                                     self.split, f'{base_name}.png')
        ll_label_path = os.path.join(self.dataset_root, 'lane_line_annotations', 
                                     self.split, f'{base_name}.png')
        
        label1 = cv2.imread(da_label_path, 0)
        label2 = cv2.imread(ll_label_path, 0)
        
        if label1 is None or label2 is None:
            raise ValueError(
                f"Failed to load labels for: {image_name}\n"
                f"DA path: {da_label_path}\n"
                f"LL path: {ll_label_path}"
            )
        
        # Apply augmentation (training only)
        if not self.valid:
            # Random perspective transformation
            if random.random() < self.prob_perspective:
                combination = (image, label1, label2)
                image, label1, label2 = random_perspective(
                    combination, self.degrees, self.translate, self.scale, self.shear
                )
            
            # HSV augmentation
            if random.random() < self.prob_hsv:
                augment_hsv(image, self.hgain, self.sgain, self.vgain)
            
            # Random horizontal flip
            if random.random() < self.prob_flip:
                image = np.fliplr(image)
                label1 = np.fliplr(label1)
                label2 = np.fliplr(label2)
        
        # Resize and pad
        image = letterbox(image, (H_, W_))
        label1 = cv2.resize(label1, (W_, H_))
        label2 = cv2.resize(label2, (W_, H_))
        
        # Create binary masks
        _, seg_b1 = cv2.threshold(label1, 1, 255, cv2.THRESH_BINARY_INV)
        _, seg_b2 = cv2.threshold(label2, 1, 255, cv2.THRESH_BINARY_INV)
        _, seg1 = cv2.threshold(label1, 1, 255, cv2.THRESH_BINARY)
        _, seg2 = cv2.threshold(label2, 1, 255, cv2.THRESH_BINARY)
        
        # Convert to tensors
        seg1 = self.Tensor(seg1)
        seg2 = self.Tensor(seg2)
        seg_b1 = self.Tensor(seg_b1)
        seg_b2 = self.Tensor(seg_b2)
        
        # Stack background and foreground channels
        seg_da = torch.stack((seg_b1[0], seg1[0]), 0)
        seg_ll = torch.stack((seg_b2[0], seg2[0]), 0)
        
        # Prepare image (BGR to RGB, HWC to CHW)
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)
        
        return image_name, torch.from_numpy(image), (seg_da, seg_ll)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_dataloader(config, valid=False, batch_size=None, num_workers=None):
    """
    Create a DataLoader for BDD100K dataset
    
    Args:
        config: Configuration dictionary
        valid: If True, create validation dataloader
        batch_size: Batch size (overrides config)
        num_workers: Number of workers (overrides config)
        
    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader
    
    dataset = BDD100KDataset(config, valid=valid)
    
    if batch_size is None:
        batch_size = config['training'].get('batch_size', 16)
    if num_workers is None:
        num_workers = config['training'].get('num_workers', 4)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(not valid),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(not valid)
    )
    
    return dataloader


if __name__ == '__main__':
    # Test dataset
    print("Testing BDD100K Dataset...")
    
    # Create test config
    config = {
        'dataset': {
            'root': './bdd100k'
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
            'batch_size': 2,
            'num_workers': 0
        }
    }
    
    try:
        # Test dataset creation
        train_dataset = BDD100KDataset(config, valid=False)
        print(f"✓ Training dataset loaded: {len(train_dataset)} images")
        
        # Test sample loading
        image_name, image, (seg_da, seg_ll) = train_dataset[0]
        print(f"\nSample loaded:")
        print(f"  Image shape: {image.shape}")
        print(f"  DA mask shape: {seg_da.shape}")
        print(f"  LL mask shape: {seg_ll.shape}")
        
        # Test dataloader
        train_loader = create_dataloader(config, valid=False)
        print(f"\n✓ DataLoader created: {len(train_loader)} batches")
        
        # Test batch loading
        for batch_idx, (names, images, targets) in enumerate(train_loader):
            print(f"\nBatch {batch_idx}:")
            print(f"  Images shape: {images.shape}")
            print(f"  DA targets shape: {targets[0].shape}")
            print(f"  LL targets shape: {targets[1].shape}")
            break
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nNote: Ensure BDD100K dataset is available in the correct structure:")
        print("  bdd100k/")
        print("  ├── images/train/")
        print("  ├── images/val/")
        print("  ├── drivable_area_annotations/train/")
        print("  ├── drivable_area_annotations/val/")
        print("  ├── lane_line_annotations/train/")
        print("  └── lane_line_annotations/val/")
