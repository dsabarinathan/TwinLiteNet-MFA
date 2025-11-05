# TwinLiteNet-MFA Project Structure

## Overview

This repository contains a modular implementation of TwinLiteNet-MFA for dual-task segmentation on BDD100K dataset.

## File Descriptions

### Core Files

#### 1. `model.py` (21 KB)
**Main model architecture file**

Contains:
- `ModelConfig`: Configuration for all model sizes (nano, small, medium, large)
  - **nano**: [8, 16, 24, 32] channels, p=2, q=3
  - **small**: [16, 24, 32, 64] channels, p=2, q=3
  - **medium**: [32, 48, 96, 192] channels, p=5, q=8
  - **large**: [48, 64, 128, 256] channels, p=7, q=10
- `ChannelSpatialAoA`: Channel and Spatial Attention module
- `MultiScaleSEBlock`: Multi-scale feature extraction with attention
- `CAAM`: Context-Aware Attention Module with GCN
- `DepthwiseESP`: Efficient Spatial Pyramid blocks
- `Encoder`: Feature extraction backbone
- `TwinLiteNetPlus`: Main dual-task segmentation model

**Usage:**
```python
from model import TwinLiteNetPlus
model = TwinLiteNetPlus('nano')  # or 'small', 'medium', 'large'
```

---

#### 2. `loss.py` (9 KB)
**Loss functions for training**

Contains:
- `FocalLoss`: Handles class imbalance (α=0.25, γ=2.0)
- `TverskyLoss`: Generalization of Dice loss (α=0.3, β=0.7)
- `DiceLoss`: Standard Dice loss
- `TotalLoss`: Combined loss for dual tasks
- `WeightedTotalLoss`: Task-weighted combined loss

**Usage:**
```python
from loss import TotalLoss
criterion = TotalLoss()
focal, tversky, total = criterion(outputs, targets)
```

---

#### 3. `dataset.py` (13 KB)
**Data loading and augmentation**

Contains:
- `BDD100KDataset`: PyTorch Dataset for BDD100K
- `letterbox()`: Resize and pad images
- `augment_hsv()`: HSV color augmentation
- `random_perspective()`: Geometric transformations
- `create_dataloader()`: Convenience function

**Features:**
- Automatic data augmentation for training
- Support for dual-task annotations
- Configurable augmentation parameters

**Usage:**
```python
from dataset import create_dataloader
train_loader = create_dataloader(config, valid=False)
```

---

#### 4. `utils.py` (14 KB)
**Utilities and evaluation metrics**

Contains:
- `SegmentationMetric`: Compute IoU, mIoU, accuracy
- `ModelEMA`: Exponential Moving Average
- `poly_lr_scheduler()`: Polynomial LR decay
- `cosine_lr_scheduler()`: Cosine annealing
- `count_parameters()`: Count model parameters
- `print_model_summary()`: Detailed model info
- `AverageMeter`: Track training metrics

**Usage:**
```python
from utils import SegmentationMetric, ModelEMA
metric = SegmentationMetric(2)
ema = ModelEMA(model)
```

---

#### 5. `train.py` (15 KB)
**Main training script**

Features:
- Complete training loop with validation
- Mixed precision training
- Checkpoint saving/loading
- Command-line interface
- Progress bars with tqdm

**Usage:**
```bash
# Basic training
python train.py --config config.yaml

# With options
python train.py --config config.yaml --model small --epochs 200 --batch-size 32
```

---

#### 5. `train.py` (15 KB)
**Main training script**

Features:
- Complete training loop with validation
- Mixed precision training
- Checkpoint saving/loading
- Command-line interface
- Progress bars with tqdm

**Usage:**
```bash
# Basic training
python train.py --config config.yaml

# With options
python train.py --config config.yaml --model small --epochs 200 --batch-size 32
```

---

#### 6. `test.py` (11 KB)
**Model evaluation script**

Features:
- Load and test trained models
- Evaluate on BDD100K validation set
- Calculate comprehensive metrics
- Measure inference speed (FPS)
- Save results to file

**Metrics Calculated:**
- Drivable Area: Pixel Accuracy, IoU, mIoU
- Lane Line: Line Accuracy, IoU, mIoU
- Performance: Inference time, FPS

**Usage:**
```bash
# Basic testing
python test.py --checkpoint checkpoints/checkpoint_best.pth --model nano

# With custom dataset
python test.py --checkpoint model.pth --model small --data-root /path/to/bdd100k

# Save results
python test.py --checkpoint model.pth --model nano --save-results --output results.txt
```

---

### Configuration Files

#### 7. `config.yaml` (3.6 KB)
**Training configuration file**

Sections:
- **model**: Architecture selection
- **dataset**: Data paths
- **augmentation**: Augmentation parameters
- **training**: Training hyperparameters
- **seed**: Random seed for reproducibility

**Key Parameters:**
```yaml
model:
  architecture: nano  # nano, small, medium, large

training:
  max_epochs: 100
  batch_size: 16
  learning_rate: 0.0005
  use_ema: true
```

---

#### 8. `requirements.txt` (304 B)
**Python dependencies**

Core dependencies:
- torch>=1.7.0
- torchvision>=0.8.0
- opencv-python>=4.5.0
- numpy>=1.19.0
- pyyaml>=5.4.0
- tqdm>=4.60.0
- albumentations>=1.0.0
- scikit-image>=0.18.0

Optional:
- thop>=0.0.31 (for FLOPs calculation)

---

### Documentation

#### 9. `README.md` (6.6 KB)
**Main documentation**

Sections:
- Features and model architectures
- Installation instructions
- Dataset preparation
- Training and testing examples
- Configuration guide
- Performance tips

---

#### 10. `train_complete.py` (52 KB)
**All-in-one training script**

A single file containing all components (model, loss, dataset, training) for easy deployment. This is the original complete version before modularization.

---

## Repository Structure for GitHub

Recommended GitHub repository layout:

```
TwinLiteNet-MFA/
├── README.md                 # Main documentation
├── requirements.txt          # Dependencies
├── config.yaml              # Configuration
├── model.py                 # Model architecture
├── loss.py                  # Loss functions
├── dataset.py               # Data loading
├── utils.py                 # Utilities
├── train.py                 # Training script
├── test.py                  # Testing script
├── train_complete.py        # All-in-one version
├── checkpoints/             # Saved models (create during training)
│   ├── checkpoint_best.pth
│   └── checkpoint_latest.pth
└── bdd100k/                 # Dataset (user provided)
    ├── images/
    ├── drivable_area_annotations/
    └── lane_line_annotations/
```

## Quick Start Workflow

1. **Setup Environment**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Dataset**
   - Download BDD100K
   - Organize as shown above
   - Update path in `config.yaml`

3. **Train Model**
   ```bash
   python train.py --config config.yaml
   ```

4. **Monitor Training**
   - Checkpoints saved to `./checkpoints/`
   - Best model: `checkpoint_best.pth`
   - Latest: `checkpoint_latest.pth`

## Model Configuration Details

### Nano (Edge Devices)
- Parameters: ~300K
- Channels: [8, 16, 24, 32]
- ESP blocks: p=2, q=3
- Use case: Real-time on mobile/edge

### Small (Mobile)
- Parameters: ~1.2M
- Channels: [16, 24, 32, 64]
- ESP blocks: p=2, q=3
- Use case: Mobile applications

### Medium (High Accuracy)
- Parameters: ~8M
- Channels: [32, 48, 96, 192]
- ESP blocks: p=5, q=8
- Use case: Server-side processing

### Large (Best Accuracy)
- Parameters: ~20M
- Channels: [48, 64, 128, 256]
- ESP blocks: p=7, q=10
- Use case: Research/benchmarking

## Key Changes from Original

The modular version includes these improvements:

1. **Separated concerns**: Model, loss, dataset, utils, training
2. **Updated ModelConfig**: Removed 'smallv2', kept 4 main variants
3. **Enhanced documentation**: Comprehensive README and comments
4. **Cleaner code**: Better organization and readability
5. **Maintained compatibility**: Same functionality as original

## Testing Individual Components

Each file can be tested independently:

```bash
# Test model
python model.py

# Test loss functions
python loss.py

# Test dataset (requires BDD100K)
python dataset.py

# Test utilities
python utils.py

# Test evaluation on validation set (requires checkpoint and BDD100K)
python test.py --checkpoint checkpoints/checkpoint_best.pth --model nano
```

## GitHub Upload Checklist

- [x] model.py - Core architecture
- [x] loss.py - Loss functions
- [x] dataset.py - Data loading
- [x] utils.py - Utilities
- [x] train.py - Training script
- [x] test.py - Testing/evaluation script
- [x] config.yaml - Configuration
- [x] requirements.txt - Dependencies
- [x] README.md - Documentation
- [x] train_complete.py - All-in-one backup

## Notes

- All files are standalone and can be used independently
- The modular structure makes it easy to customize components
- Configuration is centralized in `config.yaml`
- Command-line options override config file
- Compatible with PyTorch 1.7+ and Python 3.7+

---

**For questions or issues, please refer to README.md or open an issue on GitHub.**
