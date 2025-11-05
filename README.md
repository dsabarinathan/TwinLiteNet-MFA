# TwinLiteNet-MFA

**Multi-Scale Feature Aggregation with Channel-Spatial AoA Attention for Lightweight Scene Segmentation**

A lightweight dual-task segmentation model for autonomous driving, designed for real-time drivable area detection and lane line segmentation.

## Features

- **Dual-task segmentation**: Simultaneous drivable area and lane line detection
- **Lightweight architecture**: Multiple model sizes (nano to large)
- **Advanced attention mechanisms**: Channel-Spatial AoA and Context-Aware Attention Module (CAAM)
- **Multi-scale feature extraction**: Efficient Spatial Pyramid (ESP) blocks
- **Real-time capable**: Optimized for edge devices and mobile platforms

## Model Architectures

| Model  | Parameters | Model Size | Description                    |
|--------|-----------|------------|--------------------------------|
| Nano   | ~0.3M     | ~1.2 MB    | Smallest, best for edge devices|
| Small  | ~1.2M     | ~4.8 MB    | Good balance for mobile        |
| Medium | ~8M       | ~32 MB     | Higher accuracy                |
| Large  | ~20M      | ~80 MB     | Best accuracy                  |

## Installation

### Requirements

```bash
Python >= 3.7
PyTorch >= 1.7.0
CUDA >= 10.2 (for GPU training)
```

### Install Dependencies

```bash
pip install torch torchvision
pip install opencv-python numpy pyyaml tqdm
pip install albumentations scikit-image Pillow
pip install thop  # Optional, for FLOPs calculation
```

## Dataset Preparation

Download and prepare the BDD100K dataset:

```
bdd100k/
├── images/
│   ├── train/          # Training images
│   └── val/            # Validation images
├── drivable_area_annotations/
│   ├── train/          # Drivable area masks
│   └── val/
└── lane_line_annotations/
    ├── train/          # Lane line masks
    └── val/
```

Update the dataset path in `config.yaml`:

```yaml
dataset:
  root: ./bdd100k  # Path to your dataset
```

## Quick Start

### Basic Training

```bash
# Train with default configuration (nano model)
python train.py --config config.yaml

# Train with specific model size
python train.py --config config.yaml --model small

# Train with custom parameters
python train.py --config config.yaml --epochs 200 --batch-size 32 --lr 0.001
```

### Resume Training

```bash
python train.py --config config.yaml --resume checkpoints/checkpoint_best.pth
```

### Advanced Options

```bash
# Train without Exponential Moving Average
python train.py --config config.yaml --no-ema

# Custom save directory
python train.py --config config.yaml --save-dir ./my_checkpoints

# Override dataset path
python train.py --config config.yaml --data-root /path/to/bdd100k
```

## Configuration

Edit `config.yaml` to customize training:

```yaml
model:
  architecture: nano  # nano, small, medium, large

training:
  max_epochs: 100
  batch_size: 16
  learning_rate: 0.0005
  use_ema: true

augmentation:
  degrees: 10
  translate: 0.1
  prob_flip: 0.5
```

## Model Architecture

### Key Components

1. **Encoder**: Multi-scale feature extraction with DepthwiseESP blocks
2. **CAAM (Context-Aware Attention Module)**: Graph-based attention for global context
3. **MultiScaleSEBlock**: Multi-scale convolutions with Channel-Spatial AoA attention
4. **Dual Decoders**: Separate paths for drivable area and lane line prediction

### Attention Mechanisms

- **Channel-Spatial AoA**: Combines channel and spatial attention with refinement
- **CAAM**: Uses graph convolutions for context-aware feature aggregation

## Training Details

### Loss Function

Combined loss with Focal Loss and Tversky Loss:
- **Focal Loss**: Handles class imbalance (α=0.25, γ=2.0)
- **Tversky Loss**: Generalizes Dice loss (α=0.3, β=0.7)

### Optimizer

- **AdamW** with weight decay (default: 0.0001)
- **Polynomial LR decay** (power=1.5)
- **Mixed precision training** for faster convergence

### Data Augmentation

- Random perspective transformation
- HSV color augmentation
- Horizontal flipping
- Letterbox resizing (640×384)

## Evaluation Metrics

- **Pixel Accuracy**: Overall pixel-wise accuracy
- **IoU**: Intersection over Union for positive class
- **mIoU**: Mean IoU across all classes
- **Line Accuracy**: Average of sensitivity and specificity (for lane lines)

## File Structure

```
TwinLiteNet-MFA/
├── model.py          # Model architecture
├── loss.py           # Loss functions
├── dataset.py        # Dataset and augmentation
├── utils.py          # Utilities and metrics
├── train.py          # Training script
├── config.yaml       # Configuration file
└── README.md         # This file
```

## Usage Examples

### Testing Model

```python
from model import TwinLiteNetPlus
import torch

# Create model
model = TwinLiteNetPlus('nano')
model.eval()

# Test forward pass
x = torch.randn(1, 3, 384, 640)
out_da, out_ll = model(x)

print(f"Drivable area output: {out_da.shape}")
print(f"Lane line output: {out_ll.shape}")
```

### Custom Loss Configuration

```python
from loss import WeightedTotalLoss

# Create weighted loss (emphasize drivable area more)
criterion = WeightedTotalLoss(
    da_weight=2.0,  # Double weight for drivable area
    ll_weight=1.0,
    focal_alpha=0.25,
    focal_gamma=2.0
)
```

### Model Information

```python
from model import get_model_info

# Get model statistics
for config in ['nano', 'small', 'medium', 'large']:
    info = get_model_info(config)
    print(f"{config}: {info['parameters_M']}")
```

## Performance Tips

### For Better Accuracy
- Use `medium` or `large` model
- Increase batch size if GPU memory allows
- Train for more epochs (150-200)
- Enable EMA (default)

### For Speed
- Use `nano` or `small` model
- Reduce batch size
- Set `num_workers` appropriately (2 for Windows, 4+ for Linux)
- Disable EMA with `--no-ema`

### For Edge Devices
- Use `nano` model (~0.3M params)
- Consider quantization (PyTorch quantization tools)
- Export to ONNX for deployment

## Citation

If you use this code in your research, please cite:

```bibtex
@article{twinlitenet-mfa,
  title={TwinLiteNet-MFA: Multi-Scale Feature Aggregation with Channel-Spatial AoA Attention for Lightweight Scene Segmentation},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

This project is released under the MIT License.

## Acknowledgments

- BDD100K dataset: [https://bdd-data.berkeley.edu/](https://bdd-data.berkeley.edu/)
- Inspired by TwinLiteNet and ESPNet architectures

## Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This is a research implementation. For production use, consider additional optimizations and testing.
