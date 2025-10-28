# SDNET2018 Encoder Pre-training Project

## 🎯 Project Overview

This project implements encoder pre-training for crack detection using the SDNET2018 dataset. The pre-trained encoder weights will be used for weld seam segmentation tasks.

**Goal:** Pre-train an EfficientNet-B0 encoder on 56K crack images to achieve ≥0.95 ROC-AUC, then transfer the encoder weights to a segmentation model for weld seam analysis.

---

## 📊 Dataset Audit Summary

### SDNET2018 Statistics

- **Total Images:** 56,092
- **Resolution:** 256×256 pixels (RGB JPEGs)
- **Class Distribution:**
  - Crack: 8,484 (15.13%)
  - NonCrack: 47,608 (84.87%)

### Domain Breakdown

| Domain   | Crack | NonCrack | Total  | Crack Ratio |
|----------|-------|----------|--------|-------------|
| Bridge   | 2,025 | 11,595   | 13,620 | 14.87%      |
| Pavement | 2,608 | 21,726   | 24,334 | 10.72%      |
| Wall     | 3,851 | 14,287   | 18,138 | 21.23%      |

### Data Quality

- **Corrupt Files:** 0
- **Duplicates:** 13 files
- **File Format:** 100% JPEG
- **Color Mode:** 100% RGB
- **Image Dimensions:** Uniform 256×256

### Audit Reports

- `data_audit.md` - Comprehensive markdown report
- `data_audit.json` - Structured JSON report
- `samples_*.png` - Thumbnail grids (6 files, one per domain/class)

---

## 🛠️ Implementation

### Training Scripts

#### 1. **train_encoder.py** (Original - GPU/MPS)
- Input size: 512×512 (upscaled from 256)
- Batch size: 32
- Epochs: 30
- **Status:** Slow on MPS (~69s/batch ≈ 30 hours per epoch)
- **Recommendation:** Use with CUDA GPU

#### 2. **train_encoder_cpu.py** (Optimized - CPU)
- Input size: 256×256 (original resolution)
- Batch size: 64
- Epochs: 20
- **Status:** Ready to run, ~5-10 hours estimated
- **Recommendation:** Use for quick local training

### Key Features

✅ **Grouped Splitting** - Prevents data leakage by grouping series IDs
✅ **Class Balancing** - WeightedRandomSampler for 1:1 crack/noncrack ratio
✅ **Domain-Aware Metrics** - Per-domain AUC/F1 tracking
✅ **Rich Augmentation** - Photometric + geometric transforms
✅ **Early Stopping** - Patience=5, monitors Val AUC
✅ **Encoder Export** - Saves only encoder weights for transfer learning

### Data Pipeline

```
SDNET2018/
├── D/ (Bridge)
│   ├── CD/ (Cracked) → Label 1
│   └── UD/ (Uncracked) → Label 0
├── P/ (Pavement)
│   ├── CP/ (Cracked) → Label 1
│   └── UP/ (Uncracked) → Label 0
└── W/ (Wall)
    ├── CW/ (Cracked) → Label 1
    └── UW/ (Uncracked) → Label 0
```

**Label Extraction:** Automatic from parent directory names (case-insensitive mapping)

### Augmentations

**Training:**
- Photometric: RandomBrightnessContrast, HueSaturationValue, Blur, Noise
- Geometric: HorizontalFlip, Rotation (±7°), Perspective
- Normalize: ImageNet mean/std

**Validation/Test:**
- Resize only
- Normalize: ImageNet mean/std

---

## 🚀 Usage

### 1. Run Data Audit (Completed)

```bash
python data_audit.py
```

**Outputs:**
- `data_audit.md`
- `data_audit.json`
- `samples_*.png` (6 thumbnail grids)

### 2. Train Encoder

**Option A: CPU Training (Recommended for local)**

```bash
python train_encoder_cpu.py
```

**Option B: GPU Training (CUDA)**

```bash
# Modify train_encoder.py to use CUDA
python train_encoder.py
```

**Outputs:**
- `runs/efficientnet_b0_crack_pretraining_*/`
  - `best_model.pt` - Full checkpoint
  - `encoder_sdnet_efficientnet_b0.pt` - **Encoder weights only** ✨
  - `config.yaml` - Training configuration
  - `test_metrics.json` - Evaluation metrics
  - `confusion_matrix.png` - Confusion matrix plot
  - `training_history.png` - Loss/AUC/F1 curves
  - `USAGE.md` - Transfer learning instructions

### 3. Use Pre-trained Encoder for Segmentation

```python
import torch
from torchvision.models import efficientnet_b0

# Load pre-trained encoder
checkpoint = torch.load('encoder_sdnet_efficientnet_b0.pt')
encoder = efficientnet_b0(weights=None)
encoder.load_state_dict(checkpoint['encoder_state_dict'])

# Initialize your segmentation model (U-Net, DeepLabv3+, etc.)
# Replace encoder with pre-trained weights

# Fine-tuning strategy:
# 1. Freeze encoder for 3-5 epochs
for param in encoder.parameters():
    param.requires_grad = False

# 2. Train decoder only (lr=1e-3)

# 3. Unfreeze encoder and fine-tune jointly (lr=1e-4)
for param in encoder.parameters():
    param.requires_grad = True
```

---

## 📈 Expected Performance

### Success Criteria

- **Validation AUC:** ≥ 0.95
- **Test AUC:** ≥ 0.95
- **Domain Robustness:** No single domain AUC < 0.90

### Baseline Performance (ImageNet Pre-training)

Based on similar crack detection tasks:
- Val AUC: 0.96-0.98 (expected)
- Test AUC: 0.95-0.97 (expected)
- Per-domain variation: ±2-3% AUC

### Class Imbalance Handling

- **Positive Weight:** 5.67 (NonCrack/Crack ratio)
- **Sampling:** Balanced 1:1 during training
- **Metrics:** ROC-AUC (insensitive to imbalance)

---

## 📂 Project Structure

```
crack_model/
├── SDNET2018/                     # Dataset (56K images)
│   ├── D/, P/, W/                 # Domain folders
│   └── ...
│
├── data_audit.py                   # Dataset inspection script
├── data_audit.md                   # Audit report (markdown)
├── data_audit.json                 # Audit report (JSON)
├── samples_*.png                   # Thumbnail grids (6 files)
│
├── train_encoder.py                # GPU training script (512×512)
├── train_encoder_cpu.py            # CPU training script (256×256, optimized)
├── verify_setup.py                 # Setup verification script
│
├── runs/                           # Training outputs
│   └── efficientnet_b0_crack_*/
│       ├── encoder_*.pt            # ⭐ Encoder weights for transfer
│       ├── best_model.pt           # Full checkpoint
│       ├── test_metrics.json       # Evaluation metrics
│       └── ...
│
└── README.md                       # This file
```

---

## 🔧 Requirements

### Python Packages

```bash
pip install torch torchvision
pip install albumentations
pip install scikit-learn
pip install matplotlib seaborn
pip install pyyaml
pip install tqdm pillow numpy
```

### Hardware Recommendations

| Device       | Batch Size | Input Size | Est. Time (20 epochs) |
|--------------|------------|------------|-----------------------|
| CUDA GPU     | 32         | 512×512    | 3-5 hours             |
| Apple M-chip | 64         | 256×256    | 5-10 hours (CPU mode) |
| CPU (Intel)  | 32         | 256×256    | 10-15 hours           |

---

## 📝 Notes

### Grouped Splitting

SDNET2018 images are organized in series (e.g., `7001-*.jpg`, `7002-*.jpg`). To prevent data leakage:
- Series IDs are extracted from filenames
- Stratified Group K-Fold ensures same series stays in same split
- Train/Val/Test: 90%/4.8%/5.3% (respecting series boundaries)

### Domain Adaptation Considerations

The encoder is pre-trained on **concrete cracks**. For weld seam segmentation:
- **Similarities:** Linear defects, high contrast, binary classification
- **Differences:** Surface texture, imaging conditions, crack morphology
- **Recommendation:** Fine-tune with domain adaptation techniques if weld images differ significantly

### Known Limitations

1. **MPS Performance:** PyTorch MPS (Apple Silicon) is slow for EfficientNet (~30x slower than CUDA)
2. **Class Imbalance:** 85% NonCrack vs 15% Crack (handled via pos_weight + sampling)
3. **Resolution Trade-off:** 512×512 training provides more detail but 4x slower than 256×256

---

## 📚 References

**SDNET2018 Dataset:**
- Dorafshan, S., Thomas, R. J., & Maguire, M. (2018). SDNET2018: An annotated image dataset for training deep learning algorithms. *Data in Brief*, 21, 102623.

**EfficientNet:**
- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*.

---

## ✅ Status

### Completed Tasks

- [x] Dataset audit (56K images inspected)
- [x] Data quality report generation
- [x] Sample thumbnail grids
- [x] Grouped splitting implementation
- [x] Augmentation pipeline
- [x] EfficientNet-B0 model setup
- [x] Training loop with metrics
- [x] Early stopping
- [x] Per-domain evaluation
- [x] CPU-optimized training script

### Next Steps

- [ ] Run training (`train_encoder_cpu.py` recommended)
- [ ] Validate performance (Target: AUC ≥ 0.95)
- [ ] Export encoder weights
- [ ] Integrate with weld segmentation model

---

## 🤝 Usage Instructions

### Quick Start

```bash
# 1. Verify setup
python verify_setup.py

# 2. Run training (CPU optimized)
python train_encoder_cpu.py

# 3. Check results
cd runs/efficientnet_b0_crack_pretraining_*/
cat test_metrics.json
```

### Transfer Learning

After training completes, the encoder weights will be saved as:

```
runs/efficientnet_b0_crack_pretraining_*/encoder_sdnet_efficientnet_b0.pt
```

Load these weights into your segmentation model as shown in Section 3 above.

---

**For questions or issues, check:**
- `data_audit.md` for dataset details
- `USAGE.md` in run directory for transfer learning guide
- `test_metrics.json` for final performance

**Good luck! 🚀**
