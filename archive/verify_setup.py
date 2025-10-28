"""
Quick verification that the training pipeline is correctly set up
"""

import torch
from torchvision.models import efficientnet_b0
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

print("=" * 80)
print("SETUP VERIFICATION")
print("=" * 80)

# Check PyTorch and device
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Selected device: {device}")

# Test data loading
print("\n" + "-" * 80)
print("Testing data loading...")

data_dir = Path("/Users/niklaswegner/python_proj/crack_model/SDNET2018")
sample_image = list(data_dir.glob("**/*.jpg"))[0]

print(f"Sample image: {sample_image}")

img = Image.open(sample_image)
print(f"Original size: {img.size}")
print(f"Mode: {img.mode}")

# Test augmentations
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

img_array = np.array(img.convert('RGB'))
transformed = transform(image=img_array)
img_tensor = transformed['image']

print(f"Transformed shape: {img_tensor.shape}")
print(f"Transformed dtype: {img_tensor.dtype}")
print(f"Value range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")

# Test model
print("\n" + "-" * 80)
print("Testing model...")

model = efficientnet_b0(weights=None)
model.eval()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# Test forward pass
batch = img_tensor.unsqueeze(0)  # Add batch dimension
print(f"Input shape: {batch.shape}")

try:
    with torch.no_grad():
        features = model(batch)
    print(f"Output shape: {features.shape}")
    print("✓ Forward pass successful")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")

# Test with device
print("\n" + "-" * 80)
print(f"Testing on {device}...")

try:
    model = model.to(device)
    batch = batch.to(device)

    import time
    start = time.time()

    with torch.no_grad():
        features = model(batch)

    elapsed = time.time() - start

    print(f"✓ Device inference successful")
    print(f"  Inference time: {elapsed*1000:.1f} ms")
    print(f"  Output shape: {features.shape}")

except Exception as e:
    print(f"✗ Device inference failed: {e}")

# Estimate training time
print("\n" + "-" * 80)
print("Training time estimate...")

n_train_samples = 50458
batch_size = 32
batches_per_epoch = n_train_samples // batch_size
seconds_per_batch = elapsed * 2  # Forward + backward ~2x forward

estimated_epoch_time = batches_per_epoch * seconds_per_batch
estimated_total_time = estimated_epoch_time * 30

print(f"Batches per epoch: {batches_per_epoch}")
print(f"Estimated time per batch: {seconds_per_batch:.1f}s")
print(f"Estimated time per epoch: {estimated_epoch_time/60:.1f} minutes")
print(f"Estimated total time (30 epochs): {estimated_total_time/3600:.1f} hours")

if estimated_total_time > 36000:  # >10 hours
    print("\n⚠️  WARNING: Training will take a very long time on this device!")
    print("   Consider using a CUDA GPU or reducing input_size to 256")

# Check audit files
print("\n" + "-" * 80)
print("Checking audit files...")

audit_files = [
    "data_audit.md",
    "data_audit.json",
    "samples_Bridge_Crack.png",
    "samples_Bridge_NonCrack.png",
    "samples_Pavement_Crack.png",
    "samples_Pavement_NonCrack.png",
    "samples_Wall_Crack.png",
    "samples_Wall_NonCrack.png"
]

for f in audit_files:
    path = Path(f"/Users/niklaswegner/python_proj/crack_model/{f}")
    status = "✓" if path.exists() else "✗"
    print(f"{status} {f}")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
