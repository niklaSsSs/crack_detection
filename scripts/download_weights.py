#!/usr/bin/env python3
"""
Download EfficientNet-B0 ImageNet Weights
Für offline Training auf Server ohne Internet
"""

import os
import sys
from pathlib import Path
import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

print("=" * 70)
print("📥 DOWNLOAD EFFICIENTNET-B0 WEIGHTS")
print("=" * 70)
print()

# Ziel-Ordner im Projekt
weights_dir = Path(__file__).parent.parent / "pretrained_weights"
weights_dir.mkdir(exist_ok=True)

print(f"Ziel-Ordner: {weights_dir}")
print()

# Download Weights (wird gecacht in ~/.cache/torch)
print("⏳ Lade EfficientNet-B0 ImageNet Weights...")
print("   (wird in PyTorch Cache heruntergeladen)")
print()

try:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    print("✅ Weights geladen!")

except Exception as e:
    print(f"❌ Fehler: {e}")
    sys.exit(1)

# Finde heruntergeladene Weights
cache_dir = Path.home() / ".cache/torch/hub/checkpoints"
weight_file = "efficientnet_b0_rwightman-7f5810bc.pth"
weight_path = cache_dir / weight_file

if weight_path.exists():
    print()
    print(f"📍 Weights gefunden in PyTorch Cache:")
    print(f"   {weight_path}")
    print(f"   Größe: {weight_path.stat().st_size / (1024*1024):.1f} MB")

    # Kopiere ins Projekt
    target_path = weights_dir / weight_file

    if target_path.exists():
        print()
        print(f"✓ Weights bereits im Projekt vorhanden:")
        print(f"  {target_path}")
    else:
        import shutil
        print()
        print(f"📋 Kopiere Weights ins Projekt...")
        shutil.copy2(weight_path, target_path)
        print(f"✅ Kopiert nach: {target_path}")
else:
    print()
    print(f"⚠️  Weights nicht gefunden in Cache: {weight_path}")

print()
print("=" * 70)
print("✅ FERTIG")
print("=" * 70)
print()
print("Die Weights sind jetzt im Projekt und können offline verwendet werden.")
print()
print("Projekt-Ordner:")
print(f"  {weights_dir}")
print()
print("Beim Training auf Server werden die Weights automatisch aus diesem")
print("Ordner geladen (kein Internet nötig).")
print()
