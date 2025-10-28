"""
Quick Setup Test
Prüft ob alle Dependencies installiert sind und Training starten kann
"""

import sys
from pathlib import Path

print("=" * 80)
print("SETUP TEST")
print("=" * 80)

errors = []

# 1. Python Version
print("\n[1/8] Python Version...")
if sys.version_info < (3, 8):
    errors.append("Python 3.8+ benötigt")
    print(f"  ✗ Python {sys.version_info.major}.{sys.version_info.minor} (zu alt)")
else:
    print(f"  ✓ Python {sys.version_info.major}.{sys.version_info.minor}")

# 2. PyTorch
print("\n[2/8] PyTorch...")
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")

    # Device check
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        print(f"  ✓ MPS available (Apple Silicon)")
    else:
        print(f"  ✓ CPU only")
except ImportError:
    errors.append("PyTorch nicht installiert")
    print("  ✗ PyTorch fehlt")

# 3. TorchVision
print("\n[3/8] TorchVision...")
try:
    import torchvision
    print(f"  ✓ TorchVision {torchvision.__version__}")
except ImportError:
    errors.append("TorchVision nicht installiert")
    print("  ✗ TorchVision fehlt")

# 4. Albumentations
print("\n[4/8] Albumentations...")
try:
    import albumentations
    print(f"  ✓ Albumentations {albumentations.__version__}")
except ImportError:
    errors.append("Albumentations nicht installiert")
    print("  ✗ Albumentations fehlt")

# 5. Scikit-learn
print("\n[5/8] Scikit-learn...")
try:
    import sklearn
    print(f"  ✓ Scikit-learn {sklearn.__version__}")
except ImportError:
    errors.append("Scikit-learn nicht installiert")
    print("  ✗ Scikit-learn fehlt")

# 6. Other deps
print("\n[6/8] Other Dependencies...")
missing = []
try:
    import numpy
    import yaml
    import matplotlib
    import seaborn
    import PIL
    from tqdm import tqdm
    print("  ✓ numpy, yaml, matplotlib, seaborn, PIL, tqdm")
except ImportError as e:
    missing.append(str(e))
    print(f"  ✗ Fehlt: {e}")
    errors.append("Dependencies fehlen")

# 7. Configs
print("\n[7/8] Configs...")
config_dir = Path(__file__).parent.parent / "configs"
configs = {
    "Mac": config_dir / "config_mac.yaml",
    "Server": config_dir / "config_server_a40.yaml"
}

for name, path in configs.items():
    if path.exists():
        print(f"  ✓ {name} Config: {path.name}")
    else:
        errors.append(f"{name} Config fehlt")
        print(f"  ✗ {name} Config fehlt: {path}")

# 8. Source Files
print("\n[8/8] Source Files...")
src_dir = Path(__file__).parent.parent / "src"
src_files = ["train.py", "dataset.py", "utils.py"]

for file in src_files:
    path = src_dir / file
    if path.exists():
        print(f"  ✓ {file}")
    else:
        errors.append(f"{file} fehlt")
        print(f"  ✗ {file} fehlt")

# Dataset check (optional)
print("\n[Optional] Dataset...")
dataset_dir = Path(__file__).parent.parent / "SDNET2018"
if dataset_dir.exists():
    domains = ["D", "P", "W"]
    domain_count = sum(1 for d in domains if (dataset_dir / d).exists())
    if domain_count == 3:
        print(f"  ✓ SDNET2018 Dataset gefunden")
    else:
        print(f"  ⚠ SDNET2018 unvollständig ({domain_count}/3 Domänen)")
else:
    print(f"  ⚠ SDNET2018 nicht gefunden (benötigt für Training)")

# Summary
print("\n" + "=" * 80)
if errors:
    print("SETUP INCOMPLETE")
    print("=" * 80)
    print("\nFehler:")
    for i, err in enumerate(errors, 1):
        print(f"  {i}. {err}")

    print("\nLösung:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
else:
    print("SETUP OK")
    print("=" * 80)
    print("\nTraining kann gestartet werden!")
    print("\nMac:")
    print("  ./scripts/train_mac.sh")
    print("\nServer:")
    print("  ./scripts/train_server.sh")
    sys.exit(0)
