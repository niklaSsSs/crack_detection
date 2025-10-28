"""
Dataset und Data Loading für SDNET2018
"""

import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import StratifiedGroupKFold


# Class mapping
DOMAIN_MAP = {'D': 'Bridge', 'P': 'Pavement', 'W': 'Wall'}
CLASS_MAP = {
    'CD': 1, 'UD': 0,  # Bridge
    'CP': 1, 'UP': 0,  # Pavement
    'CW': 1, 'UW': 0   # Wall
}


class SDNET2018Dataset(Dataset):
    """SDNET2018 Dataset"""

    def __init__(self, file_paths, labels, domains, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.domains = domains
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        domain = self.domains[idx]

        # Load image
        image = np.array(Image.open(img_path).convert('RGB'))

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.float32),
            'domain': domain,
            'path': str(img_path)
        }


def extract_series_id(file_path):
    """Extract series ID from filename"""
    stem = Path(file_path).stem
    parts = stem.split('-')
    if len(parts) >= 2:
        return parts[0]
    return stem


def load_dataset(data_dir):
    """Load dataset from directory"""

    data_dir = Path(data_dir)

    file_paths = []
    labels = []
    domains = []
    series_ids = []

    print("\nLoading dataset...")

    for domain_code in ['D', 'P', 'W']:
        domain_path = data_dir / domain_code
        domain_name = DOMAIN_MAP[domain_code]

        if not domain_path.exists():
            continue

        for class_dir in domain_path.iterdir():
            if not class_dir.is_dir():
                continue

            class_code = class_dir.name
            label = CLASS_MAP.get(class_code)

            if label is None:
                continue

            for img_file in class_dir.glob("*.jpg"):
                file_paths.append(str(img_file))
                labels.append(label)
                domains.append(domain_name)
                series_ids.append(extract_series_id(img_file))

    file_paths = np.array(file_paths)
    labels = np.array(labels)
    domains = np.array(domains)
    series_ids = np.array(series_ids)

    print(f"Total images: {len(file_paths)}")
    print(f"Crack: {(labels == 1).sum()}, NonCrack: {(labels == 0).sum()}")
    print(f"Unique series: {len(np.unique(series_ids))}")

    return file_paths, labels, domains, series_ids


def create_grouped_split(file_paths, labels, domains, series_ids, seed=42):
    """Create grouped split (prevents data leakage)"""

    print("\nCreating grouped split...")

    unique_series = np.unique(series_ids)
    series_to_idx = {s: i for i, s in enumerate(unique_series)}
    group_ids = np.array([series_to_idx[s] for s in series_ids])

    n_splits = 10
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    splits = list(skf.split(file_paths, labels, groups=group_ids))
    train_idx, temp_idx = splits[0]

    temp_labels = labels[temp_idx]
    temp_groups = group_ids[temp_idx]

    skf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    temp_splits = list(skf2.split(temp_idx, temp_labels, groups=temp_groups))
    val_rel_idx, test_rel_idx = temp_splits[0]

    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    splits = {
        'train': {
            'paths': file_paths[train_idx],
            'labels': labels[train_idx],
            'domains': domains[train_idx]
        },
        'val': {
            'paths': file_paths[val_idx],
            'labels': labels[val_idx],
            'domains': domains[val_idx]
        },
        'test': {
            'paths': file_paths[test_idx],
            'labels': labels[test_idx],
            'domains': domains[test_idx]
        }
    }

    for split_name, split_data in splits.items():
        n_total = len(split_data['labels'])
        n_crack = (split_data['labels'] == 1).sum()
        n_noncrack = (split_data['labels'] == 0).sum()

        print(f"\n{split_name.upper()}:")
        print(f"  Total: {n_total} ({n_total/len(file_paths)*100:.1f}%)")
        print(f"  Crack: {n_crack} ({n_crack/n_total*100:.1f}%)")
        print(f"  NonCrack: {n_noncrack} ({n_noncrack/n_total*100:.1f}%)")

    return splits
