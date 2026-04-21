import glob
import os
from collections import defaultdict
from typing import DefaultDict

import numpy as np
import torch
from torch.utils.data import Dataset, random_split


def extract_sid(path: str) -> str | None:
    base = os.path.basename(path)
    sid, sep, _ = base.partition('__')
    return sid if sep else None


def build_pairs(matrix_dir: str, suffix: str = 'downsampled', only_positive: bool = False):
    """
    Returns list of dicts:
    [{'sid': sid, 'positive': path, 'neg': path}, ...]
    """
    npy_files = glob.glob(f'{matrix_dir}*_{suffix}.npy')

    if only_positive:
        mapping: DefaultDict[str, dict[str, str]] = defaultdict(lambda: {'positive': ''})
    else:
        mapping: DefaultDict[str, dict[str, str]] = defaultdict(
            lambda: {'positive': '', 'negative': ''}
        )

    for p in npy_files:
        sid = extract_sid(p)
        if sid is None:
            continue

        is_neg = 'negative' in p.lower()
        if is_neg and only_positive:
            continue
        elif is_neg:
            mapping[sid]['negative'] = p
        else:
            mapping[sid]['positive'] = p

    result = []
    for sid, paths in mapping.items():
        entry = {'sid': sid, 'positive': paths['positive']}
        if not only_positive:
            entry.update({'negative': paths['negative']})
        result.append(entry)
    return result


def split_pairs_torch(pairs, train_size: int = 80, valid_size: int = 10, seed: int = 42):
    n = len(pairs)
    train_size = n * train_size // 100
    valid_size = n * valid_size // 100
    test_size = n - train_size - valid_size

    train_pairs, valid_pairs, test_pairs = random_split(
        pairs, [train_size, valid_size, test_size], generator=torch.Generator().manual_seed(seed)
    )
    return train_pairs, valid_pairs, test_pairs


class MatrixDataset(Dataset):
    def __init__(
        self,
        pairs,
        transform_fn=None,
        train_mean=None,
        train_std=None,
    ):
        self.items = []
        for p in pairs:
            self.items.append((p['positive'], 1, p['sid']))
            if 'negative' in p:
                self.items.append((p['negative'], 0, p['sid']))

        self.train_mean = train_mean
        self.train_std = train_std
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, y, _ = self.items[idx]
        x = torch.from_numpy(np.load(path)).float()  # [H,W]

        if self.transform_fn is not None:
            kwargs = {}
            if self.train_mean is not None and self.train_std is not None:
                kwargs['train_mean'] = self.train_mean
                kwargs['train_std'] = self.train_std
            x = self.transform_fn(x, **kwargs)

        x = x.unsqueeze(0)  # [H, W] -> [1, H, W]
        return x, torch.tensor(y, dtype=torch.float32)
