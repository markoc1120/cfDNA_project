import glob
import re
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

SID_PATTERN = r'EE\d+'


def build_pairs(matrix_dir: str, suffix: str = 'downsampled'):
    """
    Returns list of dicts:
    [{'sid': sid, 'positive': path, 'neg': path}, ...]
    """
    npy_files = glob.glob(f'{matrix_dir}*_{suffix}.npy')

    mapping = defaultdict(lambda: {'negative': '', 'positive': ''})
    for p in npy_files:
        sid = re.search(SID_PATTERN, p)[0]
        if not sid:
            continue

        is_neg = 'negative' in p.lower()
        if is_neg:
            mapping[sid]['negative'] = p
        else:
            mapping[sid]['positive'] = p

    result = []
    for sid, paths in mapping.items():
        result.append({'sid': sid, 'positive': paths['positive'], 'negative': paths['negative']})
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
