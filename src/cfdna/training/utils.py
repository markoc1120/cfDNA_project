import logging
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .dataset import MatrixDataset, build_pairs, split_pairs_torch

logger = logging.getLogger(__name__)


class SelfTargetLoader:
    def __init__(self, loader):
        self._loader = loader

    def __iter__(self):
        for x_batch, _label in self._loader:
            yield x_batch, x_batch

    def __len__(self):
        return len(self._loader)


def load_train_pairs(pairs, transform_fn=None):
    matrixes = []
    for p in pairs:
        for path in (p['positive'], p['negative']):
            x = torch.from_numpy(np.load(path).astype(np.float32))

            if transform_fn is not None:
                x = transform_fn(x)
            matrixes.append(x)
    return torch.stack(matrixes, dim=0)


def get_dataloaders(
    output_dir: str,
    transform_fn=None,
    needs_standardization: bool = False,
    train_size: int = 80,
    valid_size: int = 10,
    batch_size: int = 32,
    seed: int = 42,
    suffix: str = 'downsampled',
    is_tiny: bool = False,
):
    pairs = build_pairs(output_dir, suffix=suffix)
    train_pairs, valid_pairs, test_pairs = split_pairs_torch(
        pairs, train_size=train_size, valid_size=valid_size, seed=seed
    )

    train_mean, train_std = None, None
    if needs_standardization:
        X = load_train_pairs(train_pairs, transform_fn)
        train_mean = X.mean().item()
        train_std = X.std(unbiased=True).item()

    DefaultMatrixDataset = partial(
        MatrixDataset,
        transform_fn=transform_fn,
        train_mean=train_mean,
        train_std=train_std,
    )
    train_ds = DefaultMatrixDataset(train_pairs)
    valid_ds = DefaultMatrixDataset(valid_pairs)
    test_ds = DefaultMatrixDataset(test_pairs)

    if is_tiny:
        train_pos_idx = [i for i, item in enumerate(train_ds.items) if item[1] == 1][:32]
        train_neg_idx = [i for i, item in enumerate(train_ds.items) if item[1] == 0][:32]
        tiny_train_ds = Subset(train_ds, train_pos_idx + train_neg_idx)

        valid_pos_idx = [i for i, item in enumerate(valid_ds.items) if item[1] == 1][:8]
        valid_neg_idx = [i for i, item in enumerate(valid_ds.items) if item[1] == 0][:8]
        tiny_valid_ds = Subset(valid_ds, valid_pos_idx + valid_neg_idx)

        tiny_train_loader = DataLoader(tiny_train_ds, batch_size=4, shuffle=True)
        tiny_valid_loader = DataLoader(tiny_valid_ds, batch_size=4)
        logger.info(f'sizes:\n {len(tiny_train_ds)} train\n {len(tiny_valid_ds)}')
        return tiny_train_loader, tiny_valid_loader, None

    DefaultDataLoader = partial(DataLoader, batch_size=batch_size)
    train_loader = DefaultDataLoader(train_ds, shuffle=True)
    valid_loader = DefaultDataLoader(valid_ds)
    test_loader = DefaultDataLoader(test_ds)

    logger.info(
        f'sizes:\n {len(train_ds)} train\n {len(valid_ds)} val\n {len(test_ds)} test\n train mean: {train_mean}\n train std: {train_std}'
    )
    return train_loader, valid_loader, test_loader
