import glob
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, random_split


def parse_sid_dhs(path: str, suffix: str) -> tuple[str, str, bool] | None:
    base = os.path.basename(path)
    sid, _, rest = base.partition('__')
    tail = f'{suffix}.npy'
    if not rest.endswith(tail):
        return None
    rest = rest[: -len(tail)]
    if rest.endswith('_negative'):
        return sid, rest[: -len('_negative')], True
    return sid, rest, False


def build_pairs(matrix_dir: str, suffix: str = '_downsampled', only_positive: bool = False):
    """
    Returns list of dicts, one per (sid, dhs):
    [{'sid': sid, 'dhs': dhs, 'positive': path, 'negative': path}, ...]
    """
    npy_files = glob.glob(f'{matrix_dir}*{suffix}.npy')

    if only_positive:
        mapping = defaultdict(lambda: {'positive': ''})
    else:
        mapping = defaultdict(lambda: {'positive': '', 'negative': ''})

    for p in npy_files:
        parsed = parse_sid_dhs(p, suffix)
        if parsed is None:
            continue
        sid, dhs, is_neg = parsed

        if is_neg and only_positive:
            continue
        elif is_neg:
            mapping[(sid, dhs)]['negative'] = p
        else:
            mapping[(sid, dhs)]['positive'] = p

    result = []
    for (sid, dhs), paths in mapping.items():
        entry = {'sid': sid, 'dhs': dhs, 'positive': paths['positive']}
        if not only_positive:
            entry['negative'] = paths['negative']
        result.append(entry)
    return result


def split_pairs_torch(pairs, train_size: int = 80, valid_size: int = 10, seed: int = 42):
    unique_sids = sorted({p['sid'] for p in pairs})
    n = len(unique_sids)
    train_n = n * train_size // 100
    valid_n = n * valid_size // 100
    test_n = n - train_n - valid_n

    train_sids, valid_sids, test_sids = random_split(
        unique_sids,
        [train_n, valid_n, test_n],
        generator=torch.Generator().manual_seed(seed),
    )
    train_set = set(train_sids)
    valid_set = set(valid_sids)
    test_set = set(test_sids)

    train_pairs = [p for p in pairs if p['sid'] in train_set]
    valid_pairs = [p for p in pairs if p['sid'] in valid_set]
    test_pairs = [p for p in pairs if p['sid'] in test_set]
    return train_pairs, valid_pairs, test_pairs


def read_coverage(path: str) -> float:
    if not os.path.exists(path):
        return 0.0
    with open(path) as f:
        return float(f.read().strip())


def apply_coverage_filter(
    pairs,
    dhs_files: list[str],
    cov_dir: str,
    coverage_threshold: float,
    max_sample_loss: float,
):
    unique_sids = sorted({p['sid'] for p in pairs})
    sid_to_row = {sid: i for i, sid in enumerate(unique_sids)}
    dhs_to_col = {dhs: j for j, dhs in enumerate(dhs_files)}

    cov = np.zeros((len(unique_sids), len(dhs_files)), dtype=np.float64)
    has_value = np.zeros_like(cov, dtype=bool)

    for p in pairs:
        sid, dhs = p['sid'], p['dhs']
        if dhs not in dhs_to_col:
            continue
        i, j = sid_to_row[sid], dhs_to_col[dhs]

        pos_cov = read_coverage(os.path.join(cov_dir, f'{sid}__{dhs}.cov.txt'))
        if 'negative' in p:
            neg_cov = read_coverage(os.path.join(cov_dir, f'{sid}__{dhs}_negative.cov.txt'))
            cov[i, j] = min(pos_cov, neg_cov)
        else:
            cov[i, j] = pos_cov
        has_value[i, j] = True

    # cells with no underlying pair are treated as below-threshold (count as missing)
    cov = np.where(has_value, cov, 0.0)

    n_samples = cov.shape[0]
    missing_fraction = (cov < coverage_threshold).sum(axis=0) / max(n_samples, 1)
    keep_dhs_mask = missing_fraction <= max_sample_loss
    kept_dhs = {dhs_files[j] for j in range(len(dhs_files)) if keep_dhs_mask[j]}
    dropped_dhs = [dhs_files[j] for j in range(len(dhs_files)) if not keep_dhs_mask[j]]

    if kept_dhs:
        cov_kept = cov[:, keep_dhs_mask]
        good_sample_mask = np.asarray((cov_kept >= coverage_threshold).all(axis=1))
    else:
        good_sample_mask = np.zeros(n_samples, dtype=bool)
    kept_sids = {unique_sids[i] for i in range(n_samples) if good_sample_mask[i]}
    dropped_sids = [unique_sids[i] for i in range(n_samples) if not good_sample_mask[i]]

    print(
        f'apply_coverage_filter: dropped {len(dropped_dhs)}/{len(dhs_files)} DHS '
        f'(missing>{max_sample_loss:.2f}): {dropped_dhs}; '
        f'dropped {len(dropped_sids)}/{n_samples} samples below {coverage_threshold:g}'
    )

    kept_pairs = [p for p in pairs if p['sid'] in kept_sids and p['dhs'] in kept_dhs]
    return kept_pairs, dropped_dhs, dropped_sids


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
