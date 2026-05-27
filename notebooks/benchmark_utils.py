import glob
import math
import os
import re

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

META_COLS = ['sample', 'binary', 'disease']
MAX_SAMPLE_LOSS = 0.2


def init_plotting(font_path: str = './AUPassata_Rg.ttf') -> None:
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'AU Passata'
    plt.rcParams['figure.figsize'] = (8, 3)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.5 * plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['savefig.dpi'] = 200
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['axes.linewidth'] = 1


def resolve_coverage(config: dict, min_cov_file: str) -> tuple[bool, int | None, str]:
    handling = config['preprocessing']['coverage_handling']
    is_downsampled = handling == 'downsample'
    threshold = None
    if is_downsampled:
        user_min_cov = config['preprocessing'].get('min_cov')
        if user_min_cov is not None:
            threshold = math.floor(float(user_min_cov))
        else:
            with open(min_cov_file) as f:
                threshold = math.floor(float(f.read().strip()))
    label = {
        'downsample': f'Coverage (downsampled to {threshold})' if is_downsampled else '',
        'normalize': 'CPM fraction (raw_count * 1e6 / sample_total)',
        'none': 'Coverage (raw counts)',
    }[handling]
    return is_downsampled, threshold, label


def discover_stats(final_matrices_dir: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(final_matrices_dir, 'feature_matrix_*.parquet')))
    return [
        os.path.basename(p).removeprefix('feature_matrix_').removesuffix('.parquet')
        for p in paths
    ]


def load_score_dfs(stats: list[str], final_matrices_dir: str) -> dict[str, pd.DataFrame]:
    score_dfs: dict[str, pd.DataFrame] = {}
    for stat in stats:
        path = os.path.join(final_matrices_dir, f'feature_matrix_{stat}.parquet')
        if os.path.exists(path):
            score_dfs[stat] = pd.read_parquet(path)
            print(f'{stat}: {score_dfs[stat].shape}')
        else:
            print(f'{stat}: not found')
    return score_dfs


def load_coverage_df(accessibility_dir: str) -> pd.DataFrame:
    cov_files = sorted(glob.glob(os.path.join(accessibility_dir, '*.cov.txt')))
    records = []
    for path in cov_files:
        basename = os.path.basename(path)
        match = re.match(r'(.+?)__(.+?)\.cov\.txt$', basename)
        if not match:
            continue
        sample, dhs = match.group(1), match.group(2)
        with open(path) as f:
            cov = int(f.read().strip())
        records.append({'sample': sample, 'dhs': dhs, 'coverage': cov})
    return pd.DataFrame(records).pivot(index='sample', columns='dhs', values='coverage')


def _dhs_values(df: pd.DataFrame, dhs_level):
    return df.columns.get_level_values(dhs_level)


def _filter_samples(dfs: dict[str, pd.DataFrame], samples: set) -> dict[str, pd.DataFrame]:
    return {
        s: df.loc[df.index.get_level_values('sample').isin(samples)]
        for s, df in dfs.items()
    }


def coverage_qc_filter(
    dfs: dict[str, pd.DataFrame],
    cov_df: pd.DataFrame,
    coverage_threshold: int,
    dhs_level,
    max_sample_loss: float = MAX_SAMPLE_LOSS,
) -> dict[str, pd.DataFrame]:
    sample_sets = [set(df.index.get_level_values('sample')) for df in dfs.values()]
    common_samples = set.intersection(*sample_sets) & set(cov_df.index)
    dfs = _filter_samples(dfs, common_samples)
    print(f'initial samples: {len(common_samples)}')

    cov_common = cov_df.loc[list(common_samples)]
    n_samples = cov_common.shape[0]
    missing_fraction = (cov_common < coverage_threshold).sum(axis=0) / n_samples
    kept_dhs = set(missing_fraction[missing_fraction <= max_sample_loss].index)
    print(f'dropped {cov_common.shape[1] - len(kept_dhs)} DHS columns globally')

    for stat, df in dfs.items():
        mask = _dhs_values(df, dhs_level).isin(kept_dhs)
        dfs[stat] = df.loc[:, mask]

    cov_kept = cov_common[sorted(kept_dhs)]
    good_samples = set(cov_kept.index[(cov_kept >= coverage_threshold).all(axis=1)])
    print(f'samples after coverage filter: {len(good_samples)}')

    return _filter_samples(dfs, good_samples)


def missing_score_filter(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    valid = None
    for df in dfs.values():
        s = set(df.index.get_level_values('sample')[~df.isna().any(axis=1)])
        valid = s if valid is None else (valid & s)
    print(f'samples after missing-score filter: {len(valid)}')
    return _filter_samples(dfs, valid)


def apply_filters(
    dfs: dict[str, pd.DataFrame],
    cov_df: pd.DataFrame,
    coverage_threshold: int | None,
    dhs_level,
    max_sample_loss: float = MAX_SAMPLE_LOSS,
) -> dict[str, pd.DataFrame]:
    if coverage_threshold is not None:
        dfs = coverage_qc_filter(dfs, cov_df, coverage_threshold, dhs_level, max_sample_loss)
    return missing_score_filter(dfs)
