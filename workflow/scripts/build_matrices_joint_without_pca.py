import json
import os

import numpy as np
import pandas as pd

STATS_KEYS = {
    'ifs': 'ifs_scores',
    'pfe': 'pfe_scores',
    'fdi': 'overlapping_fdi_scores',
    'vae': 'mu',
}


def parse_metadata(file_path: str) -> dict[str, str]:
    df = pd.read_csv(file_path, sep='\t', dtype=str)
    return {
        row['Patient']: 'healthy' if row['Patient type'].lower() == 'healthy' else 'cancer'
        for _, row in df.iterrows()
    }


def parse_sid_dhs(path: str, dhs_files: list[str]):
    basename = os.path.basename(path)
    sample, sep, rest = basename.partition('__')
    if not sep:
        return None, None
    for dhs in dhs_files:
        if rest.startswith(f'{dhs}_') or rest.startswith(f'{dhs}.'):
            return sample, dhs
    return None, None


def load_file(path: str, stat_name: str):
    if path.endswith('.npy'):
        return np.load(path).flatten()
    elif path.endswith('.npz'):
        data = np.load(path)
        key = STATS_KEYS.get(stat_name)
        return data[key].flatten()
    elif path.endswith('.txt'):
        return np.expand_dims(np.loadtxt(path), (0,)).flatten()
    return None


def load_vectors_flat(stat_name, input_files, metadata_map, dhs_files):
    entries = []
    for path in input_files:
        sid, dhs = parse_sid_dhs(path, dhs_files)
        if sid is None or sid not in metadata_map:
            continue

        binary_label = metadata_map[sid]
        # TODO: it is currently not the disease just the binary label
        # cut -f 19 DELFI_LUCAS_article_supp_table1.tsv seems to be the right value
        disease = binary_label

        vec = load_file(path, stat_name)
        if vec is None:
            continue

        entries.append(
            {
                'sample': sid,
                'dhs': dhs,
                'vector': vec,
                'disease': disease,
                'binary': binary_label,
            }
        )

    if not entries:
        return None

    # all per-(sample, dhs) vectors of one stat must share the same length
    dim = len(entries[0]['vector'])
    if not all(len(e['vector']) == dim for e in entries):
        raise ValueError(
            f'{stat_name}: per-(sample, DHS) vectors have inconsistent length'
        )

    records = []
    for e in entries:
        rec = {
            'sample': e['sample'],
            'binary': e['binary'],
            'disease': e['disease'],
            'dhs': e['dhs'],
        }
        for i, v in enumerate(e['vector']):
            rec[i] = float(v)
        records.append(rec)

    long_df = pd.DataFrame(records)
    long_df = long_df.set_index(['sample', 'binary', 'disease', 'dhs'])
    long_df.columns.name = 'dim_idx'
    wide = long_df.unstack('dhs')
    # MultiIndex columns from unstack are (dim_idx, dhs); flip to (dhs, dim_idx).
    wide = wide.reorder_levels(['dhs', 'dim_idx'], axis=1).sort_index(axis=1)
    return wide


if 'snakemake' in globals():
    cfg = snakemake.config
    metadata_path = cfg['data']['inference_metadata_path']
    final_dir = cfg['data']['final_matrices_dir']
    model = cfg['model']['name']

    dhs_files = list(snakemake.params.dhs_files)

    # dropping the same DHSs which were excluded in training
    dropped_dhs_path = getattr(snakemake.input, 'dropped_dhs', None)
    if dropped_dhs_path:
        with open(dropped_dhs_path) as f:
            dropped_set = set(json.load(f).get('dropped_dhs', []))
        if dropped_set:
            before = len(dhs_files)
            dhs_files = [d for d in dhs_files if d not in dropped_set]
            print(
                f'Filtered out {before - len(dhs_files)} DHS dropped during training '
                f'(via {dropped_dhs_path}); {len(dhs_files)} remain'
            )

    os.makedirs(final_dir, exist_ok=True)
    metadata_map = parse_metadata(metadata_path)

    input_map = {
        'lwps': 'lwps_inputs',
        'ocf': 'ocf_inputs',
        'fdi': 'fdi_inputs',
        'ifs': 'ifs_inputs',
        'pfe': 'pfe_inputs',
        model: 'model_inputs',
    }

    stat_inputs = {
        stat: getattr(snakemake.input, attr)
        for stat, attr in input_map.items()
        if hasattr(snakemake.input, attr)
    }

    for stat_name, input_files in stat_inputs.items():
        print(f'Processing: {stat_name}')
        df = load_vectors_flat(stat_name, input_files, metadata_map, dhs_files)

        if df is not None:
            out_path = os.path.join(final_dir, f'feature_matrix_{stat_name}.parquet')
            df.to_parquet(out_path)
            print(f'Saved: {out_path} (shape {df.shape})')
