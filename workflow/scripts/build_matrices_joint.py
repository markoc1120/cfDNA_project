import os
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

STATS_KEYS = {
    'ifs': 'ifs_scores',
    'pfe': 'pfe_scores',
    'fdi': 'overlapping_fdi_scores',
}


def parse_metadata(file_path: str, paper: str) -> dict:
    df = pd.read_csv(file_path)
    df = df[df.publication == paper]
    return dict(zip(df.sample_file_id, df.sample_disease))


def parse_sid_dhs(path: str):
    basename = os.path.basename(path)
    m = re.match(r'(.+?)__(.+?)_(?:downsampled_\w+\.\w+|score\.txt)$', basename)
    if m:
        return m.group(1), m.group(2)
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


def load_vectors(stat_name, input_files, metadata_map):
    entries = []
    for path in input_files:
        sid, dhs = parse_sid_dhs(path)
        if sid is None or sid not in metadata_map:
            continue

        disease = metadata_map[sid]
        binary_label = 'Healthy' if disease == 'Healthy' else 'Cancerous'

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
        return None, None

    all_vectors = np.vstack([e['vector'] for e in entries])

    loadings_df = None
    if stat_name not in ('pfe', 'cnn_score'):
        pca = PCA(n_components=2)
        pc_values = pca.fit_transform(all_vectors)
        expl_var = pca.explained_variance_ratio_
        loadings = pca.components_
        loadings_df = pd.DataFrame(loadings.T, columns=['PC1', 'PC2'])
        loadings_df.attrs['expl_var'] = expl_var

        for entry, (pc1, pc2) in zip(entries, pc_values):
            entry['pc1'] = pc1
            entry['pc2'] = pc2
            entry['pc1_var'] = expl_var[0]
            entry['pc2_var'] = expl_var[1]
    else:
        for entry in entries:
            val = entry['vector'][0] if entry['vector'].size > 0 else pd.NA
            entry['pc1'] = val
            entry['pc2'] = val

    df = pd.DataFrame(entries)
    return df.pivot(
        index=['sample', 'binary', 'disease'], columns='dhs', values=['pc1', 'pc2']
    ), loadings_df


if 'snakemake' in globals():
    cfg = snakemake.config
    metadata_path = cfg['data']['metadata_path']
    paper = cfg['data']['paper']
    final_dir = cfg['data']['final_matrices_dir']

    os.makedirs(final_dir, exist_ok=True)
    metadata_map = parse_metadata(metadata_path, paper)

    stat_inputs = {
        'lwps': snakemake.input.lwps_inputs,
        'ocf': snakemake.input.ocf_inputs,
        'fdi': snakemake.input.fdi_inputs,
        'ifs': snakemake.input.ifs_inputs,
        'pfe': snakemake.input.pfe_inputs,
        'cnn_score': snakemake.input.cnn_inputs,
    }

    for stat_name, input_files in stat_inputs.items():
        print(f'Processing: {stat_name}')
        df, loadings_df = load_vectors(stat_name, input_files, metadata_map)

        if df is not None:
            out_path = os.path.join(final_dir, f'feature_matrix_{stat_name}.parquet')
            df.to_parquet(out_path)
            print(f'Saved: {out_path}')

        if loadings_df is not None:
            loadings_path = os.path.join(final_dir, f'{stat_name}_pca_loadings.csv')
            loadings_df.to_csv(loadings_path)
            print(f'Saved: {loadings_path}')
