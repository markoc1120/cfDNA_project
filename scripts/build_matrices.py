import os
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


STATS = {
    'ocf': ('{sid}__{dhs}_sorted_ocf.npy', None),
    'lwps': ('{sid}__{dhs}_sorted_lwps.npy', None),
    'ifs': ('{sid}__{dhs}_sorted_ifs.npz', 'ifs_scores'),
    'pfe': ('{sid}__{dhs}_sorted_pfe.npz', 'pfe_scores'),
    'fdi': ('{sid}__{dhs}_sorted_fdi.npz', 'overlapping_fdi_scores'),
}


def parse_metadata(file_path: str, paper: str) -> dict:
    df = pd.read_csv(file_path)
    df = df[df.publication == paper]
    return dict(zip(df.sample_file_id, df.sample_disease))


def load_vectors(stat_name: str, metadata_map: dict, data_dir: str, dhs_files):
    pattern, key = STATS[stat_name]
    vectors, dhs_sites, disease_labels, binary_labels, sample_ids = [], [], [], [], []

    for sid, disease in metadata_map.items():
        binary_label = 'Healthy' if disease == 'Healthy' else 'Cancerous'
        for dhs in dhs_files:
            fname = pattern.format(sid=sid, dhs=dhs)
            path = os.path.join(data_dir, fname)
            try:
                if path.endswith('.npy'):
                    vec = np.load(path)
                elif path.endswith('.npz'):
                    data = np.load(path)
                    vec = data[key]
                else:
                    continue
            except FileNotFoundError:
                continue
            vectors.append(vec.flatten())
            disease_labels.append(disease)
            binary_labels.append(binary_label)
            dhs_sites.append(dhs)
            sample_ids.append(sid)

    if not vectors:
        return None

    matrix = np.vstack(vectors)
    matrix = StandardScaler().fit_transform(matrix)
    return {
        'matrix': matrix,
        'dhs_sites': np.array(dhs_sites),
        'disease_labels': np.array(disease_labels),
        'binary_labels': np.array(binary_labels),
        'sample_ids': np.array(sample_ids),
    }


def write_outputs(stat: str, payload: dict, out_dir: str):
    matrix_path = os.path.join(out_dir, f'feature_matrix_{stat}.npy')
    np.save(matrix_path, payload['matrix'])

    meta_path = os.path.join(out_dir, f'feature_matrix_{stat}_meta.npz')
    np.savez(meta_path,
             dhs_sites=payload['dhs_sites'],
             disease_labels=payload['disease_labels'],
             binary_labels=payload['binary_labels'],
             sample_ids=payload['sample_ids'])
    print(f"[ok] {stat}: matrix {payload['matrix'].shape} -> {matrix_path}; meta -> {meta_path}")


if 'snakemake' in globals():
    config_path = snakemake.input.config
    dhs_files = snakemake.params.dhs_files
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    result_dir = config['result_dir']
    os.makedirs(result_dir, exist_ok=True)
    metadata_map = parse_metadata(config['metadata_path'], config['paper'])
 
    for stat in STATS:
        print(f'Processing: {stat}')
        payload = load_vectors(stat, metadata_map, result_dir, dhs_files)
        if payload is None:
            print(f'[warn] Skipping {stat} — no data found.')
            continue
        write_outputs(stat, payload, result_dir)