import os
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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

    entries = []
    for sid, disease in metadata_map.items():
        binary_label = 'Healthy' if disease == 'Healthy' else 'Cancerous'
        for dhs in dhs_files:
            fname = pattern.format(sid=sid, dhs=dhs)
            path = os.path.join(data_dir, fname)
            
            # skip if preprocessing marked this pair as low coverage
            matrix_base = os.path.join(data_dir, f'{sid}__{dhs}_sorted.npy')
            if os.path.exists(matrix_base + '.skip'):
                continue
                
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

            flat = vec.flatten()

            entries.append({
                "sample": sid,
                "dhs": dhs,
                "vector": flat,
                "disease": disease,
                "binary": binary_label,
            })

    if not entries:
        return None

#     all_vectors = StandardScaler().fit_transform(np.vstack([e['vector'] for e in entries]))
    all_vectors = np.vstack([e['vector'] for e in entries])
    
    loadings_df = None
    if stat != 'pfe':
        pca = PCA(n_components=2)
        pc_values = pca.fit_transform(all_vectors)
        expl_var = pca.explained_variance_ratio_
        loadings = pca.components_
        loadings_df = pd.DataFrame(
            loadings.T,
            columns=['PC1', 'PC2']
        )
        loadings_df.attrs['expl_var'] = expl_var
        pc1_var = expl_var[0]
        pc2_var = expl_var[1]

        for entry, (pc1, pc2) in zip(entries, pc_values):
            entry['pc1'] = pc1
            entry['pc2'] = pc2
            entry['pc1_var'] = pc1_var
            entry['pc2_var'] = pc2_var
    else:
        for entry in entries:
            val = entry['vector'][0]
            entry['pc1'] = val
            entry['pc2'] = val

    df = pd.DataFrame(entries)
    return df.pivot(index=['sample', 'binary', 'disease'], columns='dhs', values=['pc1', 'pc2']), loadings_df


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
        df, loadings_df = load_vectors(stat, metadata_map, result_dir, dhs_files)
        
        if df is not None:
            out_path = os.path.join(result_dir, f'feature_matrix_{stat}.parquet')
            df.to_parquet(out_path)
            print(f"Saved: {out_path}")

        if loadings_df is not None:
            loadings_out_path = os.path.join(result_dir, f'{stat}_pca_loadings.csv')
            loadings_df.to_csv(loadings_out_path)
            print(f"Saved: {loadings_out_path}")